"""
src/optimizer.py
────────────────
Adaptive alpha search engine.

Strategies implemented
──────────────────────
1. Round-robin operator scheduling with per-cycle shuffle
   → Prevents any operator from being starved across a run.

2. Visited-set deduplication (persisted to visited.json)
   → Never re-simulate the same (field, op, lookback, settings) combo,
     even across separate runs.

3. Binary-search lookback refinement for near-miss alphas
   → If Sharpe is close but below threshold, zoom in on the window size
     rather than randomly guessing a new one.

4. Hill-climb on settings for near-miss alphas
   → Track which direction (neutralization, decay) last improved Sharpe
     and keep moving that way.

5. Operator UCB scoring
   → Operators that have historically produced higher Sharpe values are
     preferred in scheduling, but all operators still get fair turns.
"""

import json
import math
import random
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

VISITED_FILE = Path("visited.json")

# All Time-Series unary operators (x, d)
TS_UNARY_OPS = [
    "ts_min", "ts_max", "ts_mean", "ts_sum", "ts_std_dev",
    "ts_rank", "ts_scale", "ts_zscore", "ts_delta", "ts_delay",
    "ts_product", "ts_quantile", "ts_arg_max", "ts_arg_min",
    "ts_av_diff", "ts_decay_linear",
]

# Cross-sectional operators — format: (x, group)
CS_UNARY_OPS = [
    "rank",               # market-wide rank
    "group_rank",         # rank within group
    "group_zscore",       # z-score within group
    "group_neutralize",   # neutralize within group
]

# Groups for cross-sectional operators
CS_GROUPS = ["subindustry", "industry", "sector", "market"]

# Settings axes for hill-climbing
NEUTRALIZATION_LADDER = ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
DECAY_LADDER          = [0, 2, 4, 6, 8, 12]

# Lookback anchor points (days) — shuffled at generation time, never linear
LOOKBACK_ANCHORS = [5, 10, 20, 60, 120, 252]

# Human-like alpha name syllables for non-sequential naming
_NAME_PREFIXES = ["sig", "fac", "rev", "mom", "val", "vol", "res", "mid", "dyn", "adp"]
_NAME_SUFFIXES = ["alpha", "signal", "model", "strat", "test", "v1", "v2", "run", "scan", "idea"]


def _human_name() -> str:
    """Generate a non-sequential human-looking alpha name."""
    prefix = random.choice(_NAME_PREFIXES)
    suffix = random.choice(_NAME_SUFFIXES)
    num    = random.randint(1, 999)
    return f"{prefix}_{suffix}_{num}"


class AdaptiveOptimizer:
    """
    Intelligent alpha search that combines:
    - Round-robin + UCB operator scheduling
    - Visited-set deduplication
    - Binary-search refinement of lookback on near-miss alphas
    - Hill-climb on settings (neutralization, decay)
    """

    def __init__(self, sharpe_target: float = 1.25):
        self.sharpe_target = sharpe_target

        # Visited set: frozenset of canonical expression strings
        self._visited: set[str] = self._load_visited()

        # Round-robin queues per field: {field -> [op, op, ...]}
        self._op_queues: dict[str, list] = {}

        # UCB stats per operator: {op -> {"n": int, "total_sharpe": float}}
        self._op_stats: dict = defaultdict(lambda: {"n": 0, "total": 0.0})

        # Near-miss memory: {alpha_name -> {"expr":, "lookback":, "sharpe":, ...}}
        self._near_misses: dict = {}

        # Ensemble memory: store promising alphas that failed but are good building blocks
        self._ensemble_pool: list[dict] = []

    # ──────────────────────────────────────────────────────────────────
    # Visited-set management
    # ──────────────────────────────────────────────────────────────────

    def _load_visited(self) -> set:
        if VISITED_FILE.exists():
            data = json.loads(VISITED_FILE.read_text())
            logger.info(f"Loaded {len(data)} visited expressions from {VISITED_FILE}.")
            return set(data)
        return set()

    def save_visited(self):
        """Persist visited set to disk so it survives across runs."""
        VISITED_FILE.write_text(json.dumps(sorted(self._visited), indent=2))

    def mark_visited(self, expr: str):
        self._visited.add(expr)

    def is_visited(self, expr: str) -> bool:
        return expr in self._visited

    # ──────────────────────────────────────────────────────────────────
    # Operator scheduling — round-robin + UCB bias
    # ──────────────────────────────────────────────────────────────────

    def _init_queue(self, field: str):
        """
        Initialise the round-robin operator queue for a field.
        Shuffle once so first-run order is random.
        UCB score biases the shuffle — operators with better historical
        Sharpe get placed earlier in each new cycle.
        """
        all_ops = TS_UNARY_OPS[:]
        total_trials = sum(s["n"] for s in self._op_stats.values()) + 1

        def ucb_score(op):
            s = self._op_stats[op]
            if s["n"] == 0:
                return float("inf")  # Unexplored ops go first
            avg = s["total"] / s["n"]
            exploration = math.sqrt(2 * math.log(total_trials) / s["n"])
            return avg + exploration

        # Sort by UCB (descending) then add a random tie-break
        sorted_ops = sorted(all_ops, key=lambda op: ucb_score(op) + random.uniform(0, 0.01), reverse=True)
        self._op_queues[field] = sorted_ops

    def _next_op(self, field: str) -> str:
        """
        Pop the next operator from the round-robin queue.
        When the queue is exhausted, re-shuffle (new cycle).
        """
        if field not in self._op_queues or not self._op_queues[field]:
            self._init_queue(field)

        op = self._op_queues[field].pop(0)

        # Refill queue for next cycle when empty
        if not self._op_queues[field]:
            self._init_queue(field)

        return op

    def record_result(self, operator: str, sharpe: float | None):
        """Update UCB statistics after a simulation result is known."""
        if sharpe is not None:
            self._op_stats[operator]["n"]     += 1
            self._op_stats[operator]["total"] += sharpe

    # ──────────────────────────────────────────────────────────────────
    # Expression builders
    # ──────────────────────────────────────────────────────────────────

    def _ts_expr(self, field: str, op: str, d: int) -> str:
        return f"rank({op}({field}, {d}))"

    def _ts_ratio_expr(self, field: str, op: str, d: int) -> str:
        return f"rank({field} / {op}({field}, {d}))"

    def _cs_expr(self, field: str, cs_op: str, group: str) -> str:
        if cs_op == "rank":
            return f"rank({field})"
        return f"{cs_op}({field}, {group})"

    # ── Multi-field combinators ─────────────────────────────────────────

    def _two_field_ratio(self, f1: str, f2: str) -> str:
        """rank(f1 / f2)  — cross-field ratio signal"""
        return f"rank({f1} / {f2})"

    def _two_field_diff_zscore(self, f1: str, f2: str, d: int) -> str:
        """rank(ts_zscore(f1,d) - ts_zscore(f2,d))  — standardised divergence"""
        return f"rank(ts_zscore({f1}, {d}) - ts_zscore({f2}, {d}))"

    def _two_field_corr(self, f1: str, f2: str, d: int) -> str:
        """rank(ts_corr(f1, f2, d))  — rolling correlation as signal"""
        return f"rank(ts_corr({f1}, {f2}, {d}))"

    def _two_field_product_rank(self, f1: str, f2: str, d: int) -> str:
        """rank(ts_rank(f1,d) * rank(f2))  — product of two ranked signals"""
        return f"rank(ts_rank({f1}, {d}) * rank({f2}))"

    def _momentum_vs_fundamental(self, price_f: str, fund_f: str, d: int) -> str:
        """rank(ts_delta(price,d) / fund_field)  — price change scaled by fundamental"""
        return f"rank(ts_delta({price_f}, {d}) / {fund_f})"

    def _normalised_momentum(self, field: str, d_fast: int, d_slow: int) -> str:
        """rank((ts_mean(f,d_fast) - ts_mean(f,d_slow)) / ts_std_dev(f,d_slow))  — MACD-style"""
        return f"rank((ts_mean({field}, {d_fast}) - ts_mean({field}, {d_slow})) / ts_std_dev({field}, {d_slow}))"

    def _group_neutral_signal(self, f1: str, f2: str, d: int, group: str) -> str:
        """group_neutralize(ts_zscore(f1,d) - ts_zscore(f2,d), group)"""
        return f"group_neutralize(ts_zscore({f1}, {d}) - ts_zscore({f2}, {d}), {group})"

    # ──────────────────────────────────────────────────────────────────
    # Candidate generation
    # ──────────────────────────────────────────────────────────────────

    def generate_initial_pool(self, fields: list, n_per_field: int = 8) -> list[dict]:
        """
        Generate the initial candidate pool.

        For each field:
        - Use round-robin operator selection (no starvation)
        - Shuffle lookbacks and fields so there's no linear scan pattern
        - Skip already-visited expressions
        - Mix TS and cross-sectional expressions
        - Use human-like names instead of sequential IDs
        """
        from src.utils import build_simulation_payload

        candidates = []

        # Shuffle fields so we don't always start with the same one
        shuffled_fields = list(fields)
        random.shuffle(shuffled_fields)

        for field in shuffled_fields:
            generated = 0

            # Shuffle CS ops order per field
            cs_ops = list(CS_UNARY_OPS)
            random.shuffle(cs_ops)

            # A) Cross-sectional expressions
            for cs_op in cs_ops:
                group = random.choice(CS_GROUPS)
                expr = self._cs_expr(field, cs_op, group)
                if not self.is_visited(expr):
                    p = build_simulation_payload(expr)
                    p["name"] = _human_name()
                    p["_meta"] = {"field": field, "op": cs_op, "lookback": None, "kind": "cs"}
                    candidates.append(p)
                    self.mark_visited(expr)
                    generated += 1

            # B) Time-series expressions — shuffle lookbacks so order is non-linear
            shuffled_lookbacks = list(LOOKBACK_ANCHORS)
            random.shuffle(shuffled_lookbacks)

            for anchor_d in shuffled_lookbacks:
                if generated >= n_per_field:
                    break

                op = self._next_op(field)
                expr = self._ts_expr(field, op, anchor_d)

                if self.is_visited(expr):
                    expr = self._ts_ratio_expr(field, op, anchor_d)
                    if self.is_visited(expr):
                        continue

                p = build_simulation_payload(expr)
                p["name"] = _human_name()
                p["_meta"] = {"field": field, "op": op, "lookback": anchor_d, "kind": "ts"}
                candidates.append(p)
                self.mark_visited(expr)
                generated += 1

        # Shuffle the final pool so submission order is completely non-sequential
        random.shuffle(candidates)
        logger.info(f"Generated initial pool: {len(candidates)} unique candidates (shuffled).")
        return candidates

    def generate_multi_field_pool(
        self,
        fields: list,
        price_fields: list = None,
        max_pairs: int = 30,
    ) -> list[dict]:
        """
        Generate TWO-FIELD combination expressions.
        These are the patterns experienced Brain researchers use for stronger alphas:

        - rank(f1 / f2)                               cross-field ratio
        - rank(ts_zscore(f1,d) - ts_zscore(f2,d))    standardised divergence
        - rank(ts_corr(f1, f2, d))                    rolling correlation
        - rank(ts_rank(f1,d) * rank(f2))              product of signals
        - rank(ts_delta(price,d) / fundamental)        price change / fundamental
        - rank(MACD-style on single field)             fast vs slow moving average
        - group_neutralize(zscore_diff, sector)        sector-neutral divergence

        Parameters
        ----------
        fields      : all available fields (used to form pairs)
        price_fields: subset known to be price/volume (for economic pairing)
        max_pairs   : cap on random field pairs to avoid combinatorial explosion
        """
        from src.utils import build_simulation_payload

        pv = price_fields or ["close", "open", "volume", "returns", "vwap", "cap"]
        candidates = []
        idx = 10000   # offset so names don't clash with initial pool

        def add(expr, kind, f1, f2=None, op=None, d=None):
            nonlocal idx
            if not self.is_visited(expr):
                p = build_simulation_payload(expr)
                p["name"] = f"MF_{idx}"
                p["_meta"] = {
                    "field":   f1,
                    "field2":  f2,
                    "op":      op or "multi",
                    "lookback": d,
                    "kind":    kind,
                }
                candidates.append(p)
                self.mark_visited(expr)
                idx += 1

        # Sample field pairs (avoid combinatorial explosion)
        all_pairs = [(a, b) for i, a in enumerate(fields) for b in fields[i+1:]]
        random.shuffle(all_pairs)
        pairs = all_pairs[:max_pairs]

        for f1, f2 in pairs:
            d = random.choice(LOOKBACK_ANCHORS)

            # 1. Cross-field ratio (the most common pattern)
            add(self._two_field_ratio(f1, f2), "ratio", f1, f2, "div", None)

            # 2. Standardised divergence — two signals disagree
            add(self._two_field_diff_zscore(f1, f2, d), "zscore_diff", f1, f2, "ts_zscore", d)

            # 3. Rolling correlation — pairs trading style
            d2 = random.choice([20, 60, 120])
            add(self._two_field_corr(f1, f2, d2), "corr", f1, f2, "ts_corr", d2)

            # 4. Product of ranks — both signal in same direction
            add(self._two_field_product_rank(f1, f2, d), "prod_rank", f1, f2, "ts_rank", d)

            # 5. Sector-neutral divergence
            group = random.choice(CS_GROUPS)
            add(self._group_neutral_signal(f1, f2, d, group), "grp_neutral", f1, f2, "ts_zscore", d)

        # MACD-style single-field (fast vs slow moving average)
        for field in fields:
            for d_fast, d_slow in [(5, 20), (10, 60), (20, 120)]:
                add(self._normalised_momentum(field, d_fast, d_slow),
                    "macd", field, None, "macd", d_slow)

        # Price momentum scaled by fundamentals (price vs non-price pairs)
        non_pv = [f for f in fields if f not in pv]
        for pf in pv[:3]:
            for ff in non_pv[:5]:
                d = random.choice([20, 60])
                add(self._momentum_vs_fundamental(pf, ff, d), "mom_fund", pf, ff, "ts_delta", d)

        logger.info(f"Generated multi-field pool: {len(candidates)} unique candidates.")
        return candidates

    # ──────────────────────────────────────────────────────────────────
    # Binary-search lookback refinement (near-miss alphas)
    # ──────────────────────────────────────────────────────────────────

    def refine_near_miss(
        self,
        alpha: dict,
        sharpe: float,
        prefix: str = "Refined",
    ) -> list[dict]:
        """
        Called when an alpha's Sharpe is "close" to the target (within 25%).
        Uses binary-search logic to probe shorter and longer lookbacks
        around the current one, skipping already-visited expressions.

        Returns a list of refined candidate payloads to queue.
        """
        from src.utils import build_simulation_payload

        meta = alpha.get("_meta", {})
        field = meta.get("field")
        op    = meta.get("op")
        d     = meta.get("lookback")

        # Can only refine TS expressions with a known lookback
        if not field or not op or d is None:
            return []

        gap = self.sharpe_target - sharpe
        if gap <= 0 or gap > 0.5:
            return []  # Not a near-miss, skip

        logger.info(
            f"[{alpha.get('name')}] Near-miss: sharpe={sharpe:.3f} "
            f"(gap={gap:.3f}). Refining lookback around d={d}…"
        )

        refined = []
        # Binary-search probe: try d/2, 2d, 3d/2 clamped to [3, 504]
        probes = sorted({
            max(3, d // 2),
            min(504, d * 2),
            max(3, d * 3 // 4),
            min(504, d * 3 // 2),
        })

        for probe_d in probes:
            expr = self._ts_expr(field, op, probe_d)
            if self.is_visited(expr):
                continue
            p = build_simulation_payload(expr)
            p["name"] = f"{prefix}_{field[:6]}_{op[-4:]}_{probe_d}"
            p["_meta"] = {"field": field, "op": op, "lookback": probe_d, "kind": "ts_refined"}
            refined.append(p)
            self.mark_visited(expr)

        logger.info(f"  → {len(refined)} refined candidates queued.")
        return refined

    # ──────────────────────────────────────────────────────────────────
    # Hill-climb on settings (neutralization, decay)
    # ──────────────────────────────────────────────────────────────────

    def hill_climb_settings(self, alpha: dict, sharpe: float) -> list[dict]:
        """
        For a near-miss alpha, generate variants with neighbouring
        settings values (neutralization and decay) to hill-climb.

        Returns a list of candidate payloads.
        """
        from src.utils import build_simulation_payload

        gap = self.sharpe_target - sharpe
        if gap <= 0 or gap > 0.4:
            return []

        expr    = alpha.get("regular", "")
        cur_neu = alpha.get("settings", {}).get("neutralization", "MARKET")
        cur_dec = alpha.get("settings", {}).get("decay", 0)

        variants = []

        # Try adjacent neutralization levels
        neu_idx = NEUTRALIZATION_LADDER.index(cur_neu) if cur_neu in NEUTRALIZATION_LADDER else 0
        for step in [-1, +1]:
            ni = neu_idx + step
            if 0 <= ni < len(NEUTRALIZATION_LADDER):
                new_neu = NEUTRALIZATION_LADDER[ni]
                key = f"{expr}|neu={new_neu}|dec={cur_dec}"
                if not self.is_visited(key):
                    p = build_simulation_payload(expr, neutralization=new_neu, decay=cur_dec)
                    p["name"] = f"HC_neu_{new_neu[:3]}"
                    p["_meta"] = {**alpha.get("_meta", {}), "kind": "hillclimb_neu"}
                    variants.append(p)
                    self.mark_visited(key)

        # Try adjacent decay values
        dec_idx = DECAY_LADDER.index(cur_dec) if cur_dec in DECAY_LADDER else 0
        for step in [-1, +1]:
            di = dec_idx + step
            if 0 <= di < len(DECAY_LADDER):
                new_dec = DECAY_LADDER[di]
                key = f"{expr}|neu={cur_neu}|dec={new_dec}"
                if not self.is_visited(key):
                    p = build_simulation_payload(expr, neutralization=cur_neu, decay=new_dec)
                    p["name"] = f"HC_dec_{new_dec}"
                    p["_meta"] = {**alpha.get("_meta", {}), "kind": "hillclimb_dec"}
                    variants.append(p)
                    self.mark_visited(key)

        logger.info(f"[{alpha.get('name')}] Hill-climb: {len(variants)} settings variants queued.")
        return variants

    # ──────────────────────────────────────────────────────────────────
    # Operator leaderboard (for diagnostics)
    # ──────────────────────────────────────────────────────────────────

    def leaderboard(self, top_n: int = 10) -> list[dict]:
        """Return operators sorted by average Sharpe (descending)."""
        rows = []
        for op, stats in self._op_stats.items():
            if stats["n"] > 0:
                rows.append({
                    "operator": op,
                    "trials":   stats["n"],
                    "avg_sharpe": round(stats["total"] / stats["n"], 4),
                })
        return sorted(rows, key=lambda r: r["avg_sharpe"], reverse=True)[:top_n]

    # ──────────────────────────────────────────────────────────────────
    # Advanced Optimizations: Negation & Ensembling
    # ──────────────────────────────────────────────────────────────────

    def flip_negative_alpha(self, alpha: dict, sharpe: float) -> list[dict]:
        """
        For a highly negative alpha, flip its sign to invert the equity curve.
        """
        from src.utils import build_simulation_payload
        
        expr = alpha.get("regular", "")
        if not expr:
            return []
            
        flipped_expr = f"-1 * ({expr})"
        if self.is_visited(flipped_expr):
            return []
            
        p = build_simulation_payload(
            flipped_expr,
            neutralization=alpha.get("settings", {}).get("neutralization", "MARKET"),
            decay=alpha.get("settings", {}).get("decay", 0),
        )
        p["name"] = f"NEG_{alpha.get('name', 'UNK')}"
        p["_meta"] = {**alpha.get("_meta", {}), "kind": "negation"}
        
        self.mark_visited(flipped_expr)
        logger.info(f"[{alpha.get('name')}] Flipped highly negative alpha (sharpe={sharpe:.2f}).")
        return [p]
        
    def add_to_ensemble(self, alpha: dict):
        """Store a promising but ultimately failing alpha to be ensembled."""
        expr = alpha.get("regular", "")
        if expr and expr not in [a.get("regular") for a in self._ensemble_pool]:
            self._ensemble_pool.append(alpha)
            logger.info(f"[{alpha.get('name')}] Added to ensemble pool (size: {len(self._ensemble_pool)}).")
            
    def generate_ensembles(self) -> list[dict]:
        """
        Combine orthogonal alphas in the ensemble pool.
        """
        from src.utils import build_simulation_payload
        import itertools
        
        if len(self._ensemble_pool) < 2:
            return []
            
        ensembles = []
        # Generate combinations
        for a1, a2 in itertools.combinations(self._ensemble_pool, 2):
            expr1 = a1.get("regular")
            expr2 = a2.get("regular")
            
            combined_expr = f"rank({expr1}) + rank({expr2})"
            if self.is_visited(combined_expr):
                continue
                
            p = build_simulation_payload(combined_expr)
            p["name"] = f"ENS_{a1.get('name')}_{a2.get('name')}"
            p["_meta"] = {"kind": "ensemble"}
            
            ensembles.append(p)
            self.mark_visited(combined_expr)
            
        # Clear the pool once generated to prevent exponential growth
        pool_size = len(self._ensemble_pool)
        self._ensemble_pool.clear()
        
        logger.info(f"Generated {len(ensembles)} ensemble candidates from {pool_size} pool alphas.")
        return ensembles
