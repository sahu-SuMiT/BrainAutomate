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
import threading
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

VISITED_FILE       = Path("visited.json")
ENSEMBLE_POOL_FILE = Path("ensemble_pool.json")

# All Time-Series unary operators that produce ECONOMIC SIGNALS
# Standard form: rank(ts_OP(field, d))
# Rule: only ops whose output is a meaningful relative-value or momentum signal.
#       Data-cleaning tools (ts_backfill, ts_count_nans) are intentionally excluded.
TS_UNARY_OPS = [
    # Core statistical — proven high-Sharpe patterns on Brain
    "ts_min", "ts_max", "ts_mean", "ts_sum", "ts_std_dev",
    # Rank / scaling — most common in top-performing alphas
    "ts_rank", "ts_scale", "ts_zscore",
    # Momentum / change
    "ts_delta", "ts_delay", "ts_av_diff",
    # Advanced aggregators
    "ts_product", "ts_quantile", "ts_decay_linear",
    # Ordinal / position
    "ts_arg_max", "ts_arg_min",
]

# Special operators with non-standard signatures (handled in generate_special_ops_pool).
# These ARE valid alpha signals but need custom expression templates.
#   kth_element(x, d, k)  — percentile lookup (k=1: min, k=d//2: median)
#   hump(x, 0.01)         — noise-reduced signal wrapper (reduces turnover)
#   ts_step(t)            — time-trend correlation (monotonic day counter)
TS_SPECIAL_OPS = [
    "kth_element",  # percentile — genuinely useful alpha signal
    "hump",         # noise filter — reduces false signals from noisy data
    "ts_step",      # time-trend  — correlating a field with time = recency bias signal
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


# ──────────────────────────────────────────────────────────────────
# Pareto Front — multi-objective optimization tracker
# ──────────────────────────────────────────────────────────────────

class ParetoFront:
    """
    Tracks the non-dominated set of (Sharpe, Fitness, Margin) results.

    A result r1 dominates r2 if:
        r1.sharpe >= r2.sharpe AND
        r1.fitness >= r2.fitness AND
        r1.margin >= r2.margin  AND at least one strictly greater.

    Only non-dominated results are kept on the frontier.
    Use this to identify which alpha characteristics are
    genuinely best across all three objectives simultaneously.
    """

    def __init__(self):
        self._front: list[dict] = []   # list of {name, sharpe, fitness, margin, turnover, _meta}

    def update(self, name: str, sharpe: float, fitness: float,
               margin: float, turnover: float = 0.0, meta: dict = None):
        """Add a new result and prune the front."""
        if sharpe is None or fitness is None or margin is None:
            return
        new = {"name": name, "sharpe": sharpe, "fitness": fitness,
               "margin": margin, "turnover": turnover, "_meta": meta or {}}
        # Check if new result is dominated by any existing front member
        for existing in self._front:
            if (existing["sharpe"]  >= new["sharpe"]  and
                existing["fitness"] >= new["fitness"] and
                existing["margin"]  >= new["margin"]):
                return   # dominated — don't add
        # Remove any existing members dominated by the new result
        self._front = [
            e for e in self._front
            if not (new["sharpe"]  >= e["sharpe"]  and
                    new["fitness"] >= e["fitness"] and
                    new["margin"]  >= e["margin"])
        ]
        self._front.append(new)
        logger.debug(f"[Pareto] Front updated. Size={len(self._front)}. Added {name} "
                     f"(sharpe={sharpe:.3f}, fitness={fitness:.3f}, margin={margin:.4f})")

    def scalarized_score(self, sharpe: float, fitness: float, margin: float,
                         w_sharpe: float = 0.5, w_fitness: float = 0.3,
                         w_margin: float = 0.2) -> float:
        """Weighted combination of objectives for single-number comparison."""
        margin_score = min(1.0, max(0.0, margin * 100))  # scale margin to [0, 1] range
        return w_sharpe * sharpe + w_fitness * fitness + w_margin * margin_score

    def top(self, n: int = 5) -> list[dict]:
        """Return top-N Pareto-front members by scalarized score."""
        return sorted(
            self._front,
            key=lambda r: self.scalarized_score(r["sharpe"], r["fitness"], r["margin"]),
            reverse=True
        )[:n]

    def __len__(self):
        return len(self._front)


class AdaptiveOptimizer:
    """
    Intelligent alpha search combining:
    - Round-robin + UCB operator scheduling
    - Visited-set deduplication (persisted)
    - Binary-search lookback refinement
    - Hill-climb on settings (neutralization, decay)
    - Genetic expression evolution (mutation + crossover)
    - Correlation-aware ensemble generation
    - Multi-objective Pareto front tracking
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

        # Ensemble pool: promising near-miss alphas to combine
        self._ensemble_pool: list[dict] = []

        # Pareto front — multi-objective tracker (Sharpe + Fitness + Margin)
        self.pareto_front = ParetoFront()

        # Thread safety for visited set
        self._visited_lock = threading.Lock()

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
        with self._visited_lock:
            VISITED_FILE.write_text(json.dumps(sorted(self._visited), indent=2))

    def load_ensemble_pool(self):
        """Reload the ensemble pool from the previous run so good building blocks survive restarts."""
        if ENSEMBLE_POOL_FILE.exists():
            try:
                data = json.loads(ENSEMBLE_POOL_FILE.read_text())
                self._ensemble_pool = data
                logger.info(f"Loaded {len(self._ensemble_pool)} alphas into ensemble pool from previous run.")
            except Exception as e:
                logger.warning(f"Could not load ensemble pool: {e}")

        # Always seed from historical top performers so restarts build on best past data
        self.seed_from_history(top_n=10, min_sharpe=0.70)

    def seed_from_history(self, top_n: int = 10, min_sharpe: float = 0.70):
        """
        Scan all historical JSONL results and seed the ensemble pool with the
        top-N best-performing alphas (by Sharpe). This ensures every new run
        immediately tries to ensemble the best expressions ever discovered,
        rather than rediscovering them from scratch.

        Parameters
        ----------
        top_n     : how many top historical alphas to seed
        min_sharpe: only seed alphas above this Sharpe threshold
        """
        results_dir = Path("results")
        if not results_dir.exists():
            return

        # Collect all results with a simulation_id and expression info
        # We store simulation_id as a proxy for the alpha expression
        candidates = []
        existing_names = {a.get("name") for a in self._ensemble_pool}

        for path in sorted(results_dir.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    sharpe = row.get("sharpe")
                    sim_id = row.get("simulation_id")
                    name   = row.get("alpha_name", "")
                    status = row.get("status", "")

                    # Only use results that actually completed and cleared our threshold
                    if (sharpe is not None and sharpe >= min_sharpe
                            and sim_id and status in ("TEST_FAIL", "QUALIFIES", "SUBMITTED")
                            and name not in existing_names):
                        candidates.append({
                            "sharpe":   sharpe,
                            "name":     name,
                            "sim_id":   sim_id,
                            "operator": row.get("operator", "rank"),
                            "field":    row.get("field") or row.get("alpha_name", ""),
                            "field_category": row.get("field_category", "price_volume"),
                        })
                except Exception:
                    continue

        # Sort by Sharpe descending, take top_n
        candidates.sort(key=lambda r: r["sharpe"], reverse=True)
        top_seeds = candidates[:top_n]

        seeded = 0
        for seed in top_seeds:
            # Build a minimal alpha-like dict the ensemble generator can use
            # We use the sim_id as the "regular" expression proxy for dedup
            proxy_expr = f"__historical__{seed['sim_id']}"
            if proxy_expr not in [a.get("regular", "") for a in self._ensemble_pool]:
                self._ensemble_pool.append({
                    "name":    seed["name"],
                    "regular": proxy_expr,
                    "_meta":   {
                        "op":             seed["operator"],
                        "field":          seed["field"],
                        "field_category": seed["field_category"],
                        "kind":           "historical_seed",
                    },
                    "_seed_sharpe": seed["sharpe"],
                })
                existing_names.add(seed["name"])
                seeded += 1

        if seeded > 0:
            logger.info(
                f"Seeded ensemble pool with {seeded} historical top performers "
                f"(min_sharpe={min_sharpe}, pool size now={len(self._ensemble_pool)})."
            )
            self.save_ensemble_pool()

    def save_ensemble_pool(self):
        """Persist the current ensemble pool to disk."""
        try:
            ENSEMBLE_POOL_FILE.write_text(json.dumps(self._ensemble_pool, indent=2))
        except Exception as e:
            logger.warning(f"Could not save ensemble pool: {e}")


    def mark_visited(self, expr: str):
        with self._visited_lock:
            self._visited.add(expr)

    def is_visited(self, expr: str) -> bool:
        with self._visited_lock:
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
        """rank(ts_corr(f1, f2, d))  — rolling Pearson correlation signal"""
        return f"rank(ts_corr({f1}, {f2}, {d}))"

    def _two_field_covariance(self, f1: str, f2: str, d: int) -> str:
        """rank(ts_covariance(f1, f2, d))  — rolling covariance signal"""
        return f"rank(ts_covariance({f1}, {f2}, {d}))"

    def _two_field_regress_residual(self, f1: str, f2: str, d: int) -> str:
        """
        rank(ts_regress(f1, f2, d, lag=0, retype=0))  — regression residual.
        retype=0 returns the intercept; retype=4 returns residuals (alpha signal).
        """
        return f"rank(ts_regress({f1}, {f2}, {d}, 0, 4))"

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

    def generate_initial_pool(
        self,
        fields: list,
        n_per_field: int = 6,
        field_category_map: dict = None,
        quota_map: dict = None,
    ) -> list[dict]:
        """
        Generate the initial candidate pool.

        For each field, generates cross-sectional (CS) expressions first,
        then time-series (TS) expressions up to the field's quota.

        Parameters
        ----------
        n_per_field : int
            Default number of TS candidates per field (used when quota_map is None).
        field_category_map : dict
            {field_id: category_str} — tags _meta[field_category] for the Learner.
        quota_map : dict, optional
            {field_id: n} — per-field simulation budget from learner.field_quota_map().
            When provided, proven fields (historically high Sharpe) get more candidates.
            Overrides n_per_field on a per-field basis.

        Goal alignment
        ──────────────
        quota_map channels more simulations into fields that have historically
        produced high Sharpe — getting us closer to 1.25 faster.
        """
        from src.utils import build_simulation_payload

        cat_map = field_category_map or {}
        candidates = []

        # Shuffle fields so we don't always start with the same one
        shuffled_fields = list(fields)
        random.shuffle(shuffled_fields)

        for field in shuffled_fields:
            generated = 0
            field_cat = cat_map.get(field, "price_volume")

            # Per-field quota: use quota_map if available, else flat n_per_field
            field_quota = (quota_map or {}).get(field, n_per_field)

            # Shuffle CS ops order per field
            cs_ops = list(CS_UNARY_OPS)
            random.shuffle(cs_ops)

            # A) Cross-sectional expressions (always 1-2 per field, count toward quota)
            for cs_op in cs_ops:
                if generated >= max(2, field_quota // 3):
                    break  # CS limited to ~1/3 of quota; rest goes to TS
                group = random.choice(CS_GROUPS)
                expr = self._cs_expr(field, cs_op, group)
                if not self.is_visited(expr):
                    p = build_simulation_payload(expr)
                    p["name"] = _human_name()
                    p["_meta"] = {"field": field, "op": cs_op, "lookback": None,
                                  "kind": "cs", "field_category": field_cat}
                    candidates.append(p)
                    self.mark_visited(expr)
                    generated += 1

            # B) Time-series expressions — up to field_quota
            shuffled_lookbacks = list(LOOKBACK_ANCHORS)
            random.shuffle(shuffled_lookbacks)

            for anchor_d in shuffled_lookbacks:
                if generated >= field_quota:
                    break

                op = self._next_op(field)
                expr = self._ts_expr(field, op, anchor_d)

                if self.is_visited(expr):
                    expr = self._ts_ratio_expr(field, op, anchor_d)
                    if self.is_visited(expr):
                        continue

                p = build_simulation_payload(expr)
                p["name"] = _human_name()
                p["_meta"] = {"field": field, "op": op, "lookback": anchor_d,
                              "kind": "ts", "field_category": field_cat}
                candidates.append(p)
                self.mark_visited(expr)
                generated += 1

        # Shuffle the final pool so submission order is non-sequential
        random.shuffle(candidates)
        if quota_map:
            proven = sum(1 for f in fields if (quota_map.get(f, n_per_field) >= n_per_field * 3))
            logger.info(
                f"Generated initial pool: {len(candidates)} unique candidates "
                f"({proven} proven fields × {n_per_field*3} quota, rest × {n_per_field})."
            )
        else:
            logger.info(f"Generated initial pool: {len(candidates)} unique candidates (flat quota={n_per_field}).")
        return candidates

    def generate_special_ops_pool(
        self,
        fields: list,
        field_category_map: dict = None,
    ) -> list[dict]:
        """
        Generate candidates for Brain operators with NON-STANDARD signatures.

        Operators covered (all produce genuine alpha signals):
        ─────────────────────────────────────────────────────
        kth_element(x, d, k)   — percentile lookup over d days
                                  k=1: recent minimum; k=d//2: median; k=d: maximum
                                  Useful for ranking stocks by their recent range position

        hump(x, hump=0.01)     — noise filter: smooths out tiny daily changes
                                  Useful with fundamental data to suppress noise
                                  Pattern: rank(ts_mean(hump(field, 0.01), d))

        ts_step(t)             — monotonic day counter (1, 2, 3, ...)
                                  ts_corr(field, ts_step(1), d) = is this field trending up over time?
                                  Useful for detecting persistent trends vs mean-reversion

        NOT included (data quality tools, not alpha signals):
          days_from_last_change — measures data staleness, not stock quality
          ts_count_nans         — counts missing values, not meaningful as alpha
          ts_backfill           — data imputation utility, output is same as input
        """
        from src.utils import build_simulation_payload

        candidates = []
        field_category_map = field_category_map or {}

        def add(expr, kind, field, op, lookback=None):
            if not self.is_visited(expr):
                p = build_simulation_payload(expr)
                p["name"]  = _human_name()
                p["_meta"] = {
                    "field":          field,
                    "op":             op,
                    "kind":           kind,
                    "field_category": field_category_map.get(field, "price_volume"),
                    "lookback":       lookback,
                }
                candidates.append(p)
                self.mark_visited(expr)

        # Sample fields — prefer non-price fields for special ops
        # (fundamental/analyst fields benefit most from noise filtering)
        non_price = [f for f in fields
                     if field_category_map.get(f, "price_volume") != "price_volume"]
        price     = [f for f in fields
                     if field_category_map.get(f, "price_volume") == "price_volume"]
        # 70% non-price, 30% price for special ops sampling
        sample_np = random.sample(non_price, min(8, len(non_price)))
        sample_pv = random.sample(price,     min(4, len(price)))
        sample_fields = sample_np + sample_pv

        for field in sample_fields:
            # hump limits small changes to 0 — reduces turnover
            for hump_val in [0.01, 0.05]:
                for d in [20, 60]:
                    expr = f"rank(ts_mean(hump({field}, {hump_val}), {d}))"
                    add(expr, "hump_mean", field, "hump")

            # ── kth_element: percentile lookup ───────────────────────────────
            # k=1 → minimum; k=d//2 → median; k=d → maximum
            for d in [20, 60, 120]:
                for k in [1, d // 4, d // 2]:
                    k = max(1, k)
                    expr = f"rank(kth_element({field}, {d}, {k}))"
                    add(expr, "kth_element", field, "kth_element")

        # ── ts_step: time-decay weight ────────────────────────────────────────
        # ts_step(t) increments each day — useful for recency weighting
        for field in random.sample(fields, min(4, len(fields))):
            for d in [60, 120]:
                expr = f"rank(ts_corr({field}, ts_step(1), {d}))"
                add(expr, "time_corr", field, "ts_step")

        random.shuffle(candidates)
        logger.info(f"Generated {len(candidates)} special-op candidates "
                    f"(days_from_last_change, hump, kth_element, ts_step).")
        return candidates

    def generate_multi_field_pool(
        self,
        fields: list,
        price_fields: list = None,
        max_pairs: int = 30,
        field_category_map: dict = None,
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
        field_category_map : {field_id: category_str} — for tagging _meta
        """
        from src.utils import build_simulation_payload

        pv     = price_fields or ["close", "open", "volume", "returns", "vwap", "cap"]
        cat_map = field_category_map or {}
        candidates = []
        idx = 10000   # offset so names don't clash with initial pool

        def add(expr, kind, f1, f2=None, op=None, d=None):
            nonlocal idx
            if not self.is_visited(expr):
                p = build_simulation_payload(expr)
                p["name"] = f"MF_{idx}"
                # Use f1's category; for cross-field combos tag as "mixed" if different
                cat1 = cat_map.get(f1, "price_volume")
                cat2 = cat_map.get(f2, "price_volume") if f2 else cat1
                field_cat = cat1 if cat1 == cat2 else "mixed"
                p["_meta"] = {
                    "field":          f1,
                    "field2":         f2,
                    "op":             op or "multi",
                    "lookback":       d,
                    "kind":           kind,
                    "field_category": field_cat,
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

            # 4. Rolling covariance — complements correlation (scale-sensitive)
            add(self._two_field_covariance(f1, f2, d2), "cov", f1, f2, "ts_covariance", d2)

            # 5. Regression residual — pure alpha after removing f2 from f1
            d3 = random.choice([60, 120])
            add(self._two_field_regress_residual(f1, f2, d3), "regress", f1, f2, "ts_regress", d3)

            # 6. Product of ranks — both signals agree in same direction
            add(self._two_field_product_rank(f1, f2, d), "prod_rank", f1, f2, "ts_rank", d)

            # 7. Sector-neutral divergence
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
            self.save_ensemble_pool()   # persist immediately so restarts don't lose this
            
    def generate_ensembles(self) -> list[dict]:
        """
        Correlation-aware ensemble generation.

        Instead of combining any two alphas, we prefer ORTHOGONAL pairs:
        - Low-turnover (value/fundamental signals) × High-turnover (momentum/price signals)
          These are least correlated and produce the best combined alpha.
        - Pairs with very different field categories score highest.
        - Same-style pairs (both momentum or both value) are deprioritized.

        Pair scoring:
            orthogonality = |turnover_A - turnover_B| × category_bonus
        """
        from src.utils import build_simulation_payload
        import itertools

        if len(self._ensemble_pool) < 2:
            return []

        MAX_ENSEMBLES = 20   # cap — don't waste all quota on ensembles

        def _turnover_proxy(alpha: dict) -> float:
            """Estimate turnover from metadata. High = momentum, Low = value."""
            kind = alpha.get("_meta", {}).get("kind", "")
            op   = alpha.get("_meta", {}).get("op", "")
            cat  = alpha.get("_meta", {}).get("field_category", "price_volume")
            # Momentum operators tend to have high turnover
            if op in ("ts_delta", "ts_diff", "ts_zscore", "ts_rank"):
                return 0.8
            # Value-like operators tend to have low turnover
            if op in ("ts_mean", "ts_sum", "rank") or cat in ("fundamental", "analyst"):
                return 0.2
            return 0.5  # default

        def _category_bonus(a1: dict, a2: dict) -> float:
            """Bonus for pairing different field categories (cross-category = more orthogonal)."""
            cat1 = a1.get("_meta", {}).get("field_category", "price_volume")
            cat2 = a2.get("_meta", {}).get("field_category", "price_volume")
            return 1.5 if cat1 != cat2 else 1.0

        # Score all pairs by orthogonality
        pair_scores = []
        for a1, a2 in itertools.combinations(self._ensemble_pool, 2):
            expr1 = a1.get("regular", "")
            expr2 = a2.get("regular", "")
            if not expr1 or not expr2:
                continue
            t1 = _turnover_proxy(a1)
            t2 = _turnover_proxy(a2)
            ortho = abs(t1 - t2) * _category_bonus(a1, a2)
            pair_scores.append((ortho, a1, a2))

        # Sort by orthogonality descending — most orthogonal pairs first
        pair_scores.sort(key=lambda t: t[0], reverse=True)

        ensembles = []
        for _, a1, a2 in pair_scores[:MAX_ENSEMBLES * 2]:  # try more than needed (some may be visited)
            if len(ensembles) >= MAX_ENSEMBLES:
                break
            expr1 = a1.get("regular", "")
            expr2 = a2.get("regular", "")
            combined_expr = f"rank({expr1}) + rank({expr2})"
            if self.is_visited(combined_expr):
                continue
            p = build_simulation_payload(combined_expr)
            p["name"]  = f"ENS_{a1.get('name')}_{a2.get('name')}"
            p["_meta"] = {
                "kind":     "ensemble",
                "op":       "ensemble",
                "field":    a1.get("_meta", {}).get("field", ""),
                "field_category": "mixed",
            }
            ensembles.append(p)
            self.mark_visited(combined_expr)

        # Clear pool after generation
        pool_size = len(self._ensemble_pool)
        self._ensemble_pool.clear()
        logger.info(
            f"Generated {len(ensembles)} correlation-aware ensemble candidates "
            f"from {pool_size} pool alphas (max orthogonality-scored pairs)."
        )
        return ensembles

    def generate_evolved_pool(
        self,
        fields: list[str],
        field_category_map: dict,
        max_candidates: int = 50,
        field_sharpe_map: dict = None,
    ) -> list[dict]:
        """
        Generate evolved candidates by mutating + crossbreeding the
        best expressions in the ensemble pool.

        This is called at startup (after load_ensemble_pool + seed_from_history)
        to immediately generate promising improvements on past performers,
        rather than waiting for the current run to accumulate near-misses.

        field_sharpe_map: {field_id: avg_sharpe} from learner.field_sharpe_map().
        When provided, field-swap mutations prefer proven fields over random ones.
        """
        from src.evolver import GeneticEvolver

        evolver = GeneticEvolver()
        evolver.set_visited_fns(self.is_visited, self.mark_visited)

        candidates = evolver.evolve(
            top_alphas=self._ensemble_pool,
            fields=fields,
            field_category_map=field_category_map,
            max_candidates=max_candidates,
            field_sharpe_map=field_sharpe_map,
        )
        logger.info(
            f"Evolved pool: {len(candidates)} mutated candidates from "
            f"{len(self._ensemble_pool)} ensemble pool alphas."
        )
        return candidates
