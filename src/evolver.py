"""
src/evolver.py
──────────────
Genetic Expression Evolver — generates new alpha candidates by
mutating and crossbreeding the best historically-performing expressions.

Mutation types
──────────────
1. Operator swap     — rank(ts_mean(f,d))  → rank(ts_zscore(f,d))
2. Lookback shift    — rank(ts_mean(f,20)) → rank(ts_mean(f,60))
3. Field swap        — rank(ts_mean(A,d))  → rank(ts_mean(B,d))  (same category)
4. Outer wrap        — rank(ts_mean(f,d))  → rank(ts_delta(ts_mean(f,d), 5))
5. Settings mutation — MARKET neu → SECTOR neu, decay 0 → decay 4
6. Crossover         — op/lookback from A  × field from B

Crossover types
───────────────
7. Field crossover   — op+lookback from parent A, field from parent B
"""

import re
import math
import random
import logging

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

# Operators the evolver can swap TO (must produce meaningful alpha signals)
# Kept in sync with optimizer.py TS_UNARY_OPS intentionally
TS_UNARY_OPS = [
    "ts_min", "ts_max", "ts_mean", "ts_sum", "ts_std_dev",
    "ts_rank", "ts_scale", "ts_zscore",
    "ts_delta", "ts_delay", "ts_av_diff",
    "ts_product", "ts_quantile", "ts_decay_linear",
    "ts_arg_max", "ts_arg_min",
]

NEUTRALIZATIONS = ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
DECAYS          = [0, 2, 4, 6, 8]
LOOKBACKS       = [5, 10, 20, 60, 120, 252]

# ── Expression Parser ──────────────────────────────────────────────────────

# Matches: rank(ts_OP(FIELD, D))
_TS_PAT = re.compile(r"^rank\((ts_\w+)\((\w+),\s*(\d+)\)\)$")

# Matches: rank(FIELD / ts_OP(FIELD2, D))
_RATIO_PAT = re.compile(r"^rank\((\w+)\s*/\s*(ts_\w+)\((\w+),\s*(\d+)\)\)$")

# Matches: rank(FIELD) or group_OP(FIELD, GROUP)
_CS_PAT = re.compile(r"^(rank|group_\w+)\((\w+)(?:,\s*(\w+))?\)$")


def parse_expression(expr: str) -> dict | None:
    """
    Try to parse an expression string into a structured dict.
    Returns None for complex/nested expressions we can't handle.
    """
    expr = expr.strip()
    m = _TS_PAT.match(expr)
    if m:
        return {
            "kind": "ts",
            "op": m.group(1),
            "field": m.group(2),
            "lookback": int(m.group(3)),
            "expr": expr,
        }
    m = _RATIO_PAT.match(expr)
    if m:
        return {
            "kind": "ratio",
            "field": m.group(1),
            "op": m.group(2),
            "field2": m.group(3),
            "lookback": int(m.group(4)),
            "expr": expr,
        }
    m = _CS_PAT.match(expr)
    if m:
        return {
            "kind": "cs",
            "op": m.group(1),
            "field": m.group(2),
            "group": m.group(3),
            "expr": expr,
        }
    return None


# ── Genetic Evolver ────────────────────────────────────────────────────────

class GeneticEvolver:
    """
    Generates new alpha candidates by mutating and crossbreeding
    the best historically-performing expressions.

    Usage
    ─────
    evolver = GeneticEvolver()
    evolver.set_visited_fns(optimizer.is_visited, optimizer.mark_visited)
    new_candidates = evolver.evolve(
        top_alphas=optimizer._ensemble_pool[:20],
        fields=all_field_ids,
        field_category_map=field_category_map,
        max_candidates=50,
    )
    """

    def __init__(self):
        self._visited_check = lambda e: False
        self._mark_visited  = lambda e: None

    def set_visited_fns(self, check_fn, mark_fn):
        self._visited_check = check_fn
        self._mark_visited  = mark_fn

    def _add(self, expr: str, name: str, meta: dict, out: list) -> bool:
        """Add a candidate if expression hasn't been visited."""
        if self._visited_check(expr):
            return False
        try:
            from src.utils import build_simulation_payload
            p = build_simulation_payload(expr)
            p["name"] = name
            p["_meta"] = meta
            out.append(p)
            self._mark_visited(expr)
            return True
        except Exception:
            return False

    # ── Mutation 1: Operator Swap ──────────────────────────────────────────

    def mutate_operator(self, parsed: dict, out: list, limit: int = 3) -> int:
        """Swap the time-series operator for a different one."""
        if parsed.get("kind") != "ts":
            return 0
        field, d = parsed["field"], parsed["lookback"]
        cat  = parsed.get("field_category", "price_volume")
        added = 0
        candidates = [op for op in TS_UNARY_OPS if op != parsed["op"]]
        random.shuffle(candidates)
        for new_op in candidates[:limit]:
            expr = f"rank({new_op}({field}, {d}))"
            meta = {"op": new_op, "field": field, "lookback": d,
                    "kind": "evo_op_swap", "field_category": cat}
            if self._add(expr, f"EVO_op_{new_op[-4:]}_{field[:8]}_{d}", meta, out):
                added += 1
        return added

    # ── Mutation 2: Lookback Shift ─────────────────────────────────────────

    def mutate_lookback(self, parsed: dict, out: list) -> int:
        """Shift lookback to ±50%, ±25% of original."""
        if parsed.get("kind") != "ts" or not parsed.get("lookback"):
            return 0
        op, field, d = parsed["op"], parsed["field"], parsed["lookback"]
        cat = parsed.get("field_category", "price_volume")
        probes = sorted({
            max(3, d // 2),
            max(3, d * 3 // 4),
            min(504, d * 3 // 2),
            min(504, d * 2),
        } - {d})
        added = 0
        for probe_d in probes[:3]:
            expr = f"rank({op}({field}, {probe_d}))"
            meta = {"op": op, "field": field, "lookback": probe_d,
                    "kind": "evo_lb_shift", "field_category": cat}
            if self._add(expr, f"EVO_lb_{field[:8]}_{op[-4:]}_{probe_d}", meta, out):
                added += 1
        return added

    # ── Mutation 3: Field Swap ─────────────────────────────────────────────

    def mutate_field(
        self,
        parsed: dict,
        fields: list,
        field_category_map: dict,
        out: list,
        limit: int = 3,
        field_sharpe_map: dict = None,
    ) -> int:
        """
        Swap field for another — preferring proven fields if field_sharpe_map is provided.

        field_sharpe_map: {field_id: best_sharpe} from learner.field_quota_map().
        When provided, candidate fields are sorted best-Sharpe-first so the evolver
        prefers swapping to a field that has historically performed well.
        """
        if parsed.get("kind") != "ts":
            return 0
        op, d = parsed["op"], parsed["lookback"]
        orig_field = parsed["field"]
        orig_cat   = field_category_map.get(orig_field, "price_volume")

        same_cat = [f for f in fields
                    if field_category_map.get(f, "price_volume") == orig_cat
                    and f != orig_field]

        # Sort by historical Sharpe if map is available — proven fields first
        if field_sharpe_map:
            same_cat = sorted(
                same_cat,
                key=lambda f: field_sharpe_map.get(f, -99.0),
                reverse=True,
            )
        else:
            random.shuffle(same_cat)

        added = 0
        for new_field in same_cat[:limit]:
            expr = f"rank({op}({new_field}, {d}))"
            meta = {"op": op, "field": new_field, "lookback": d,
                    "kind": "evo_field_swap", "field_category": orig_cat}
            if self._add(expr, f"EVO_fld_{new_field[:8]}_{op[-4:]}_{d}", meta, out):
                added += 1
        return added


    # ── Mutation 4: Outer Wrap ─────────────────────────────────────────────

    def mutate_wrap(self, parsed: dict, out: list) -> int:
        """Wrap the inner expression in an additional outer operator."""
        if parsed.get("kind") != "ts":
            return 0
        op, field, d = parsed["op"], parsed["field"], parsed["lookback"]
        cat   = parsed.get("field_category", "price_volume")
        inner = f"{op}({field}, {d})"
        added = 0
        # Wrap in ts_delta (momentum of the signal)
        for delta_d in [5, 10]:
            expr = f"rank(ts_delta({inner}, {delta_d}))"
            meta = {"op": "ts_delta", "field": field, "lookback": delta_d,
                    "kind": "evo_wrap_delta", "field_category": cat}
            if self._add(expr, f"EVO_wdelta_{field[:6]}_{delta_d}", meta, out):
                added += 1
        # Wrap in ts_zscore (standardize the signal over time)
        for z_d in [20, 60]:
            expr = f"rank(ts_zscore({inner}, {z_d}))"
            meta = {"op": "ts_zscore", "field": field, "lookback": z_d,
                    "kind": "evo_wrap_zscore", "field_category": cat}
            if self._add(expr, f"EVO_wzscore_{field[:6]}_{z_d}", meta, out):
                added += 1
        # Ratio: field / ts_op(field, d) — deviation from historical
        expr = f"rank({field} / {op}({field}, {d}))"
        meta = {"op": op, "field": field, "lookback": d,
                "kind": "evo_wrap_ratio", "field_category": cat}
        if self._add(expr, f"EVO_ratio_{field[:8]}_{d}", meta, out):
            added += 1
        return added

    # ── Mutation 5: Settings ───────────────────────────────────────────────

    def mutate_settings(self, parsed: dict, out: list) -> int:
        """Try different neutralization + decay combos on a promising expression."""
        if parsed.get("kind") != "ts":
            return 0
        from src.utils import build_simulation_payload
        op, field, d = parsed["op"], parsed["field"], parsed["lookback"]
        cat  = parsed.get("field_category", "price_volume")
        base = f"rank({op}({field}, {d}))"
        added = 0
        for neu in ["SECTOR", "INDUSTRY", "SUBINDUSTRY"]:
            for dec in [2, 4]:
                key = f"{base}|neu={neu}|dec={dec}"
                if not self._visited_check(key):
                    p = build_simulation_payload(base, neutralization=neu, decay=dec)
                    name = f"EVO_set_{field[:6]}_{op[-4:]}_{neu[:3]}{dec}"
                    p["name"] = name
                    p["_meta"] = {"op": op, "field": field, "lookback": d,
                                  "decay": dec, "neutralization": neu,
                                  "kind": "evo_settings", "field_category": cat}
                    out.append(p)
                    self._mark_visited(key)
                    added += 1
        return added

    # ── Crossover: Field from B, Op+Lookback from A ────────────────────────

    def crossover_fields(self, parsed_a: dict, parsed_b: dict, out: list) -> int:
        """Take operator + lookback from A, field from B."""
        if parsed_a.get("kind") != "ts" or parsed_b.get("kind") != "ts":
            return 0
        op, d   = parsed_a["op"], parsed_a["lookback"]
        field_b = parsed_b["field"]
        cat_b   = parsed_b.get("field_category", "price_volume")
        expr = f"rank({op}({field_b}, {d}))"
        meta = {"op": op, "field": field_b, "lookback": d,
                "kind": "evo_crossover", "field_category": cat_b}
        return 1 if self._add(expr, f"EVO_xo_{field_b[:8]}_{op[-4:]}_{d}", meta, out) else 0

    # ── Main Evolution API ─────────────────────────────────────────────────

    def evolve(
        self,
        top_alphas: list,
        fields: list,
        field_category_map: dict,
        max_candidates: int = 50,
        field_sharpe_map: dict = None,
    ) -> list:
        """
        Generate evolved candidates from the best historical alphas.

        Goal alignment
        ──────────────
        Parents sorted by seed Sharpe (best-first). Near-miss parents
        (Sharpe ≥ 1.0, gap ≤ 0.25 from target 1.25) receive 5x more
        operator mutations vs weak parents — maximizing the chance
        children cross the 1.25 threshold.

        field_sharpe_map: {field_id: best_sharpe} — when provided,
        mutate_field() swaps toward proven fields first.
        """
        SHARPE_TARGET = 1.25
        out = []
        parseable = []

        # Sort parents: best Sharpe first (near-misses get the most mutations)
        sorted_alphas = sorted(
            top_alphas,
            key=lambda a: a.get("_seed_sharpe", a.get("sharpe", 0.0)),
            reverse=True,
        )

        for alpha in sorted_alphas:
            if len(out) >= max_candidates:
                break

            expr = alpha.get("regular", "")
            if not expr or expr.startswith("__historical__"):
                continue

            parsed = parse_expression(expr)
            if not parsed:
                continue

            # Inject field_category
            if "field_category" not in parsed:
                parsed["field_category"] = field_category_map.get(
                    parsed.get("field", ""), "price_volume"
                )
            parseable.append(parsed)

            # Mutation budget: proportional to proximity to 1.25
            seed_sharpe = alpha.get("_seed_sharpe", alpha.get("sharpe", 0.0))
            gap = max(0.0, SHARPE_TARGET - seed_sharpe)

            if gap <= 0.25:       # Sharpe ≥ 1.0  — very close, mutate heavily
                op_limit = 5
                do_all   = True
            elif gap <= 0.55:     # Sharpe ≥ 0.7  — near-miss range
                op_limit = 3
                do_all   = True
            else:                 # Sharpe < 0.7  — weak, minimal mutations
                op_limit = 1
                do_all   = False

            self.mutate_operator(parsed, out, limit=op_limit)
            self.mutate_lookback(parsed, out)
            if do_all:
                self.mutate_field(parsed, fields, field_category_map, out)
                self.mutate_wrap(parsed, out)
                self.mutate_settings(parsed, out)

        # Crossover between the best parents only (top-10 by Sharpe)
        best = parseable[:10]
        for i in range(0, len(best) - 1, 2):
            if len(out) >= max_candidates:
                break
            self.crossover_fields(best[i], best[i + 1], out)

        random.shuffle(out)
        out = out[:max_candidates]
        logger.info(
            f"GeneticEvolver: {len(out)} evolved candidates from "
            f"{len(parseable)} parseable parents "
            f"(of {len(sorted_alphas)} total). Goal=Sharpe≥{SHARPE_TARGET}"
        )
        return out
