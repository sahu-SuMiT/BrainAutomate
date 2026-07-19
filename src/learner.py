"""
src/learner.py
──────────────
Surrogate model that learns from simulation history to predict Sharpe.

How it works
────────────
1. After each simulation, the result (Sharpe, Fitness, Turnover, etc.)
   is stored in results/*.jsonl by ResultLogger.

2. The Learner reads all historical results and builds feature vectors:
     [operator_idx, lookback_norm, decay_norm, neutralization_idx,
      field_category_idx, sharpe_of_best_same_field, ...]

3. A simple weighted ridge regression predicts Sharpe from those features.
   No external ML library needed — implemented with pure Python math.

4. Before the next batch is queued, Learner.rank_candidates() scores each
   candidate and returns them sorted best-first.
   → The queue now prioritises the most promising experiments.

5. After each result, the model is updated online (incremental update).

This is NOT just a lookup table — it generalises across unseen combinations
by learning weights for each feature dimension.
"""

import json
import math
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

# ──────────────────────────────────────────────────────────────────
# Feature encoding
# ──────────────────────────────────────────────────────────────────

OPERATORS = [
    "ts_min", "ts_max", "ts_mean", "ts_sum", "ts_std_dev",
    "ts_rank", "ts_scale", "ts_zscore", "ts_delta", "ts_delay",
    "ts_product", "ts_quantile", "ts_arg_max", "ts_arg_min",
    "ts_av_diff", "ts_decay_linear",
    "rank", "group_rank", "group_zscore", "group_neutralize",
]
OP_IDX = {op: i for i, op in enumerate(OPERATORS)}
N_OPS  = len(OPERATORS)

NEUTRALIZATIONS = ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
NEU_IDX = {n: i for i, n in enumerate(NEUTRALIZATIONS)}

LOOKBACK_MAX = 504   # clip normalisation at 504 days (~2 trading years)
DECAY_MAX    = 12


def _feature_vector(op: str, lookback, decay: int, neutralization: str,
                    field_category: str = "unknown") -> list[float]:
    """
    Build a fixed-length numeric feature vector for one alpha configuration.

    Dimensions
    ──────────
    0..N_OPS-1   : one-hot for operator
    N_OPS        : normalised lookback  (0 if no lookback, e.g. CS ops)
    N_OPS+1      : normalised decay
    N_OPS+2      : normalised neutralization index
    N_OPS+3      : is cross-sectional op?     (0 or 1)
    N_OPS+4..+7  : one-hot for field category
    """
    vec = [0.0] * (N_OPS + 8)

    # One-hot operator
    op_i = OP_IDX.get(op, -1)
    if op_i >= 0:
        vec[op_i] = 1.0

    # Continuous features
    vec[N_OPS]     = min(lookback, LOOKBACK_MAX) / LOOKBACK_MAX if lookback else 0.0
    vec[N_OPS + 1] = min(decay, DECAY_MAX) / DECAY_MAX
    vec[N_OPS + 2] = NEU_IDX.get(neutralization, 0) / max(len(NEUTRALIZATIONS) - 1, 1)
    vec[N_OPS + 3] = 1.0 if op in {"rank", "group_rank", "group_zscore", "group_neutralize"} else 0.0

    # Field category one-hot (4 categories)
    cat_map = {"price_volume": 0, "fundamental": 1, "analyst": 2, "sentiment": 3}
    cat_i = cat_map.get(field_category, 0)
    vec[N_OPS + 4 + cat_i] = 1.0

    return vec


# ──────────────────────────────────────────────────────────────────
# Ridge Regression (pure Python — no external deps)
# ──────────────────────────────────────────────────────────────────

class _RidgeRegression:
    """
    Incremental ridge regression using the normal equations.
    w = (X^T X + λI)^{-1} X^T y

    Solved via Cholesky / simple Gaussian elimination for small feature sets.
    """

    def __init__(self, n_features: int, alpha: float = 0.5):
        self.n  = n_features
        self.alpha = alpha          # regularisation strength
        self._XtX = [[0.0] * n_features for _ in range(n_features)]
        self._Xty = [0.0] * n_features
        self._w   = [0.0] * n_features
        self._n_samples = 0

    def fit_one(self, x: list[float], y: float):
        """Incremental update with one new observation."""
        # Accumulate X^T X and X^T y
        for i in range(self.n):
            self._Xty[i] += x[i] * y
            for j in range(self.n):
                self._XtX[i][j] += x[i] * x[j]
        self._n_samples += 1
        self._solve()

    def _solve(self):
        """Solve (X^T X + λI) w = X^T y via Gaussian elimination."""
        n = self.n
        # Build augmented matrix [A | b] where A = XtX + λI
        A = [row[:] for row in self._XtX]
        b = self._Xty[:]
        for i in range(n):
            A[i][i] += self.alpha

        # Forward elimination
        for col in range(n):
            pivot = A[col][col]
            if abs(pivot) < 1e-12:
                continue
            for row in range(col + 1, n):
                factor = A[row][col] / pivot
                for k in range(col, n):
                    A[row][k] -= factor * A[col][k]
                b[row] -= factor * b[col]

        # Back substitution
        w = [0.0] * n
        for row in range(n - 1, -1, -1):
            if abs(A[row][row]) < 1e-12:
                w[row] = 0.0
                continue
            w[row] = b[row]
            for k in range(row + 1, n):
                w[row] -= A[row][k] * w[k]
            w[row] /= A[row][row]

        self._w = w

    def predict(self, x: list[float]) -> float:
        return sum(xi * wi for xi, wi in zip(x, self._w))

    @property
    def n_samples(self):
        return self._n_samples


# ──────────────────────────────────────────────────────────────────
# Learner — public API
# ──────────────────────────────────────────────────────────────────

class Learner:
    """
    Learns from every simulation result to predict Sharpe for unseen
    (field, operator, lookback, decay, neutralization) combinations.

    Usage
    ─────
    learner = Learner()
    learner.load_history()             # load all past results

    # Before queuing a new batch:
    ranked = learner.rank_candidates(candidates)

    # After each simulation:
    learner.record(alpha_meta, sharpe, fitness, turnover)
    """

    def __init__(self):
        n_features = N_OPS + 8
        self._model = _RidgeRegression(n_features, alpha=0.5)

        # Field-level statistics: {field -> {"n": int, "sharpe_sum": float, "best": float}}
        self._field_stats = defaultdict(lambda: {"n": 0, "sum": 0.0, "best": -99.0})

        # Correlation matrix between (field, op) pairs → Sharpe  (used for exploration bonus)
        self._pair_results = defaultdict(list)

    # ──────────────────────────────────────────────────────────────
    # History loading
    # ──────────────────────────────────────────────────────────────

    def load_history(self):
        """
        Read all JSONL result files and train on historical Sharpe values.
        Call this once at startup.
        """
        loaded = 0
        if not RESULTS_DIR.exists():
            return

        for path in sorted(RESULTS_DIR.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    sharpe = row.get("sharpe")
                    if sharpe is None:
                        continue
                    meta = {
                        "op":              row.get("operator", "rank"),
                        "lookback":        row.get("lookback"),
                        "decay":           row.get("decay", 0),
                        "neutralization":  row.get("neutralization", "MARKET"),
                        "field_category":  row.get("field_category", "price_volume"),
                        "field":           row.get("alpha_name", ""),
                    }
                    self._update_model(meta, sharpe)
                    loaded += 1
                except Exception:
                    continue

        logger.info(f"Learner: loaded {loaded} historical results. Model has {self._model.n_samples} samples.")

    # ──────────────────────────────────────────────────────────────
    # Online update after each simulation
    # ──────────────────────────────────────────────────────────────

    def record(self, alpha: dict, sharpe: float | None, fitness: float | None = None,
               turnover: float | None = None):
        """Call this after every simulation to update the model."""
        if sharpe is None:
            return

        meta = alpha.get("_meta", {})
        self._update_model(meta, sharpe)

        # Field-level stats
        field = meta.get("field", "")
        if field:
            fs = self._field_stats[field]
            fs["n"] += 1
            fs["sum"] += sharpe
            fs["best"] = max(fs["best"], sharpe)

        # Pair stats for exploration bonus
        op = meta.get("op", "")
        if field and op:
            self._pair_results[(field, op)].append(sharpe)

        logger.info(
            f"[Learner] Updated on sharpe={sharpe:.3f}  "
            f"op={op}  lookback={meta.get('lookback')}  "
            f"(total samples={self._model.n_samples})"
        )

    def _update_model(self, meta: dict, sharpe: float):
        x = _feature_vector(
            op             = meta.get("op", "rank"),
            lookback       = meta.get("lookback"),
            decay          = meta.get("decay", 0),
            neutralization = meta.get("neutralization", "MARKET"),
            field_category = meta.get("field_category", "price_volume"),
        )
        self._model.fit_one(x, sharpe)

    # ──────────────────────────────────────────────────────────────
    # Prediction & candidate ranking
    # ──────────────────────────────────────────────────────────────

    def predict_sharpe(self, alpha: dict) -> float:
        """
        Predict Sharpe for an unseen candidate.
        Returns raw model output (may be unreliable with < 10 samples).
        """
        meta = alpha.get("_meta", {})
        x = _feature_vector(
            op             = meta.get("op", "rank"),
            lookback       = meta.get("lookback"),
            decay          = meta.get("decay", 0),
            neutralization = meta.get("neutralization", "MARKET"),
            field_category = meta.get("field_category", "price_volume"),
        )
        return self._model.predict(x)

    def _exploration_bonus(self, alpha: dict) -> float:
        """
        UCB-style exploration bonus: under-explored (field, op) pairs
        get a positive bonus to encourage diversity.
        """
        meta  = alpha.get("_meta", {})
        field = meta.get("field", "")
        op    = meta.get("op", "")
        key   = (field, op)
        n     = len(self._pair_results.get(key, []))
        total = self._model.n_samples + 1
        if n == 0:
            return 1.0   # Large bonus for completely unexplored pairs
        return math.sqrt(2 * math.log(total) / n) * 0.3

    def score(self, alpha: dict) -> float:
        """
        Combined score = predicted_sharpe + exploration_bonus.
        Used to rank candidates before queuing.
        """
        pred  = self.predict_sharpe(alpha)
        bonus = self._exploration_bonus(alpha)
        return pred + bonus

    def rank_candidates(self, candidates: list[dict]) -> list[dict]:
        """
        Sort candidates by predicted Sharpe + exploration bonus (descending).
        The queue will process the most promising ones first.

        Falls back to random shuffle if model has fewer than 5 samples
        (not enough data to predict reliably yet).
        """
        if self._model.n_samples < 5:
            import random
            logger.info("Learner: insufficient history — using random ordering.")
            shuffled = candidates[:]
            random.shuffle(shuffled)
            return shuffled

        scored = [(self.score(c), c) for c in candidates]
        scored.sort(key=lambda t: t[0], reverse=True)

        logger.info(
            f"Learner: ranked {len(candidates)} candidates. "
            f"Top predicted sharpe: {scored[0][0]:.3f}  "
            f"Bottom: {scored[-1][0]:.3f}"
        )
        return [c for _, c in scored]

    # ──────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────

    def top_fields(self, n: int = 10) -> list[dict]:
        """Return fields sorted by their best historical Sharpe."""
        rows = []
        for field, stats in self._field_stats.items():
            if stats["n"] > 0:
                rows.append({
                    "field":     field,
                    "trials":    stats["n"],
                    "avg_sharpe": round(stats["sum"] / stats["n"], 4),
                    "best_sharpe": round(stats["best"], 4),
                })
        return sorted(rows, key=lambda r: r["best_sharpe"], reverse=True)[:n]

    def field_sharpe_map(self) -> dict:
        """
        Return {field_id: avg_sharpe} for all fields with at least one result.
        Pass this to client.fetch_top_fields(learner_stats=...) so the composite
        field ranking incorporates YOUR personal success rate per field.

        On the first run (no history), returns an empty dict — fetch_top_fields
        will fall back to coverage + novelty signals only.
        """
        return {
            field: round(stats["sum"] / stats["n"], 4)
            for field, stats in self._field_stats.items()
            if stats["n"] > 0
        }

    def feature_weights(self) -> dict:
        """Return the learned weight for each operator (for diagnostics)."""
        w = {}
        for op, idx in OP_IDX.items():
            w[op] = round(self._model._w[idx], 4)
        return dict(sorted(w.items(), key=lambda kv: kv[1], reverse=True))
