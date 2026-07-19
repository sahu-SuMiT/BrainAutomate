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
import threading
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

# ──────────────────────────────────────────────────────────────────
# Feature encoding
# ──────────────────────────────────────────────────────────────────

OPERATORS = [
    # Core statistical
    "ts_min", "ts_max", "ts_mean", "ts_sum", "ts_std_dev",
    # Rank / scaling
    "ts_rank", "ts_scale", "ts_zscore",
    # Momentum / change
    "ts_delta", "ts_delay", "ts_av_diff",
    # Advanced aggregators
    "ts_product", "ts_quantile", "ts_decay_linear",
    # Ordinal / position
    "ts_arg_max", "ts_arg_min",
    # Special-signature ops (also appear in results)
    "kth_element", "hump",
    # Cross-sectional
    "rank", "group_rank", "group_zscore", "group_neutralize",
]
OP_IDX = {op: i for i, op in enumerate(OPERATORS)}

N_OPS  = len(OPERATORS)

NEUTRALIZATIONS = ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
NEU_IDX = {n: i for i, n in enumerate(NEUTRALIZATIONS)}

LOOKBACK_MAX    = 504   # clip normalisation at 504 days (~2 trading years)
DECAY_MAX       = 12
N_FIELD_BUCKETS = 16    # hash-embedding buckets for field identity
                        # 16 buckets for ~30 fields = mild collision, good resolution


def _field_bucket(field_id: str) -> int:
    """
    Map a field name to a stable bucket index in [0, N_FIELD_BUCKETS).

    Uses MD5 for stability across sessions — Python's built-in hash() is
    randomised per session via PYTHONHASHSEED.
    """
    import hashlib
    h = int(hashlib.md5(field_id.encode()).hexdigest(), 16)
    return h % N_FIELD_BUCKETS


def _feature_vector(
    op: str,
    lookback,
    decay: int,
    neutralization: str,
    field_category: str = "unknown",
    field_id: str = "",
) -> list[float]:
    """
    Build a fixed-length numeric feature vector for one alpha configuration.

    Dimensions
    ──────────
    0 .. N_OPS-1         : one-hot for operator              (22 dims)
    N_OPS                : normalised lookback               (1 dim)
    N_OPS+1              : normalised decay                  (1 dim)
    N_OPS+2              : normalised neutralization index   (1 dim)
    N_OPS+3              : is cross-sectional op?            (1 dim)
    N_OPS+4 .. N_OPS+7   : one-hot for field category        (4 dims)
    N_OPS+8 .. N_OPS+23  : field hash-embedding              (16 dims)
                           Lets the model learn FIELD-SPECIFIC Sharpe patterns.
                           e.g. "anl4_fs_* fields cluster in bucket 3 and score 0.9+"

    Total: N_OPS + 8 + N_FIELD_BUCKETS
    """
    vec = [0.0] * (N_OPS + 8 + N_FIELD_BUCKETS)

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

    # Field fingerprint: hash-embedding into N_FIELD_BUCKETS slots
    # The model learns weights per bucket — fields with historically high Sharpe
    # share a bucket and reinforce each other's weight.
    if field_id:
        bucket = _field_bucket(field_id)
        vec[N_OPS + 8 + bucket] = 1.0

    return vec


# ──────────────────────────────────────────────────────────────────
# Ridge Regression — base learner (used by bootstrap ensemble)
# ──────────────────────────────────────────────────────────────────

class _RidgeRegression:
    """
    Batch ridge regression using the normal equations.
    w = (X^T X + λI)^{-1} X^T y
    """

    def __init__(self, n_features: int, alpha: float = 0.5):
        self.n     = n_features
        self.alpha = alpha
        self._XtX  = [[0.0] * n_features for _ in range(n_features)]
        self._Xty  = [0.0] * n_features
        self._w    = [0.0] * n_features
        self._n_samples = 0

    def fit_one(self, x: list, y: float):
        """Incremental update — accumulates XtX and Xty then re-solves."""
        for i in range(self.n):
            self._Xty[i] += x[i] * y
            for j in range(self.n):
                self._XtX[i][j] += x[i] * x[j]
        self._n_samples += 1
        self._solve()

    def fit_batch(self, X: list, y: list):
        """Reset and train from scratch on a batch."""
        self._XtX = [[0.0] * self.n for _ in range(self.n)]
        self._Xty = [0.0] * self.n
        self._n_samples = 0
        for xi, yi in zip(X, y):
            for i in range(self.n):
                self._Xty[i] += xi[i] * yi
                for j in range(self.n):
                    self._XtX[i][j] += xi[i] * xi[j]
            self._n_samples += 1
        self._solve()

    def _solve(self):
        n = self.n
        A = [row[:] for row in self._XtX]
        b = self._Xty[:]
        for i in range(n):
            A[i][i] += self.alpha
        for col in range(n):
            pivot = A[col][col]
            if abs(pivot) < 1e-12:
                continue
            for row in range(col + 1, n):
                f = A[row][col] / pivot
                for k in range(col, n):
                    A[row][k] -= f * A[col][k]
                b[row] -= f * b[col]
        w = [0.0] * n
        for row in range(n - 1, -1, -1):
            if abs(A[row][row]) < 1e-12:
                continue
            w[row] = b[row]
            for k in range(row + 1, n):
                w[row] -= A[row][k] * w[k]
            w[row] /= A[row][row]
        self._w = w

    def predict(self, x: list) -> float:
        return sum(xi * wi for xi, wi in zip(x, self._w))

    @property
    def n_samples(self):
        return self._n_samples


# ──────────────────────────────────────────────────────────────────
# Bootstrap Ensemble Surrogate — UCB exploration via uncertainty
# ──────────────────────────────────────────────────────────────────

class _BootstrapEnsemble:
    """
    Bootstrap aggregation (bagging) of ridge regressors.

    - N_MODELS models each trained on a 70% random subsample.
    - Mean across models = point prediction.
    - Std dev across models = uncertainty estimate.
    - UCB score = mean + κ × std  (where κ decays with more data).

    This enables proper exploration-exploitation balance:
    under-explored regions have high uncertainty → high UCB score.
    Proven winners have low uncertainty → UCB ≈ mean prediction.
    """

    N_MODELS       = 7
    SUBSAMPLE_FRAC = 0.70
    REBUILD_EVERY  = 5   # rebuild bootstraps every N new samples

    def __init__(self, n_features: int, alpha: float = 0.5):
        self.n      = n_features
        self.alpha  = alpha
        self._models = [_RidgeRegression(n_features, alpha) for _ in range(self.N_MODELS)]
        self._X: list = []
        self._y: list = []
        self._n_samples   = 0
        self._since_rebuild = 0

    def fit_one(self, x: list, y: float):
        self._X.append(x[:])
        self._y.append(y)
        self._n_samples += 1
        self._since_rebuild += 1
        if self._since_rebuild >= self.REBUILD_EVERY or self._n_samples <= 20:
            self._rebuild()
            self._since_rebuild = 0

    def _rebuild(self):
        import random as _rng
        n = len(self._X)
        if n == 0:
            return
        k = max(1, int(n * self.SUBSAMPLE_FRAC))
        for model in self._models:
            idxs  = [_rng.randrange(n) for _ in range(k)]
            model.fit_batch([self._X[i] for i in idxs],
                            [self._y[i] for i in idxs])

    def predict_ucb(self, x: list, kappa: float = 0.8) -> float:
        """UCB score = mean + kappa × std across bootstrap models."""
        if self._n_samples < 3:
            return 0.0
        preds = [m.predict(x) for m in self._models]
        mu  = sum(preds) / len(preds)
        var = sum((p - mu) ** 2 for p in preds) / len(preds)
        return mu + kappa * math.sqrt(var)

    def predict(self, x: list) -> float:
        if self._n_samples < 3:
            return 0.0
        preds = [m.predict(x) for m in self._models]
        return sum(preds) / len(preds)

    def uncertainty(self, x: list) -> float:
        if self._n_samples < 3:
            return 1.0
        preds = [m.predict(x) for m in self._models]
        mu  = sum(preds) / len(preds)
        var = sum((p - mu) ** 2 for p in preds) / len(preds)
        return math.sqrt(var)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def _w(self) -> list:
        """Average weights across models (for diagnostics)."""
        if not self._models or not self._models[0]._w:
            return []
        n = len(self._models[0]._w)
        avg = [sum(m._w[i] for m in self._models) / len(self._models) for i in range(n)]
        return avg


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

    # Primary Sharpe target — every scoring decision aims at this
    SHARPE_TARGET = 1.25

    def __init__(self, sharpe_target: float = 1.25):
        self.sharpe_target = sharpe_target
        n_features = N_OPS + 8 + N_FIELD_BUCKETS  # op(22) + meta(8) + field_hash(16)
        self._model = _BootstrapEnsemble(n_features, alpha=0.5)

        # Field-level statistics: {field -> {"n": int, "sum": float, "best": float}}
        self._field_stats = defaultdict(lambda: {"n": 0, "sum": 0.0, "best": -99.0})

        # Category-level win rates: {category -> {"n": int, "passes": int}}
        self._cat_wins = defaultdict(lambda: {"n": 0, "passes": 0})

        # (field, op) pair results — used for exploration diversity
        self._pair_results = defaultdict(list)

        # Thread safety for parallel workers
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────
    # History loading
    # ──────────────────────────────────────────────────────────────

    def load_history(self):
        """
        Read all JSONL result files and train on historical Sharpe values.
        Call this once at startup.

        Keys read from each JSONL line (written by ResultLogger.record()):
            operator       - the alpha operator (e.g. ts_mean, rank)
            lookback       - lookback window in days
            decay          - decay value
            neutralization - neutralization level
            field          - the data field used (written since the fix)
            field_category - price_volume / fundamental / analyst / mixed
            sharpe         - the simulation Sharpe result
        """
        loaded = 0
        skipped_no_meta = 0
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
                        # "field" key is written by the new result_logger;
                        # old entries fall back to alpha_name (still useful for field stats)
                        "field":           row.get("field") or row.get("alpha_name", ""),
                    }
                    self._update_model(meta, sharpe)

                    # Also update field-level stats so top_fields() works on reload
                    field = meta["field"]
                    if field:
                        fs = self._field_stats[field]
                        fs["n"]   += 1
                        fs["sum"] += sharpe
                        fs["best"] = max(fs["best"], sharpe)

                    # Update pair stats for exploration bonus
                    op = meta["op"]
                    if field and op:
                        self._pair_results[(field, op)].append(sharpe)

                    loaded += 1
                except Exception:
                    continue

        logger.info(
            f"Learner: loaded {loaded} historical results "
            f"({skipped_no_meta} had no meta). Model has {self._model.n_samples} samples."
        )

    # ──────────────────────────────────────────────────────────────
    # Online update after each simulation
    # ──────────────────────────────────────────────────────────────

    def record(self, alpha: dict, sharpe: float | None, fitness: float | None = None,
               turnover: float | None = None):
        """Call this after every simulation to update the model."""
        if sharpe is None:
            return

        with self._lock:
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
            f"op={meta.get('op','')}  lookback={meta.get('lookback')}  "
            f"(total samples={self._model.n_samples})"
        )

    def record_outcome(self, alpha: dict, sharpe: float | None,
                       fitness: float | None = None, tests_pass: bool = False):
        """
        Extended record that also tracks category-level win rates.
        Call this after every simulation for multi-objective learning.
        """
        if sharpe is None:
            return
        meta = alpha.get("_meta", {})
        cat  = meta.get("field_category", "price_volume")
        with self._lock:
            self._cat_wins[cat]["n"] += 1
            if tests_pass:
                self._cat_wins[cat]["passes"] += 1

    def _update_model(self, meta: dict, sharpe: float):
        x = _feature_vector(
            op             = meta.get("op", "rank"),
            lookback       = meta.get("lookback"),
            decay          = meta.get("decay", 0),
            neutralization = meta.get("neutralization", "MARKET"),
            field_category = meta.get("field_category", "price_volume"),
            field_id       = meta.get("field", ""),
        )
        self._model.fit_one(x, sharpe)

    # ──────────────────────────────────────────────────────────────
    # Prediction & candidate ranking
    # ──────────────────────────────────────────────────────────────

    def predict_sharpe(self, alpha: dict) -> float:
        """Point-estimate of Sharpe (mean across bootstrap models)."""
        meta = alpha.get("_meta", {})
        x = _feature_vector(
            op             = meta.get("op", "rank"),
            lookback       = meta.get("lookback"),
            decay          = meta.get("decay", 0),
            neutralization = meta.get("neutralization", "MARKET"),
            field_category = meta.get("field_category", "price_volume"),
            field_id       = meta.get("field", ""),
        )
        return self._model.predict(x)

    def _ucb_kappa(self) -> float:
        """
        Exploration-exploitation trade-off coefficient.
        Starts at 1.2 (explore aggressively early) and decays to 0.3
        as more data accumulates (exploit proven combinations).
        κ = 1.2 × exp(-n / 200) + 0.3
        """
        n = self._model.n_samples
        return 1.2 * math.exp(-n / 200.0) + 0.3

    def score(self, alpha: dict) -> float:
        """
        Goal-aligned UCB score.

        This is NOT generic "predict highest Sharpe" — it specifically aims
        at Sharpe ≥ 1.25 (the target). The score is designed so that:

        1. Candidates whose mean prediction is close to OR above the target
           score highest — they are most likely to cross 1.25.

        2. Candidates with HIGH uncertainty (σ) near the target also score
           high — they might be above 1.25 despite the uncertain estimate.

        3. Candidates whose mean is far BELOW the target (e.g. 0.2) get
           deprioritized even if they have high uncertainty, because the
           probability of reaching 1.25 from 0.2 is near zero.

        Formula:
            ucb_raw   = mean + κ × σ          (standard UCB)
            target_gap = target - mean         (distance from goal)
            gap_weight = exp(-max(gap,0) / 0.4)  # sharp drop for hopeless cases
            score      = ucb_raw × gap_weight

        Near-misses (mean 0.8–1.24) get the highest gap_weight ≈ 1.0
        because they could plausibly reach 1.25 with the right settings.
        """
        meta = alpha.get("_meta", {})
        x = _feature_vector(
            op             = meta.get("op", "rank"),
            lookback       = meta.get("lookback"),
            decay          = meta.get("decay", 0),
            neutralization = meta.get("neutralization", "MARKET"),
            field_category = meta.get("field_category", "price_volume"),
            field_id       = meta.get("field", ""),
        )
        kappa = self._ucb_kappa()
        mean  = self._model.predict(x)
        sigma = self._model.uncertainty(x)
        ucb   = mean + kappa * sigma

        # Gap between prediction and target
        gap = self.sharpe_target - mean
        # Candidates already predicted above target: gap_weight = 1.0 (keep)
        # Near-misses (gap ≤ 0.4): weight ≈ 0.9 → 0.37 (still very competitive)
        # Far below (gap > 1.0): weight ≈ 0.08 (heavily deprioritized)
        gap_weight = math.exp(-max(gap, 0.0) / 0.4)

        return ucb * gap_weight

    def rank_candidates(self, candidates: list[dict]) -> list[dict]:
        """
        Sort candidates by goal-aligned UCB score (descending).

        Candidates most likely to reach Sharpe ≥ 1.25 come first.
        Thread-safe: acquires lock before reading model weights.
        """
        with self._lock:
            if self._model.n_samples < 5:
                import random
                logger.info("Learner: insufficient history — using random ordering.")
                shuffled = candidates[:]
                random.shuffle(shuffled)
                return shuffled

            kappa = self._ucb_kappa()
            scored = [(self.score(c), c) for c in candidates]
            scored.sort(key=lambda t: t[0], reverse=True)

            # Diagnostics: show how many are predicted near-miss range
            near_miss = sum(1 for s, _ in scored if s >= self.sharpe_target * 0.7)
            logger.info(
                f"Learner: ranked {len(candidates)} candidates  "
                f"(UCB κ={kappa:.2f}, goal=Sharpe≥{self.sharpe_target}, "
                f"predicted near-miss≥{self.sharpe_target*0.7:.2f}: {near_miss}).  "
                f"Top: {scored[0][0]:.3f}  Bottom: {scored[-1][0]:.3f}"
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
        """Return the average learned weight for each operator (across bootstrap models)."""
        w = {}
        weights = self._model._w
        for op, idx in OP_IDX.items():
            if idx < len(weights):
                w[op] = round(weights[idx], 4)
        return dict(sorted(w.items(), key=lambda kv: kv[1], reverse=True))

    def field_quota_map(self, all_fields: list, base_n: int = 4) -> dict:
        """
        Route simulation quota to fields based on historical Sharpe performance.

        Tier system (aligned with Sharpe ≥ 1.25 goal):
        ────────────────────────────────────────────────
        PROVEN   best_sharpe ≥ 0.70  → base_n × 3   (most likely to reach 1.25)
        MODERATE best_sharpe ≥ 0.30  → base_n × 1.5 (worth more exploration)
        UNKNOWN  never tested         → base_n × 1   (equal opportunity)
        WEAK     best_sharpe < 0.00  → max(1, base_n // 4)  (minimal quota)

        Returns
        -------
        dict: {field_id: n_per_field}  — how many candidates to generate per field
        """
        result = {}
        for field in all_fields:
            stats = self._field_stats.get(field)
            if stats is None or stats["n"] == 0:
                # Never tested — give base allocation
                result[field] = base_n
            else:
                best = stats["best"]
                if best >= 0.70:
                    # Proven field — triple quota
                    result[field] = base_n * 3
                elif best >= 0.30:
                    # Moderate field — 1.5x quota
                    result[field] = max(base_n, int(base_n * 1.5))
                elif best < 0.0:
                    # Historically weak — minimum quota (don't skip entirely)
                    result[field] = max(1, base_n // 4)
                else:
                    result[field] = base_n

        proven  = sum(1 for v in result.values() if v >= base_n * 3)
        moderate = sum(1 for v in result.values() if base_n <= v < base_n * 3)
        unknown  = sum(1 for v in result.values() if v == base_n)
        weak     = sum(1 for v in result.values() if v < base_n)
        total_sims = sum(result.values())
        logger.info(
            f"Field quota map: {proven} proven (n={base_n*3}), "
            f"{moderate} moderate, {unknown} unknown, {weak} weak. "
            f"Total budget: {total_sims} candidates."
        )
        return result

    def category_win_rates(self) -> dict:
        """Return win rate (tests_pass fraction) per field category."""
        return {
            cat: round(v["passes"] / v["n"], 3) if v["n"] > 0 else 0.0
            for cat, v in self._cat_wins.items()
        }
