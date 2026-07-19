import random
import logging
from src.utils import build_simulation_payload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full operator catalogue scraped from platform.worldquantbrain.com/learn/operators
# ---------------------------------------------------------------------------

# Time-series operators — format: (x, d)
TS_UNARY_OPS = [
    "ts_min",           # minimum of x over past d days
    "ts_max",           # maximum of x over past d days
    "ts_mean",          # simple average of x over past d days
    "ts_sum",           # sum of x over past d days
    "ts_std_dev",       # standard deviation of x over past d days
    "ts_rank",          # rank of today's x within the past d days
    "ts_scale",         # scale x to 0-1 range over past d days
    "ts_zscore",        # z-score of x over past d days
    "ts_delta",         # x - ts_delay(x, d)  → momentum
    "ts_delay",         # value of x from d days ago
    "ts_product",       # product of x over past d days (geometric mean proxy)
    "ts_quantile",      # quantile-normalise x over past d days (Gaussian default)
    "ts_arg_max",       # days since the max occurred in the last d days
    "ts_arg_min",       # days since the min occurred in the last d days
    "ts_av_diff",       # x - ts_mean(x, d)
    "ts_count_nans",    # count of NaN values in x over past d days
]

# Time-series operators — format: (x, y, d)  [two-field ops]
TS_BINARY_OPS = [
    "ts_corr",          # Pearson correlation of x and y over past d days
    "ts_covariance",    # covariance of x and y over past d days
]

# Time-series operators — special signatures
TS_SPECIAL_OPS = {
    "ts_decay_linear":  "(x, d)",          # linear decay of x over past d days
    "hump":             "(x, 0.01)",       # reduce turnover — hump(x, threshold)
    "days_from_last_change": "(x)",        # days since x last changed value
    "ts_regress":       "(y, x, d, 0, 0)", # regression of y on x over d days
}

# Fields available on Brain — grouped by category (most relevant first)
PRICE_VOLUME_FIELDS = [
    "close", "open", "high", "low", "volume", "returns",
    "vwap", "cap", "adv5", "adv10", "adv20",
]

ANALYST_FIELDS = [
    "actual_cashflow_per_share_value_quarterly",
    "actual_dividend_value_quarterly",
    "anl_actual_eps_value_quarterly",
    "actual_sales_value_annual",
    "actual_sales_value_quarterly",
    "adj_net_income_avg",
    "adj_net_income_median",
]

FUNDAMENTAL_FIELDS = [
    "assets",
    "debt",
    "equity",
    "revenue",
    "net_income",
    "ebitda",
    "bookvalue",
    "pe",
    "pb",
    "cash",
]

SENTIMENT_FIELDS = [
    "sentiment",
]

# Combine all for "everything" mode
ALL_FIELDS = PRICE_VOLUME_FIELDS + ANALYST_FIELDS + FUNDAMENTAL_FIELDS + SENTIMENT_FIELDS

# Lookback windows (days) to try
LOOKBACKS = [5, 10, 20, 60, 120, 252]


class AlphaGenerator:
    """
    Generates alpha simulation payloads by combining data fields with the
    full suite of WorldQuant Brain operators.

    SAFE: Only builds and writes JSON — does NOT call any API.
    """

    def __init__(
        self,
        fields: list = None,
        lookbacks: list = None,
        regions: list = None,
        universes: list = None,
        mode: str = "price_volume",   # "price_volume" | "analyst" | "fundamental" | "all"
    ):
        self.regions = regions or ["USA"]
        self.universes = universes or ["TOP3000"]
        self.lookbacks = lookbacks or LOOKBACKS

        if fields:
            self.fields = fields
        elif mode == "analyst":
            self.fields = ANALYST_FIELDS
        elif mode == "fundamental":
            self.fields = FUNDAMENTAL_FIELDS
        elif mode == "all":
            self.fields = ALL_FIELDS
        else:
            self.fields = PRICE_VOLUME_FIELDS

    # ------------------------------------------------------------------
    # Expression builders
    # ------------------------------------------------------------------

    def _rank_raw(self, field: str) -> str:
        """rank(field)"""
        return f"rank({field})"

    def _ts_unary_rank(self, field: str, op: str, d: int) -> str:
        """rank(ts_op(field, d))"""
        return f"rank({op}({field}, {d}))"

    def _ts_ratio_rank(self, field: str, op: str, d: int) -> str:
        """rank(field / ts_op(field, d))  — relative-to-history"""
        return f"rank({field} / {op}({field}, {d}))"

    def _ts_delta_rank(self, field: str, d: int) -> str:
        """rank(field - ts_delay(field, d))  — momentum / change"""
        return f"rank({field} - ts_delay({field}, {d}))"

    def _ts_zscore_rank(self, field: str, d: int) -> str:
        """rank(ts_zscore(field, d))  — standardised signal"""
        return f"rank(ts_zscore({field}, {d}))"

    def _ts_scale_rank(self, field: str, d: int) -> str:
        """rank(ts_scale(field, d))  — 0-1 normalised"""
        return f"rank(ts_scale({field}, {d}))"

    def _ts_arg_max_rank(self, field: str, d: int) -> str:
        """rank(ts_arg_max(field, d))  — days since peak"""
        return f"rank(ts_arg_max({field}, {d}))"

    def _ts_arg_min_rank(self, field: str, d: int) -> str:
        """rank(ts_arg_min(field, d))  — days since trough"""
        return f"rank(ts_arg_min({field}, {d}))"

    def _ts_corr_rank(self, field_x: str, field_y: str, d: int) -> str:
        """rank(ts_corr(x, y, d))  — two-field correlation"""
        return f"rank(ts_corr({field_x}, {field_y}, {d}))"

    def _ts_regress_rank(self, field_y: str, field_x: str, d: int) -> str:
        """rank(ts_regress(y, x, d, 0, 0))  — regression residual"""
        return f"rank(ts_regress({field_y}, {field_x}, {d}, 0, 0))"

    def _hump_rank(self, field: str, d: int) -> str:
        """rank(ts_mean(hump(field, 0.01), d)) — low-turnover smoothed"""
        return f"rank(ts_mean(hump({field}, 0.01), {d}))"

    # ------------------------------------------------------------------
    # Pool builder
    # ------------------------------------------------------------------

    def build_generation_pool(self, count_per_field: int = 8) -> list:
        """
        Builds a list of candidate alpha payloads without executing anything.

        Parameters
        ----------
        count_per_field : int
            How many expressions to generate per data field.
        """
        alphas = []
        seen_exprs = set()  # Deduplicate
        alpha_index = 1

        def add(expr, field=None):
            nonlocal alpha_index
            if expr not in seen_exprs:
                seen_exprs.add(expr)
                region = random.choice(self.regions)
                universe = random.choice(self.universes)
                payload = self._to_payload(expr, f"Gen_{alpha_index}", region, universe)
                alphas.append(payload)
                alpha_index += 1

        for field in self.fields:
            # 1. Simple cross-sectional rank
            add(self._rank_raw(field))

            # 2. Time-series unary operators (pick random subset)
            ops_sample = random.sample(TS_UNARY_OPS, min(4, len(TS_UNARY_OPS)))
            for op in ops_sample:
                d = random.choice(self.lookbacks)
                add(self._ts_unary_rank(field, op, d))

            # 3. Relative-to-history ratios (use mean, std_dev)
            for op in ["ts_mean", "ts_std_dev"]:
                d = random.choice(self.lookbacks)
                add(self._ts_ratio_rank(field, op, d))

            # 4. Momentum / change signals
            for d in random.sample(self.lookbacks, min(2, len(self.lookbacks))):
                add(self._ts_delta_rank(field, d))

            # 5. Z-score and scale normalised signals
            for d in random.sample(self.lookbacks, min(2, len(self.lookbacks))):
                add(self._ts_zscore_rank(field, d))
                add(self._ts_scale_rank(field, d))

            # 6. Days-since-peak / trough signals (reversal indicators)
            d = random.choice(self.lookbacks)
            add(self._ts_arg_max_rank(field, d))
            add(self._ts_arg_min_rank(field, d))

            # 7. Hump (low-turnover version)
            d = random.choice(self.lookbacks)
            add(self._hump_rank(field, d))

        # 8. Two-field cross signals: corr & regression (price volume pairs)
        pv = PRICE_VOLUME_FIELDS
        for _ in range(20):
            x, y = random.sample(pv, 2)
            d = random.choice(self.lookbacks)
            add(self._ts_corr_rank(x, y, d))
            add(self._ts_regress_rank(x, y, d))

        logger.info(f"Generated pool: {len(alphas)} unique candidate alphas.")
        return alphas

    # ------------------------------------------------------------------
    # Payload helper
    # ------------------------------------------------------------------

    def _to_payload(
        self, expr: str, name: str, region: str = "USA", universe: str = "TOP3000"
    ) -> dict:
        payload = build_simulation_payload(
            expression=expr, region=region, universe=universe
        )
        payload["name"] = name
        return payload
