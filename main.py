import time
import logging
from src.client import BrainAPIClient
from src.automation import AlphaAutomation
from src.optimizer import AdaptiveOptimizer
from src.learner import Learner
from src.guardian import Guardian
from config import CREDENTIALS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Fallback field list — used ONLY if the Brain API is unreachable.
# These are known-good fields from each category. The real run uses live data.
# ---------------------------------------------------------------------------
FALLBACK_FIELDS = {
    "price_volume": [
        "close", "open", "high", "low", "volume", "returns",
        "vwap", "cap", "adv5", "adv10", "adv20",
    ],
    "analyst": [
        "actual_cashflow_per_share_value_quarterly",
        "actual_dividend_value_quarterly",
        "anl_actual_eps_value_quarterly",
        "actual_sales_value_annual",
        "actual_sales_value_quarterly",
    ],
    "fundamental": [
        "assets", "debt", "equity", "revenue", "net_income",
        "ebitda", "bookvalue", "cash", "pe", "pb",
    ],
}

PRICE_VOLUME_KEYWORDS = {
    "close", "open", "high", "low", "volume", "returns", "vwap",
    "cap", "adv", "price", "turnover", "sharesout", "sharefloat",
}


def _is_price_volume(field_id: str) -> bool:
    """Heuristic: check if a field ID looks like a price/volume field."""
    fid = field_id.lower()
    return any(kw in fid for kw in PRICE_VOLUME_KEYWORDS)


def fetch_fields_by_category(client, learner_stats: dict, top_n_each: int = 10) -> dict:
    """
    Fetch top fields from each category separately so we always get a
    diverse mix — not just price-volume fields dominating the list.

    Returns
    -------
    dict: {category: [field_ids]}
    """
    categories = {
        "price_volume": "close",       # Brain API search parameter mapping
        "fundamental":  "fundamental",
        "analyst":      "analyst",
    }

    result = {}

    for cat_label, api_cat in categories.items():
        try:
            logging.info(f"  Fetching top {top_n_each} {cat_label} fields...")
            fields = client.fetch_top_fields(
                category=api_cat,
                top_n=top_n_each,
                min_coverage=0.5,
                learner_stats=learner_stats,
            )
            ids = [f["id"] for f in fields]
            result[cat_label] = ids
            logging.info(f"  ✓ {cat_label}: {ids[:3]}{'...' if len(ids) > 3 else ''}")
        except Exception as e:
            logging.warning(f"  ✗ Could not fetch {cat_label} fields: {e}")
            logging.warning(f"    Using fallback list for {cat_label}.")
            result[cat_label] = FALLBACK_FIELDS.get(cat_label, [])

        time.sleep(2.0)   # avoid rapid-fire 429s between category fetches

    return result


def print_field_inventory(fields_by_cat: dict):
    """Print a clear summary of which fields will be used."""
    print("\n── Fields Selected ──────────────────────────────────────────────")
    total = 0
    for cat, ids in fields_by_cat.items():
        print(f"  {cat:<15} ({len(ids)} fields)")
        for fid in ids:
            print(f"    • {fid}")
        total += len(ids)
    print(f"  {'TOTAL':<15} {total} unique fields")
    print("─────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Configuration — edit these before running
    # ----------------------------------------------------------------
    AUTO_SUBMIT   = False
    SHARPE_TARGET = 1.25
    DAILY_LIMIT   = 100     # Brain simulation quota per day (adjust for your tier)
    MIN_GAP_SEC   = 3.0     # Minimum seconds between simulation submissions

    # How many top fields to fetch per category (analyst, fundamental, price_volume)
    TOP_N_PER_CATEGORY = 8

    THRESHOLDS = {
        "min_sharpe":   SHARPE_TARGET,
        "min_fitness":  1.00,
        "max_turnover": 0.70,
        "min_margin":   0.005,
    }

    # ----------------------------------------------------------------
    # Bootstrap
    # ----------------------------------------------------------------
    client    = BrainAPIClient(CREDENTIALS)
    optimizer = AdaptiveOptimizer(sharpe_target=SHARPE_TARGET)

    learner = Learner(sharpe_target=SHARPE_TARGET)
    learner.load_history()

    guardian = Guardian(
        daily_limit=DAILY_LIMIT,
        min_gap_seconds=MIN_GAP_SEC,
        max_self_correlation=0.80,
        max_intra_batch_similarity=0.90,
    )
    guardian.initialise(client)

    automation = AlphaAutomation(
        api_client=client,
        auto_submit=AUTO_SUBMIT,
        thresholds=THRESHOLDS,
        optimizer=optimizer,
        learner=learner,
        guardian=guardian,
    )

    # ----------------------------------------------------------------
    # Fetch top fields — one API call per category so we always get
    # a balanced mix of price_volume + analyst + fundamental fields
    # ----------------------------------------------------------------
    print("\nFetching top fields from Brain API (by category)...")
    fields_by_cat = fetch_fields_by_category(
        client,
        learner_stats=learner.field_sharpe_map(),
        top_n_each=TOP_N_PER_CATEGORY,
    )

    # Print exactly what we're using — full transparency
    print_field_inventory(fields_by_cat)

    # Combine all into one list; separate price_volume for multi-field pairing
    pv_fields    = fields_by_cat.get("price_volume", [])
    other_fields = (
        fields_by_cat.get("analyst", []) +
        fields_by_cat.get("fundamental", [])
    )
    all_field_ids = pv_fields + other_fields

    # Build a field → category lookup so _meta gets correctly tagged for the Learner
    field_category_map = {}
    for cat, fids in fields_by_cat.items():
        for fid in fids:
            field_category_map[fid] = cat

    # Deduplicate (in case an API call returned overlapping fields)
    seen = set()
    all_field_ids = [f for f in all_field_ids if not (f in seen or seen.add(f))]

    # ----------------------------------------------------------------
    # Build field quota map — channels budget to proven fields
    # Proven fields (best_sharpe >= 0.70): 3x candidates
    # Unknown fields (never tested):        1x candidates
    # Weak fields (best_sharpe < 0.00):     1 candidate minimum
    # ----------------------------------------------------------------
    BASE_N = 4   # base candidates per field
    quota_map = learner.field_quota_map(all_field_ids, base_n=BASE_N)

    # ----------------------------------------------------------------
    # Generate candidate pool using fetched fields
    # Pool 1: Single-field (TS + CS operators) — quota-routed per field
    # Pool 2: Multi-field (ratio, zscore-diff, MACD, corr, group-neutral)
    # Pool 3: Evolved (mutations + crossover of best historical expressions)
    # Pool 4: Special ops (hump, kth_element, ts_step)
    # ----------------------------------------------------------------
    single_field = optimizer.generate_initial_pool(
        fields=all_field_ids,
        n_per_field=BASE_N,
        field_category_map=field_category_map,
        quota_map=quota_map,
    )
    multi_field = optimizer.generate_multi_field_pool(
        fields=all_field_ids,
        price_fields=pv_fields,
        max_pairs=20,
        field_category_map=field_category_map,
    )
    evolved = optimizer.generate_evolved_pool(
        fields=all_field_ids,
        field_category_map=field_category_map,
        max_candidates=50,
        field_sharpe_map=learner.field_sharpe_map(),
    )
    special = optimizer.generate_special_ops_pool(
        fields=all_field_ids,
        field_category_map=field_category_map,
    )

    # Merge — guardian will expression-validate, deduplicate, filter toxic + correlated
    all_alphas = single_field + multi_field + evolved + special
    automation.add_tasks(all_alphas)

    print(f"\n{'='*60}")
    print(f"  BrainAuto  —  {len(all_alphas)} candidates generated")
    print(f"    {len(single_field)} single-field  +  {len(multi_field)} multi-field")
    print(f"    {len(evolved)} evolved  +  {len(special)} special-ops")
    print(f"    Operators: {len(__import__('src.optimizer', fromlist=['TS_UNARY_OPS']).TS_UNARY_OPS)} TS + 3 special (hump/kth_element/ts_step)")
    print(f"    {len(pv_fields)} price/vol fields  +  {len(other_fields)} analyst/fundamental")
    # Show quota distribution
    proven_fields  = [f for f in all_field_ids if quota_map.get(f, BASE_N) >= BASE_N * 3]
    unknown_fields = [f for f in all_field_ids if quota_map.get(f, BASE_N) == BASE_N]
    weak_fields    = [f for f in all_field_ids if quota_map.get(f, BASE_N) < BASE_N]
    print(f"    Field quota: {len(proven_fields)} proven×{BASE_N*3}  {len(unknown_fields)} unknown×{BASE_N}  {len(weak_fields)} weak×1")
    print(f"  Auto-submit : {'ON' if AUTO_SUBMIT else 'OFF (dry run)'}")
    print(f"  Sharpe target: {SHARPE_TARGET}")
    print(f"  Strategy: quota-routed + multi-field + genetic-evolution + goal-aligned UCB + Pareto")
    print(f"{'='*60}\n")

    automation.run_with_backoff(max_retries=5)

    # ----------------------------------------------------------------
    # End-of-run diagnostics
    # ----------------------------------------------------------------
    print(f"\n── Guardian Status ──")
    print(f"  {guardian.status_report()}")
    print(f"  Toxic (field, op) patterns blocked: {len(guardian.toxic._toxic)}")

    print("\n── Operator Leaderboard ──")
    for row in optimizer.leaderboard(top_n=10):
        print(f"  {row['operator']:<24} avg_sharpe={row['avg_sharpe']}  trials={row['trials']}")

    print("\n── Field Leaderboard (learned from your results) ──")
    for row in learner.top_fields(n=10):
        print(f"  {row['field']:<45} best={row['best_sharpe']}  avg={row['avg_sharpe']}  trials={row['trials']}")

    print("\n── Learned Operator Weights ──")
    weights = learner.feature_weights()
    for op, w in list(weights.items())[:10]:
        print(f"  {op:<24} weight={w}")

    print("\nAll tasks finished.")
