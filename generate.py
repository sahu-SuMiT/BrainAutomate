import json
import logging
from src.client import BrainAPIClient
from src.generator import AlphaGenerator
from config import CREDENTIALS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

OUTPUT_FILE = "alphas.json"

CATEGORY_MAP = {
    "1": ("price_volume",  None),           # Brain API category key for PV
    "2": ("fundamental",   None),
    "3": ("analyst",       None),
    "4": ("sentiment",     None),
    "5": ("all",           None),           # fetch from all categories
}


def main():
    print("=" * 60)
    print("         WorldQuant Brain Auto-Generator Tool")
    print("=" * 60)
    print()
    print("This tool will:")
    print("  1. Authenticate with the Brain API")
    print("  2. Fetch the TOP data fields for your chosen category")
    print("     (sorted by # of existing alphas — proven fields first)")
    print("  3. Generate alpha expression candidates LOCALLY")
    print("  4. Save them to alphas.json for your review")
    print("  ⚠️  No simulations are run at this stage.")
    print()
    print("Categories:")
    print("  1) Price Volume  — close, open, high, low, volume, returns, vwap ...")
    print("  2) Fundamental   — assets, debt, equity, revenue, PE, PB ...")
    print("  3) Analyst       — EPS, cashflow, dividends, sales (quarterly/annual) ...")
    print("  4) Sentiment     — sentiment scores")
    print("  5) All           — all of the above combined")
    print()
    choice = input("Select category [1-5] (default: 1): ").strip() or "1"
    category_key, api_cat = CATEGORY_MAP.get(choice, CATEGORY_MAP["1"])

    top_n = input("How many top fields to fetch? (default: 30): ").strip()
    try:
        top_n = int(top_n)
    except ValueError:
        top_n = 30

    count = input("Expressions per field (default: 8): ").strip()
    try:
        count = int(count)
    except ValueError:
        count = 8

    min_cov = input("Minimum field coverage % (default: 50): ").strip()
    try:
        min_cov = float(min_cov) / 100.0
    except ValueError:
        min_cov = 0.50

    print(f"\n[1/3] Authenticating with Brain API...")
    client = BrainAPIClient(CREDENTIALS)

    print(f"[2/3] Fetching top {top_n} fields (category: {category_key}, min coverage: {min_cov:.0%})...")
    raw_fields = client.fetch_top_fields(
        category=api_cat,
        min_coverage=min_cov,
        top_n=top_n,
    )

    if not raw_fields:
        print("[!] No fields returned. Check your credentials or adjust the filters.")
        return

    # Extract just the field IDs (e.g. "close", "actual_cashflow_per_share_value_quarterly")
    field_ids = [f["id"] for f in raw_fields]

    print(f"\n  Top fields fetched ({len(field_ids)}):")
    for i, f in enumerate(raw_fields[:10]):
        print(f"    {i+1:>2}. {f['id']:<45}  alphas={f.get('alphas', '?'):>6}  coverage={f.get('coverage', 0):.0%}")
    if len(raw_fields) > 10:
        print(f"    ... and {len(raw_fields) - 10} more")

    print(f"\n[3/3] Generating expression candidates...")
    generator = AlphaGenerator(
        fields=field_ids,        # <-- real fields from the API
        lookbacks=[5, 10, 20, 60, 120, 252],
        regions=["USA"],
        universes=["TOP3000"],
    )
    candidates = generator.build_generation_pool(count_per_field=count)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"\n[✓] Generated {len(candidates)} unique candidate alphas")
    print(f"[✓] Saved to: {OUTPUT_FILE}")
    print("\nPreview (first 15 expressions):")
    print("-" * 60)
    for c in candidates[:15]:
        print(f"  {c['name']:12s}  {c['regular']}")
    print(f"  ... ({len(candidates) - 15} more)")
    print("-" * 60)
    print(
        "\n👉 NEXT STEPS:\n"
        " 1. Open alphas.json — inspect and remove any expressions you don't like.\n"
        " 2. When ready: run `python3 main.py` to start the simulation queue.\n"
    )


if __name__ == "__main__":
    main()
