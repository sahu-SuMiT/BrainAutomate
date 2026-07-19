"""
verify_learning.py - End-to-end verification that the model truly learns from past data.
"""
import json, sys
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("results")

# ── 1. JSONL inventory ──────────────────────────────────────────────
print("=" * 65)
print("  STEP 1: JSONL File Inventory")
print("=" * 65)

all_entries, old_no_meta, new_with_meta = [], 0, 0
for path in sorted(RESULTS_DIR.glob("*.jsonl")):
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    count, has_meta = 0, 0
    for line in lines:
        try:
            row = json.loads(line)
            if row.get("sharpe") is not None:
                all_entries.append(row)
                count += 1
                if row.get("operator") or row.get("field_category"):
                    has_meta += 1
                else:
                    old_no_meta += 1
        except: pass
    new_with_meta += has_meta
    print(f"  {path.name:45s}  {count:3d} entries  ({has_meta} with meta)")

print(f"\n  Total entries with sharpe : {len(all_entries)}")
print(f"  WITH meta fields          : {new_with_meta}  <- newly fixed")
print(f"  WITHOUT meta fields       : {old_no_meta}  <- old entries (pre-fix)")

# ── 2. Meta field quality check ─────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 2: Meta Field Quality (newest 15 meta-tagged entries)")
print("=" * 65)
meta_entries = [e for e in all_entries if e.get("operator") or e.get("field_category")]
field_cats_seen, operators_seen = Counter(), Counter()
for e in meta_entries[-15:]:
    op  = e.get("operator", "MISSING")
    fc  = e.get("field_category", "MISSING")
    fld = e.get("field", "MISSING")
    sh  = e.get("sharpe", "?")
    field_cats_seen[fc] += 1
    operators_seen[op]  += 1
    print(f"  op={op:18s} cat={fc:15s} field={fld[:22]:22s} sharpe={sh}")
print(f"\n  Field categories: {dict(field_cats_seen)}")
print(f"  Operators       : {dict(operators_seen)}")

# ── 3. Learner warm-start ────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 3: Learner Warm-Start Quality")
print("=" * 65)
sys.path.insert(0, ".")
from src.learner import Learner
learner = Learner()
learner.load_history()
model = learner._model
print(f"  Samples trained on     : {model.n_samples}")
print(f"  Non-zero weights       : {sum(1 for w in model._w if abs(w)>1e-6)}/{len(model._w)}")
all_zero = all(abs(w) < 1e-9 for w in model._w)
print(f"  All weights zero?      : {'YES <- NOT LEARNING' if all_zero else 'NO <- Learning happening'}")

print(f"\n  Top 5 operator weights:")
for op, w in list(learner.feature_weights().items())[:5]:
    bar = "+" * max(0, int(abs(w)*15))
    print(f"    {op:22s}  {w:+.4f}  {bar}")

print(f"\n  Top 8 fields by best Sharpe:")
for row in learner.top_fields(n=8):
    print(f"    {row['field'][:38]:38s}  best={row['best_sharpe']:+.3f}  avg={row['avg_sharpe']:+.3f}  trials={row['trials']}")

# ── 4. Ensemble pool ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 4: Ensemble Pool Persistence")
print("=" * 65)
pool_file = Path("ensemble_pool.json")
if pool_file.exists():
    pool = json.loads(pool_file.read_text())
    print(f"  ensemble_pool.json EXISTS  ({len(pool)} alphas persisted)")
    for a in pool[:5]:
        print(f"    {a.get('name','?'):30s} expr={a.get('regular','')[:45]}")
else:
    print("  ensemble_pool.json NOT FOUND yet (will be created once near-miss alphas accumulate)")

# ── 5. Prediction differentiation test ──────────────────────────────
print("\n" + "=" * 65)
print("  STEP 5: Prediction Quality (does model differentiate?)")
print("=" * 65)
test_cases = [
    ("ts_mean  pv  lb=20",  {"op":"ts_mean",   "lookback":20,  "decay":0, "neutralization":"MARKET",      "field_category":"price_volume"}),
    ("ts_mean  fnd lb=20",  {"op":"ts_mean",   "lookback":20,  "decay":0, "neutralization":"MARKET",      "field_category":"fundamental"}),
    ("rank     anl lb=None",{"op":"rank",      "lookback":None,"decay":0, "neutralization":"SUBINDUSTRY", "field_category":"analyst"}),
    ("ts_zscore pv lb=60",  {"op":"ts_zscore", "lookback":60,  "decay":4, "neutralization":"INDUSTRY",    "field_category":"price_volume"}),
    ("ts_delta  fnd lb=5",  {"op":"ts_delta",  "lookback":5,   "decay":0, "neutralization":"SECTOR",      "field_category":"fundamental"}),
    ("ts_rank  anl lb=252", {"op":"ts_rank",   "lookback":252, "decay":0, "neutralization":"MARKET",      "field_category":"analyst"}),
]
preds = []
for label, meta in test_cases:
    pred = learner.predict_sharpe({"_meta": meta})
    preds.append(pred)
    print(f"  {label:25s}  predicted_sharpe = {pred:+.4f}")

spread = max(preds) - min(preds)
print(f"\n  Prediction spread (max-min): {spread:.4f}")
print(f"  {'Good  <- model differentiates between candidates' if spread > 0.01 else 'FLAT  <- model may not be differentiating yet (needs more data)'}")

print("\n" + "=" * 65)
print("  VERIFICATION COMPLETE")
print("=" * 65)
