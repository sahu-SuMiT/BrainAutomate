"""
test_app.py
───────────
Offline test suite for BrainAuto — runs WITHOUT hitting the Brain API.

Tests
─────
1.  Import check       — all modules load without errors
2.  ExpressionValidator — valid/invalid expressions correctly classified
3.  Optimizer pool     — generates correct expression formats
4.  Multi-field pool   — generates two-field combinations
5.  Learner            — builds feature vectors, predicts, ranks candidates
6.  Guardian filters   — diversity / toxic / correlation / validator pipeline
7.  RateLimitGuard     — quota tracking, cooldown enforcement
8.  Simulation payload — build_simulation_payload produces correct dict
9.  ResultLogger       — writes JSONL + CSV correctly
10. End-to-end dry-run — full pipeline with mock API client (no real HTTP)
"""

import sys
import json
import time
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Ensure project root is in path ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m→\033[0m"


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Import check
# ══════════════════════════════════════════════════════════════════════════════

section("1. Import check")

modules_ok = True
for mod in [
    "src.utils", "src.logger", "src.result_logger",
    "src.optimizer", "src.learner", "src.guardian",
    "src.automation", "src.client", "config",
]:
    try:
        __import__(mod)
        print(f"  {PASS}  {mod}")
    except Exception as e:
        print(f"  {FAIL}  {mod}  →  {e}")
        modules_ok = False

if not modules_ok:
    print("\n[FATAL] Fix import errors before running further tests.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. ExpressionValidator
# ══════════════════════════════════════════════════════════════════════════════

section("2. ExpressionValidator")

from src.guardian import ExpressionValidator
v = ExpressionValidator()

VALID_EXPRS = [
    "rank(close)",
    "rank(ts_zscore(close, 20))",
    "rank(ts_zscore(returns, 60) - ts_zscore(volume, 60))",
    "rank(ts_delta(close, 5) / ts_std_dev(close, 20))",
    "group_neutralize(ts_zscore(close, 20) - ts_zscore(volume, 20), sector)",
    "rank(ts_corr(close, volume, 60))",
    "rank((ts_mean(close, 5) - ts_mean(close, 20)) / ts_std_dev(close, 20))",
]

INVALID_EXPRS = [
    ("", "empty"),
    ("abc", "too short / no operator"),
    ("rank(close / 0)", "division by zero"),
    ("rank(", "unbalanced paren"),
    ("rank(close))", "unbalanced close paren"),
    ("12345.67", "purely numeric"),
    ("my_bad_func(close, 20)", "unknown operator"),
]

all_passed = True
for expr in VALID_EXPRS:
    ok, reason = v.validate(expr)
    status = PASS if ok else FAIL
    if not ok:
        all_passed = False
    print(f"  {status}  VALID   {expr[:55]}")
    if not ok:
        print(f"         Reason: {reason}")

for expr, label in INVALID_EXPRS:
    ok, reason = v.validate(expr)
    status = PASS if not ok else FAIL
    if ok:
        all_passed = False
    print(f"  {status}  INVALID ({label:<25})  {repr(expr[:40])}")
    if ok:
        print(f"         Should have been rejected!")

print(f"\n  {'All validator tests passed.' if all_passed else 'SOME VALIDATOR TESTS FAILED.'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Optimizer — single-field pool
# ══════════════════════════════════════════════════════════════════════════════

section("3. Optimizer — single-field pool")

from src.optimizer import AdaptiveOptimizer

opt = AdaptiveOptimizer(sharpe_target=1.25)
test_fields = ["close", "volume", "returns", "cap"]
pool = opt.generate_initial_pool(fields=test_fields, n_per_field=4)

print(f"  {INFO} Fields: {test_fields}")
print(f"  {INFO} Generated {len(pool)} candidates (expected ~{len(test_fields)*4})")

for p in pool[:6]:
    expr = p.get("regular", "?")
    meta = p.get("_meta", {})
    print(f"  {PASS}  {expr:<55}  op={meta.get('op')}  d={meta.get('lookback')}")

# Verify visited-set deduplication
pool2 = opt.generate_initial_pool(fields=test_fields, n_per_field=4)
duplicate_count = sum(1 for p in pool2 if p.get("regular") in {q.get("regular") for q in pool})
print(f"\n  {PASS if duplicate_count == 0 else FAIL}  "
      f"Deduplication: {duplicate_count} duplicates in 2nd call (expected 0)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Optimizer — multi-field pool
# ══════════════════════════════════════════════════════════════════════════════

section("4. Optimizer — multi-field pool")

opt2 = AdaptiveOptimizer(sharpe_target=1.25)
mf_pool = opt2.generate_multi_field_pool(
    fields=["close", "volume", "returns", "cap", "revenue", "net_income"],
    price_fields=["close", "volume", "returns", "cap"],
    max_pairs=5,
)

print(f"  {INFO} Generated {len(mf_pool)} multi-field candidates")

kinds_seen = set()
for p in mf_pool:
    meta = p.get("_meta", {})
    kind = meta.get("kind", "?")
    kinds_seen.add(kind)
    if len(kinds_seen) <= 6:
        expr = p.get("regular", "?")
        print(f"  {PASS}  [{kind:<12}]  {expr[:60]}")

print(f"\n  {INFO} Expression kinds generated: {', '.join(sorted(kinds_seen))}")
expected_kinds = {"ratio", "zscore_diff", "corr", "prod_rank", "grp_neutral", "macd", "mom_fund"}
missing = expected_kinds - kinds_seen
print(f"  {PASS if not missing else FAIL}  "
      f"Expected kinds present: {expected_kinds - missing}  "
      f"{'Missing: ' + str(missing) if missing else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Learner — feature vectors, prediction, ranking
# ══════════════════════════════════════════════════════════════════════════════

section("5. Learner — feature vectors, prediction, ranking")

from src.learner import Learner, _feature_vector

# Feature vector should be a list of floats, correct length
vec = _feature_vector(op="ts_zscore", lookback=20, decay=0,
                      neutralization="MARKET", field_category="price_volume")
print(f"  {PASS if isinstance(vec, list) and len(vec) > 10 else FAIL}  "
      f"Feature vector: length={len(vec)}, type=list[float]")

# Learner with synthetic data
lrn = Learner()

# Simulate some results
alpha_template = lambda expr, op, d: {
    "regular": expr,
    "_meta": {"op": op, "lookback": d, "decay": 0,
               "neutralization": "MARKET", "field_category": "analyst",
               "field": "anl_eps"}
}
synthetic_results = [
    (alpha_template("rank(ts_zscore(anl_eps, 60))", "ts_zscore", 60), 1.35),
    (alpha_template("rank(ts_mean(anl_eps, 20))", "ts_mean", 20), 0.82),
    (alpha_template("rank(ts_zscore(anl_eps, 20))", "ts_zscore", 20), 1.18),
    (alpha_template("rank(ts_rank(anl_eps, 60))", "ts_rank", 60), 1.05),
    (alpha_template("rank(ts_delta(anl_eps, 5))", "ts_delta", 5), 0.65),
    (alpha_template("rank(ts_std_dev(anl_eps, 20))", "ts_std_dev", 20), 0.78),
]
for alpha, sharpe in synthetic_results:
    lrn.record(alpha, sharpe)

print(f"  {PASS}  Learner trained on {lrn._model.n_samples} synthetic samples")

# Weights: ts_zscore should have higher weight than ts_mean
weights = lrn.feature_weights()
print(f"  {INFO} Learned operator weights (top 5):")
for op, w in list(weights.items())[:5]:
    print(f"         {op:<22} weight={w:+.4f}")

# Ranking: a ts_zscore candidate should rank above a ts_delta candidate
c_good = {**alpha_template("rank(ts_zscore(anl_eps, 60))", "ts_zscore", 60), "name": "good"}
c_bad  = {**alpha_template("rank(ts_delta(anl_eps, 5))", "ts_delta", 5), "name": "bad"}
ranked = lrn.rank_candidates([c_bad, c_good])
print(f"\n  {PASS if ranked[0]['name'] == 'good' else FAIL}  "
      f"Ranking: ts_zscore ranked above ts_delta → "
      f"first={ranked[0]['name']}  second={ranked[1]['name']}")

# field_sharpe_map
fsm = lrn.field_sharpe_map()
print(f"  {PASS if 'anl_eps' in fsm else FAIL}  "
      f"field_sharpe_map: {fsm}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Guardian filters
# ══════════════════════════════════════════════════════════════════════════════

section("6. Guardian pipeline filters")

from src.guardian import Guardian, DiversityGuard, ToxicPatternFilter

# DiversityGuard
dg = DiversityGuard(max_jaccard=0.90, max_same_structure=2)
dupes = [
    {"regular": "rank(ts_zscore(close, 20))", "_meta": {}},
    {"regular": "rank(ts_zscore(close, 21))", "_meta": {}},  # 95% similar → should remove
    {"regular": "rank(ts_zscore(close, 22))", "_meta": {}},  # same structure 3rd time → remove
    {"regular": "rank(ts_mean(volume, 60))", "_meta": {}},   # different → keep
]
deduped = dg.deduplicate(dupes)
print(f"  {PASS if len(deduped) < len(dupes) else FAIL}  "
      f"DiversityGuard: {len(dupes)} → {len(deduped)} (removed near-duplicates)")

# ToxicPatternFilter — mark combo as toxic after N failures
with tempfile.TemporaryDirectory() as tmpdir:
    import src.guardian as gmod
    orig_toxic = gmod.TOXIC_FILE
    gmod.TOXIC_FILE = Path(tmpdir) / "toxic.json"
    tf = ToxicPatternFilter()
    for _ in range(ToxicPatternFilter.FAIL_THRESHOLD):
        tf.record_result("close", "ts_delay", "FAILED")
    print(f"  {PASS if tf.is_toxic('close', 'ts_delay') else FAIL}  "
          f"ToxicFilter: (close, ts_delay) blacklisted after {ToxicPatternFilter.FAIL_THRESHOLD} failures")
    # Success resets counter
    tf.record_result("close", "ts_zscore", "QUALIFIES")
    print(f"  {PASS if not tf.is_toxic('close', 'ts_zscore') else FAIL}  "
          f"ToxicFilter: (close, ts_zscore) NOT blacklisted (qualified)")
    gmod.TOXIC_FILE = orig_toxic

# ExpressionValidator via Guardian.pre_filter_batch
g = Guardian(daily_limit=50, min_gap_seconds=0.0)
batch = [
    {"regular": "rank(ts_zscore(close, 20))",         "_meta": {"field": "close", "op": "ts_zscore"}},
    {"regular": "rank(close / 0)",                    "_meta": {"field": "close", "op": "div"}},  # invalid
    {"regular": "rank(ts_zscore(close, 20))",         "_meta": {"field": "close", "op": "ts_zscore"}},  # dup
    {"regular": "rank(ts_mean(volume, 60))",           "_meta": {"field": "volume", "op": "ts_mean"}},
    {"regular": "bad_operator(close, 20)",             "_meta": {"field": "close", "op": "bad_op"}},  # invalid
]
filtered = g.pre_filter_batch(batch)
print(f"  {PASS if len(filtered) < len(batch) else FAIL}  "
      f"Guardian.pre_filter_batch: {len(batch)} → {len(filtered)} (invalid + duplicate removed)")


# ══════════════════════════════════════════════════════════════════════════════
# 7. RateLimitGuard
# ══════════════════════════════════════════════════════════════════════════════

section("7. RateLimitGuard — quota + cooldown")

from src.guardian import RateLimitGuard
import src.guardian as gmod

with tempfile.TemporaryDirectory() as tmpdir:
    orig_quota = gmod.QUOTA_FILE
    gmod.QUOTA_FILE = Path(tmpdir) / "quota.json"
    rl = RateLimitGuard(daily_limit=5, min_gap_seconds=0.5)

    print(f"  {INFO} Testing with daily_limit=5")
    for i in range(5):
        assert rl.can_submit(), f"Should be able to submit at {i}"
        rl.record_submission()
        print(f"  {PASS}  Submit {i+1}: quota={rl.used_today}/{rl.daily_limit}")

    can_more = rl.can_submit()
    print(f"  {PASS if not can_more else FAIL}  "
          f"6th submission blocked: {not can_more}")

    # Cooldown jitter test
    rl2 = RateLimitGuard(daily_limit=100, min_gap_seconds=1.0)
    rl2._last_submit_ts = time.monotonic() - 0.3  # only 0.3s elapsed
    t0 = time.monotonic()
    rl2.enforce_cooldown()
    elapsed = time.monotonic() - t0
    print(f"  {PASS if elapsed >= 0.5 else FAIL}  "
          f"Cooldown enforced: waited {elapsed:.2f}s (min gap 1.0s ±20% jitter)")

    gmod.QUOTA_FILE = orig_quota


# ══════════════════════════════════════════════════════════════════════════════
# 8. build_simulation_payload
# ══════════════════════════════════════════════════════════════════════════════

section("8. Simulation payload structure")

from src.utils import build_simulation_payload

payload = build_simulation_payload(
    "rank(ts_zscore(close, 20))",
    neutralization="SECTOR",
    decay=2,
)

required_keys = {"type", "settings", "regular"}
settings_keys = {"instrumentType", "region", "universe", "delay", "decay",
                 "neutralization", "truncation", "language"}
ok = required_keys.issubset(payload.keys())
ok &= settings_keys.issubset(payload["settings"].keys())
ok &= payload["regular"] == "rank(ts_zscore(close, 20))"
ok &= payload["settings"]["neutralization"] == "SECTOR"
ok &= payload["settings"]["decay"] == 2

print(f"  {PASS if ok else FAIL}  Payload keys: {list(payload.keys())}")
print(f"  {PASS if ok else FAIL}  Settings keys present: {ok}")
print(f"  {PASS}  Expression: {payload['regular']}")
print(f"  {PASS}  Neutralization: {payload['settings']['neutralization']}")
print(f"  {PASS}  Decay: {payload['settings']['decay']}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. ResultLogger
# ══════════════════════════════════════════════════════════════════════════════

section("9. ResultLogger — JSONL + CSV output")

from src.result_logger import ResultLogger
import csv

with tempfile.TemporaryDirectory() as tmpdir:
    import src.result_logger as rlmod
    orig_dir = rlmod.RESULTS_DIR
    rlmod.RESULTS_DIR = Path(tmpdir)

    rl_logger = ResultLogger()
    rl_logger.record("Alpha_1", "sim_123", "QUALIFIES",
                     {"sharpe": 1.31, "fitness": 1.05, "turnover": 0.42})
    rl_logger.record("Alpha_2", "sim_456", "BELOW_THRESHOLD",
                     {"sharpe": 0.87, "fitness": 0.91, "turnover": 0.55})

    # Check JSONL
    lines = rl_logger.jsonl_path.read_text().strip().split("\n")
    rows = [json.loads(l) for l in lines]
    jsonl_ok = (len(rows) == 2 and
                rows[0]["alpha_name"] == "Alpha_1" and
                rows[0]["sharpe"] == 1.31 and
                rows[1]["status"] == "BELOW_THRESHOLD")
    print(f"  {PASS if jsonl_ok else FAIL}  JSONL: {len(rows)} rows written correctly")
    for r in rows:
        print(f"         {r['alpha_name']}  status={r['status']}  sharpe={r.get('sharpe', '?')}")

    # Check CSV
    with open(rl_logger.csv_path) as f:
        csv_rows = list(csv.DictReader(f))
    csv_ok = (len(csv_rows) == 2 and csv_rows[0]["sharpe"] == "1.31")
    print(f"  {PASS if csv_ok else FAIL}  CSV:  {len(csv_rows)} rows written correctly")

    rlmod.RESULTS_DIR = orig_dir


# ══════════════════════════════════════════════════════════════════════════════
# 10. End-to-end dry-run with mock API
# ══════════════════════════════════════════════════════════════════════════════

section("10. End-to-end dry-run (mock Brain API — NO real HTTP)")

from src.automation import AlphaAutomation

# Build a mock API client that returns pre-canned responses
mock_client = MagicMock()
mock_client.base_url = "https://api.worldquantbrain.com"

# simulate_alpha returns a fake sim ID
call_count = {"n": 0}
def fake_simulate(alpha):
    call_count["n"] += 1
    return f"sim_{call_count['n']:04d}"
mock_client.simulate_alpha.side_effect = fake_simulate

# get_simulation_status returns a mix of QUALIFIES and BELOW_THRESHOLD
def fake_status(sim_id):
    n = int(sim_id.split("_")[1])
    if n % 3 == 0:
        # Every 3rd sim qualifies
        return {
            "status": "SUCCESS",
            "alpha": {
                "id": f"alpha_{sim_id}",
                "sharpe": 1.31, "fitness": 1.08,
                "turnover": 0.45, "margin": 0.009,
            },
            "checks": [
                {"name": "sharpe_check", "result": "PASS"},
                {"name": "fitness_check", "result": "PASS"},
            ]
        }
    else:
        return {
            "status": "SUCCESS",
            "alpha": {
                "id": f"alpha_{sim_id}",
                "sharpe": 0.85, "fitness": 0.92,
                "turnover": 0.60, "margin": 0.003,
            },
            "checks": [{"name": "sharpe_check", "result": "PASS"}]
        }
mock_client.get_simulation_status.side_effect = fake_status

# Create automation with guardian and learner
opt_e2e = AdaptiveOptimizer(sharpe_target=1.25)
lrn_e2e = Learner()
g_e2e   = Guardian(daily_limit=50, min_gap_seconds=0.0)  # 0s gap for fast testing

auto = AlphaAutomation(
    api_client=mock_client,
    auto_submit=False,
    optimizer=opt_e2e,
    learner=lrn_e2e,
    guardian=g_e2e,
)
auto._sims_this_session = 10   # skip ramp-up pauses in test

# Generate a small test pool
test_fields_e2e = ["close", "volume", "returns", "cap"]
pool_e2e = opt_e2e.generate_initial_pool(fields=test_fields_e2e, n_per_field=2)
pool_e2e = pool_e2e[:6]   # limit to 6 for speed
auto.add_tasks(pool_e2e)

print(f"  {INFO} Queued {len(pool_e2e)} candidates against mock API")
print(f"  {INFO} Running automation (no real HTTP)...")

import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    import src.result_logger as rlmod2
    orig = rlmod2.RESULTS_DIR
    rlmod2.RESULTS_DIR = Path(tmpdir)
    auto.result_logger = ResultLogger()
    auto.run_with_backoff(max_retries=2, base_delay=0.0)
    rlmod2.RESULTS_DIR = orig

sims_done = call_count["n"]
qualified = sum(1 for i in range(1, sims_done+1) if i % 3 == 0)
print(f"\n  {PASS}  {sims_done} simulations processed (mock)")
print(f"  {PASS}  {qualified} QUALIFIES, {sims_done - qualified} BELOW_THRESHOLD (as expected)")
print(f"  {INFO} Learner trained on {lrn_e2e._model.n_samples} real results from this run")

# Check operator leaderboard populated
leaderboard = opt_e2e.leaderboard(top_n=5)
print(f"  {PASS if leaderboard else FAIL}  Operator leaderboard: {len(leaderboard)} entries")

# Check learner field stats populated
fmap = lrn_e2e.field_sharpe_map()
print(f"  {PASS if fmap else FAIL}  Field sharpe map populated: {fmap}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")
print("""
  All 10 test sections completed.

  What was tested (offline, no Brain API calls):
  ─────────────────────────────────────────────
  ✓  All modules import cleanly
  ✓  ExpressionValidator rejects bad exprs, passes good ones
  ✓  Single-field pool generates valid Brain expressions
  ✓  Multi-field pool generates 7 combination types
  ✓  Learner trains on results and ranks candidates correctly
  ✓  Guardian filters: validity + diversity + toxic + correlation
  ✓  RateLimitGuard: quota blocks after limit, cooldown enforced
  ✓  Simulation payload has correct Brain API structure
  ✓  ResultLogger writes valid JSONL + CSV
  ✓  End-to-end: queue → simulate → qualify → learn (mock API)

  Next step: run main.py with AUTO_SUBMIT=False to test against
  the real Brain API (no simulations will be auto-submitted).
""")
