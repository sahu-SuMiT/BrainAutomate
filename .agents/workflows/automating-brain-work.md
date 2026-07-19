---
description: Step-by-step workflow for improving the BrainAuto optimizer.

PRIMARY GOAL: Every decision must produce alphas with Sharpe >= 1.25 and Fitness >= 1.0.

---

BEFORE ANY CHANGE — Always Start Here

Step 0: Operator sync check
  python3 -c "
  from src.optimizer import TS_UNARY_OPS
  from src.evolver import TS_UNARY_OPS as EVO_OPS
  diff = set(TS_UNARY_OPS).symmetric_difference(set(EVO_OPS))
  print('Sync:', 'CLEAN' if not diff else diff)
  "

Step 1: Check historical data
  python3 -c "
  from src.learner import Learner
  l = Learner(); l.load_history()
  print(f'Samples: {l._model.n_samples}, kappa={l._ucb_kappa():.3f}')
  for r in l.top_fields(5): print(r)
  "

Step 2: Run full tests
  python3 test_optimizer.py

---

WORKFLOW A — Adding a New Operator
1. Does it produce an ECONOMIC SIGNAL? (not a data quality tool)
2. If YES: add to TS_UNARY_OPS in optimizer.py, evolver.py, AND OPERATORS in learner.py
3. NEVER add: ts_backfill, ts_count_nans, days_from_last_change, last_diff_value, ts_diff
4. Run operator sync check after every change

---

WORKFLOW B — Debugging Low Sharpe
1. Check top fields: python3 -c "from src.learner import Learner; l=Learner(); l.load_history(); print(l.top_fields(10))"
2. Are data-cleaning ops appearing in results? Remove them.
3. Try SUBINDUSTRY neutralization for fundamental alphas (+0.1-0.2 Sharpe)
4. Lookback ranges: fundamental=60-252d, price momentum=5-20d, mixed=20-60d

---

WORKFLOW C — Improving a Near-Miss (0.85 <= Sharpe < 1.25)
This is the highest-ROI workflow. Steps in order:
1. Binary search lookback: probe d//2, d*3//4, d*3//2, d*2
2. Hill-climb ALL 20 settings combos: 4 neutralizations x 5 decay values (0,2,4,6,8)
3. Seed into ensemble pool, run GeneticEvolver with 5x mutation budget
4. Try same operator on fundamental/analyst field if currently on price_volume

---

WORKFLOW D — Field Quota Routing
Proven fields (best_sharpe >= 0.70) get 3x simulation budget.
Moderate fields (best_sharpe 0.30-0.69) get 1.5x.
Unknown fields get 1x. Weak fields (best_sharpe < 0.00) get minimum 1.
Use: learner.field_quota_map(all_fields, base_n=4)

---

WORKFLOW E — Evolver Health Check
Verify: near-miss parents (seed_sharpe >= 1.0) get op_limit=5 and all 6 mutations.
Weak parents (seed_sharpe < 0.7) get op_limit=1 (operator swap only).
Crossover only between top-10 parents.

---

WORKFLOW F — System Health Check
  python3 -c "
  from src.optimizer import AdaptiveOptimizer, TS_UNARY_OPS, TS_SPECIAL_OPS
  from src.evolver import GeneticEvolver
  from src.learner import Learner
  print('All imports OK')
  print('TS_UNARY_OPS:', len(TS_UNARY_OPS), 'ops')
  print('TS_SPECIAL_OPS:', TS_SPECIAL_OPS)
  "

---

KEY CONSTANTS (need >= 50 sim results to change):
  SHARPE_TARGET = 1.25     (NEVER change)
  near_miss_min = 0.85
  kappa = 1.2*exp(-n/200) + 0.3
  gap_decay = 0.4
  n_bootstrap = 7
  pareto_weights = [0.5 Sharpe, 0.3 Fitness, 0.2 Margin]
---
