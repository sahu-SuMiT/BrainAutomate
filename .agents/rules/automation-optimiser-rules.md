---
trigger: always_on
glob:
description: Rules for BrainAuto — WorldQuant Brain alpha automation.

PRIMARY GOAL: Produce alphas with Sharpe >= 1.25, Fitness >= 1.0, smooth equity curve.

---

RULE 1 — Operator Hygiene

ALLOWED signal operators (use these as primary expressions):
  ts_mean, ts_sum, ts_std_dev, ts_min, ts_max
  ts_rank, ts_scale, ts_zscore
  ts_delta, ts_delay, ts_av_diff
  ts_product, ts_quantile, ts_decay_linear
  ts_arg_max, ts_arg_min
  rank, group_rank, group_zscore, group_neutralize
  kth_element (percentile signal)
  hump (noise-reduction wrapper ONLY: rank(ts_mean(hump(field,0.01), d)))
  ts_step (time-trend: rank(ts_corr(field, ts_step(1), d)))

NEVER USE as primary signals (data quality tools, not alpha):
  ts_backfill, ts_count_nans, days_from_last_change, last_diff_value, ts_diff

---

RULE 2 — Field Selection Priority

  TIER 1 (3x quota):   best_sharpe >= 0.70  (proven — most quota)
  TIER 2 (1.5x quota): best_sharpe 0.30-0.69 (moderate)
  TIER 3 (1x quota):   never tested (unknown potential)
  TIER 4 (min 1 sim):  best_sharpe < 0.00 (weak — minimal quota)

Priority categories: analyst (anl4_fs_*) > fundamental (fnd6_*) > price_volume

---

RULE 3 — Near-Miss Protocol (0.85 <= Sharpe < 1.25)

  0.85 <= sharpe < 1.00: binary search on lookback + settings hill-climb
  1.00 <= sharpe < 1.25: all above + seed into ensemble pool + genetic mutation
  sharpe < 0.85:         settings hill-climb only
  sharpe >= 1.25:        QUALIFYING ALPHA — log it

Actions: (1) Binary search lookback, (2) Hill-climb all 4 neutralizations x 5 decays,
(3) If sharpe >= 1.0, immediately seed into ensemble pool for GeneticEvolver.

---

RULE 4 — Genetic Evolver Budget

  gap <= 0.25 (Sharpe >= 1.0): op_limit=5, all 6 mutations
  gap <= 0.55 (Sharpe >= 0.7): op_limit=3, all mutations
  gap >  0.55 (Sharpe <  0.7): op_limit=1, operator swap only

6 mutation types: operator swap, lookback shift, field swap, outer wrap,
settings mutation, crossover (op+lookback from A, field from B)

---

RULE 5 — UCB Scoring Formula (always goal-aligned)

  ucb_raw    = mean + kappa * sigma
  gap        = max(1.25 - mean, 0)
  gap_weight = exp(-gap / 0.4)
  score      = ucb_raw * gap_weight

  kappa = 1.2 * exp(-n / 200) + 0.3  (decays from 1.5 to 0.3)

---

RULE 6 — Pareto Tracking
Track 3 objectives on EVERY result: Sharpe (0.5), Fitness (0.3), Margin (0.2)

RULE 7 — Ensemble Pairing
Prefer cross-style pairs: momentum x fundamental. Avoid: momentum x momentum.

RULE 8 — Constants (need 50+ sims to change)
  near_miss_min=0.85, kappa=(1.2, 0.3), gap_decay=0.4, n_bootstrap=7

RULE 9 — Quota Discipline
Before any new operator: Is it an economic signal? Would Brain researchers use it?

RULE 10 — Code Sync Invariants
  TS_UNARY_OPS in optimizer.py == evolver.py == OPERATORS in learner.py (always)
  generate_special_ops_pool() generates ONLY: kth_element, hump, ts_step
  automation.py feedback loop must call BOTH learner.record() AND learner.record_outcome()
  Pareto front updated on every simulation result
  visited.json saved after every run
---
