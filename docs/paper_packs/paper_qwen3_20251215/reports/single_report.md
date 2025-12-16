# Ecology Run Analysis

- Generations: 50
- Total episodes: 25
- Average ROI: 0.064 (median -0.012, range -0.051 – 0.975, σ 0.222)
- Average total reward: -0.198 (range -1.368 – 1.113)
- Average energy cost: 0.341
- Energy balance mean: 1.088 (range 0.000 – 11.705)
- Curriculum lp_mix active: mean 0.439 | last 0.379 (base mean 0.450)
- Active organelles per generation: 0 – 1 (bankrupt: 0 – 1)
- Bankruptcy culls: total 0 (max per generation 0)
- Assimilation merges (per summary): 0
- Team routes / promotions: 0 / 0
- Assimilation gating totals:
  - canary_failed: 0
  - low_energy: 36
  - low_power: 0
  - no_best_cell: 0
  - no_activity: 0
  - cooldown: 0
  - uplift_below_threshold: 14
  - cell_merges_exceeded: 0
  - insufficient_scores: 0
  - global_probe_failed: 0
  - holdout_failed: 0
  - topup_success: 8
  - topup_roi_blocked: 36
  - topup_cap_blocked: 0
  - topup_already_sufficient: 5
  - topup_disabled: 1
  - reserve_guard: 0
  - cautious_skip: 0
- Assimilation energy tuning: floor 0.500, ROI threshold 1.000
- Knowledge cache: writes 4 (denied 0); reads 7 (denied 3, hits 7)
  - Entries mean 3.28, latest 0; expired 4
- Power economics:
  - Episodes tracked: 25; avg power need 1.000; avg price multiplier 1.250
  - Evidence tokens minted/used: 0 / 0
- Assimilation tests: none recorded
- Colony selection: dissolved 0 / replicated 0
  - Pool mean members 0.00; pool mean pot 0.00
- Colony tier mean: avg 0.00, last 0.00
- Colony reserve guard (mean active colonies): 0.00
- Colony winter mode (mean active colonies): 0.00
- Colony hazard z-score (mean): 0.000
- Foraging traits (mean): beta 1.50, decay 0.30, ucb 0.20, budget 0.50
  - org_14a6b9b0 top cells: word.count:short (0.38), word.count:medium (0.37), math:short (0.18)
- Colony bandwidth: mean 0.000, last 0.000
  total members mean 0.00, last 0
  ΔROI mean 0.0000; variance ratio mean 1.000
  hazard members (max) 0
- Budget totals: mean 0.50, median 0.00, last 1
  cap max 120 (hit-rate 0.0%)
  zero-alloc mean 0.00; energy ratio mean 8.55; trait mean 0.50; policy mean 1.00
- Comms totals: posts 0 / reads 0 / credits 0
- Mutation operators invoked:
  - rank noise: 0, dropout masks: 0, duplications: 0
- Recent gating snapshots:
  - gen 046 org_14a6b9b0: low_energy | {"balance": 0.0, "generation": 46, "ticket": 0.5, "top_up": {"after": 0.0, "before": 0.0, "credited": 0.0, "fail_streak": 0.0, "floor": 0.5, "relief": 0.0, "roi": -0.006090130336985087, "roi_std": 0.010070606452270333, "roi_threshold": 1.0, "roi_threshold_effective": 0.0, "status": "skip_low_roi", "tokens_available": 0}}
  - gen 047 org_14a6b9b0: low_energy | {"balance": 0.0, "generation": 47, "ticket": 0.5, "top_up": {"after": 0.0, "before": 0.0, "credited": 0.0, "fail_streak": 0.0, "floor": 0.5, "relief": 0.0, "roi": -0.006090130336985087, "roi_std": 0.010070606452270333, "roi_threshold": 1.0, "roi_threshold_effective": 0.0, "status": "skip_low_roi", "tokens_available": 0}}
  - gen 048 org_14a6b9b0: low_energy | {"balance": 0.0, "generation": 48, "ticket": 0.5, "top_up": {"after": 0.0, "before": 0.0, "credited": 0.0, "fail_streak": 0.0, "floor": 0.5, "relief": 0.0, "roi": -0.006090130336985087, "roi_std": 0.010070606452270333, "roi_threshold": 1.0, "roi_threshold_effective": 0.0, "status": "skip_low_roi", "tokens_available": 0}}
  - gen 049 org_14a6b9b0: uplift_below_threshold | {"ema": 0.29184000000000004, "generation": 49, "relief": 0.0, "tau": 0.32, "threshold": 1.0, "uplift_gate": -0.08815999999999996}
  - gen 050 org_14a6b9b0: low_energy | {"balance": 0.0, "generation": 50, "ticket": 0.5, "top_up": {"after": 0.0, "before": 0.0, "credited": 0.0, "fail_streak": 0.0, "floor": 0.5, "relief": 0.0, "roi": -0.018736099422864336, "roi_std": 0.031179340044421806, "roi_threshold": 1.0, "roi_threshold_effective": 0.0, "status": "skip_low_roi", "tokens_available": 0}}
- Assimilation tests: none recorded
- ROI volatility (std across organelles): 0.000
- Evaluation accuracy: 66.67% (2/3)
  (reward weight 0.75)
  - Evaluation by family:
    * math: 100.00% (2/2), ΔROI μ -0.106, cost μ 1.067
    * word.count: 0.00% (0/1), ΔROI μ -1.161, cost μ 1.161
