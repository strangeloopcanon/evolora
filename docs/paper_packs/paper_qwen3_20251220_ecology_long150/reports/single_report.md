# Ecology Run Analysis

- Generations: 50
- Total episodes: 63
- Average ROI: 0.076 (median -0.048, range -0.078 – 0.709, σ 0.222)
- Average total reward: -1.463 (range -3.306 – 0.928)
- Average energy cost: 1.607
- Energy balance mean: 0.584 (range 0.000 – 8.394)
- Curriculum lp_mix active: mean 0.456 | last 0.430 (base mean 0.450)
- Active organelles per generation: 1 – 1 (bankrupt: 0 – 0)
- Bankruptcy culls: total 0 (max per generation 0)
- Assimilation merges (per summary): 0
- Team routes / promotions: 0 / 0
- Assimilation gating totals:
  - canary_failed: 0
  - low_energy: 0
  - low_power: 0
  - no_best_cell: 0
  - no_activity: 0
  - cooldown: 7
  - uplift_below_threshold: 43
  - cell_merges_exceeded: 0
  - insufficient_scores: 0
  - global_probe_failed: 0
  - holdout_failed: 0
  - topup_success: 4
  - topup_roi_blocked: 40
  - topup_cap_blocked: 0
  - topup_already_sufficient: 4
  - topup_disabled: 2
  - reserve_guard: 0
  - cautious_skip: 0
- Assimilation energy tuning: floor 0.259, ROI threshold 1.000
- Knowledge cache: writes 4 (denied 2); reads 4 (denied 2, hits 4)
  - Entries mean 0.52, latest 0; expired 0
- Power economics:
  - Episodes tracked: 63; avg power need 1.000; avg price multiplier 1.250
  - Evidence tokens minted/used: 0 / 0
- Assimilation tests: none recorded
- Colony selection: dissolved 0 / replicated 0
  - Pool mean members 0.00; pool mean pot 0.00
- Colony tier mean: avg 0.00, last 0.00
- Colony reserve guard (mean active colonies): 0.00
- Colony winter mode (mean active colonies): 0.00
- Colony hazard z-score (mean): 0.000
- Foraging traits (mean): beta 1.50, decay 0.30, ucb 0.20, budget 0.50
  - org_7805811b top cells: word.count:short (0.06), word.count:medium (-0.04), math:short (-0.05)
- Colony bandwidth: mean 0.000, last 0.000
  total members mean 0.00, last 0
  ΔROI mean 0.0000; variance ratio mean 1.000
  hazard members (max) 0
- Budget totals: mean 1.26, median 1.00, last 1
  cap max 120 (hit-rate 0.0%)
  zero-alloc mean 0.00; energy ratio mean 883632.21; trait mean 0.50; policy mean 1.00
- Comms totals: posts 0 / reads 0 / credits 0
- Mutation operators invoked:
  - rank noise: 0, dropout masks: 0, duplications: 0
- Recent gating snapshots:
  - gen 036 org_7805811b: uplift_below_threshold | {"ema": 0.09334609443553282, "generation": 36, "relief": 0.0, "tau": 0.32, "threshold": 1.0, "uplift_gate": -0.2866539055644672}
  - gen 037 org_7805811b: uplift_below_threshold | {"ema": 0.09334609443553282, "generation": 37, "relief": 0.0, "tau": 0.32, "threshold": 1.0, "uplift_gate": -0.2866539055644672}
  - gen 038 org_7805811b: uplift_below_threshold | {"ema": 0.09334609443553282, "generation": 38, "relief": 0.0, "tau": 0.32, "threshold": 1.0, "uplift_gate": -0.2866539055644672}
  - gen 039 org_7805811b: uplift_below_threshold | {"ema": 0.09334609443553282, "generation": 39, "relief": 0.0, "tau": 0.32, "threshold": 1.0, "uplift_gate": -0.2866539055644672}
  - gen 040 org_7805811b: uplift_below_threshold | {"ema": 0.09334609443553282, "generation": 40, "relief": 0.0, "tau": 0.32, "threshold": 1.0, "uplift_gate": -0.2866539055644672}
- Assimilation tests: none recorded
- ROI volatility (std across organelles): 0.000
- Evaluation accuracy: 0.00% (0/2)
  (reward weight 0.75)
  - Evaluation by family:
    * math: 0.00% (0/1), ΔROI μ -0.668, cost μ 0.668
    * word.count: 0.00% (0/1), ΔROI μ -3.758, cost μ 3.758
