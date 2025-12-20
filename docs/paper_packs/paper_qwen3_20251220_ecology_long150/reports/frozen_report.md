# Ecology Run Analysis

- Generations: 50
- Total episodes: 1727
- Average ROI: 0.996 (median 0.980, range 0.304 – 1.605, σ 0.282)
- Average total reward: -1.117 (range -2.257 – 0.000)
- Average energy cost: 1.428
- Energy balance mean: 2.828 (range 2.375 – 3.211)
- Curriculum lp_mix active: mean 0.670 | last 0.700 (base mean 0.450)
- Active organelles per generation: 0 – 16 (bankrupt: 0 – 16)
- Bankruptcy culls: total 0 (max per generation 0)
- Assimilation merges (per summary): 0
- Team routes / promotions: 0 / 0
- Prompt scaffolds applied (latest generation): math:18, word.count:9
- Assimilation gating totals:
  - canary_failed: 0
  - low_energy: 356
  - low_power: 0
  - no_best_cell: 0
  - no_activity: 36
  - cooldown: 0
  - uplift_below_threshold: 396
  - cell_merges_exceeded: 0
  - insufficient_scores: 0
  - global_probe_failed: 0
  - holdout_failed: 0
  - topup_success: 159
  - topup_roi_blocked: 357
  - topup_cap_blocked: 0
  - topup_already_sufficient: 268
  - topup_disabled: 4
  - reserve_guard: 0
  - cautious_skip: 0
- Population refreshes: 4 (latest: {'count': 1, 'reason': 'no_merges', 'no_merge_counter': 12})
- Assimilation energy tuning: floor 0.612, ROI threshold 1.041
- Power economics:
  - Episodes tracked: 1727; avg power need 1.000; avg price multiplier 1.250
  - Evidence tokens minted/used: 0 / 0
- Assimilation tests: none recorded
- Colony selection: dissolved 0 / replicated 0
  - Pool mean members 0.00; pool mean pot 0.00
- Colony tier mean: avg 0.00, last 0.00
- Colony reserve guard (mean active colonies): 0.00
- Colony winter mode (mean active colonies): 0.00
- Colony hazard z-score (mean): 0.000
- Foraging traits (mean): beta 2.02, decay 0.78, ucb 0.23, budget 0.82
  - org_67351eff top cells: math:short (1.28)
- Colony bandwidth: mean 0.000, last 0.000
  total members mean 0.00, last 0
  ΔROI mean 0.0000; variance ratio mean 1.000
  hazard members (max) 0
- Budget totals: mean 34.54, median 32.00, last 32
  cap max 120 (hit-rate 0.0%)
  zero-alloc mean 0.00; energy ratio mean 0.00; trait mean 0.80; policy mean 1.00
- Comms totals: posts 0 / reads 0 / credits 0
- Mutation operators invoked:
  - rank noise: 552, dropout masks: 496, duplications: 433
- Recent gating snapshots:
  - gen 050 org_28d70847: low_energy | {"balance": 0.0, "generation": 50, "ticket": 0.5, "top_up": {"after": 0.0, "before": 0.0, "credited": 0.0, "fail_streak": 0.0, "floor": 0.512, "relief": 0.0, "roi": -0.048272530324015975, "roi_std": 0.007289522429474973, "roi_threshold": 1.0, "roi_threshold_effective": 0.0, "status": "skip_low_roi", "tokens_available": 0}}
  - gen 050 org_78652548: uplift_below_threshold | {"ema": 0.40480000000000005, "generation": 50, "relief": 0.13999999999999999, "tau": 0.35, "threshold": 1.0, "uplift_gate": 0.11480000000000001}
  - gen 050 org_f9b81826: low_energy | {"balance": 0.0, "generation": 50, "ticket": 0.5, "top_up": {"after": 0.0, "before": 0.0, "credited": 0.0, "fail_streak": 0.0, "floor": 0.512, "relief": 0.0, "roi": -0.03801761847924419, "roi_std": 0.0033981381357222167, "roi_threshold": 1.0, "roi_threshold_effective": 0.0, "status": "skip_low_roi", "tokens_available": 0}}
  - gen 050 org_1803d0b3: low_energy | {"balance": 0.0, "generation": 50, "ticket": 0.5, "top_up": {"after": 0.0, "before": 0.0, "credited": 0.0, "fail_streak": 0.0, "floor": 0.512, "relief": 0.0, "roi": -0.04795522930468657, "roi_std": 0.004485619789642785, "roi_threshold": 1.0, "roi_threshold_effective": 0.0, "status": "skip_low_roi", "tokens_available": 0}}
  - gen 050 org_bda92f55: uplift_below_threshold | {"ema": 0.456, "generation": 50, "relief": 0.13999999999999999, "tau": 0.35, "threshold": 1.0, "uplift_gate": 0.16599999999999998}
- Assimilation tests: none recorded
- ROI volatility (std across organelles): 0.439
- Evaluation accuracy: 66.67% (2/3)
  (reward weight 0.75)
  - Evaluation by family:
    * math: 100.00% (2/2), ΔROI μ -25.483, cost μ 26.606
    * word.count: 0.00% (0/1), ΔROI μ -26.706, cost μ 26.706
