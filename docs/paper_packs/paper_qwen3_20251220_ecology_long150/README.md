# Paper Candidate Pack — Qwen3-0.6B (2025-12-20)

- git_commit: `c80aaa7`
- source_runs:
  - frozen: `artifacts_paper_qwen3_frozen_20251215_094901`
  - single: `artifacts_paper_qwen3_single_20251216_115655`
  - ecology: `artifacts_paper_qwen3_ecology_long_20251216_222921`

## Summary Table

| condition | gens | records | episodes | avg_roi_last | merges_total* | colonies_max* | qd_cov_last* | holdout_acc | holdout_avg_cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 50 | 50 | 1727 | 0.943 | 0 | 0 | 0.0% | 0.133 | 1.910 |
| single | 50 | 50 | 63 | -0.060 | 0 | 0 | 0.0% | 0.133 | 1.935 |
| ecology | 150 | 35 | 7413 | 1.391 | 52 | 1 | 16.7% | 0.142 | 1.138 |

*Runs with `records < gens` were resumed after interruption; metrics marked `*` are derived from recorded generations only.

## Takeaways

- holdout_acc: ecology 0.142 vs best baseline 0.133 (Δ +0.008)
- holdout_avg_cost: ecology 1.138 vs best baseline 1.910 (×0.60)

## ROI Comparison

![ROI comparison](plots/roi_comparison.png)

## Included Files

- `reports/`: copied `report.md` from each run
- `reports/`: copied `final_holdout.md` when present
- `plots/`: curated plots copied from each run + an aggregate comparison plot
- `summary.json`: machine-readable summary extracted from `gen_summaries.jsonl`
