# Paper Candidate Pack — Qwen3-0.6B (2025-12-16)

- git_commit: `a099f94`
- source_runs:
  - frozen: `artifacts_paper_qwen3_frozen_20251215_094901`
  - single: `artifacts_paper_qwen3_single_20251216_115655`
  - ecology: `artifacts_paper_qwen3_ecology_20251215_162621`

## Summary Table

| condition | gens | episodes | avg_roi (mean) | merges_total | colonies_max | qd_cov_max | holdout_acc | holdout_avg_cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 50 | 1727 | 0.996 | 0 | 0 | 0.0% | 0.133 | 1.910 |
| single | 50 | 63 | 0.076 | 0 | 0 | 0.0% | 0.133 | 1.935 |
| ecology | 50 | 2058 | 1.421 | 20 | 1 | 20.8% | 0.133 | 1.769 |

## ROI Comparison

![ROI comparison](plots/roi_comparison.png)

## Ecology Highlights

### Ecology ROI

![Ecology ROI](plots/ecology/avg_roi.png)

### Ecology merges

![Ecology merges](plots/ecology/merges.png)

### Ecology colonies

![Ecology colonies](plots/ecology/colonies_count.png)

### Ecology QD coverage

![Ecology QD coverage](plots/ecology/qd_archive_coverage.png)

## Included Files

- `reports/`: copied `report.md` from each run
- `reports/`: copied `final_holdout.md` when present
- `plots/`: curated plots copied from each run + an aggregate comparison plot
- `summary.json`: machine-readable summary extracted from `gen_summaries.jsonl`
