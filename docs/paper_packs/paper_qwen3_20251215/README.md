# Paper Candidate Pack — Qwen3-0.6B (2025-12-15)

- git_commit: `7df1278`
- source_runs:
  - frozen: `artifacts_paper_qwen3_frozen_20251215_094901`
  - single: `artifacts_paper_qwen3_single_20251215_162455`
  - ecology: `artifacts_paper_qwen3_ecology_20251215_162621`

## Summary Table

| condition | gens | episodes | avg_roi (mean) | merges_total | colonies_max | qd_cov_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 50 | 1727 | 0.996 | 0 | 0 | 0.0% |
| single | 50 | 25 | 0.064 | 0 | 0 | 0.0% |
| ecology | 50 | 2058 | 1.421 | 20 | 1 | 20.8% |

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
- `plots/`: curated plots copied from each run + an aggregate comparison plot
- `summary.json`: machine-readable summary extracted from `gen_summaries.jsonl`
