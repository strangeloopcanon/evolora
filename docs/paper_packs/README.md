# Paper Packs

This folder contains small, tracked “paper packs”: summary tables plus curated plots copied out of
gitignored `artifacts_*` run directories.

- `paper_qwen3_20251215/README.md` — Qwen3‑0.6B paper-suite baselines (frozen vs single vs ecology).
- `paper_qwen3_20251220_ecology_long150/README.md` — Same baselines, but swapping in a longer ecology run (150 gens) + final holdout metrics.

If a run directory contains `final_holdout.json` / `final_holdout.md`, the pack also includes
holdout accuracy + cost columns and copies the holdout plots.

## Generate a new pack

1. Run the three paper baselines (this creates three `artifacts_*` run directories and writes plots + reports into each):
```bash
scripts/run_paper_ecology_suite.sh all
```

Optional: include a final, measurement-only holdout suite in each run:
```bash
FINAL_HOLDOUT_TASKS=config/evaluation/paper_qwen3_holdout_v1.jsonl \
FINAL_HOLDOUT_SAMPLE_SIZE=120 \
scripts/run_paper_ecology_suite.sh all
```

2. Package the three run directories into a tracked folder under `docs/paper_packs/`:
```bash
python scripts/paper_pack.py \
  --frozen  artifacts_paper_qwen3_frozen_<timestamp> \
  --single  artifacts_paper_qwen3_single_<timestamp> \
  --ecology artifacts_paper_qwen3_ecology_<timestamp> \
  --output  docs/paper_packs/paper_qwen3_$(date +%Y%m%d)
```

The output pack includes:
- `README.md` with a summary table + key plots (and holdout deltas if present)
- `summary.json` (machine-readable)
- `plots/` and `reports/` copied out of the source runs

## What a run directory must contain

`scripts/paper_pack.py` expects each run directory to include:
- `gen_summaries.jsonl` (written by `scripts/run_evolution.py`)
- `visuals/` + `report.md` (written by `scripts/analyze_ecology_run.py --plots --report`)
- optional: `final_holdout.json` / `final_holdout.md` (written by `scripts/run_evolution.py --final-holdout ...`)

## Common pitfalls

- **Matplotlib cache issues on macOS**: always run analysis/plotting with `MPLCONFIGDIR="$(mktemp -d)"` (the suite scripts already do).
- **Hugging Face cache size**: long runs will download model weights. If you run out of disk, move the cache:
  - `export HF_HOME=/path/to/big_disk/hf_cache` (or `HUGGINGFACE_HUB_CACHE=...`)
- **Pack naming**: include `YYYYMMDD` in the pack folder name (e.g. `paper_qwen3_20251220_...`) so the generated README header shows the correct date.
- **Resumed runs**: packs may show `records < gens` if a run was resumed after interruption; `scripts/paper_pack.py` marks some metrics with `*` in that case.
