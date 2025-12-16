# Paper Packs

This folder contains small, tracked “paper packs”: summary tables plus curated plots copied out of
gitignored `artifacts_*` run directories.

- `paper_qwen3_20251215/README.md` — Qwen3‑0.6B paper-suite baselines (frozen vs single vs ecology).

If a run directory contains `final_holdout.json` / `final_holdout.md`, the pack also includes
holdout accuracy + cost columns and copies the holdout plots.

To generate a new pack from your own run directories, use `scripts/paper_pack.py`.
