# Evolora — Symbiotic LLM Ecology
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/evolora)

Evolora is a research prototype where a frozen small LLM (e.g. Gemma‑270M or Qwen3‑0.6B) hosts a population of tiny LoRA adapters (“organelles”). Organelles compete for energy by solving tasks; the ecology tracks reward‑on‑cost (ROI), retires bankrupt organelles, and occasionally merges adapters when uplift + holdout checks pass.

## What you get
- YAML-configured experiments under `config/experiments/`
- CI-safe benchmark harness (stubbed backend; no model downloads)
- Run artifacts under `artifacts_*` (telemetry + plots)
- Analysis + visualization: `scripts/analyze_ecology_run.py`, `scripts/evoscope.py`
- Paper packs (tracked summaries): `docs/paper_packs/`

## Quickstart (CI-safe)
```bash
make all
AGENT_MODE=baseline .venv/bin/python scripts/benchmark_suite.py --mode ci
```

## Reproduce the paper-style Qwen3 suite
```bash
scripts/run_paper_ecology_suite.sh all
```

Optional: add a fixed, measurement-only holdout suite (paper packs will include holdout metrics):
```bash
FINAL_HOLDOUT_TASKS=config/evaluation/paper_qwen3_holdout_v1.jsonl \
FINAL_HOLDOUT_SAMPLE_SIZE=120 \
scripts/run_paper_ecology_suite.sh all
```

## Analyze a run
```bash
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/analyze_ecology_run.py <run_dir> --plots --report
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/evoscope.py <run_dir>
```

## Docs
- `docs/running.md` — how to run longer experiments (resume/holdout)
- `docs/ecology_overview.md` — conceptual overview + baselines framing
- `docs/architecture.md` — module-level architecture sketch
- `docs/paper_packs/README.md` — paper packs + examples

## Notes
- Full experiments download model weights from Hugging Face; use the CI benchmark mode above for a fast smoke test without downloads.
- On macOS, `MPLCONFIGDIR="$(mktemp -d)"` avoids matplotlib font-cache issues in long runs/plots.
