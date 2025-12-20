# Running Evolora

This is a practical runbook for running longer experiments locally.

## Setup

CI-style setup (creates `.venv`):
```bash
make setup
```

On macOS for longer runs, a dedicated Python 3.11 venv is often more stable:
```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -U pip
pip install -r requirements-dev.txt
```

## Long runs (resumable)

Use `scripts/eval_gemma_long.py` for resumable runs via `checkpoint.pt`.

Fresh run:
```bash
PYTHONPATH=src MPLCONFIGDIR="$(mktemp -d)" AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py \
  --config config/experiments/paper_qwen3_ecology.yaml \
  --generations 50 \
  --checkpoint-every 5 \
  --seed 777 \
  --disable-human \
  --output artifacts_paper_qwen3_ecology_long_$(date +%Y%m%d_%H%M%S)
```

Resume an existing run directory (continues from `<run_dir>/checkpoint.pt`):
```bash
PYTHONPATH=src MPLCONFIGDIR="$(mktemp -d)" AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py \
  --config config/experiments/paper_qwen3_ecology.yaml \
  --resume-from <run_dir> \
  --generations 50 \
  --checkpoint-every 5 \
  --seed 777 \
  --disable-human
```

If a chunk is killed mid-run (e.g., macOS memory pressure), rerun the same resume command; it will pick up from the last checkpoint.

Notes:
- `gen_summaries.jsonl` is appended per generation, so summaries survive hard kills.
- On resume, telemetry files are truncated back to the last checkpoint to avoid duplicate episodes when `--checkpoint-every > 1`.

## Calibration → resume helper

This runs a short calibration segment first, then resumes the same run directory to the full length:
```bash
scripts/run_calibration_then_resume.sh \
  --config config/experiments/paper_qwen3_ecology.yaml \
  --calib-gens 10 \
  --full-gens 150 \
  --checkpoint-every 5 \
  --seed 777 \
  --disable-human \
  --output artifacts_paper_qwen3_ecology_long_$(date +%Y%m%d_%H%M%S)
```

## Final holdout evaluation (measurement-only)

After the run completes, you can optionally score a fixed holdout and write `final_holdout.json` / `final_holdout.md` into the run directory:
```bash
PYTHONPATH=src MPLCONFIGDIR="$(mktemp -d)" AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py \
  --config config/experiments/paper_qwen3_ecology.yaml \
  --resume-from <run_dir> \
  --generations 0 \
  --disable-human \
  --final-holdout config/evaluation/paper_qwen3_holdout_v1.jsonl \
  --final-holdout-sample-size 120
```

## Analyze a run
```bash
MPLCONFIGDIR="$(mktemp -d)" .venv311/bin/python scripts/analyze_ecology_run.py <run_dir> --plots --report
MPLCONFIGDIR="$(mktemp -d)" .venv311/bin/python scripts/evoscope.py <run_dir>
```

## macOS stability notes
- Always set `MPLCONFIGDIR="$(mktemp -d)"` for analysis/plots.
- If Metal/MPS memory pressure is still a problem, try `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before starting a long run.
