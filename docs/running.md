# Running Evolora

This is a practical runbook for running longer experiments locally.

## Setup

CI-style setup (creates `.venv`):
```bash
make setup
```

On macOS for longer runs, Python 3.11 is often more stable. To force 3.11, recreate `.venv` with it:
```bash
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-dev.txt
```

## Long runs (resumable)

Use `scripts/run_evolution.py` for resumable runs via `checkpoint.pt`.

Fresh run:
```bash
PYTHONPATH=src MPLCONFIGDIR="$(mktemp -d)" AGENT_MODE=baseline .venv/bin/python scripts/run_evolution.py \
  --config config/experiments/paper_qwen3_ecology.yaml \
  --generations 50 \
  --checkpoint-every 5 \
  --seed 777 \
  --disable-human \
  --output artifacts_paper_qwen3_ecology_long_$(date +%Y%m%d_%H%M%S)
```

Resume an existing run directory (continues from `<run_dir>/checkpoint.pt`):
```bash
PYTHONPATH=src MPLCONFIGDIR="$(mktemp -d)" AGENT_MODE=baseline .venv/bin/python scripts/run_evolution.py \
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
  --full-gens 50 \
  --checkpoint-every 5 \
  --seed 777 \
  --disable-human \
  --output artifacts_paper_qwen3_ecology_long_$(date +%Y%m%d_%H%M%S)
```

You can also run the same workflow via `make`:
```bash
make calibrate-resume CONFIG=config/experiments/paper_qwen3_ecology.yaml FULL_GENS=50
```

## Package results (paper packs)

To copy a small, tracked summary (tables + curated plots) out of gitignored `artifacts_*` run directories, use:
```bash
python scripts/paper_pack.py --help
```

## Final holdout evaluation (measurement-only)

After the run completes, you can optionally score a fixed holdout and write `final_holdout.json` / `final_holdout.md` into the run directory:
```bash
PYTHONPATH=src MPLCONFIGDIR="$(mktemp -d)" AGENT_MODE=baseline .venv/bin/python scripts/run_evolution.py \
  --config config/experiments/paper_qwen3_ecology.yaml \
  --resume-from <run_dir> \
  --generations 0 \
  --disable-human \
  --final-holdout config/evaluation/paper_qwen3_holdout_v1.jsonl \
  --final-holdout-sample-size 120
```

## SFT baseline (compute-matched)

Train a standard SFT LoRA with a compute-matched budget from an evolution run for fair comparison:

```bash
# Recommended: match evo "training" compute via FLOPs (more stable than tokens)
.venv/bin/python scripts/run_sft.py \
  --checkpoint <run_dir>/checkpoint.pt \
  --match-budget-field train_flops \
  --backprop-multiplier 3.0 \
  --attn-implementation eager \
  --engine manual \
  --resume \
  --data training_data.jsonl \
  --output sft_baseline_matched

# Alternatives: wall-clock match or an explicit token budget
.venv/bin/python scripts/run_sft.py \
  --checkpoint <run_dir>/checkpoint.pt \
  --match-budget-field wall_clock_seconds \
  --attn-implementation eager \
  --engine manual \
  --resume \
  --data training_data.jsonl \
  --output sft_baseline_wallclock

.venv/bin/python scripts/run_sft.py \
  --token-budget 100000 \
  --data training_data.jsonl \
  --output sft_baseline_100k
```

The training data JSONL should have lines like:
```json
{"prompt": "What is 2+2?", "completion": "4"}
```

Output includes:
- `lora_adapter.safetensors` - compatible with `HostKernel.load_organelle_adapter()`
- `peft_adapter/` - HuggingFace PEFT format
- `sft_metadata.json` - training stats and token counts

## Analyze a run
```bash
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/analyze_ecology_run.py <run_dir> --plots --report
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/evoscope.py <run_dir>
```

## Backprop plasticity (optional)

The default ecology uses Hebbian-like updates inside each organelle. For regex generalization experiments,
you can switch organelles to per-organelle backprop updates:

```bash
EVOLORA_PLASTICITY=backprop scripts/run_regex_generalization_evo_vs_sft.sh --no-sft --no-eval-id
```

Notes:
- `scripts/evaluate_holdout.py` can evaluate a specific organelle via `--evo-organelle-id org_...` if you want to sanity-check selection.

## macOS stability notes
- Always set `MPLCONFIGDIR="$(mktemp -d)"` for analysis/plots.
- If Metal/MPS memory pressure is still a problem, try `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before starting a long run.
- If SFT training hits NaNs on MPS, use `--engine manual` (default on MPS) and `--attn-implementation eager` (SDPA backward can be unstable on some setups).
