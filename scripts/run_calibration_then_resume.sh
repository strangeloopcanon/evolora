#!/usr/bin/env bash
set -euo pipefail

# Run a short calibration segment, then resume the same run directory to full length.
#
# This is a safe way to kick off long runs before sleeping:
# - Calibration validates the config + model load + basic dynamics in minutes.
# - Resume continues from checkpoint.pt in the same run_id directory (no restart).

VENV_BIN="${VENV_BIN:-.venv311/bin}"
PY="${PY:-$VENV_BIN/python}"

if [ ! -x "$PY" ]; then
  echo "Expected $PY – set VENV_BIN or create/refresh .venv311 and install deps first." >&2
  exit 1
fi

timestamp() {
  date +%Y%m%d_%H%M%S
}

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_calibration_then_resume.sh --config <yaml> [options]

Options:
  --output <dir>            Run directory (default: artifacts_calib_<timestamp>)
  --calib-gens <int>        Calibration generations (default: 10)
  --full-gens <int>         Full generations after resume (default: 50)
  --checkpoint-every <int>  Checkpoint cadence in generations (default: 1)
  --seed <int>              Grid seed (default: 777)
  --device <str>            Torch device override (e.g. mps/cpu/cuda)
  --batch-size <int>        Override synthetic batch size per generation
  --disable-human           Disable human bandit even if config enables it
  --final-holdout <jsonl>   Run a measurement-only holdout after the full run
  --final-holdout-sample-size <int>  Optional sample size for holdout tasks (default: all)
  --no-analyze              Skip final analyze step
EOF
}

CONFIG=""
OUTPUT=""
CALIB_GENS=10
FULL_GENS=50
CHECKPOINT_EVERY=1
SEED=777
DEVICE=""
BATCH_SIZE=""
DISABLE_HUMAN=0
ANALYZE=1
FINAL_HOLDOUT=""
FINAL_HOLDOUT_SAMPLE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"; shift 2 ;;
    --output)
      OUTPUT="${2:-}"; shift 2 ;;
    --calib-gens)
      CALIB_GENS="${2:-}"; shift 2 ;;
    --full-gens)
      FULL_GENS="${2:-}"; shift 2 ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="${2:-}"; shift 2 ;;
    --seed)
      SEED="${2:-}"; shift 2 ;;
    --device)
      DEVICE="${2:-}"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="${2:-}"; shift 2 ;;
    --disable-human)
      DISABLE_HUMAN=1; shift ;;
    --final-holdout)
      FINAL_HOLDOUT="${2:-}"; shift 2 ;;
    --final-holdout-sample-size)
      FINAL_HOLDOUT_SAMPLE="${2:-}"; shift 2 ;;
    --no-analyze)
      ANALYZE=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [ -z "$CONFIG" ]; then
  echo "--config is required" >&2
  usage
  exit 2
fi

if [ -z "$OUTPUT" ]; then
  OUTPUT="artifacts_calib_$(timestamp)"
fi

if ! [ -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

if [ "$FULL_GENS" -le 0 ] || [ "$CALIB_GENS" -le 0 ]; then
  echo "--calib-gens and --full-gens must be positive" >&2
  exit 2
fi

if [ "$CALIB_GENS" -ge "$FULL_GENS" ]; then
  echo "--calib-gens must be < --full-gens (got $CALIB_GENS vs $FULL_GENS)" >&2
  exit 2
fi

RESUME_GENS=$((FULL_GENS - CALIB_GENS))

EXTRA_ARGS=()
if [ -n "$DEVICE" ]; then
  EXTRA_ARGS+=(--device "$DEVICE")
fi
if [ -n "$BATCH_SIZE" ]; then
  EXTRA_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [ "$DISABLE_HUMAN" -eq 1 ]; then
  EXTRA_ARGS+=(--disable-human)
fi

FINAL_HOLDOUT_ARGS=()
if [ -n "$FINAL_HOLDOUT" ]; then
  if ! [ -f "$FINAL_HOLDOUT" ]; then
    echo "Holdout tasks not found: $FINAL_HOLDOUT" >&2
    exit 2
  fi
  FINAL_HOLDOUT_ARGS+=(--final-holdout "$FINAL_HOLDOUT")
  if [ -n "$FINAL_HOLDOUT_SAMPLE" ]; then
    FINAL_HOLDOUT_ARGS+=(--final-holdout-sample-size "$FINAL_HOLDOUT_SAMPLE")
  fi
fi

echo "[calibration] config=$CONFIG output=$OUTPUT gens=$CALIB_GENS seed=$SEED checkpoint_every=$CHECKPOINT_EVERY"
MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline "$PY" scripts/eval_gemma_long.py \
  --config "$CONFIG" \
  --generations "$CALIB_GENS" \
  --output "$OUTPUT" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

echo "[resume] resume_from=$OUTPUT gens=$RESUME_GENS seed=$SEED checkpoint_every=$CHECKPOINT_EVERY"
MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline "$PY" scripts/eval_gemma_long.py \
  --config "$CONFIG" \
  --resume-from "$OUTPUT" \
  --generations "$RESUME_GENS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$SEED" \
  "${FINAL_HOLDOUT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

if [ "$ANALYZE" -eq 1 ]; then
  echo "[analyze] $OUTPUT"
  MPLCONFIGDIR=$(mktemp -d) "$PY" scripts/analyze_ecology_run.py "$OUTPUT" --plots --report
fi

echo "Done: $OUTPUT"
