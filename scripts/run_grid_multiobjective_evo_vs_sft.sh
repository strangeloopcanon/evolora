#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner: grid multiobjective evolution → compute-matched SFT → in-distribution evaluation.
#
# This script is meant to test whether evolution shines as a *portfolio* on mixed objectives
# (specialists + routing), even if a single SFT adapter wins on any individual family.
#
# Defaults:
# - EVOLORA_PLASTICITY=backprop
# - Routed evo evaluation by family (--evo-eval-routing family)

VENV_BIN="${VENV_BIN:-}"
if [ -z "$VENV_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    VENV_BIN=".venv/bin"
  else
    VENV_BIN=".venv311/bin"
  fi
fi
PY="${PY:-$VENV_BIN/python}"

if [ ! -x "$PY" ]; then
  resolved_py="$(command -v "$PY" 2>/dev/null || true)"
  if [ -n "$resolved_py" ] && [ -x "$resolved_py" ]; then
    PY="$resolved_py"
  fi
fi

if [ ! -x "$PY" ]; then
  echo "Expected python at $PY – run \`make setup\` (creates .venv), or set PY to a python executable." >&2
  exit 1
fi

timestamp() {
  date +%Y%m%d_%H%M%S
}

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_grid_multiobjective_evo_vs_sft.sh [options]

Options:
  --config <yaml>            EcologyConfig YAML (default: config/experiments/qwen25_grid_multiobjective.yaml)
  --output <dir>             Output run directory (default: artifacts_grid_multiobj_evo_sft_<timestamp>)
  --train-size <int>         Generated SFT train tasks (default: 20000)
  --selection-size <int>     Generated selection tasks for evo organelle picking (default: 256)
  --id-holdout-size <int>    Generated in-distribution holdout tasks (default: 512)
  --calib-gens <int>         Calibration generations (default: 5)
  --full-gens <int>          Total generations target (default: 50)
  --checkpoint-every <int>   Checkpoint cadence in generations (default: 5)
  --seed <int>               Grid seed (default: 777)
  --device <str>             Torch device override (e.g. mps/cpu/cuda)
  --batch-size <int>         Override synthetic batch size per generation
  --disable-human            Disable human bandit even if config enables it
  --weights-json <json>      Optional JSON mapping of family -> weight for dataset generation
  --evo-eval-routing <mode>  One of: single,family,cell (default: family)
  --evo-selection-max-samples <int>  Optional cap on selection tasks evaluated (default: all)
  --evo-selection-top-k-by-roi <int> Shortlist organelles by ROI before selection (default: 0)
  --evo-selection-max-new-tokens <int> Max new tokens per selection generation (default: 96)
  --sft-match-budget-field   One of: total_tokens,train_tokens,total_flops,train_flops,wall_clock_seconds (default: train_flops)
  --backprop-multiplier      Convert evo budget → SFT budget (default: 3.0)
  --no-sft                   Skip SFT baseline training
  --no-eval                  Skip final holdout evaluation
EOF
}

CONFIG="config/experiments/qwen25_grid_multiobjective.yaml"
OUTPUT=""
TRAIN_SIZE=20000
SELECTION_SIZE=256
ID_HOLDOUT_SIZE=512
CALIB_GENS=5
FULL_GENS=50
CHECKPOINT_EVERY=5
SEED=777
DEVICE=""
BATCH_SIZE=""
DISABLE_HUMAN=0
WEIGHTS_JSON=""
EVO_EVAL_ROUTING="family"
EVO_SELECTION_MAX_SAMPLES=""
EVO_SELECTION_TOP_K_BY_ROI="0"
EVO_SELECTION_MAX_NEW_TOKENS="96"
RUN_SFT=1
RUN_EVAL=1
SFT_MATCH_BUDGET_FIELD="train_flops"
BACKPROP_MULTIPLIER="3.0"

while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"; shift 2 ;;
    --output)
      OUTPUT="${2:-}"; shift 2 ;;
    --train-size)
      TRAIN_SIZE="${2:-}"; shift 2 ;;
    --selection-size)
      SELECTION_SIZE="${2:-}"; shift 2 ;;
    --id-holdout-size)
      ID_HOLDOUT_SIZE="${2:-}"; shift 2 ;;
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
    --weights-json)
      WEIGHTS_JSON="${2:-}"; shift 2 ;;
    --evo-eval-routing)
      EVO_EVAL_ROUTING="${2:-}"; shift 2 ;;
    --evo-selection-max-samples)
      EVO_SELECTION_MAX_SAMPLES="${2:-}"; shift 2 ;;
    --evo-selection-top-k-by-roi)
      EVO_SELECTION_TOP_K_BY_ROI="${2:-}"; shift 2 ;;
    --evo-selection-max-new-tokens)
      EVO_SELECTION_MAX_NEW_TOKENS="${2:-}"; shift 2 ;;
    --sft-match-budget-field)
      SFT_MATCH_BUDGET_FIELD="${2:-}"; shift 2 ;;
    --backprop-multiplier)
      BACKPROP_MULTIPLIER="${2:-}"; shift 2 ;;
    --no-sft)
      RUN_SFT=0; shift ;;
    --no-eval)
      RUN_EVAL=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [ -z "$OUTPUT" ]; then
  OUTPUT="artifacts_grid_multiobj_evo_sft_$(timestamp)"
fi

if ! [ -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

MPLCONFIGDIR="$OUTPUT/mplconfig"
export MPLCONFIGDIR

if [ "$FULL_GENS" -le 0 ] || [ "$CALIB_GENS" -le 0 ]; then
  echo "--calib-gens and --full-gens must be positive" >&2
  exit 2
fi

export EVOLORA_PLASTICITY="${EVOLORA_PLASTICITY:-backprop}"

MODEL="$("$PY" -c "from symbiont_ecology import load_ecology_config; from pathlib import Path; cfg=load_ecology_config(Path('$CONFIG')); print(cfg.host.backbone_model)" | tr -d '\r')"
if [ -z "$MODEL" ]; then
  echo "Failed to infer base model from config: $CONFIG" >&2
  exit 2
fi

echo "[config] $CONFIG"
echo "[model]  $MODEL"
echo "[out]    $OUTPUT"
echo "[plasticity] $EVOLORA_PLASTICITY"

DATASET_DIR="$OUTPUT/datasets"
SFT_DATA="$DATASET_DIR/sft_train.jsonl"
ID_SELECTION_TASKS="$DATASET_DIR/selection_tasks.jsonl"
ID_HOLDOUT="$DATASET_DIR/holdout_tasks.jsonl"

if ! [ -f "$SFT_DATA" ] || ! [ -f "$ID_SELECTION_TASKS" ] || ! [ -f "$ID_HOLDOUT" ]; then
  echo "[data] generating distribution-matched datasets into $DATASET_DIR"
  EXTRA_DATA_ARGS=()
  if [ -n "$WEIGHTS_JSON" ]; then
    EXTRA_DATA_ARGS+=(--weights-json "$WEIGHTS_JSON")
  fi
  if [ "${#EXTRA_DATA_ARGS[@]}" -gt 0 ]; then
    AGENT_MODE=baseline "$PY" scripts/generate_grid_datasets.py \
      --config "$CONFIG" \
      --seed "$SEED" \
      --train-size "$TRAIN_SIZE" \
      --selection-size "$SELECTION_SIZE" \
      --holdout-size "$ID_HOLDOUT_SIZE" \
      --out-dir "$DATASET_DIR" \
      "${EXTRA_DATA_ARGS[@]}"
  else
    AGENT_MODE=baseline "$PY" scripts/generate_grid_datasets.py \
      --config "$CONFIG" \
      --seed "$SEED" \
      --train-size "$TRAIN_SIZE" \
      --selection-size "$SELECTION_SIZE" \
      --holdout-size "$ID_HOLDOUT_SIZE" \
      --out-dir "$DATASET_DIR"
  fi
fi

CHECKPOINT="$OUTPUT/checkpoint.pt"
current_generation() {
  local checkpoint_path="$1"
  if ! [ -f "$checkpoint_path" ]; then
    echo "0"
    return 0
  fi
  "$PY" - <<'PY' "$checkpoint_path" || echo "0"
import pickle
import sys
from pathlib import Path

checkpoint_path = Path(sys.argv[1])
try:
    state = pickle.loads(checkpoint_path.read_bytes())
    gen = int(state.get("generation", 0) or 0)
    print(str(gen))
except Exception:
    print("0")
PY
}

budget_field_is_positive() {
  local checkpoint_path="$1"
  local field="$2"
  if ! [ -f "$checkpoint_path" ]; then
    return 1
  fi
  "$PY" - <<'PY' "$checkpoint_path" "$field"
import pickle
import sys
from pathlib import Path

checkpoint_path = Path(sys.argv[1])
field = str(sys.argv[2])
try:
    state = pickle.loads(checkpoint_path.read_bytes())
    budget = state.get("compute_budget") or {}
    value = budget.get(field, 0.0) or 0.0
    try:
        value = float(value)
    except Exception:
        value = 0.0
    sys.exit(0 if value > 0.0 else 1)
except Exception:
    sys.exit(1)
PY
}

EXTRA_EVO_ARGS=()
if [ -n "$DEVICE" ]; then
  EXTRA_EVO_ARGS+=(--device "$DEVICE")
fi
if [ -n "$BATCH_SIZE" ]; then
  EXTRA_EVO_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [ "$DISABLE_HUMAN" -eq 1 ]; then
  EXTRA_EVO_ARGS+=(--disable-human)
fi

GEN_DONE="$(current_generation "$CHECKPOINT")"
if [ -z "$GEN_DONE" ]; then
  GEN_DONE="0"
fi

if [ "$GEN_DONE" -lt "$CALIB_GENS" ]; then
  GENS_TO_RUN=$((CALIB_GENS - GEN_DONE))
  if [ "$GEN_DONE" -eq 0 ]; then
    echo "[calibration] gens=$CALIB_GENS seed=$SEED checkpoint_every=$CHECKPOINT_EVERY"
    if [ "${#EXTRA_EVO_ARGS[@]}" -gt 0 ]; then
      AGENT_MODE=baseline "$PY" scripts/run_evolution.py \
        --config "$CONFIG" \
        --generations "$CALIB_GENS" \
        --output "$OUTPUT" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --seed "$SEED" \
        "${EXTRA_EVO_ARGS[@]}"
    else
      AGENT_MODE=baseline "$PY" scripts/run_evolution.py \
        --config "$CONFIG" \
        --generations "$CALIB_GENS" \
        --output "$OUTPUT" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --seed "$SEED"
    fi
  else
    echo "[calibration-resume] gens=$GENS_TO_RUN (done=$GEN_DONE/$CALIB_GENS) seed=$SEED checkpoint_every=$CHECKPOINT_EVERY"
    if [ "${#EXTRA_EVO_ARGS[@]}" -gt 0 ]; then
      AGENT_MODE=baseline "$PY" scripts/run_evolution.py \
        --config "$CONFIG" \
        --resume-from "$OUTPUT" \
        --generations "$GENS_TO_RUN" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --seed "$SEED" \
        "${EXTRA_EVO_ARGS[@]}"
    else
      AGENT_MODE=baseline "$PY" scripts/run_evolution.py \
        --config "$CONFIG" \
        --resume-from "$OUTPUT" \
        --generations "$GENS_TO_RUN" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --seed "$SEED"
    fi
  fi
fi

GEN_DONE="$(current_generation "$CHECKPOINT")"
if [ -z "$GEN_DONE" ]; then
  GEN_DONE="0"
fi

if [ "$GEN_DONE" -lt "$FULL_GENS" ]; then
  REMAINING=$((FULL_GENS - GEN_DONE))
  echo "[resume] gens=$REMAINING (done=$GEN_DONE/$FULL_GENS) seed=$SEED checkpoint_every=$CHECKPOINT_EVERY"
  if [ "${#EXTRA_EVO_ARGS[@]}" -gt 0 ]; then
    AGENT_MODE=baseline "$PY" scripts/run_evolution.py \
      --config "$CONFIG" \
      --resume-from "$OUTPUT" \
      --generations "$REMAINING" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --seed "$SEED" \
      --final-holdout "$ID_HOLDOUT" \
      "${EXTRA_EVO_ARGS[@]}"
  else
    AGENT_MODE=baseline "$PY" scripts/run_evolution.py \
      --config "$CONFIG" \
      --resume-from "$OUTPUT" \
      --generations "$REMAINING" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --seed "$SEED" \
      --final-holdout "$ID_HOLDOUT"
  fi
fi

if [ "$RUN_SFT" -eq 1 ]; then
  if budget_field_is_positive "$CHECKPOINT" "$SFT_MATCH_BUDGET_FIELD"; then
    echo "[sft] compute-matched baseline from $CHECKPOINT"
    AGENT_MODE=baseline "$PY" scripts/run_sft.py \
      --checkpoint "$CHECKPOINT" \
      --match-budget-field "$SFT_MATCH_BUDGET_FIELD" \
      --backprop-multiplier "$BACKPROP_MULTIPLIER" \
      --data "$SFT_DATA" \
      --model "$MODEL" \
      --lora-rank 8 \
      --output "$OUTPUT/sft" \
      --resume
  else
    echo "[sft] Skipping: checkpoint compute_budget.$SFT_MATCH_BUDGET_FIELD is missing/zero; cannot match budget."
  fi
fi

if [ "$RUN_EVAL" -eq 1 ]; then
  echo "[eval] in-distribution holdout: $ID_HOLDOUT"
  EVAL_CMD=(
    "$PY"
    scripts/evaluate_holdout.py
    --holdout
    "$ID_HOLDOUT"
    --model
    "$MODEL"
    --evo-checkpoint
    "$CHECKPOINT"
    --evo-selection-tasks
    "$ID_SELECTION_TASKS"
    --evo-selection-family
    any
    --evo-eval-routing
    "$EVO_EVAL_ROUTING"
    --evo-selection-max-new-tokens
    "$EVO_SELECTION_MAX_NEW_TOKENS"
    --output
    "$OUTPUT/eval_id.json"
  )
  if [ "$RUN_SFT" -eq 1 ] && [ -d "$OUTPUT/sft/peft_adapter" ]; then
    EVAL_CMD+=(--sft-adapter "$OUTPUT/sft/peft_adapter")
  fi
  if [ -n "$EVO_SELECTION_MAX_SAMPLES" ]; then
    EVAL_CMD+=(--evo-selection-max-samples "$EVO_SELECTION_MAX_SAMPLES")
  fi
  if [ -n "$EVO_SELECTION_TOP_K_BY_ROI" ] && [ "$EVO_SELECTION_TOP_K_BY_ROI" -gt 0 ]; then
    EVAL_CMD+=(--evo-selection-top-k-by-roi "$EVO_SELECTION_TOP_K_BY_ROI")
  fi
  AGENT_MODE=baseline "${EVAL_CMD[@]}"
  echo "Wrote: $OUTPUT/eval_id.json"
fi

echo "Done: $OUTPUT"
