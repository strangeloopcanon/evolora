#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner: regex evolution → compute-matched SFT → holdout evaluation.
#
# This wraps long runs with a short calibration segment first, so failures show up early,
# and then resumes into the same output directory. All intermediate artifacts are saved.

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
  echo "Expected $PY – run \`make setup\` (creates .venv), or set VENV_BIN/PY to your venv." >&2
  exit 1
fi

timestamp() {
  date +%Y%m%d_%H%M%S
}

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_regex_evo_vs_sft.sh [options]

Options:
  --config <yaml>            EcologyConfig YAML (default: config/experiments/qwen3_regex.yaml)
  --output <dir>             Output run directory (default: artifacts_regex_evo_sft_<timestamp>)
  --train-size <int>         Generated SFT train tasks (default: 20000)
  --selection-size <int>     Generated selection tasks for evo organelle picking (default: 64)
  --id-holdout-size <int>    Generated in-distribution holdout tasks (default: 256)
  --calib-gens <int>         Calibration generations (default: 5)
  --full-gens <int>          Total generations after resume (default: 50)
  --checkpoint-every <int>   Checkpoint cadence in generations (default: 5)
  --seed <int>               Grid seed (default: 777)
  --device <str>             Torch device override (e.g. mps/cpu/cuda)
  --batch-size <int>         Override synthetic batch size per generation
  --disable-human            Disable human bandit even if config enables it
  --sft-data <jsonl>         Optional SFT data file (default: generated from GridEnvironment)
  --holdout <jsonl>          OOD holdout eval tasks (default: config/evaluation/regex_generalization.jsonl)
  --eval-max-samples <int>   Optional cap on holdout task count (default: all)
  --no-eval-id               Skip in-distribution regex holdout evaluation
  --no-eval-ood              Skip OOD holdout evaluation
  --sft-match-budget-field   One of: total_tokens,train_tokens,total_flops,train_flops,wall_clock_seconds (default: train_flops)
  --backprop-multiplier      Convert evo budget → SFT budget (default: 3.0)
  --no-sft                   Skip SFT baseline training
  --no-eval                  Skip final holdout evaluation
EOF
}

CONFIG="config/experiments/qwen3_regex.yaml"
OUTPUT=""
TRAIN_SIZE=20000
SELECTION_SIZE=64
ID_HOLDOUT_SIZE=256
CALIB_GENS=5
FULL_GENS=50
CHECKPOINT_EVERY=5
SEED=777
DEVICE=""
BATCH_SIZE=""
DISABLE_HUMAN=0
SFT_DATA=""
HOLDOUT="config/evaluation/regex_generalization.jsonl"
EVAL_MAX_SAMPLES=""
RUN_SFT=1
RUN_EVAL=1
RUN_EVAL_ID=1
RUN_EVAL_OOD=1
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
    --sft-data)
      SFT_DATA="${2:-}"; shift 2 ;;
    --holdout)
      HOLDOUT="${2:-}"; shift 2 ;;
    --eval-max-samples)
      EVAL_MAX_SAMPLES="${2:-}"; shift 2 ;;
    --no-eval-id)
      RUN_EVAL_ID=0; shift ;;
    --no-eval-ood)
      RUN_EVAL_OOD=0; shift ;;
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
  OUTPUT="artifacts_regex_evo_sft_$(timestamp)"
fi

if ! [ -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

if ! [ -f "$HOLDOUT" ]; then
  echo "Holdout tasks not found: $HOLDOUT" >&2
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

MODEL="$("$PY" -c "from symbiont_ecology import load_ecology_config; from pathlib import Path; cfg=load_ecology_config(Path('$CONFIG')); print(cfg.host.backbone_model)" | tr -d '\r')"
if [ -z "$MODEL" ]; then
  echo "Failed to infer base model from config: $CONFIG" >&2
  exit 2
fi

echo "[config] $CONFIG"
echo "[model]  $MODEL"
echo "[out]    $OUTPUT"

DATASET_DIR="$OUTPUT/datasets"
ID_SELECTION_TASKS="$DATASET_DIR/selection_tasks.jsonl"
ID_HOLDOUT="$DATASET_DIR/holdout_tasks.jsonl"
if [ -z "$SFT_DATA" ]; then
  SFT_DATA="$DATASET_DIR/sft_train.jsonl"
fi

if ! [ -f "$SFT_DATA" ] || ! [ -f "$ID_SELECTION_TASKS" ] || ! [ -f "$ID_HOLDOUT" ]; then
  echo "[data] generating distribution-matched regex datasets into $DATASET_DIR"
  AGENT_MODE=baseline "$PY" scripts/generate_regex_grid_datasets.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --train-size "$TRAIN_SIZE" \
    --selection-size "$SELECTION_SIZE" \
    --holdout-size "$ID_HOLDOUT_SIZE" \
    --out-dir "$DATASET_DIR"
fi

if ! [ -f "$SFT_DATA" ]; then
  echo "SFT data not found: $SFT_DATA" >&2
  exit 2
fi

echo "[calibration] gens=$CALIB_GENS seed=$SEED checkpoint_every=$CHECKPOINT_EVERY"
MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline "$PY" scripts/run_evolution.py \
  --config "$CONFIG" \
  --generations "$CALIB_GENS" \
  --output "$OUTPUT" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

echo "[resume] resume_from=$OUTPUT gens=$RESUME_GENS seed=$SEED checkpoint_every=$CHECKPOINT_EVERY"
MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline "$PY" scripts/run_evolution.py \
  --config "$CONFIG" \
  --resume-from "$OUTPUT" \
  --generations "$RESUME_GENS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

CHECKPOINT="$OUTPUT/checkpoint.pt"
if ! [ -f "$CHECKPOINT" ]; then
  echo "Expected checkpoint at $CHECKPOINT" >&2
  exit 1
fi

if [ "$RUN_SFT" -eq 1 ]; then
  SFT_OUT="$OUTPUT/sft"
  echo "[sft] checkpoint=$CHECKPOINT data=$SFT_DATA output=$SFT_OUT"
  AGENT_MODE=baseline "$PY" scripts/run_sft.py \
    --checkpoint "$CHECKPOINT" \
    --data "$SFT_DATA" \
    --model "$MODEL" \
    --match-budget-field "$SFT_MATCH_BUDGET_FIELD" \
    --backprop-multiplier "$BACKPROP_MULTIPLIER" \
    --attn-implementation eager \
    --optim adamw_torch \
    --engine manual \
    --resume \
    --output "$SFT_OUT"
fi

if [ "$RUN_EVAL" -eq 1 ]; then
  if [ "$RUN_EVAL_ID" -eq 1 ]; then
    EVAL_ID_OUT="$OUTPUT/eval_holdout_id.json"
    EVAL_ID_ARGS=(--holdout "$ID_HOLDOUT" --model "$MODEL" --evo-checkpoint "$CHECKPOINT" --output "$EVAL_ID_OUT" --evo-selection-tasks "$ID_SELECTION_TASKS")
    if [ "$RUN_SFT" -eq 1 ]; then
      EVAL_ID_ARGS+=(--sft-adapter "$OUTPUT/sft/peft_adapter")
    fi
    if [ -n "$EVAL_MAX_SAMPLES" ]; then
      EVAL_ID_ARGS+=(--max-samples "$EVAL_MAX_SAMPLES")
    fi
    echo "[eval-id] holdout=$ID_HOLDOUT output=$EVAL_ID_OUT"
    AGENT_MODE=baseline "$PY" scripts/evaluate_holdout.py \
      --attn-implementation eager \
      "${EVAL_ID_ARGS[@]}"
  fi
  if [ "$RUN_EVAL_OOD" -eq 1 ]; then
    EVAL_OOD_OUT="$OUTPUT/eval_holdout_ood.json"
    EVAL_OOD_ARGS=(--holdout "$HOLDOUT" --model "$MODEL" --evo-checkpoint "$CHECKPOINT" --output "$EVAL_OOD_OUT" --evo-selection-tasks "$ID_SELECTION_TASKS")
    if [ "$RUN_SFT" -eq 1 ]; then
      EVAL_OOD_ARGS+=(--sft-adapter "$OUTPUT/sft/peft_adapter")
    fi
    if [ -n "$EVAL_MAX_SAMPLES" ]; then
      EVAL_OOD_ARGS+=(--max-samples "$EVAL_MAX_SAMPLES")
    fi
    echo "[eval-ood] holdout=$HOLDOUT output=$EVAL_OOD_OUT"
    AGENT_MODE=baseline "$PY" scripts/evaluate_holdout.py \
      --attn-implementation eager \
      "${EVAL_OOD_ARGS[@]}"
  fi
fi

echo "Done: $OUTPUT"
