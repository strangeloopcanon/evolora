#!/usr/bin/env bash
set -euo pipefail

# Small helper to reproduce the three main Qwen3‑0.6B baselines used in the ecology writeup.
# It assumes you're running this from repo root and have a local venv ready (`make setup` creates `.venv`).

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
  echo "Expected $PY – run `make setup` (creates .venv), or set VENV_BIN/PY to your venv." >&2
  exit 1
fi

EXTRA_EVAL_ARGS=()
if [ -n "${FINAL_HOLDOUT_TASKS:-}" ]; then
  if [ ! -f "$FINAL_HOLDOUT_TASKS" ]; then
    echo "FINAL_HOLDOUT_TASKS not found: $FINAL_HOLDOUT_TASKS" >&2
    exit 2
  fi
  EXTRA_EVAL_ARGS+=(--final-holdout "$FINAL_HOLDOUT_TASKS")
  if [ -n "${FINAL_HOLDOUT_SAMPLE_SIZE:-}" ]; then
    EXTRA_EVAL_ARGS+=(--final-holdout-sample-size "$FINAL_HOLDOUT_SAMPLE_SIZE")
  fi
fi

timestamp() {
  date +%Y%m%d_%H%M%S
}

run_frozen() {
  local run_id="artifacts_paper_qwen3_frozen_$(timestamp)"
  echo "[frozen] run_id=$run_id"
  MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline "$PY" scripts/eval_gemma_long.py \
    --config config/experiments/paper_qwen3_frozen.yaml \
    --generations 50 \
    --output "$run_id" \
    --checkpoint-every 5 \
    ${EXTRA_EVAL_ARGS[@]+"${EXTRA_EVAL_ARGS[@]}"}
  MPLCONFIGDIR=$(mktemp -d) "$PY" scripts/analyze_ecology_run.py "$run_id" --plots --report
}

run_single() {
  local run_id="artifacts_paper_qwen3_single_$(timestamp)"
  echo "[single] run_id=$run_id"
  MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline "$PY" scripts/eval_gemma_long.py \
    --config config/experiments/paper_qwen3_single.yaml \
    --generations 50 \
    --output "$run_id" \
    --checkpoint-every 5 \
    ${EXTRA_EVAL_ARGS[@]+"${EXTRA_EVAL_ARGS[@]}"}
  MPLCONFIGDIR=$(mktemp -d) "$PY" scripts/analyze_ecology_run.py "$run_id" --plots --report
}

run_ecology() {
  local run_id="artifacts_paper_qwen3_ecology_$(timestamp)"
  echo "[ecology] run_id=$run_id"
  MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline "$PY" scripts/eval_gemma_long.py \
    --config config/experiments/paper_qwen3_ecology.yaml \
    --generations 50 \
    --output "$run_id" \
    --checkpoint-every 5 \
    ${EXTRA_EVAL_ARGS[@]+"${EXTRA_EVAL_ARGS[@]}"}
  MPLCONFIGDIR=$(mktemp -d) "$PY" scripts/analyze_ecology_run.py "$run_id" --plots --report
}

case "${1:-all}" in
  frozen)  run_frozen ;;
  single)  run_single ;;
  ecology) run_ecology ;;
  all)
    run_frozen
    run_single
    run_ecology
    ;;
  *)
    echo "Usage: $0 [frozen|single|ecology|all]" >&2
    exit 1
    ;;
esac
