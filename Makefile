PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
POETRY := $(VENV)/bin/poetry
PY := $(VENV)/bin/python
BLACK := $(VENV)/bin/black
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest
BANDIT := $(VENV)/bin/bandit
DETECT_SECRETS := $(VENV)/bin/detect-secrets
DETECT_SECRETS_HOOK := $(VENV)/bin/detect-secrets-hook
PIP_AUDIT := $(VENV)/bin/pip-audit
PRECOMMIT := $(VENV)/bin/pre-commit

.DEFAULT_GOAL := all

$(VENV)/bin/activate:
	@if [ ! -d $(VENV) ]; then $(PYTHON) -m venv $(VENV); fi
	$(PIP) install --upgrade pip wheel
	$(PIP) install -e .[dev]
	$(PRECOMMIT) install || true

.PHONY: setup bootstrap
setup: $(VENV)/bin/activate
	@echo "Environment ready"

bootstrap: setup

.PHONY: check
check: setup
	$(BLACK) --check src tests
	$(RUFF) check src tests
	$(MYPY) src
	$(BANDIT) -q -r src/symbiont_ecology --severity-level medium
	git ls-files -z | xargs -0 $(DETECT_SECRETS_HOOK) --baseline .secrets.baseline

.PHONY: test
test: setup
	$(PYTEST)

.PHONY: llm-live
llm-live: setup
	$(PY) scripts/llm_live.py --dry-run

.PHONY: deps-audit
deps-audit: setup
	$(PIP_AUDIT) || true

.PHONY: all
all: check test llm-live

.PHONY: clean
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage

.PHONY: format
format: setup
	$(BLACK) src tests
	$(RUFF) check src tests --fix

.PHONY: calibrate-resume
calibrate-resume: setup
	./scripts/run_calibration_then_resume.sh \
	  --config $(CONFIG) \
	  --output $(OUTPUT) \
	  --calib-gens $(CALIB_GENS) \
	  --full-gens $(FULL_GENS) \
	  --checkpoint-every $(CHECKPOINT_EVERY) \
	  --seed $(SEED) \
	  $(if $(DEVICE),--device $(DEVICE),) \
	  $(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
	  $(if $(DISABLE_HUMAN),--disable-human,) \
	  $(if $(FINAL_HOLDOUT),--final-holdout $(FINAL_HOLDOUT),) \
	  $(if $(FINAL_HOLDOUT_SAMPLE_SIZE),--final-holdout-sample-size $(FINAL_HOLDOUT_SAMPLE_SIZE),) \
	  $(if $(NO_ANALYZE),--no-analyze,)

CONFIG ?= config/experiments/paper_qwen3_ecology.yaml
OUTPUT ?= artifacts_calib_$(shell date +%Y%m%d_%H%M%S)
CALIB_GENS ?= 10
FULL_GENS ?= 50
CHECKPOINT_EVERY ?= 5
SEED ?= 777
DEVICE ?=
BATCH_SIZE ?=
DISABLE_HUMAN ?= 1
FINAL_HOLDOUT ?=
FINAL_HOLDOUT_SAMPLE_SIZE ?=
NO_ANALYZE ?=

.PHONY: regex-evo-vs-sft
regex-evo-vs-sft: setup
	./scripts/run_regex_evo_vs_sft.sh \
	  --config $(REGEX_CONFIG) \
	  --output $(REGEX_OUTPUT) \
	  --calib-gens $(REGEX_CALIB_GENS) \
	  --full-gens $(REGEX_FULL_GENS) \
	  --checkpoint-every $(REGEX_CHECKPOINT_EVERY) \
	  --seed $(REGEX_SEED) \
	  $(if $(REGEX_DEVICE),--device $(REGEX_DEVICE),) \
	  $(if $(REGEX_BATCH_SIZE),--batch-size $(REGEX_BATCH_SIZE),) \
	  $(if $(REGEX_DISABLE_HUMAN),--disable-human,) \
	  $(if $(REGEX_EVAL_MAX_SAMPLES),--eval-max-samples $(REGEX_EVAL_MAX_SAMPLES),)

REGEX_CONFIG ?= config/experiments/qwen3_regex.yaml
REGEX_OUTPUT ?= artifacts_regex_evo_sft_$(shell date +%Y%m%d_%H%M%S)
REGEX_CALIB_GENS ?= 5
REGEX_FULL_GENS ?= 50
REGEX_CHECKPOINT_EVERY ?= 5
REGEX_SEED ?= 777
REGEX_DEVICE ?=
REGEX_BATCH_SIZE ?=
REGEX_DISABLE_HUMAN ?= 1
REGEX_EVAL_MAX_SAMPLES ?=

.PHONY: regex-generalization-evo-vs-sft
regex-generalization-evo-vs-sft: setup
	./scripts/run_regex_generalization_evo_vs_sft.sh \
	  --config $(REGEX_GEN_CONFIG) \
	  --output $(REGEX_GEN_OUTPUT) \
	  --calib-gens $(REGEX_GEN_CALIB_GENS) \
	  --full-gens $(REGEX_GEN_FULL_GENS) \
	  --checkpoint-every $(REGEX_GEN_CHECKPOINT_EVERY) \
	  --seed $(REGEX_GEN_SEED) \
	  $(if $(REGEX_GEN_DEVICE),--device $(REGEX_GEN_DEVICE),) \
	  $(if $(REGEX_GEN_BATCH_SIZE),--batch-size $(REGEX_GEN_BATCH_SIZE),) \
	  $(if $(REGEX_GEN_DISABLE_HUMAN),--disable-human,) \
	  $(if $(REGEX_GEN_EVAL_MAX_SAMPLES),--eval-max-samples $(REGEX_GEN_EVAL_MAX_SAMPLES),)

REGEX_GEN_CONFIG ?= config/experiments/qwen3_regex_generalization.yaml
REGEX_GEN_OUTPUT ?= artifacts_regex_gen_evo_sft_$(shell date +%Y%m%d_%H%M%S)
REGEX_GEN_CALIB_GENS ?= 5
REGEX_GEN_FULL_GENS ?= 50
REGEX_GEN_CHECKPOINT_EVERY ?= 5
REGEX_GEN_SEED ?= 777
REGEX_GEN_DEVICE ?=
REGEX_GEN_BATCH_SIZE ?=
REGEX_GEN_DISABLE_HUMAN ?= 1
REGEX_GEN_EVAL_MAX_SAMPLES ?=

.PHONY: grid-multiobjective-evo-vs-sft
grid-multiobjective-evo-vs-sft: setup
	./scripts/run_grid_multiobjective_evo_vs_sft.sh \
	  --config $(GRID_MO_CONFIG) \
	  --output $(GRID_MO_OUTPUT) \
	  --calib-gens $(GRID_MO_CALIB_GENS) \
	  --full-gens $(GRID_MO_FULL_GENS) \
	  --checkpoint-every $(GRID_MO_CHECKPOINT_EVERY) \
	  --seed $(GRID_MO_SEED) \
	  $(if $(GRID_MO_DEVICE),--device $(GRID_MO_DEVICE),) \
	  $(if $(GRID_MO_BATCH_SIZE),--batch-size $(GRID_MO_BATCH_SIZE),) \
	  $(if $(GRID_MO_DISABLE_HUMAN),--disable-human,) \
	  $(if $(GRID_MO_EVO_EVAL_ROUTING),--evo-eval-routing $(GRID_MO_EVO_EVAL_ROUTING),)

GRID_MO_CONFIG ?= config/experiments/qwen25_grid_multiobjective.yaml
GRID_MO_OUTPUT ?= artifacts_grid_multiobj_evo_sft_$(shell date +%Y%m%d_%H%M%S)
GRID_MO_CALIB_GENS ?= 5
GRID_MO_FULL_GENS ?= 50
GRID_MO_CHECKPOINT_EVERY ?= 5
GRID_MO_SEED ?= 777
GRID_MO_DEVICE ?=
GRID_MO_BATCH_SIZE ?=
GRID_MO_DISABLE_HUMAN ?= 1
GRID_MO_EVO_EVAL_ROUTING ?= family

.PHONY: grid-multiobjective-full-ecology-evo-vs-sft
grid-multiobjective-full-ecology-evo-vs-sft: setup
	./scripts/run_grid_multiobjective_evo_vs_sft.sh \
	  --config $(GRID_MO_FULL_ECOLOGY_CONFIG) \
	  --output $(GRID_MO_FULL_ECOLOGY_OUTPUT) \
	  --calib-gens $(GRID_MO_FULL_ECOLOGY_CALIB_GENS) \
	  --full-gens $(GRID_MO_FULL_ECOLOGY_FULL_GENS) \
	  --checkpoint-every $(GRID_MO_FULL_ECOLOGY_CHECKPOINT_EVERY) \
	  --seed $(GRID_MO_FULL_ECOLOGY_SEED) \
	  $(if $(GRID_MO_FULL_ECOLOGY_DEVICE),--device $(GRID_MO_FULL_ECOLOGY_DEVICE),) \
	  $(if $(GRID_MO_FULL_ECOLOGY_BATCH_SIZE),--batch-size $(GRID_MO_FULL_ECOLOGY_BATCH_SIZE),) \
	  $(if $(GRID_MO_FULL_ECOLOGY_DISABLE_HUMAN),--disable-human,) \
	  $(if $(GRID_MO_FULL_ECOLOGY_EVO_EVAL_ROUTING),--evo-eval-routing $(GRID_MO_FULL_ECOLOGY_EVO_EVAL_ROUTING),)

GRID_MO_FULL_ECOLOGY_CONFIG ?= config/experiments/qwen25_grid_multiobjective_full_ecology.yaml
GRID_MO_FULL_ECOLOGY_OUTPUT ?= artifacts_grid_multiobj_full_ecology_evo_sft_$(shell date +%Y%m%d_%H%M%S)
GRID_MO_FULL_ECOLOGY_CALIB_GENS ?= 5
GRID_MO_FULL_ECOLOGY_FULL_GENS ?= 50
GRID_MO_FULL_ECOLOGY_CHECKPOINT_EVERY ?= 5
GRID_MO_FULL_ECOLOGY_SEED ?= 777
GRID_MO_FULL_ECOLOGY_DEVICE ?=
GRID_MO_FULL_ECOLOGY_BATCH_SIZE ?=
GRID_MO_FULL_ECOLOGY_DISABLE_HUMAN ?= 1
GRID_MO_FULL_ECOLOGY_EVO_EVAL_ROUTING ?= family
