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
