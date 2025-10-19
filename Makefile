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
	$(BANDIT) -q -r src/symbiont_ecology
	$(DETECT_SECRETS) scan --baseline .secrets.baseline

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

