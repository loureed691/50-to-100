.PHONY: install dev test lint fmt help

PYTHON ?= python

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install runtime + development dependencies (ruff, pytest)
	$(PYTHON) -m pip install -r requirements-dev.txt

dev: install-dev ## Install dev deps and run the bot in paper mode
	PAPER_MODE=true $(PYTHON) bot.py

lint: install-dev ## Run ruff linter (no auto-fix)
	$(PYTHON) -m ruff check .

fmt: install-dev ## Auto-format and fix safe lint issues
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test: install-dev ## Run the full test suite (lint + unit tests)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest tests/ -v
