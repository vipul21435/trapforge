.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test test-fast cov check demo clean

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Create the uv environment (dev tools included) and install git hooks
	$(UV) sync
	$(UV) run pre-commit install

lint:  ## Lint and check formatting with ruff
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Auto-fix lint findings and reformat
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:  ## Type-check src/ with mypy --strict
	$(UV) run mypy

test:  ## Run the full test suite
	$(UV) run pytest -q

test-fast:  ## Skip slow and Docker-dependent tests
	$(UV) run pytest -q -m "not slow and not docker"

cov:  ## Run the tests with branch coverage (fails under 85%)
	$(UV) run pytest -q --cov --cov-report=term-missing

check: lint typecheck cov  ## Everything CI runs

demo:  ## End-to-end demo of the CLI
	$(UV) run trapforge --version
	$(UV) run trapforge info

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage .coverage.* htmlcov dist build bundles reports out
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
