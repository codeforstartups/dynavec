# dynavec developer commands. Run `make help` to see everything.
#
# These wrap the exact tools CI uses (uv + ruff + pytest), so `make run-ci`
# locally is the same pipeline that runs on your PR.

.DEFAULT_GOAL := help
.PHONY: help install install-all format lint check test test-live docs clean run-ci

PY_DIRS := src benchmarks tests
CI_DIRS := src benchmarks           # what CI lints (keep in sync with .github/workflows/ci.yml)

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install dynavec + dev tools in editable mode (recommended)
	uv pip install -e ".[dev,ingest]"

install-all:  ## Install everything: all embedders, adapters, and dev tools
	uv pip install -e ".[all,ingest,dev]"

format:  ## Auto-format and fix lint issues (ruff format + ruff --fix)
	uv run --no-sync ruff format $(PY_DIRS)
	uv run --no-sync ruff check --fix $(PY_DIRS)

lint:  ## Lint without changing files (mirrors CI exactly)
	uv run --no-sync ruff check $(CI_DIRS)

check: lint  ## Quick health check (lint, no tests)

test:  ## Run the unit test suite (offline, no AWS needed)
	uv run --no-sync pytest -q

test-live:  ## Run the opt-in end-to-end test against real AWS (costs money)
	DYNAVEC_LIVE=1 uv run --no-sync pytest tests/integration -v

docs:  ## Regenerate the static docs site into opensource/dynavec/docs/
	uv run --no-sync python tools/build_docs.py

run-ci:  ## Run the full CI pipeline locally (lint + test)
	$(MAKE) lint
	$(MAKE) test

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
