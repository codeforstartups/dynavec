# Contributing to dynavec

Thank you for your interest in contributing to **dynavec**! This guide covers the
development workflow for both human developers and AI coding agents. We welcome bug
fixes, documentation improvements, test additions, and new features.

```
+----------------------------------------------------------------------------+
|     +----------------------------------------------------------------+     |
|     | Developers: Those who built with `dynavec`.                    |     |
|     | (You have `import dynavec` somewhere in your project)          |     |
|     |     +----------------------------------------------------+     |     |
|     |     | Contributors: Those who make `dynavec` better.     |     |     |
|     |     | (You make a PR to this repo)                       |     |     |
|     |     +----------------------------------------------------+     |     |
|     +----------------------------------------------------------------+     |
+----------------------------------------------------------------------------+
```

---

## Quick Start (for Developers)

```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/dynavec.git
cd dynavec

# 2. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create a virtual environment and install dev dependencies
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
make install                       # editable install with dev + ingest extras

# 4. Verify everything works
make check                         # lint
make test                          # run the offline test suite

# 5. See every available command
make help
```

No AWS account is needed for development — the test suite runs fully offline against
in-memory fakes.

## Quick Start (for AI Agents)

AI agents working in this repo should use the standardized `make` targets:

```bash
make help            # See all available targets
make install         # Install dev environment (editable)
make check           # Quick health check (lint)
make test            # Run all unit tests (offline)
make run-ci          # Full CI pipeline locally (lint + test)
make format          # Auto-format and fix lint issues
make docs            # Regenerate the static docs site
make clean           # Remove caches and build artifacts
```

**Key points for AI agents:**

- Prefer `make` targets over invoking tools directly — they match CI exactly.
- For direct tool calls, use the `uv run --no-sync` prefix (plain `uv run` can trigger a
  universal resolve that pulls yanked optional deps).
- Run `make run-ci` before declaring a change complete; it is the same pipeline CI runs.
- Prefer editing existing files over creating new ones, and follow the conventions in
  neighboring modules.

---

## Repository Architecture

dynavec is a single Python package (not a monorepo):

```
/
├── src/dynavec/            # The library
│   ├── client.py           # Dynavec orchestrator (upsert / search / delete)
│   ├── stores/             # DynamoDB + S3 Vectors backends
│   ├── embeddings/         # Pluggable BYO-key embedder backends
│   ├── integrations/       # LangChain / LlamaIndex / agent-framework adapters
│   ├── eval/               # RAG evaluation (LLM-judge faithfulness / relevance)
│   ├── ingest.py           # Document ingestion sources (files, PDF, URL, ...)
│   ├── graph.py            # GraphRAG entity/relationship layer
│   ├── cache.py            # Semantic / DynamoDB-TTL / Redis query caches
│   ├── telemetry.py        # In-process telemetry recorder + aggregation
│   └── ...                 # config, provisioning, retrieval, quantization, mcp, ...
├── tests/                  # Unit tests (offline); tests/integration/ is opt-in
├── benchmarks/             # Recall / latency / cost benchmarking + cost model
├── examples/               # Runnable usage examples
├── tools/build_docs.py     # Generates the static docs site
├── opensource/dynavec/     # Landing page + generated docs site
├── dashboard/              # Next.js observability dashboard
├── .github/workflows/      # CI, PyPI publish, GitHub Pages
├── Makefile                # Developer commands (this guide uses them)
└── pyproject.toml          # Dependencies, extras, and tool configuration
```

---

## Development Environment

**Prerequisites:** Python 3.9+, [uv](https://docs.astral.sh/uv/) (recommended) or pip, Git.

**Using Make (recommended):**

```bash
make install        # uv pip install -e ".[dev,ingest]"
make install-all    # everything: all embedders + adapters + dev tools
```

**Manual setup:**

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # add ,ingest / ,all as needed
```

### Pre-commit hooks (optional but encouraged)

We ship a [pre-commit](https://pre-commit.com/) config that runs `ruff` and basic hygiene
checks on every commit:

```bash
pre-commit install                  # set up the git hook
pre-commit run --all-files          # run across the whole tree once
```

---

## Command Reference

| Command | What it does |
|---------|--------------|
| `make help` | List all targets |
| `make install` | Editable install with dev + ingest extras |
| `make install-all` | Editable install with **all** extras |
| `make format` | Auto-format (`ruff format`) and auto-fix lint (`ruff --fix`) |
| `make lint` / `make check` | Lint with ruff — mirrors CI exactly |
| `make test` | Run the offline unit suite (`pytest -q`) |
| `make test-live` | Opt-in end-to-end test against **real AWS** (costs money) |
| `make run-ci` | The full CI pipeline locally: lint + test |
| `make docs` | Regenerate the static docs site |
| `make clean` | Remove caches and build artifacts |

---

## Development Workflow

**Daily development:**

```bash
git checkout development
git pull upstream development
git checkout -b feat/your-feature    # branch off development

# ... make changes ...

make format                          # tidy up
make check                           # lint
make test                            # verify
```

**Before submitting a PR:**

```bash
make run-ci                          # must pass — same checks as CI
```

Keep your branch current without merge commits:

```bash
git fetch upstream development
git rebase upstream/development
```

---

## Testing

- **Unit tests** (`tests/`) are fast, isolated, and run fully **offline** against in-memory
  fakes — `make test`. New code should come with tests.
- **Integration test** (`tests/integration/test_live_aws.py`) is **opt-in** and exercises the
  real path (provision → S3 Vectors → DynamoDB → hydration) against your own AWS account. It
  is skipped by default; enable it explicitly:

  ```bash
  export DYNAVEC_LIVE=1
  export AWS_REGION=us-east-1           # a region where S3 Vectors is available
  make test-live                       # or: pytest tests/integration -v -s
  ```

Run a single test while iterating:

```bash
uv run --no-sync pytest tests/test_cache.py -k "jitter" -v
```

---

## Code Quality & CI/CD

- **Style/linting:** [ruff](https://docs.astral.sh/ruff/) (config in `pyproject.toml`, rule
  sets `E, F, I, UP, B`, line length 100). `make format` fixes most issues automatically.
- **Type hints:** dynavec ships a `py.typed` marker — please add type hints to new public APIs.
- **CI** (`.github/workflows/ci.yml`) runs on every push/PR: **ruff + pytest across Python
  3.9, 3.11, and 3.12**. `make run-ci` reproduces it locally.

---

## Pull Request Process

1. **Fork** the repository.
2. **Branch** off `development`: `git checkout -b feat/amazing-feature`.
3. **Develop** using the workflow above.
4. **Test** thoroughly: `make run-ci` must pass.
5. **Open a PR** targeting the `development` branch and reference the issue it closes
   (`Closes #123`).

**Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Voyage AI embedder
fix: drain QueryVectors pages fully
docs: expand the caching guide
test: cover graph traversal cycles
ci: add Python 3.13 to the matrix
```

**Review checklist:**

- [ ] Tests pass (`make test`)
- [ ] Lint passes (`make check`)
- [ ] New/changed behavior has tests
- [ ] Docs updated where relevant

---

## Troubleshooting

**`uv` not found** — install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

**Import errors after pulling** — reinstall the editable package: `make install`.

**Lint or test failures you can't reproduce** — run the exact CI pipeline: `make run-ci`.

**Dependency conflicts** — clean and reinstall:

```bash
make clean
rm -rf .venv && uv venv && source .venv/bin/activate
make install
```

**Python 3.13** — dynavec's CI currently targets **Python 3.9–3.12**; 3.13 support is being
added (see the open CI matrix work). If you're on 3.13 and a dependency fails to build a
wheel, use 3.12 for development:

```bash
uv python install 3.12
rm -rf .venv && uv venv -p 3.12 && source .venv/bin/activate
make install
```

---

## Getting Help

- **Issues:** browse [good first issues](https://github.com/codeforstartups/dynavec/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  or open a new one.
- **Community:** join the [WhatsApp community](https://chat.whatsapp.com/D73Mf1aDyIZHHMCgd32XHg).
- **Security:** please report vulnerabilities privately — see [SECURITY.md](SECURITY.md).

Happy coding! 🚀
