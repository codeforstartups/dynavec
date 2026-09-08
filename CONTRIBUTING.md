# Contributing to dynavec

Thank you for your interest in contributing to **dynavec**! We welcome bug fixes, documentation improvements, test additions, and new features.

---

## Getting Started

### 1. Fork and Clone
Fork the repository on GitHub, then clone your fork locally:
```bash
git clone https://github.com/<your-username>/dynavec.git
cd dynavec
git remote add upstream https://github.com/codeforstartups/dynavec.git
```

### 2. Set Up a Virtual Environment
Create and activate a virtual environment (Python 3.9+):
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies in Editable Mode
Install the package along with developer tools:
```bash
pip install -e ".[dev]"
```

---

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically enforce code style and formatting (`ruff`, trailing whitespace, end-of-file newlines, and valid YAML/TOML).

### Install the Hooks
After installing `.[dev]`, set up pre-commit to run on every `git commit`:
```bash
pre-commit install
```

### Run Manually on All Files
You can run all hooks across the codebase at any time:
```bash
pre-commit run --all-files
```

---

## Running Tests and Linting

Before pushing changes or submitting a Pull Request, make sure all tests pass and code checks succeed:

### Run Linter and Formatter
```bash
# Check code with ruff
ruff check src benchmarks tests

# Auto-fix lint issues where possible
ruff check --fix src benchmarks tests

# Format code
ruff format src benchmarks tests
```

### Run the Test Suite
```bash
pytest
```

---

## Making Changes & Submitting a PR

1. **Branching**: Always branch off the `development` branch:
   ```bash
   git checkout development
   git pull upstream development
   git checkout -b feat/your-feature-name
   ```

2. **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat:`, `fix:`, `docs:`, `test:`, `ci:`).

3. **Rebasing**: Keep your branch up to date with `upstream/development` without merge commits:
   ```bash
   git fetch upstream development
   git rebase upstream/development
   ```

4. **Pull Request**: Open your PR targeting the `development` branch of `codeforstartups/dynavec` and reference any related issue with `Closes #<issue_number>`.
