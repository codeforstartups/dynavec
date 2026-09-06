# Contributing to dynavec

Welcome! We're glad you're interested in contributing to dynavec.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Run ruff linter
ruff check .

# Run ruff formatter
ruff format .
```

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality:

- `ruff` - Linting with automatic fixes
- `ruff-format` - Code formatting
- `trailing-whitespace` - Removes trailing whitespace
- `end-of-file-fixer` - Ensures files end with a newline
- `check-case-conflict` - Checks for case conflicts
- `check-added-large-files` - Prevents adding large files

Install hooks with:
```bash
pre-commit install
```

Run hooks manually:
```bash
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dynavec --cov-report=term-missing
```

### Submitting Changes

1. Create a new branch for your changes
2. Make your changes and commit them
3. Ensure pre-commit hooks pass
4. Push to your fork and open a Pull Request
5. Reference the issue number in your PR description

## Good First Issues

Looking for a place to start? Check out issues labeled [good first issue](https://github.com/codeforstartups/dynavec/labels/good%20first%20issue).

## Questions?

Feel free to open an issue for any questions or discussions.
