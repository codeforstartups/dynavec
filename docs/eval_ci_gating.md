# Evaluation Regression Runs & CI Gating

`dynaevals` includes built-in regression detection and CI gating to prevent performance drops (retrieval quality, faithfulness, relevance, or latency regressions) from slipping into production.

---

## 1. Overview

When developing agent workflows, RAG pipelines, or retrieval systems, code changes, chunking adjustments, or prompt changes can silently degrade search accuracy or answer quality.

Regression gating solves this by comparing an **evaluation run** against an established **baseline**:
- **Automatic Polarity**: Quality metrics (`recall@k`, `mrr`, `ndcg@k`, `faithfulness`, `relevance`) must not decrease below a threshold. Latency metrics (`p50_latency_ms`, `p95_latency_ms`) must not increase beyond a threshold.
- **CI Gate**: Non-zero exit code (`1`) immediately fails CI builds when any gated metric regresses.
- **Trend History**: Persists evaluation snapshots across versions/commits in `EvalRunStore`.

---

## 2. CLI Usage (`dynavec eval check`)

The `dynavec eval check` command compares a current eval result against a baseline JSON snapshot.

```bash
dynavec eval check \
  --current current_eval.json \
  --baseline baseline_eval.json \
  --max-drop 0.05
```

### CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--current`, `-c` | Path to current run eval summary JSON | **Required** |
| `--baseline`, `-b` | Path to baseline run eval summary JSON | **Required** (unless `--allow-missing-baseline`) |
| `--max-drop` | Max allowed relative degradation | `0.05` (5%) |
| `--threshold METRIC=VAL` | Override threshold for a specific metric (repeatable) | None |
| `--absolute` | Compare absolute drop instead of relative change | `False` |
| `--store FILE` | Path to `EvalRunStore` JSON file to persist history | None |
| `--tag KEY=VALUE` | Attach metadata tag to the stored run (repeatable) | None |
| `--json` | Emit structured JSON report instead of text table | `False` |
| `--allow-missing-baseline` | Succeed with warning if baseline file does not exist | `False` |

### Exit Codes

- `0`: All metrics passed (within thresholds or improved).
- `1`: One or more metrics regressed beyond threshold (CI fails).
- `2`: Missing file or invalid CLI arguments.

### Output Example

```text
Metric         | Baseline | Current | Delta   | % Change | Threshold | Status
---------------+----------+---------+---------+----------+-----------+-----------
mrr            | 0.8200   | 0.8350  | +0.0150 | +1.83%   | <= 5.0%   | IMPROVED
p95_latency_ms | 32.0000  | 33.5000 | +1.5000 | +4.69%   | <= 5.0%   | PASS
recall@10      | 0.9100   | 0.8200  | -0.0900 | -9.89%   | <= 5.0%   | REGRESSION

Overall: FAILED: 1 metric(s) regressed beyond threshold.
```

---

## 3. GitHub Actions Workflow Example

Create `.github/workflows/eval-gate.yml` in your repository:

```yaml
name: Evaluation Regression Gate

on:
  pull_request:
    branches: [main, development]
  push:
    branches: [main]

jobs:
  eval-gate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Cache eval baseline
        id: cache-baseline
        uses: actions/cache@v4
        with:
          path: .eval_cache/baseline.json
          key: eval-baseline-${{ runner.os }}

      - name: Run benchmark evaluation
        run: |
          python -m tests.run_eval_benchmark --output current_eval.json

      - name: Gate on regressions
        run: |
          dynavec eval check \
            --current current_eval.json \
            --baseline .eval_cache/baseline.json \
            --max-drop 0.05 \
            --threshold p95_latency_ms=0.15 \
            --allow-missing-baseline \
            --store .eval_cache/eval_history.json \
            --tag commit=${{ github.sha }} \
            --tag branch=${{ github.ref_name }}

      - name: Update baseline on main branch
        if: github.ref == 'refs/heads/main' && success()
        run: |
          mkdir -p .eval_cache
          cp current_eval.json .eval_cache/baseline.json
```

---

## 4. Python Testing API (`assert_no_regression`)

You can also enforce regression gates inside `pytest` test suites:

```python
from dynavec.eval import assert_no_regression, EvalRunner, MockJudge

def test_rag_quality_no_regression():
    runner = EvalRunner(judge=MockJudge())
    current_summary = runner.run(test_dataset)

    # Compare against saved baseline
    assert_no_regression(
        current=current_summary,
        baseline="benchmarks/baselines/v0.7.0.json",
        max_drop=0.05,
        metric_thresholds={"faithfulness": 0.03},
    )
```

---

## 5. Cross-Version Trend Persistence (`EvalRunStore`)

`EvalRunStore` can persist evaluation history to disk and reload it:

```python
from dynavec.eval import EvalRunStore

# Load existing history (or create new if file doesn't exist)
store = EvalRunStore.load_json("eval_history.json")

# Record new evaluation run
store.record(summary, tags={"version": "0.7.1", "git_commit": "abc1234"})

# Persist to disk atomically
store.save_json("eval_history.json")

# Query trends over time
recall_trend = store.trend("recall", k=10)  # [(timestamp, value), ...]
```
