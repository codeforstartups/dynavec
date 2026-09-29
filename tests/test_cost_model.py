"""Tests for benchmark cost-model regression protection."""

from __future__ import annotations

import sys

import pytest

from benchmarks.cost_model import Workload, compare


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Syrupy requires Python 3.10+")
def test_cost_model_snapshot(snapshot) -> None:
    workload = Workload(
        vectors=5_000_000,
        dim=768,
        queries_per_month=1_000_000,
        writes_per_month=200_000,
    )

    costs = {name: round(value, 2) for name, value in compare(workload).items()}

    assert costs == snapshot
