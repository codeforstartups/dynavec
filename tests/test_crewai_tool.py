"""Tests for the CrewAI tool wrapper.

crewai isn't in the dev/ingest test extras (only the optional [crewai]/[all]
extras), so these tests inject a minimal fake crewai module into sys.modules
rather than requiring the real (heavy) dependency to be installed.
"""

import sys
import types

import pytest

from dynavec.exceptions import MissingDependencyError
from dynavec.models import SearchResult


class _FakeClient:
    """Stands in for a Dynavec client's .search()."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [SearchResult(id="doc-1", score=0.87, text="CrewAI-relevant passage")]


class _FakeTool:
    """Minimal stand-in for crewai.tools.Tool — real Tool is invoked via .run(),
    not by calling it directly (see crewai.tools.base_tool.Tool.run)."""

    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
        self.description = func.__doc__

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _install_fake_crewai(monkeypatch):
    def fake_tool_decorator(name=None):
        def decorator(func):
            return _FakeTool(func, name=name)

        return decorator

    fake_crewai = types.ModuleType("crewai")
    fake_crewai_tools = types.ModuleType("crewai.tools")
    fake_crewai_tools.tool = fake_tool_decorator
    fake_crewai.tools = fake_crewai_tools

    monkeypatch.setitem(sys.modules, "crewai", fake_crewai)
    monkeypatch.setitem(sys.modules, "crewai.tools", fake_crewai_tools)


def test_as_crewai_tool_runs_search_and_returns_text(monkeypatch):
    _install_fake_crewai(monkeypatch)
    from dynavec.integrations.tools import as_crewai_tool

    client = _FakeClient()
    tool = as_crewai_tool(client, name="dynavec_search", top_k=2, namespace="kb")

    assert tool.name == "dynavec_search"

    result = tool.run("machine learning")

    assert result == "CrewAI-relevant passage"
    assert client.calls == [
        (
            "machine learning",
            {"top_k": 2, "namespace": "kb", "filter": None, "rescore": None},
        )
    ]


def test_as_crewai_tool_missing_dependency_raises():
    """crewai isn't in the dev/ingest test extras, so it's genuinely absent here —
    confirms we get our own clear error instead of a raw ImportError."""
    from dynavec.integrations.tools import as_crewai_tool

    with pytest.raises(MissingDependencyError):
        as_crewai_tool(_FakeClient())