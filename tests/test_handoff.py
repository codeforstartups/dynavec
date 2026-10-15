"""Handoff primitives for multi-agent workflows."""

from __future__ import annotations

from dynavec.agents import AgentResult, AgentState, Handoff
from dynavec.agents.handoff import HandoffRequest


def test_handoff_creates_named_routing_request():
    handoff = Handoff()

    request = handoff("policy", "Review transaction TX123.")

    assert isinstance(request, HandoffRequest)
    assert request.target == "policy"
    assert request.goal == "Review transaction TX123."


def test_handoff_rejects_empty_target_or_goal():
    handoff = Handoff()

    try:
        handoff("", "Review transaction.")
    except ValueError as exc:
        assert "target" in str(exc)
    else:
        raise AssertionError("Expected an empty target to be rejected.")

    try:
        handoff("policy", "")
    except ValueError as exc:
        assert "goal" in str(exc)
    else:
        raise AssertionError("Expected an empty goal to be rejected.")


def test_handoff_invokes_target_agent_with_shared_state():
    class PolicyAgent:
        def run(self, goal: str, *, state: AgentState) -> AgentResult:
            assert goal == "Review transaction TX123."
            assert state.data["decline_code"] == "51"
            state.data["policy_result"] = "Manual review required."
            return AgentResult(output=state.data["policy_result"])

    state = AgentState(data={"transaction_id": "TX123", "decline_code": "51"})
    handoff = Handoff()
    target = PolicyAgent()

    result = handoff.transfer(
        target,
        "Review transaction TX123.",
        state=state,
    )

    assert result.output == "Manual review required."
    assert state.data["policy_result"] == "Manual review required."
