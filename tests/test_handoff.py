from dynavec.agents import AgentState, Handoff


def test_handoff_preserves_shared_state():
    state = AgentState()

    state.data["transaction_id"] = "TX123"
    state.data["decline_code"] = "51"

    handoff = Handoff()

    next_state = handoff(state)

    assert next_state is state
    assert next_state.data == {
        "transaction_id": "TX123",
        "decline_code": "51",
    }
