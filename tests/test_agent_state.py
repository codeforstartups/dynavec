from dynavec.agents import AgentState


def test_agent_state_stores_shared_data():
    state = AgentState()

    state.data["customer_id"] = "C123"
    state.data["decline_code"] = "51"

    assert state.data["customer_id"] == "C123"
    assert state.data["decline_code"] == "51"


def test_agent_state_has_independent_data():
    first = AgentState()
    second = AgentState()

    first.data["customer_id"] = "C123"

    assert second.data == {}
