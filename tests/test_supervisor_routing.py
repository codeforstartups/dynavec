from dynavec.agents import AgentGraph, AgentResult, AgentState, GraphState, Supervisor


def test_supervisor_routes_only_to_selected_specialist():
    calls: list[str] = []

    def route(state: GraphState) -> str:
        return state["selected_specialist"]

    def transaction_agent(state: GraphState) -> dict:
        calls.append("transaction")
        return {"transaction_result": "Declined with code 51"}

    def policy_agent(state: GraphState) -> dict:
        calls.append("policy")
        return {"policy_result": "Review required"}

    graph = (
        AgentGraph("supervisor-routing")
        .add_node("supervisor", lambda state: {})
        .add_node("transaction", transaction_agent)
        .add_node("policy", policy_agent)
        .add_conditional_edges(
            "supervisor",
            route,
            branches={
                "transaction": "transaction",
                "policy": "policy",
            },
        )
        .set_entry("supervisor")
    )

    result = graph.run({"selected_specialist": "transaction"})

    assert result.success
    assert calls == ["transaction"]
    assert result.state["transaction_result"] == "Declined with code 51"
    assert "policy_result" not in result.state


def test_handoff_preserves_findings_for_next_specialist():
    def transaction_agent(state: GraphState) -> dict:
        return {
            "transaction_result": "Declined",
            "decline_code": "51",
        }

    def policy_agent(state: GraphState) -> dict:
        assert state["decline_code"] == "51"
        return {"policy_result": "Manual review required"}

    graph = (
        AgentGraph("supervisor-handoff")
        .add_node("transaction", transaction_agent)
        .add_node("policy", policy_agent)
        .add_edge("transaction", "policy")
        .set_entry("transaction")
    )

    result = graph.run({"customer_query": "Why was my transaction declined?"})

    assert result.success
    assert result.state["decline_code"] == "51"
    assert result.state["policy_result"] == "Manual review required"


def test_supervisor_routes_to_multiple_specialists_with_shared_state():
    state = AgentState()
    calls = []

    class TransactionAgent:
        def run(self, goal: str, *, state: AgentState) -> AgentResult:
            calls.append("transaction")
            state.data["transaction_id"] = "TX123"
            state.data["decline_code"] = "51"
            return AgentResult(output="Transaction analyzed.")

    class PolicyAgent:
        def run(self, goal: str, *, state: AgentState) -> AgentResult:
            calls.append("policy")
            assert state.data["transaction_id"] == "TX123"
            assert state.data["decline_code"] == "51"
            return AgentResult(output="Policy reviewed.")

    supervisor = Supervisor(
        agents={
            "transaction": TransactionAgent(),
            "policy": PolicyAgent(),
        },
        router=lambda goal, state: ["transaction", "policy"],
    )

    results = supervisor.run(
        "Review transaction TX123.",
        state=state,
    )

    assert calls == ["transaction", "policy"]
    assert [result.output for result in results] == [
        "Transaction analyzed.",
        "Policy reviewed.",
    ]


def test_handoff_request_routes_to_named_specialist_with_shared_state():
    from dynavec.agents import Handoff

    handoff = Handoff()
    request = handoff("policy", "Review transaction TX123.")

    calls = []

    def supervisor_node(state: GraphState) -> dict:
        return {
            "handoff_target": request.target,
            "handoff_goal": request.goal,
            "decline_code": "51",
        }

    def transaction_agent(state: GraphState) -> dict:
        calls.append("transaction")
        return {"unexpected": True}

    def policy_agent(state: GraphState) -> dict:
        calls.append("policy")
        assert state["handoff_goal"] == "Review transaction TX123."
        assert state["decline_code"] == "51"
        return {"policy_result": "Manual review required."}

    graph = (
        AgentGraph("handoff-request-routing")
        .add_node("supervisor", supervisor_node)
        .add_node("transaction", transaction_agent)
        .add_node("policy", policy_agent)
        .add_conditional_edges(
            "supervisor",
            lambda state: state["handoff_target"],
            branches={
                "transaction": "transaction",
                "policy": "policy",
            },
        )
        .set_entry("supervisor")
    )

    result = graph.run()

    assert result.success
    assert calls == ["policy"]
    assert result.state["policy_result"] == "Manual review required."
    assert "unexpected" not in result.state
