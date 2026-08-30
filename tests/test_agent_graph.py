from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from agent.graph import build_graph, create_initial_state, route_after_model, tool_node
from agent.prompts import SYSTEM_PROMPT


def test_create_initial_state_maps_request_to_messages_and_empty_memory() -> None:
    state = create_initial_state("hello")

    assert isinstance(state["messages"][0], SystemMessage)
    assert state["messages"][0].content == SYSTEM_PROMPT
    assert isinstance(state["messages"][1], HumanMessage)
    assert state["messages"][1].content == "hello"
    assert state["customer_id"] == ""
    assert state["order_id"] == ""
    assert state["tool_results"] == {}
    assert state["approval_required"] is False
    assert state["step_count"] == 0


def test_route_after_model_ends_when_last_message_has_no_tool_calls() -> None:
    state = create_initial_state("hello")
    state["messages"].append(AIMessage("done"))

    assert route_after_model(state) == END


def test_route_after_model_routes_to_tools_when_last_message_requests_tool() -> None:
    state = create_initial_state("hello")
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_customer",
                    "args": {"customer_id": "1842"},
                    "id": "call_1",
                }
            ],
        )
    )

    assert route_after_model(state) == "tool_node"


def test_tool_node_executes_allowlisted_tool_and_updates_state() -> None:
    state = create_initial_state("hello")
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_customer",
                    "args": {"customer_id": "1842"},
                    "id": "call_1",
                }
            ],
        )
    )

    update = tool_node(state)

    assert update["customer_id"] == "1842"
    assert update["order_id"] == ""
    assert update["tool_results"]["get_customer"]["name"] == "Acme Logistics"
    assert update["approval_required"] is False
    assert update["step_count"] == 1
    assert update["messages"][-1].type == "tool"


def test_tool_node_marks_approval_required_from_action_result() -> None:
    state = create_initial_state("hello")
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "prepare_refund",
                    "args": {"order_id": "O-2000"},
                    "id": "call_1",
                }
            ],
        )
    )

    update = tool_node(state)

    assert update["order_id"] == "O-2000"
    assert update["tool_results"]["prepare_refund"]["status"] == "approval_required"
    assert update["approval_required"] is True


def test_build_graph_returns_invokable_graph() -> None:
    graph = build_graph()

    assert callable(graph.invoke)
