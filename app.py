from pprint import pprint
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: list[BaseMessage]


CUSTOMERS = {
    "1842": {
        "name": "Acme Logistics",
        "tier": "enterprise",
        "status": "active",
    }
}


@tool
def get_customer(customer_id: str) -> dict[str, str]:
    """Look up a customer by customer ID."""
    return CUSTOMERS.get(customer_id, {"error": "customer not found"})


TOOLS = {"get_customer": get_customer}


def summarize_state(state: AgentState) -> list[dict[str, Any]]:
    return [
        {
            "type": message.type,
            "content": message.content,
            "tool_calls": getattr(message, "tool_calls", None),
        }
        for message in state["messages"]
    ]


def model_node(state: AgentState) -> AgentState:
    # Node functions receive graph state and return only the fields they update.
    model = ChatOpenAI(model="gpt-4.1-mini").bind_tools([get_customer])
    response = model.invoke(state["messages"])
    update = {"messages": [*state["messages"], response]}

    print("\nmodel_node update:")
    pprint(summarize_state(update))

    return update


def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_messages = []

    # Tool calls are model requests; Python still owns execution.
    for tool_call in last_message.tool_calls:
        result = TOOLS[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(ToolMessage(str(result), tool_call_id=tool_call["id"]))

    update = {"messages": [*state["messages"], *tool_messages]}

    print("\ntool_node update:")
    pprint(summarize_state(update))

    return update


def route_after_model(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tool_node" if last_message.tool_calls else END


def main() -> None:
    # Loads OPENAI_API_KEY from local .env into the process environment.
    load_dotenv()

    initial_state: AgentState = {
        "messages": [
            SystemMessage(
                "You are a support agent. Use get_customer when a request includes a customer ID."
            ),
            HumanMessage("Customer 1842 says they need help. Who is this customer?"),
        ]
    }

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("model_node", model_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_edge(START, "model_node")
    graph_builder.add_conditional_edges("model_node", route_after_model)
    graph_builder.add_edge("tool_node", "model_node")

    # compile() turns the graph definition into something runnable.
    graph = graph_builder.compile()

    print("Initial state:")
    print(initial_state)

    final_state = graph.invoke(initial_state)

    print("\nFinal state:")
    pprint(summarize_state(final_state))


if __name__ == "__main__":
    main()
