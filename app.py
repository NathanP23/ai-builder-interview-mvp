from pprint import pprint
import sys

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: list[BaseMessage]


model_pass_count = 0


# Demo business data lives in normal Python structures for now.
# These are exact operational records, so tools should look them up by ID.
CUSTOMERS = {
    "1842": {
        "name": "Acme Logistics",
        "tier": "enterprise",
        "status": "active",
    }
}

ORDERS = {
    "O-991": {
        "customer_id": "1842",
        "item": "Warehouse scanner subscription",
        "amount": "$499.00",
        "status": "paid",
    }
}

TRANSACTIONS = [
    {"customer_id": "1842", "order_id": "O-991", "kind": "charge", "amount": "$499.00"},
    {"customer_id": "1842", "order_id": "O-991", "kind": "charge", "amount": "$499.00"},
]


@tool
def get_customer(customer_id: str) -> dict[str, str]:
    """Look up a customer by customer ID."""
    return CUSTOMERS.get(customer_id, {"error": "customer not found"})


@tool
def get_order(order_id: str) -> dict[str, str]:
    """Look up an order by order ID."""
    return ORDERS.get(order_id, {"error": "order not found"})


@tool
def search_transactions(customer_id: str) -> list[dict[str, str]]:
    """Find transactions for a customer ID."""
    return [
        transaction
        for transaction in TRANSACTIONS
        if transaction["customer_id"] == customer_id
    ]


TOOLS = {
    "get_customer": get_customer,
    "get_order": get_order,
    "search_transactions": search_transactions,
}


def model_node(state: AgentState) -> AgentState:
    global model_pass_count
    model_pass_count += 1

    print("\n============================================================")
    print(f"MODEL NODE PASS #{model_pass_count}")
    print("LangGraph now calls model_node(state).")
    print("This function gives the LLM the conversation so far.")
    print("The LLM can either answer, or ask Python to run tools.")
    print("\nConversation the LLM will see:")
    for index, message in enumerate(state["messages"]):
        print(f"- {index}: {message.type}: {message.content!r}")
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("  tool calls on this message:")
            pprint(tool_calls)

    # LangGraph gives control to this node with the conversation so far.
    # The model sees the messages plus tool schemas; it can answer directly
    # or ask Python to run one of the bound tools.
    available_tools = list(TOOLS.values())
    print("\nAvailable tools shown to the LLM:")
    for available_tool in available_tools:
        print(f"- {available_tool.name}: {available_tool.description}")
        print(f"  args: {available_tool.args}")

    # @tool let LangChain derive each tool name/description/schema from Python
    # instead of us manually constructing the schema sent to the model.
    print("\nPython creates a ChatOpenAI wrapper.")
    print("LangChain hides the raw HTTP request, but the model still runs remotely.")
    model = ChatOpenAI(model="gpt-4.1-mini")

    print("Python binds the tools.")
    print("bind_tools only exposes schemas. It does not run any tool.")
    model = model.bind_tools(available_tools)

    print("\nLLM TURN:")
    print("Python sends messages + tool schemas to OpenAI.")
    print("The API key is used for auth, but never printed.")
    response = model.invoke(state["messages"])

    if response.tool_calls:
        print("\nLLM DECISION:")
        print("The LLM is not ready to answer yet.")
        print("It asks Python to run these tools:")
        for tool_call in response.tool_calls:
            print(f"- {tool_call['name']}({tool_call['args']})")
    else:
        print("\nLLM DECISION:")
        print("The LLM has enough observations and writes the final answer:")
        print(response.content)

    print("\nmodel_node returns an updated state:")
    print("old messages + this new AI message")
    update = {"messages": [*state["messages"], response]}
    print(f"message count is now {len(update['messages'])}")
    print("Next: LangGraph asks route_after_model where to go.")
    return update


def tool_node(state: AgentState) -> AgentState:
    print("\n============================================================")
    print("TOOL NODE")
    print("LangGraph routed here because the LLM requested tools.")
    print("Important: the LLM did not execute anything.")
    print("Python now dispatches each requested tool from the allowlist.")

    # The LLM requested tools, but Python is in control here.
    # Only names in TOOLS can run, and each result becomes a ToolMessage
    # observation for the next model step.
    last_message = state["messages"][-1]

    tool_messages = []

    for index, tool_call in enumerate(last_message.tool_calls):
        print("\n------------------------------------------------------------")
        print(f"Tool request #{index + 1} from the LLM:")
        print(f"- requested tool: {tool_call['name']}")
        print(f"- requested args: {tool_call['args']}")
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        selected_tool = TOOLS[tool_name]

        print("Python dispatch:")
        print(f"- TOOLS[{tool_name!r}] selects the allowlisted function wrapper")
        print("- now Python executes it with the LLM-provided args")
        tool_result = selected_tool.invoke(tool_args)
        print("Observation produced by Python:")
        pprint(tool_result)

        print("Python wraps that observation in a ToolMessage.")
        print("ToolMessage links the result back to the original tool_call_id.")
        tool_message = ToolMessage(
            str(tool_result), tool_call_id=tool_call["id"]
        )
        print(f"- ToolMessage content: {tool_message.content!r}")
        tool_messages.append(tool_message)

    update = {"messages": [*state["messages"], *tool_messages]}
    print("\ntool_node returns updated state:")
    print("old messages + ToolMessage observations")
    print(f"message count is now {len(update['messages'])}")
    print("Next: LangGraph follows tool_node -> model_node.")
    print("The second model pass will see the observations and answer.")
    return update


def route_after_model(state: AgentState) -> str:
    print("\n============================================================")
    print("ROUTING")
    print("LangGraph asks: did the last LLM message request tools?")

    # This edge decides whether the graph needs a Python tool step.
    # No tool calls means the model has produced the final answer.
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    route = "tool_node" if tool_calls else END

    if tool_calls:
        print("Yes. Route to tool_node so Python can execute them.")
    else:
        print("No. Route to END because the model produced the final answer.")
    print(f"Route returned to LangGraph: {route!r}")
    return route


def main() -> None:
    global model_pass_count
    model_pass_count = 0

    print("\n============================================================")
    print("PROGRAM START")
    print("Python starts locally on your machine.")

    # Loads OPENAI_API_KEY from local .env into the process environment.
    print("\nPython loads .env so the OpenAI client can authenticate.")
    print("The API key is never printed.")
    load_dotenv()

    support_request = (
        " ".join(sys.argv[1:])
        or "Customer 1842 says order O-991 was charged twice. Check the customer, "
        "order and transactions, then explain what happened and what should happen next."
    )
    print("\nUser request entering the system:")
    print(support_request)

    # Initial state is just the conversation entering the graph.
    # In VS Code: launch.json args -> sys.argv -> support_request
    # -> HumanMessage -> graph state.
    # Later nodes append AI and tool messages until the answer is complete.
    print("\nPython creates initial_state.")
    print("This is the conversation LangGraph will carry through the graph.")
    initial_state: AgentState = {
        "messages": [
            SystemMessage(
                "You are a support agent. Use get_customer for customer IDs, "
                "get_order for order IDs, and search_transactions when checking charge history."
            ),
            HumanMessage(support_request),
        ]
    }
    for index, message in enumerate(initial_state["messages"]):
        print(f"- messages[{index}] {message.type}: {message.content!r}")

    # The graph is still tiny: model -> optional tools -> model.
    # The conditional edge is the agent loop in its smallest useful form.
    print("\n============================================================")
    print("BUILD GRAPH")
    print("Nothing is executing yet. Python is only describing the workflow.")
    graph_builder = StateGraph(AgentState)

    print("- Register model_node: this stores the function for later.")
    graph_builder.add_node("model_node", model_node)

    print("- Register tool_node: this stores the function for later.")
    graph_builder.add_node("tool_node", tool_node)

    print("- Edge START -> model_node: first, ask the LLM what to do.")
    graph_builder.add_edge(START, "model_node")

    print("- Conditional edge after model_node: tools needed or finished?")
    graph_builder.add_conditional_edges("model_node", route_after_model)

    print("- Edge tool_node -> model_node: send observations back to the LLM.")
    graph_builder.add_edge("tool_node", "model_node")

    # compile() freezes the graph definition into a runnable workflow.
    print("\nCompile graph.")
    print("This turns the definition into a runnable workflow.")
    print("Still nothing has executed.")
    graph = graph_builder.compile()

    print("\n============================================================")
    print("RUN GRAPH")
    print("graph.invoke(initial_state) is where execution starts.")
    print("LangGraph now controls which node runs next.")
    final_state = graph.invoke(initial_state)

    print("\n============================================================")
    print("GRAPH FINISHED")
    print("LangGraph returned final_state to normal Python.")
    print("\nFinal message sequence:")
    for index, message in enumerate(final_state["messages"]):
        print(f"- {index}: {message.type}: {message.content!r}")
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("  tool calls:")
            pprint(tool_calls)

    print("\nFinal natural-language answer:")
    print(final_state["messages"][-1].content)

    print("\n============================================================")
    print("EXECUTION SUMMARY")
    print("main")
    print("-> create support_request")
    print("-> create initial_state")
    print("-> build graph definition")
    print("-> compile graph")
    print("-> invoke graph")
    print("-> model_node pass 1")
    print("-> LLM requested tools")
    print("-> route_after_model -> tool_node")
    print("-> Python dispatched and executed tools")
    print("-> ToolMessages created")
    print("-> model_node pass 2")
    print("-> LLM produced final answer")
    print("-> route_after_model -> END")
    print("-> graph.invoke returned final_state")


if __name__ == "__main__":
    main()
