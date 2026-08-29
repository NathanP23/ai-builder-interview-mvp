from pprint import pprint
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing_extensions import TypedDict


POLICY_DIR = Path("policies")


class AgentState(TypedDict):
    messages: list[BaseMessage]
    customer_id: str
    order_id: str
    tool_results: dict[str, Any]
    approval_required: bool
    step_count: int


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
        "amount_cents": 49900,
        "status": "paid",
    },
    "O-2000": {
        "customer_id": "1842",
        "item": "Enterprise hardware bundle",
        "amount": "$2,500.00",
        "amount_cents": 250000,
        "status": "paid",
    },
    "O-123": {
        "customer_id": "1842",
        "item": "Support seat",
        "amount": "$99.00",
        "amount_cents": 9900,
        "status": "paid",
    }
}

TRANSACTIONS = [
    {"customer_id": "1842", "order_id": "O-991", "kind": "charge", "amount": "$499.00"},
    {"customer_id": "1842", "order_id": "O-991", "kind": "charge", "amount": "$499.00"},
    {"customer_id": "1842", "order_id": "O-2000", "kind": "charge", "amount": "$2,500.00"},
    {"customer_id": "1842", "order_id": "O-2000", "kind": "charge", "amount": "$2,500.00"},
    {"customer_id": "1842", "order_id": "O-123", "kind": "charge", "amount": "$99.00"},
]

AUTO_REFUND_LIMIT_CENTS = 100000


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


@tool
def prepare_refund(order_id: str) -> dict[str, Any]:
    """Prepare a refund only if deterministic backend policy allows it."""
    order = ORDERS.get(order_id)
    if not order:
        return {
            "status": "blocked",
            "reason": "order not found",
            "order_id": order_id,
        }

    matching_charges = [
        transaction
        for transaction in TRANSACTIONS
        if transaction["order_id"] == order_id and transaction["kind"] == "charge"
    ]
    if len(matching_charges) < 2:
        return {
            "status": "blocked",
            "reason": "no duplicate charge found",
            "order_id": order_id,
        }

    if order["amount_cents"] > AUTO_REFUND_LIMIT_CENTS:
        return {
            "status": "approval_required",
            "reason": "duplicate charge is above the automatic refund limit",
            "order_id": order_id,
            "amount": order["amount"],
        }

    return {
        "status": "prepared",
        "reason": "duplicate charge is under the automatic refund limit",
        "order_id": order_id,
        "amount": order["amount"],
    }


def load_policy_documents() -> list[Document]:
    print("\n============================================================")
    print("LOAD POLICY DOCUMENTS")
    print("Python reads fake policy markdown files from policies/.")
    print("Each file becomes one document chunk for this tiny demo.")
    print("Later, chunking can become smarter if the files get large.")

    documents = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text()
        document = Document(page_content=text, metadata={"source": path.name})
        documents.append(document)
        print(f"- loaded {path.name} as one chunk")

    return documents


def find_policy_matches(query: str, k: int = 2) -> list[Document]:
    documents = load_policy_documents()

    print("\nEMBEDDINGS")
    print("Python sends each policy chunk to OpenAIEmbeddings.")
    print("The hosted embedding model returns vectors: lists of numbers meaning text position.")
    print("Secrets are not printed. Policy text does leave the machine for embedding.")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\nVECTOR STORE")
    print("InMemoryVectorStore stores the document vectors only in this Python process.")
    print("No database or persistence yet; when the program exits, this store disappears.")
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)

    print("\nQUERY")
    print(f"Policy search query: {query!r}")
    print("Python embeds the query, compares it to policy vectors, and returns top-k matches.")
    return vector_store.similarity_search(query, k=k)


@tool
def search_policy(query: str) -> list[dict[str, str]]:
    """Search refund, escalation, account, and privacy policy text."""
    print("\n============================================================")
    print("SEARCH POLICY TOOL")
    print("The LLM asked for policy context instead of guessing the rule.")
    print("Python now runs semantic retrieval over the policy documents.")

    matches = find_policy_matches(query)

    print("\nPOLICY OBSERVATIONS RETURNED TO THE LLM")
    print("These snippets become a ToolMessage, so the next LLM turn can use them.")
    results = [
        {"source": match.metadata["source"], "content": match.page_content.strip()}
        for match in matches
    ]
    pprint(results)
    return results


TOOLS = {
    "get_customer": get_customer,
    "get_order": get_order,
    "search_transactions": search_transactions,
    "search_policy": search_policy,
    "prepare_refund": prepare_refund,
}


def run_policy_retrieval_demo() -> None:
    print("\n============================================================")
    print("POLICY RETRIEVAL DEMO")
    print("This is Checkpoint 8: retrieval exists, but it is not an agent tool yet.")
    print("Goal: turn policy text into vectors, then search for relevant policy chunks.")

    query = "When can a duplicate charge be refunded?"
    matches = find_policy_matches(query)

    print("\nTOP POLICY MATCHES")
    for index, match in enumerate(matches, start=1):
        print(f"{index}. source={match.metadata['source']}")
        print(match.page_content.strip())


def model_node(state: AgentState) -> dict[str, Any]:
    global model_pass_count
    model_pass_count += 1

    print("\n============================================================")
    print(f"MODEL NODE PASS #{model_pass_count}")
    print("LangGraph now calls model_node(state).")
    print("This function gives the LLM the conversation so far.")
    print("The LLM can either answer, or ask Python to run tools.")
    print("\nExplicit state memory entering this node:")
    print(f"- customer_id: {state['customer_id']!r}")
    print(f"- order_id: {state['order_id']!r}")
    print(f"- tool_results keys: {list(state['tool_results'])}")
    print(f"- approval_required: {state['approval_required']}")
    print(f"- step_count: {state['step_count']}")
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
    print("It also increments step_count so we can see progress through the graph.")
    update = {
        "messages": [*state["messages"], response],
        "step_count": state["step_count"] + 1,
    }
    print(f"message count is now {len(update['messages'])}")
    print(f"step_count is now {update['step_count']}")
    print("Next: LangGraph asks route_after_model where to go.")
    return update


def tool_node(state: AgentState) -> dict[str, Any]:
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
    customer_id = state["customer_id"]
    order_id = state["order_id"]
    tool_results = dict(state["tool_results"])

    print("\nExplicit state memory before tools:")
    print(f"- customer_id: {customer_id!r}")
    print(f"- order_id: {order_id!r}")
    print(f"- tool_results: {tool_results}")

    for index, tool_call in enumerate(last_message.tool_calls):
        print("\n------------------------------------------------------------")
        print(f"Tool request #{index + 1} from the LLM:")
        print(f"- requested tool: {tool_call['name']}")
        print(f"- requested args: {tool_call['args']}")
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        selected_tool = TOOLS[tool_name]

        if "customer_id" in tool_args:
            customer_id = tool_args["customer_id"]
            print(f"State memory learns customer_id={customer_id!r} from tool args.")
        if "order_id" in tool_args:
            order_id = tool_args["order_id"]
            print(f"State memory learns order_id={order_id!r} from tool args.")

        print("Python dispatch:")
        print(f"- TOOLS[{tool_name!r}] selects the allowlisted function wrapper")
        print("- now Python executes it with the LLM-provided args")
        tool_result = selected_tool.invoke(tool_args)
        print("Observation produced by Python:")
        pprint(tool_result)
        tool_results[tool_name] = tool_result
        print(f"State memory stores this observation under tool_results[{tool_name!r}].")
        if tool_name == "prepare_refund":
            print("\nACTION TOOL POLICY RESULT:")
            print("The LLM requested a refund action, but Python made the policy decision.")
            print(f"- policy status: {tool_result['status']!r}")
            print(f"- policy reason: {tool_result['reason']!r}")

        print("Python wraps that observation in a ToolMessage.")
        print("ToolMessage links the result back to the original tool_call_id.")
        tool_message = ToolMessage(
            str(tool_result), tool_call_id=tool_call["id"]
        )
        print(f"- ToolMessage content: {tool_message.content!r}")
        tool_messages.append(tool_message)

    update = {
        "messages": [*state["messages"], *tool_messages],
        "customer_id": customer_id,
        "order_id": order_id,
        "tool_results": tool_results,
        "approval_required": any(
            result.get("status") == "approval_required"
            for result in tool_results.values()
            if isinstance(result, dict)
        ),
        "step_count": state["step_count"] + 1,
    }
    print("\ntool_node returns updated state:")
    print("old messages + ToolMessage observations")
    print(f"message count is now {len(update['messages'])}")
    print("Explicit state memory after tools:")
    print(f"- customer_id: {update['customer_id']!r}")
    print(f"- order_id: {update['order_id']!r}")
    print("- tool_results:")
    pprint(update["tool_results"])
    print(f"- approval_required: {update['approval_required']}")
    print(f"- step_count: {update['step_count']}")
    print("Next: LangGraph follows tool_node -> model_node.")
    print("The next model pass will see the observations and decide what to do next.")
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
        "order, transactions, and refund policy, then prepare a refund if policy allows."
    )
    print("\nUser request entering the system:")
    print(support_request)

    # Initial state is just the conversation entering the graph.
    # In VS Code: launch.json args -> sys.argv -> support_request
    # -> HumanMessage -> graph state.
    # Later nodes append AI and tool messages until the answer is complete.
    print("\nPython creates initial_state.")
    print("This is the conversation LangGraph will carry through the graph.")
    print("It also starts explicit state memory as empty/default values.")
    initial_state: AgentState = {
        "messages": [
            SystemMessage(
                "You are a support agent. Use get_customer for customer IDs, "
                "get_order for order IDs, search_transactions when checking charge history, "
                "search_policy when you need refund or escalation rules, "
                "and prepare_refund only after observations show a duplicate charge. "
                "Never claim a refund was executed; it can only be prepared or require approval."
            ),
            HumanMessage(support_request),
        ],
        "customer_id": "",
        "order_id": "",
        "tool_results": {},
        "approval_required": False,
        "step_count": 0,
    }
    for index, message in enumerate(initial_state["messages"]):
        print(f"- messages[{index}] {message.type}: {message.content!r}")
    print("- customer_id: ''")
    print("- order_id: ''")
    print("- tool_results: {}")
    print("- approval_required: False")
    print("- step_count: 0")

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

    print("\nFinal explicit state memory:")
    print(f"- customer_id: {final_state['customer_id']!r}")
    print(f"- order_id: {final_state['order_id']!r}")
    print("- tool_results:")
    pprint(final_state["tool_results"])
    print(f"- approval_required: {final_state['approval_required']}")
    print(f"- step_count: {final_state['step_count']}")

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
    print("-> explicit state stored customer_id, order_id, and tool_results")
    print("-> if policy was needed, search_policy returned retrieved policy snippets")
    print("-> Python policy decided whether refund preparation was allowed")
    print("-> later model pass produced final answer")
    print("-> route_after_model -> END")
    print("-> graph.invoke returned final_state")


if __name__ == "__main__":
    main()
