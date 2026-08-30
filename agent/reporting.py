from pprint import pprint
import os

from agent.state import AgentState


def print_langsmith_status() -> None:
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    api_key_configured = bool(os.getenv("LANGSMITH_API_KEY"))
    project = os.getenv("LANGSMITH_PROJECT", "default")

    print("\nLangSmith tracing check:")
    print("LangSmith is the hosted timeline for this run: model calls, tools, retrieval, errors.")
    print(f"- tracing enabled: {tracing_enabled}")
    print(f"- API key configured: {api_key_configured}")
    print(f"- project: {project!r}")
    print("No LangSmith or OpenAI secret values are printed.")


def print_final_report(final_state: AgentState) -> None:
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
