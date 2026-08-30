from pprint import pprint

from langchain_core.tools import tool

from retrieval.policy_retrieval import policy_results_for_query


@tool
def search_policy(query: str) -> list[dict[str, str]]:
    """Search refund, escalation, account, and privacy policy text."""
    print("\n============================================================")
    print("SEARCH POLICY TOOL")
    print("The LLM asked for policy context instead of guessing the rule.")
    print("Python now runs keyword + semantic retrieval over the policy documents.")

    print("\nPOLICY OBSERVATIONS RETURNED TO THE LLM")
    print("These snippets become a ToolMessage, so the next LLM turn can use them.")
    results = policy_results_for_query(query)
    pprint(results)
    return results
