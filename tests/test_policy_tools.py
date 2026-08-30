from tools import policy_tools
from tools.policy_tools import search_policy


def test_search_policy_tool_returns_policy_results(monkeypatch) -> None:
    monkeypatch.setattr(
        policy_tools,
        "policy_results_for_query",
        lambda query: [{"source": "refund.md", "content": query}],
    )

    assert search_policy.invoke({"query": "refund rule"}) == [
        {"source": "refund.md", "content": "refund rule"}
    ]
