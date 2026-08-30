from agent.reporting import print_final_report, print_langsmith_status
from agent.state import AgentState
from langchain_core.messages import AIMessage


def test_print_langsmith_status_never_prints_api_key_value(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret-test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "demo-project")

    print_langsmith_status()

    output = capsys.readouterr().out
    assert "API key configured: True" in output
    assert "demo-project" in output
    assert "secret-test-key" not in output


def test_print_final_report_includes_answer_and_state_summary(capsys) -> None:
    final_state: AgentState = {
        "messages": [AIMessage("final answer")],
        "customer_id": "1842",
        "order_id": "O-991",
        "approval_required": False,
        "step_count": 1,
        "tool_results": {"prepare_refund": {"status": "prepared"}},
    }

    print_final_report(final_state)

    output = capsys.readouterr().out
    assert "Final natural-language answer:" in output
    assert "final answer" in output
    assert "customer_id: '1842'" in output
