from langchain_core.messages import AIMessage

from agent.state import AgentState
from evals.eval_runner import actual_from_state, load_eval_cases, score_case


def test_load_eval_cases_reads_moved_eval_dataset() -> None:
    cases = load_eval_cases()

    assert cases[0]["id"] == "duplicate_low_value_refund"
    assert any(case["id"] == "malicious_refund_override" for case in cases)


def test_actual_from_state_handles_no_policy_or_refund_tool_results() -> None:
    final_state: AgentState = {
        "messages": [AIMessage("no tools needed")],
        "customer_id": "",
        "order_id": "",
        "approval_required": False,
        "step_count": 1,
        "tool_results": {},
    }

    assert actual_from_state(final_state) == {
        "tools": [],
        "customer_id": "",
        "order_id": "",
        "sources": [],
        "refund_status": "",
        "final_answer": "no tools needed",
    }


def test_score_case_treats_missing_refund_status_as_blocked_when_expected() -> None:
    case = {
        "expected_tools": [],
        "expected_customer_id": "",
        "expected_order_id": "",
        "expected_sources": [],
        "expected_refund_status": "blocked",
        "forbidden_actions": [],
    }
    actual = {
        "tools": [],
        "customer_id": "",
        "order_id": "",
        "sources": [],
        "refund_status": "",
    }

    assert score_case(case, actual)["refund_status"] is True
