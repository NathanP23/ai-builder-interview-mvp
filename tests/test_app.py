import json

from langchain_core.messages import AIMessage

from agent.state import AgentState
from data.fake_business_data import AUTO_REFUND_LIMIT_CENTS, ORDERS, TRANSACTIONS
from evals.eval_runner import EVAL_CASES_PATH, actual_from_state, score_case
from retrieval.policy_retrieval import keyword_policy_matches, load_policy_documents
from tools.business_tools import (
    get_customer,
    get_order,
    prepare_refund,
    search_transactions,
)
from tools.registry import TOOLS


def test_eval_cases_have_required_fields() -> None:
    with EVAL_CASES_PATH.open() as file:
        cases = json.load(file)

    required_fields = {
        "id",
        "input",
        "expected_tools",
        "expected_order_id",
        "expected_customer_id",
        "expected_sources",
        "expected_refund_status",
        "forbidden_actions",
    }

    assert len(cases) >= 8
    for case in cases:
        assert required_fields <= case.keys()
        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["input"], str) and case["input"]
        assert isinstance(case["expected_tools"], list)
        assert isinstance(case["expected_sources"], list)
        assert isinstance(case["forbidden_actions"], list)


def test_eval_case_ids_are_unique_and_expected_tools_are_known() -> None:
    with EVAL_CASES_PATH.open() as file:
        cases = json.load(file)

    case_ids = [case["id"] for case in cases]
    assert len(case_ids) == len(set(case_ids))

    for case in cases:
        assert set(case["expected_tools"]) <= set(TOOLS)


def test_eval_scoring_detects_expected_behavior() -> None:
    case = {
        "expected_tools": ["get_customer", "search_policy"],
        "expected_customer_id": "1842",
        "expected_order_id": "",
        "expected_sources": ["refund_policy.md"],
        "expected_refund_status": "",
        "forbidden_actions": ["prepare_refund"],
    }
    actual = {
        "tools": ["get_customer", "search_policy"],
        "customer_id": "1842",
        "order_id": "",
        "sources": ["refund_policy.md", "escalation_policy.md"],
        "refund_status": "",
    }

    assert all(score_case(case, actual).values())


def test_eval_scoring_detects_failures_and_forbidden_actions() -> None:
    case = {
        "expected_tools": ["get_customer"],
        "expected_customer_id": "1842",
        "expected_order_id": "",
        "expected_sources": ["refund_policy.md"],
        "expected_refund_status": "",
        "forbidden_actions": ["prepare_refund"],
    }
    actual = {
        "tools": ["get_customer", "prepare_refund"],
        "customer_id": "9999",
        "order_id": "O-991",
        "sources": [],
        "refund_status": "prepared",
    }

    assert score_case(case, actual) == {
        "tools": False,
        "customer_id": False,
        "order_id": False,
        "sources": False,
        "refund_status": False,
        "forbidden_actions": False,
    }


def test_actual_from_state_extracts_eval_signals() -> None:
    final_state: AgentState = {
        "messages": [AIMessage("done")],
        "customer_id": "1842",
        "order_id": "O-991",
        "approval_required": False,
        "step_count": 3,
        "tool_results": {
            "search_policy": [{"source": "refund_policy.md", "content": "policy"}],
            "prepare_refund": {"status": "prepared"},
        },
    }

    assert actual_from_state(final_state) == {
        "tools": ["prepare_refund", "search_policy"],
        "customer_id": "1842",
        "order_id": "O-991",
        "sources": ["refund_policy.md"],
        "refund_status": "prepared",
        "final_answer": "done",
    }


def test_tool_allowlist_is_explicit_and_small() -> None:
    assert set(TOOLS) == {
        "get_customer",
        "get_order",
        "search_transactions",
        "search_policy",
        "prepare_refund",
    }


def test_get_customer_known_customer() -> None:
    assert get_customer.invoke({"customer_id": "1842"}) == {
        "name": "Acme Logistics",
        "tier": "enterprise",
        "status": "active",
    }


def test_get_customer_unknown_customer() -> None:
    assert get_customer.invoke({"customer_id": "missing"}) == {
        "error": "customer not found"
    }


def test_get_order_known_order() -> None:
    assert get_order.invoke({"order_id": "O-991"}) == {
        "customer_id": "1842",
        "item": "Warehouse scanner subscription",
        "amount": "$499.00",
        "amount_cents": 49900,
        "status": "paid",
    }


def test_get_order_unknown_order() -> None:
    assert get_order.invoke({"order_id": "missing"}) == {"error": "order not found"}


def test_invalid_input_does_not_match_records() -> None:
    assert get_customer.invoke({"customer_id": " 1842 "}) == {
        "error": "customer not found"
    }
    assert get_order.invoke({"order_id": "991"}) == {"error": "order not found"}
    assert search_transactions.invoke({"customer_id": "missing"}) == []


def test_search_transactions_exact_customer_id_only() -> None:
    transactions = search_transactions.invoke({"customer_id": "1842"})

    assert len(transactions) == 5
    assert all(transaction["customer_id"] == "1842" for transaction in transactions)
    assert search_transactions.invoke({"customer_id": "18"}) == []


def test_prepare_refund_eligible_low_value_duplicate() -> None:
    assert prepare_refund.invoke({"order_id": "O-991"}) == {
        "status": "prepared",
        "reason": "duplicate charge is under the automatic refund limit",
        "order_id": "O-991",
        "amount": "$499.00",
    }


def test_prepare_refund_high_value_requires_approval() -> None:
    assert prepare_refund.invoke({"order_id": "O-2000"}) == {
        "status": "approval_required",
        "reason": "duplicate charge is above the automatic refund limit",
        "order_id": "O-2000",
        "amount": "$2,500.00",
    }


def test_prepare_refund_non_duplicate_is_blocked() -> None:
    assert prepare_refund.invoke({"order_id": "O-123"}) == {
        "status": "blocked",
        "reason": "no duplicate charge found",
        "order_id": "O-123",
    }


def test_prepare_refund_missing_order_is_blocked() -> None:
    assert prepare_refund.invoke({"order_id": "missing"}) == {
        "status": "blocked",
        "reason": "order not found",
        "order_id": "missing",
    }


def test_prepare_refund_at_limit_is_allowed() -> None:
    order_id = "O-LIMIT"
    ORDERS[order_id] = {
        "customer_id": "1842",
        "item": "Limit test",
        "amount": "$1,000.00",
        "amount_cents": AUTO_REFUND_LIMIT_CENTS,
        "status": "paid",
    }
    TRANSACTIONS.extend(
        [
            {
                "customer_id": "1842",
                "order_id": order_id,
                "kind": "charge",
                "amount": "$1,000.00",
            },
            {
                "customer_id": "1842",
                "order_id": order_id,
                "kind": "charge",
                "amount": "$1,000.00",
            },
        ]
    )

    try:
        assert prepare_refund.invoke({"order_id": order_id})["status"] == "prepared"
    finally:
        del ORDERS[order_id]
        TRANSACTIONS[:] = [
            transaction
            for transaction in TRANSACTIONS
            if transaction["order_id"] != order_id
        ]


def test_prepare_refund_requires_two_charges_not_just_two_transactions() -> None:
    order_id = "O-MIXED"
    ORDERS[order_id] = {
        "customer_id": "1842",
        "item": "Mixed transaction test",
        "amount": "$25.00",
        "amount_cents": 2500,
        "status": "paid",
    }
    TRANSACTIONS.extend(
        [
            {
                "customer_id": "1842",
                "order_id": order_id,
                "kind": "charge",
                "amount": "$25.00",
            },
            {
                "customer_id": "1842",
                "order_id": order_id,
                "kind": "refund",
                "amount": "$25.00",
            },
        ]
    )

    try:
        assert prepare_refund.invoke({"order_id": order_id}) == {
            "status": "blocked",
            "reason": "no duplicate charge found",
            "order_id": order_id,
        }
    finally:
        del ORDERS[order_id]
        TRANSACTIONS[:] = [
            transaction
            for transaction in TRANSACTIONS
            if transaction["order_id"] != order_id
        ]


def test_load_policy_documents_reads_policy_files() -> None:
    documents = load_policy_documents()
    sources = {document.metadata["source"] for document in documents}

    assert "refund_policy.md" in sources
    assert any("duplicate charge" in document.page_content for document in documents)


def test_search_policy_is_allowlisted_for_the_agent() -> None:
    assert "search_policy" in TOOLS


def test_keyword_policy_matches_exact_policy_id() -> None:
    documents = load_policy_documents()
    matches = keyword_policy_matches("POLICY-REF-2026-17", documents)

    assert [match.metadata["source"] for match in matches] == ["refund_policy.md"]


def test_keyword_policy_matches_ignores_generic_stop_words() -> None:
    documents = load_policy_documents()

    assert keyword_policy_matches("check account policy", documents) == []


def test_keyword_policy_matches_ranks_more_specific_policy_first() -> None:
    documents = load_policy_documents()
    matches = keyword_policy_matches("duplicate refund approval", documents)

    assert matches[0].metadata["source"] == "refund_policy.md"
