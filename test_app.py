from app import (
    TOOLS,
    keyword_policy_matches,
    load_policy_documents,
    get_customer,
    get_order,
    prepare_refund,
    search_transactions,
)


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
