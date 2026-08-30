from agent.prompts import DEFAULT_SUPPORT_REQUEST, SYSTEM_PROMPT
from data.fake_business_data import (
    AUTO_REFUND_LIMIT_CENTS,
    CUSTOMERS,
    ORDERS,
    TRANSACTIONS,
)


def test_fake_business_data_has_expected_demo_customer_order_and_duplicate() -> None:
    assert CUSTOMERS["1842"]["status"] == "active"
    assert ORDERS["O-991"]["customer_id"] == "1842"
    assert ORDERS["O-991"]["amount_cents"] == 49900

    matching_charges = [
        transaction
        for transaction in TRANSACTIONS
        if transaction["order_id"] == "O-991" and transaction["kind"] == "charge"
    ]
    assert len(matching_charges) == 2


def test_refund_limit_matches_policy_boundary() -> None:
    assert AUTO_REFUND_LIMIT_CENTS == 100000


def test_prompts_name_the_current_tool_flow() -> None:
    assert "O-991" in DEFAULT_SUPPORT_REQUEST
    assert "search_policy" in SYSTEM_PROMPT
    assert "prepare_refund" in SYSTEM_PROMPT
    assert "Never claim a refund was executed" in SYSTEM_PROMPT
