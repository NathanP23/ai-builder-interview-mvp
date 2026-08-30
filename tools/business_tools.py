from typing import Any

from langchain_core.tools import tool

from data.fake_business_data import (
    AUTO_REFUND_LIMIT_CENTS,
    CUSTOMERS,
    ORDERS,
    TRANSACTIONS,
)


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
