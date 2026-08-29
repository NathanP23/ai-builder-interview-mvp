from app import get_customer, get_order, search_transactions


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
