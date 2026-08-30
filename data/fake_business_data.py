CUSTOMERS = {
    "1842": {
        "name": "Acme Logistics",
        "tier": "enterprise",
        "status": "active",
    }
}

ORDERS = {
    "O-991": {
        "customer_id": "1842",
        "item": "Warehouse scanner subscription",
        "amount": "$499.00",
        "amount_cents": 49900,
        "status": "paid",
    },
    "O-2000": {
        "customer_id": "1842",
        "item": "Enterprise hardware bundle",
        "amount": "$2,500.00",
        "amount_cents": 250000,
        "status": "paid",
    },
    "O-123": {
        "customer_id": "1842",
        "item": "Support seat",
        "amount": "$99.00",
        "amount_cents": 9900,
        "status": "paid",
    },
}

TRANSACTIONS = [
    {"customer_id": "1842", "order_id": "O-991", "kind": "charge", "amount": "$499.00"},
    {"customer_id": "1842", "order_id": "O-991", "kind": "charge", "amount": "$499.00"},
    {"customer_id": "1842", "order_id": "O-2000", "kind": "charge", "amount": "$2,500.00"},
    {"customer_id": "1842", "order_id": "O-2000", "kind": "charge", "amount": "$2,500.00"},
    {"customer_id": "1842", "order_id": "O-123", "kind": "charge", "amount": "$99.00"},
]

AUTO_REFUND_LIMIT_CENTS = 100000
