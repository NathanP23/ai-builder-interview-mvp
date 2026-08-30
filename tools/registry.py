from tools.business_tools import (
    get_customer,
    get_order,
    prepare_refund,
    search_transactions,
)
from tools.policy_tools import search_policy


TOOLS = {
    "get_customer": get_customer,
    "get_order": get_order,
    "search_transactions": search_transactions,
    "search_policy": search_policy,
    "prepare_refund": prepare_refund,
}
