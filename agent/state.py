from typing import Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: list[BaseMessage]
    customer_id: str
    order_id: str
    tool_results: dict[str, Any]
    approval_required: bool
    step_count: int
