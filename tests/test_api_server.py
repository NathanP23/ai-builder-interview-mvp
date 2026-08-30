from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from api import server


client = TestClient(server.app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_agent_state_as_json(monkeypatch) -> None:
    def fake_run_agent(message: str):
        assert message == "hello"
        return {
            "messages": [AIMessage("final answer")],
            "customer_id": "1842",
            "order_id": "O-991",
            "tool_results": {
                "search_policy": [{"source": "refund_policy.md", "content": "policy"}],
                "prepare_refund": {"status": "prepared"},
            },
            "approval_required": False,
            "step_count": 5,
        }

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "final answer",
        "customer_id": "1842",
        "order_id": "O-991",
        "tools_used": ["prepare_refund", "search_policy"],
        "sources": ["refund_policy.md"],
        "refund_status": "prepared",
        "approval_required": False,
        "step_count": 5,
    }


def test_chat_endpoint_rejects_empty_message() -> None:
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422
