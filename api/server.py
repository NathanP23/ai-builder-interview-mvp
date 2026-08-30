from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.graph import run_agent
from agent.reporting import print_langsmith_status


load_dotenv()

app = FastAPI(title="Enterprise Support Agent MVP")
app.mount("/web", StaticFiles(directory="web"), name="web")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    customer_id: str
    order_id: str
    tools_used: list[str]
    sources: list[str]
    refund_status: str
    approval_required: bool
    step_count: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse("web/index.html")


def response_from_state(final_state: dict[str, Any]) -> ChatResponse:
    tool_results = final_state["tool_results"]
    policy_results = tool_results.get("search_policy", [])
    refund_result = tool_results.get("prepare_refund", {})

    return ChatResponse(
        answer=final_state["messages"][-1].content,
        customer_id=final_state["customer_id"],
        order_id=final_state["order_id"],
        tools_used=sorted(tool_results),
        sources=sorted(result["source"] for result in policy_results),
        refund_status=refund_result.get("status", ""),
        approval_required=final_state["approval_required"],
        step_count=final_state["step_count"],
    )


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    print("\n============================================================")
    print("FASTAPI /chat")
    print("FastAPI received JSON from an HTTP client, now likely the React UI.")
    print("Pydantic already validated that message is a non-empty string.")
    print("Now Python hands the message to the same LangGraph agent used by app.py.")
    print_langsmith_status()

    final_state = run_agent(request.message)

    print("\nFASTAPI RESPONSE")
    print("The agent returned final_state.")
    print("FastAPI now turns the useful parts into a JSON response.")
    return response_from_state(final_state)
