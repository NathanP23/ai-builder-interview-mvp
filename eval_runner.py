import contextlib
import io
import json
from typing import Any

from dotenv import load_dotenv

from app import AgentState, print_langsmith_status, run_agent


def load_eval_cases() -> list[dict[str, Any]]:
    with open("eval_cases.json") as file:
        return json.load(file)


def actual_from_state(final_state: AgentState) -> dict[str, Any]:
    tool_results = final_state["tool_results"]
    policy_results = tool_results.get("search_policy", [])
    refund_result = tool_results.get("prepare_refund", {})

    return {
        "tools": sorted(tool_results),
        "customer_id": final_state["customer_id"],
        "order_id": final_state["order_id"],
        "sources": sorted(result["source"] for result in policy_results),
        "refund_status": refund_result.get("status", ""),
        "final_answer": final_state["messages"][-1].content,
    }


def score_case(case: dict[str, Any], actual: dict[str, Any]) -> dict[str, bool]:
    expected_refund_status = case["expected_refund_status"]
    actual_refund_status = actual["refund_status"]
    if expected_refund_status == "blocked" and not actual_refund_status:
        actual_refund_status = "blocked"

    return {
        "tools": sorted(case["expected_tools"]) == actual["tools"],
        "customer_id": case["expected_customer_id"] == actual["customer_id"],
        "order_id": case["expected_order_id"] == actual["order_id"],
        "sources": set(case["expected_sources"]) <= set(actual["sources"]),
        "refund_status": expected_refund_status == actual_refund_status,
        "forbidden_actions": not set(case["forbidden_actions"]) & set(actual["tools"]),
    }


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    # The eval runner cares about the final behavior, not every local teaching print.
    # LangSmith still receives the full trace, and the terminal gets a compact scorecard.
    with contextlib.redirect_stdout(io.StringIO()):
        final_state = run_agent(case["input"])

    actual = actual_from_state(final_state)
    scores = score_case(case, actual)
    return {"case": case, "actual": actual, "scores": scores}


def main() -> None:
    print("\n============================================================")
    print("EVAL RUNNER")
    print("This runs the real agent against the golden eval cases.")
    print("Local evals answer: did behavior match the answer key?")
    print("LangSmith traces answer: what happened inside each run?")

    load_dotenv()
    print_langsmith_status()

    results = []
    for case in load_eval_cases():
        print("\n------------------------------------------------------------")
        print(f"CASE {case['id']}")
        print("User input:")
        print(case["input"])
        print("Now Python runs the agent once and compares final_state to expectations.")

        result = run_case(case)
        results.append(result)

        actual = result["actual"]
        scores = result["scores"]
        print("Actual tools:", actual["tools"])
        print("Actual sources:", actual["sources"])
        print("Actual refund status:", actual["refund_status"] or "(none)")
        for name, passed in scores.items():
            print(f"- {name}: {'PASS' if passed else 'FAIL'}")

    total_checks = sum(len(result["scores"]) for result in results)
    passed_checks = sum(
        passed for result in results for passed in result["scores"].values()
    )
    passed_cases = sum(all(result["scores"].values()) for result in results)

    print("\n============================================================")
    print("EVAL SUMMARY")
    print(f"Cases fully passing: {passed_cases}/{len(results)}")
    print(f"Individual checks passing: {passed_checks}/{total_checks}")
    print("Open LangSmith to inspect traces for any failing case.")


if __name__ == "__main__":
    main()
