import sys

from dotenv import load_dotenv

from agent.graph import run_agent
from agent.prompts import DEFAULT_SUPPORT_REQUEST
from agent.reporting import print_final_report, print_langsmith_status


def main() -> None:
    print("\n============================================================")
    print("PROGRAM START")
    print("Python starts locally on your machine.")

    # Loads local secrets into the process environment without printing them.
    print("\nPython loads .env so OpenAI and LangSmith can authenticate.")
    print("Secrets are never printed.")
    load_dotenv()
    print_langsmith_status()

    # In VS Code: launch.json args -> sys.argv -> support_request
    # -> HumanMessage -> graph state.
    support_request = " ".join(sys.argv[1:]) or DEFAULT_SUPPORT_REQUEST
    final_state = run_agent(support_request)
    print_final_report(final_state)


if __name__ == "__main__":
    main()
