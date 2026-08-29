from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def main() -> None:
    # Loads OPENAI_API_KEY from local .env into the process environment.
    load_dotenv()

    # LangChain wraps the OpenAI chat API; the model still runs remotely.
    model = ChatOpenAI(model="gpt-4.1-mini")
    response = model.invoke("Reply in one short sentence: what is this MVP?")

    print(response.content)


if __name__ == "__main__":
    main()
