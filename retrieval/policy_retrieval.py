from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


POLICY_DIR = Path("policies")
KEYWORD_STOP_WORDS = {
    "account",
    "allowed",
    "charged",
    "check",
    "customer",
    "order",
    "policy",
    "prepare",
    "says",
    "twice",
}


def load_policy_documents() -> list[Document]:
    print("\n============================================================")
    print("LOAD POLICY DOCUMENTS")
    print("Python reads fake policy markdown files from policies/.")
    print("Each file becomes one document chunk for this tiny demo.")
    print("Later, chunking can become smarter if the files get large.")

    documents = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text()
        document = Document(page_content=text, metadata={"source": path.name})
        documents.append(document)
        print(f"- loaded {path.name} as one chunk")

    return documents


def keyword_policy_matches(query: str, documents: list[Document]) -> list[Document]:
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9-]+", query)
        if len(term) >= 4 and term.lower() not in KEYWORD_STOP_WORDS
    }

    scored_matches = []
    for document in documents:
        text = document.page_content.lower()
        score = sum(term in text for term in query_terms)
        if score:
            scored_matches.append((score, document))

    return [
        document
        for _, document in sorted(
            scored_matches,
            key=lambda scored_match: (
                -scored_match[0],
                scored_match[1].metadata["source"],
            ),
        )
    ]


def find_policy_matches(query: str, k: int = 2) -> list[Document]:
    documents = load_policy_documents()

    print("\nKEYWORD CHECK")
    print("Python first checks literal words/codes in the policy text.")
    print("This helps exact IDs like POLICY-REF-2026-17, where meaning alone is not enough.")
    keyword_matches = keyword_policy_matches(query, documents)
    if keyword_matches:
        print("Keyword matches found:")
        for match in keyword_matches:
            print(f"- {match.metadata['source']}")
    else:
        print("No keyword matches found. Semantic search still runs next.")

    print("\nEMBEDDINGS")
    print("Python sends each policy chunk to OpenAIEmbeddings.")
    print("The hosted embedding model returns vectors: lists of numbers meaning text position.")
    print("Secrets are not printed. Policy text does leave the machine for embedding.")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\nVECTOR STORE")
    print("InMemoryVectorStore stores the document vectors only in this Python process.")
    print("No database or persistence yet; when the program exits, this store disappears.")
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)

    print("\nQUERY")
    print(f"Policy search query: {query!r}")
    print("Python embeds the query, compares it to policy vectors, and returns top-k matches.")
    semantic_matches = vector_store.similarity_search(query, k=k)

    print("\nMERGE RETRIEVAL RESULTS")
    print("Keyword hits go first. Semantic hits fill the rest.")
    merged_matches = []
    seen_sources = set()
    for match in [*keyword_matches, *semantic_matches]:
        source = match.metadata["source"]
        if source not in seen_sources:
            merged_matches.append(match)
            seen_sources.add(source)

    return merged_matches[:k]


def run_policy_retrieval_demo() -> None:
    print("\n============================================================")
    print("POLICY RETRIEVAL DEMO")
    print("This is Checkpoint 8: retrieval exists, but it is not an agent tool yet.")
    print("Goal: turn policy text into vectors, then search for relevant policy chunks.")

    query = "When can a duplicate charge be refunded?"
    matches = find_policy_matches(query)

    print("\nTOP POLICY MATCHES")
    for index, match in enumerate(matches, start=1):
        print(f"{index}. source={match.metadata['source']}")
        print(match.page_content.strip())


def policy_results_for_query(query: str) -> list[dict[str, str]]:
    matches = find_policy_matches(query)
    return [
        {"source": match.metadata["source"], "content": match.page_content.strip()}
        for match in matches
    ]
