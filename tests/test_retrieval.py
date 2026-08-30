from langchain_core.documents import Document

from retrieval import policy_retrieval
from retrieval.policy_retrieval import (
    find_policy_matches,
    keyword_policy_matches,
    load_policy_documents,
    policy_results_for_query,
)


def test_load_policy_documents_attaches_source_metadata() -> None:
    documents = load_policy_documents()

    assert all(document.page_content for document in documents)
    assert all(document.metadata["source"].endswith(".md") for document in documents)


def test_keyword_policy_matches_returns_empty_when_query_has_no_meaningful_terms() -> None:
    documents = [Document(page_content="Account Policy", metadata={"source": "account.md"})]

    assert keyword_policy_matches("check account policy", documents) == []


def test_find_policy_matches_merges_keyword_before_semantic(monkeypatch) -> None:
    docs = [
        Document(page_content="Policy ID: EXACT-ID", metadata={"source": "exact.md"}),
        Document(page_content="Semantic fallback", metadata={"source": "semantic.md"}),
    ]

    class FakeVectorStore:
        @classmethod
        def from_documents(cls, documents, embeddings):
            return cls()

        def similarity_search(self, query, k):
            return [docs[1], docs[0]]

    monkeypatch.setattr(policy_retrieval, "load_policy_documents", lambda: docs)
    monkeypatch.setattr(policy_retrieval, "OpenAIEmbeddings", lambda model: object())
    monkeypatch.setattr(policy_retrieval, "InMemoryVectorStore", FakeVectorStore)

    matches = find_policy_matches("EXACT-ID", k=2)

    assert [match.metadata["source"] for match in matches] == ["exact.md", "semantic.md"]


def test_policy_results_for_query_returns_source_and_content(monkeypatch) -> None:
    monkeypatch.setattr(
        policy_retrieval,
        "find_policy_matches",
        lambda query: [
            Document(page_content=" Policy text \n", metadata={"source": "refund.md"})
        ],
    )

    assert policy_results_for_query("refund") == [
        {"source": "refund.md", "content": "Policy text"}
    ]
