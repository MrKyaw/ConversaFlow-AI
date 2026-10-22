"""Tests for BM25 knowledge base."""

from conversaflow.knowledge.bm25 import KnowledgeBase
from conversaflow.models import KnowledgeDocumentIn


def _docs():
    return [
        KnowledgeDocumentIn(
            id="a", title="Support hours", text="We are open Monday to Friday.", tags=["support"]
        ),
        KnowledgeDocumentIn(
            id="b", title="Refunds", text="Refunds take 3 to 5 days.", tags=["refund"]
        ),
    ]


def test_ingest_and_search():
    kb = KnowledgeBase()
    assert kb.ingest("t", _docs()) == 2
    hits = kb.search("t", "support hours", top_k=2)
    assert hits and hits[0].document_id == "a"


def test_delete():
    kb = KnowledgeBase()
    kb.ingest("t", _docs())
    assert kb.delete("t", "a") is True
    assert kb.delete("t", "a") is False
    assert kb.count_documents("t") == 1


def test_search_empty():
    kb = KnowledgeBase()
    assert kb.search("t", "anything") == []


def test_search_min_score():
    kb = KnowledgeBase()
    kb.ingest("t", _docs())
    assert kb.search("t", "xyznonexistent", min_score=100.0) == []


def test_count_chunks_and_ids():
    kb = KnowledgeBase()
    kb.ingest("t", _docs())
    assert kb.count_chunks("t") >= 2
    assert "a" in kb.document_ids("t")


def test_total_counts():
    kb = KnowledgeBase()
    kb.ingest("t", _docs())
    assert kb.count_documents() == 2
    assert kb.count_chunks() >= 2


def test_split_long_text():
    long_text = "Sentence one. " + ("word " * 200)
    doc = KnowledgeDocumentIn(id="long", title="Long", text=long_text)
    kb = KnowledgeBase(max_chars=50)
    kb.ingest("t", [doc])
    assert kb.count_chunks("t") > 1
