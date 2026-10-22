"""
Tests for `PineconeVectorStore`, using a fake Pinecone client — no network.

`test_delete_all_tolerates_a_namespace_that_never_existed` pins down a real
bug found while running Phase 3 against a live Pinecone project: the very
first `--recreate` ingestion against a brand-new index called
`delete(delete_all=True, ...)` on a namespace that had never been written
to. Pinecone raises 404 NotFoundError for that rather than treating it as
already empty, which aborted ingestion before any chunk was embedded or
upserted — the index stayed empty, and every downstream dense test failed
with "no results" for reasons that had nothing to do with retrieval quality.
"""

from __future__ import annotations

import pytest

from app.rag.vector_store import PineconeVectorStore


class _FakeIndex:
    def __init__(self, raise_not_found_on_delete: bool = False):
        self.raise_not_found_on_delete = raise_not_found_on_delete
        self.delete_calls: list[dict] = []

    def delete(self, *, delete_all=False, namespace=""):
        from pinecone import NotFoundError

        self.delete_calls.append({"delete_all": delete_all, "namespace": namespace})
        if self.raise_not_found_on_delete:
            raise NotFoundError(reason="Namespace not found")


def _store_with_fake_index(fake_index: _FakeIndex) -> PineconeVectorStore:
    """Build a PineconeVectorStore without running __init__ (needs a real client)."""
    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store._index_name = "industrial-intelligence"
    store._namespace = "maintenance-docs"
    store._cloud = "aws"
    store._region = "us-east-1"
    store._index = fake_index
    store._client = None
    return store


def test_delete_all_tolerates_a_namespace_that_never_existed():
    pytest.importorskip("pinecone")
    fake_index = _FakeIndex(raise_not_found_on_delete=True)
    store = _store_with_fake_index(fake_index)

    store.delete_all()  # must not raise

    assert fake_index.delete_calls == [{"delete_all": True, "namespace": "maintenance-docs"}]


def test_delete_all_passes_through_on_a_populated_namespace():
    pytest.importorskip("pinecone")
    fake_index = _FakeIndex(raise_not_found_on_delete=False)
    store = _store_with_fake_index(fake_index)

    store.delete_all()

    assert fake_index.delete_calls == [{"delete_all": True, "namespace": "maintenance-docs"}]


# ---------------------------------------------------------------------------
# query() — regression test for the real ScoredVector struct
# ---------------------------------------------------------------------------


class _FakeQueryIndex:
    """Returns a real pinecone QueryResponse/ScoredVector, not a mock.

    The bug this guards against — `dict(match)` silently corrupting or
    raising on a real `ScoredVector` — was invisible to a test built on a
    hand-rolled dict or namespace stand-in, because those don't share
    ScoredVector's `__iter__` (which yields field *names*, not pairs).
    Only a test using the actual SDK class would have caught it, so this
    one constructs a real `ScoredVector` via the installed `pinecone`
    package rather than faking its shape.
    """

    def __init__(self, scored_vectors):
        self._scored_vectors = scored_vectors

    def query(self, *, vector, top_k, namespace, include_metadata, filter):
        from pinecone.db_data.dataclasses.query_response import QueryResponse

        return QueryResponse(matches=self._scored_vectors, namespace=namespace)


def test_query_reads_real_scored_vectors_without_corruption():
    pytest.importorskip("pinecone")
    from pinecone.models.vectors.vector import ScoredVector

    scored_vectors = [
        ScoredVector(
            id="SOP-102--003",
            score=0.87,
            values=[],
            sparse_values=None,
            metadata={"doc_id": "SOP-102", "title": "Heat Dissipation", "text": "passage text"},
        ),
        ScoredVector(
            id="SOP-101--000",
            score=0.61,
            values=[],
            sparse_values=None,
            metadata={"doc_id": "SOP-101", "title": "Tool Wear", "text": "other passage"},
        ),
    ]
    store = _store_with_fake_index(_FakeQueryIndex(scored_vectors))

    results = store.query(vector=[0.1] * 8, top_k=2)

    assert results == [
        {"id": "SOP-102--003", "score": 0.87, "metadata": {"doc_id": "SOP-102", "title": "Heat Dissipation", "text": "passage text"}},
        {"id": "SOP-101--000", "score": 0.61, "metadata": {"doc_id": "SOP-101", "title": "Tool Wear", "text": "other passage"}},
    ]


def test_query_handles_missing_metadata():
    """A match with no metadata (include_metadata=False, or nothing stored) must not crash."""
    pytest.importorskip("pinecone")
    from pinecone.models.vectors.vector import ScoredVector

    scored_vectors = [ScoredVector(id="X--0", score=0.5, values=[], sparse_values=None, metadata=None)]
    store = _store_with_fake_index(_FakeQueryIndex(scored_vectors))

    results = store.query(vector=[0.1] * 8, top_k=1)

    assert results == [{"id": "X--0", "score": 0.5, "metadata": {}}]
