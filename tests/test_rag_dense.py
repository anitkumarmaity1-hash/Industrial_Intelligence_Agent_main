"""
Dense-backend tests — Vertex AI embeddings + Pinecone.

These are the only tests in the suite that touch the network, cost money
(fractions of a cent) and can fail for reasons outside this repository.
They are therefore skipped unless credentials are actually configured, so
`pytest` stays green offline and in CI.

Run them with:

    pip install -r requirements-cloud.txt
    gcloud auth application-default login
    # PINECONE_API_KEY and GOOGLE_CLOUD_PROJECT set in .env
    python scripts/ingest_documents.py --backend pinecone --recreate
    pytest tests/test_rag_dense.py -v

The key assertion is not "dense retrieval works" but "dense retrieval is
at least as good as the free, dependency-less baseline". A dense backend
that loses to BM25 on this corpus is a misconfiguration — wrong task
type, wrong dimension, or a stale index — not a reason to accept worse
retrieval while paying for it.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.rag.chunking import load_chunks
from app.rag.evaluation import RELEVANCE_CASES, evaluate_retriever
from app.rag.retriever import DenseRetriever, LexicalRetriever

pytestmark = pytest.mark.dense


@pytest.fixture(scope="module")
def settings() -> Settings:
    config = Settings()
    if not config.pinecone_configured:
        pytest.skip("PINECONE_API_KEY and/or GOOGLE_CLOUD_PROJECT not set — skipping dense tests.")
    return config


@pytest.fixture(scope="module")
def dense_retriever(settings) -> DenseRetriever:
    pytest.importorskip("google.genai")
    pytest.importorskip("pinecone")

    from app.rag.embeddings import VertexEmbedder
    from app.rag.vector_store import PineconeVectorStore

    embedder = VertexEmbedder(
        project=settings.gcp_project or "",
        location=settings.gcp_location,
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    store = PineconeVectorStore(
        api_key=settings.pinecone_api_key or "",
        index_name=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    store.ensure_index(dimension=settings.embedding_dimension)
    return DenseRetriever(embedder, store)


@pytest.fixture(scope="module")
def lexical_retriever(settings) -> LexicalRetriever:
    return LexicalRetriever(load_chunks(settings.chunks_path))


def test_embedding_dimension_matches_configuration(dense_retriever, settings):
    """Catches the single most common dense-retrieval misconfiguration."""
    vector = dense_retriever.embedder.embed_query("tool wear replacement")
    assert len(vector) == settings.embedding_dimension


def test_index_is_populated(dense_retriever):
    hits = dense_retriever.search("tool wear replacement", top_k=3)
    assert hits, (
        "Pinecone returned nothing. Has ingestion run? "
        "`python scripts/ingest_documents.py --backend pinecone --recreate`"
    )


def test_results_carry_text_and_citations(dense_retriever):
    """Metadata must survive the round trip, or the agent has no evidence to cite."""
    hit = dense_retriever.search("heat dissipation failure condition", top_k=1)[0]
    assert hit.text.strip()
    assert hit.doc_id and hit.section
    assert "§" in hit.citation()


def test_scores_are_cosine_similarities(dense_retriever):
    hits = dense_retriever.search("coolant pump inspection", top_k=3)
    assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_doc_type_filter_is_applied_server_side(dense_retriever):
    hits = dense_retriever.search("isolation procedure", top_k=5, filters={"doc_type": "safety"})
    assert hits
    assert {hit.doc_id for hit in hits} == {"SAF-300"}


def test_failure_mode_filter_is_applied_server_side(dense_retriever):
    hits = dense_retriever.search("what should I inspect", top_k=10, filters={"failure_mode": "HDF"})
    assert hits
    assert "SAF-300" not in {hit.doc_id for hit in hits}


@pytest.mark.parametrize("case", RELEVANCE_CASES, ids=lambda c: c.expected_doc_id + ": " + c.query[:40])
def test_dense_retrieval_meets_the_lexical_baseline(dense_retriever, case):
    """The same ten cases the lexical backend already passes."""
    hits = dense_retriever.search(case.query, top_k=5)
    doc_ids = [hit.doc_id for hit in hits]
    assert case.expected_doc_id in doc_ids[: case.within_top], (
        f"expected {case.expected_doc_id} within top {case.within_top}, got {doc_ids}. "
        f"{case.note}"
    )


def test_dense_is_not_worse_than_lexical_overall(dense_retriever, lexical_retriever):
    """Whole-set comparison, which a per-case assertion can hide.

    Dense may lose an individual case and still be the better retriever;
    it should not lose on aggregate. If it does, suspect task-type
    configuration before concluding embeddings are unhelpful here.
    """
    dense = evaluate_retriever(dense_retriever, top_k=5)
    lexical = evaluate_retriever(lexical_retriever, top_k=5)
    assert dense.hit_rate >= lexical.hit_rate, (
        f"dense hit_rate={dense.hit_rate:.2f} < lexical {lexical.hit_rate:.2f}; "
        f"dense failures: {[(r.case.query, r.top_doc_id) for r in dense.failures]}"
    )
