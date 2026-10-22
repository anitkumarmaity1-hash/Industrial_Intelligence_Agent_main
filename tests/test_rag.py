"""
Phase 3 tests — document loading, chunking and retrieval.

Retrieval is tested *before* any agent exists, per the phase plan. All of
these run offline against the BM25 backend: the dense path's own logic
(filter translation, result mapping) is tested with fakes, because
asserting on Vertex/Pinecone network behaviour would test their uptime,
not our code.

The relevance assertions are the important ones. They encode "a
maintenance engineer asking X should be shown document Y", which is the
thing that silently breaks when chunk size, header handling or the
embedding-text format changes.
"""

from __future__ import annotations

import pytest

from app.core.config import PROJECT_ROOT, Settings
from app.rag.chunking import Chunk, chunk_documents, load_chunks, save_chunks
from app.rag.evaluation import RELEVANCE_CASES, evaluate_retriever
from app.rag.documents import (
    ALLOWED_DOC_TYPES,
    DocumentValidationError,
    load_document,
    load_documents,
    parse_frontmatter,
)
from app.rag.retriever import (
    DenseRetriever,
    LexicalRetriever,
    RetrievedChunk,
    build_retriever,
    search_maintenance_documents,
)

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
EXPECTED_DOC_IDS = {"SOP-101", "SOP-102", "SOP-103",
                    "MAN-200", "SAF-300", "TRB-400", "INC-500"}


@pytest.fixture(scope="module")
def documents():
    return load_documents(DOCUMENTS_DIR)


@pytest.fixture(scope="module")
def chunks(documents):
    return chunk_documents(documents, chunk_size=900, chunk_overlap=150)


@pytest.fixture(scope="module")
def retriever(chunks):
    return LexicalRetriever(chunks, min_coverage=Settings().rag_min_term_coverage_lexical)


@pytest.fixture(scope="module")
def unfiltered_retriever(chunks):
    """Same index, no coverage floor — used to prove the floor is
    actually doing something (F3), not just to test the calibrated
    default."""
    return LexicalRetriever(chunks)


# ---------------------------------------------------------------------------
# Document loading and validation
# ---------------------------------------------------------------------------


def test_all_expected_documents_load(documents):
    assert {doc.doc_id for doc in documents} == EXPECTED_DOC_IDS


def test_every_document_is_labelled_synthetic(documents):
    """Data-honesty rule: no knowledge-base document may pass as real documentation."""
    assert all(doc.is_synthetic for doc in documents)


def test_doc_types_are_within_the_allowed_set(documents):
    assert {doc.doc_type for doc in documents} <= ALLOWED_DOC_TYPES


def test_failure_modes_are_uppercased_and_known(documents):
    known = {"TWF", "HDF", "PWF", "OSF", "RNF"}
    for doc in documents:
        assert set(doc.failure_modes) <= known, doc.doc_id


def test_frontmatter_parses_scalars_and_lists():
    metadata, body = parse_frontmatter(
        "---\ndoc_id: X-1\nfailure_modes: TWF, HDF\n---\n\n# Heading\n\nBody text.\n"
    )
    assert metadata["doc_id"] == "X-1"
    assert metadata["failure_modes"] == ["TWF", "HDF"]
    assert body.startswith("# Heading")


def test_missing_frontmatter_is_rejected():
    with pytest.raises(DocumentValidationError):
        parse_frontmatter("# No frontmatter here\n")


def test_unterminated_frontmatter_is_rejected():
    with pytest.raises(DocumentValidationError):
        parse_frontmatter("---\ndoc_id: X-1\n")


def test_missing_required_field_is_rejected(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text(
        "---\ndoc_id: X-1\ntitle: T\n---\n\nBody\n", encoding="utf-8")
    with pytest.raises(DocumentValidationError, match="missing frontmatter field"):
        load_document(path)


def test_unknown_doc_type_is_rejected(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text(
        "---\ndoc_id: X-1\ntitle: T\ndoc_type: memo\nversion: 1\n"
        "effective_date: 2026-01-01\ndata_class: SYNTHETIC\n---\n\nBody\n",
        encoding="utf-8",
    )
    with pytest.raises(DocumentValidationError, match="doc_type"):
        load_document(path)


def test_duplicate_doc_ids_are_rejected(tmp_path):
    frontmatter = (
        "---\ndoc_id: DUP-1\ntitle: T\ndoc_type: sop\nversion: 1\n"
        "effective_date: 2026-01-01\ndata_class: SYNTHETIC\n---\n\nBody\n"
    )
    (tmp_path / "a.md").write_text(frontmatter, encoding="utf-8")
    (tmp_path / "b.md").write_text(frontmatter, encoding="utf-8")
    with pytest.raises(DocumentValidationError, match="duplicate doc_id"):
        load_documents(tmp_path)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_every_document_produces_chunks(documents, chunks):
    assert {chunk.doc_id for chunk in chunks} == {
        doc.doc_id for doc in documents}


def test_chunk_ids_are_unique(chunks):
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_chunking_is_deterministic(documents):
    first = chunk_documents(documents, chunk_size=900, chunk_overlap=150)
    second = chunk_documents(documents, chunk_size=900, chunk_overlap=150)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_chunks_respect_size_limit_and_are_not_empty(chunks):
    for chunk in chunks:
        assert chunk.text.strip(), chunk.chunk_id
        assert len(chunk.text) <= 900, chunk.chunk_id


def test_every_chunk_carries_a_section_path(chunks):
    """A chunk without a section cannot produce a usable citation."""
    assert all(chunk.section.strip() for chunk in chunks)


def test_embedding_text_includes_provenance(chunks):
    chunk = chunks[0]
    assert chunk.doc_id in chunk.embedding_text
    assert chunk.section in chunk.embedding_text


def test_citation_marks_synthetic_documents(chunks):
    citation = chunks[0].citation()
    assert chunks[0].doc_id in citation
    assert "synthetic" in citation.lower()


def test_threshold_text_survives_chunking(chunks):
    """The numeric thresholds are the whole point of the knowledge base.

    If a splitter change ever severs a threshold from its rule, retrieval
    still 'works' and the evidence quietly becomes useless — so assert the
    key numbers are still present somewhere in the corpus.
    """
    corpus = "\n".join(chunk.text for chunk in chunks)
    for token in ("8.6 K", "1380 rpm", "3500 W", "9000 W", "200–240", "11,000", "150 minutes"):
        assert token in corpus, token


def test_chunks_round_trip_through_jsonl(chunks, tmp_path):
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)
    reloaded = load_chunks(path)
    assert reloaded == chunks


def test_load_chunks_missing_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest_documents"):
        load_chunks(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Retrieval relevance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", RELEVANCE_CASES, ids=lambda c: c.expected_doc_id + ": " + c.query[:40])
def test_retrieval_surfaces_the_right_document(retriever, case):
    """The evaluation set, asserted case by case against the lexical backend."""
    hits = retriever.search(case.query, top_k=5)
    assert hits, f"no results for {case.query!r}"
    assert case.expected_doc_id in [hit.doc_id for hit in hits[: case.within_top]], [
        (h.doc_id, h.section, round(h.score, 2)) for h in hits
    ]


class _EmptyRetriever:
    backend_name = "empty"

    def search(self, query, top_k=5, filters=None):
        return []


def test_evaluation_detects_a_failing_backend():
    """The harness must report failure, not average it away."""
    report = evaluate_retriever(_EmptyRetriever(), top_k=5)
    assert report.hit_rate == 0.0
    assert report.strict_pass_rate == 0.0
    assert report.mrr == 0.0
    assert len(report.failures) == len(RELEVANCE_CASES)
    assert all(result.rank is None for result in report.results)


def test_evaluation_report_metrics_are_coherent(retriever):
    """The harness itself must be right before it is used to judge a backend."""
    report = evaluate_retriever(retriever, top_k=5)
    assert report.backend == "local-bm25"
    assert len(report.results) == len(RELEVANCE_CASES)
    assert report.strict_pass_rate == 1.0, [
        (r.case.query, r.rank, r.top_doc_id) for r in report.failures
    ]
    assert 0.0 <= report.mrr <= 1.0
    assert report.hit_rate >= report.strict_pass_rate


def test_results_are_ranked_descending(retriever):
    hits = retriever.search("coolant filter swarf restriction", top_k=5)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(retriever):
    assert len(retriever.search("tool wear", top_k=2)) == 2


def test_query_with_no_shared_vocabulary_returns_nothing(retriever):
    """Zero-scoring chunks are dropped, not padded.

    Returning arbitrary passages to an LLM told to ground its answer in
    evidence is how a grounded system starts hallucinating with citations.
    """
    assert retriever.search(
        "cryptocurrency blockchain airdrop wallet", top_k=5) == []


def test_incidental_word_overlap_is_dropped_by_the_coverage_floor(retriever, unfiltered_retriever):
    """Audit F3 regression test.

    "quarterly revenue forecast" is off-topic, but "quarterly" appears in
    MAN-200's maintenance-interval table, so BM25 finds a coincidental
    match on a single shared term (1 of 4 query terms = 0.25 coverage).
    Without a floor (unfiltered_retriever) that coincidental hit used to
    come back and get treated as retrieved evidence. With the calibrated
    coverage floor (retriever, see Settings.rag_min_term_coverage_lexical)
    it's dropped.
    """
    off_topic_unfiltered = unfiltered_retriever.search(
        "quarterly revenue forecast marketing budget", top_k=1)
    assert off_topic_unfiltered, "expected the coincidental match to still exist pre-floor"

    off_topic_floored = retriever.search(
        "quarterly revenue forecast marketing budget", top_k=1)
    assert off_topic_floored == []

    on_topic = retriever.search("heat dissipation failure condition", top_k=1)
    assert on_topic, "the floor must not also drop genuinely relevant matches"

    # A *short*, fully on-topic query must survive even though its raw
    # BM25 score is lower than the coincidental "quarterly" hit's score —
    # this is exactly why the floor is a coverage ratio and not a raw
    # score cutoff (see Settings.rag_min_term_coverage_lexical).
    short_on_topic = retriever.search("tool wear", top_k=1)
    assert short_on_topic
    assert short_on_topic[0].score < off_topic_unfiltered[0].score


@pytest.mark.parametrize("query", [
    "banana bread recipe oven temperature",
    "best pizza toppings in Naples",
    "weather forecast tomorrow",
    "javascript async await tutorial",
    "quarterly tax filing deadline for freelancers",
    "how do I bake sourdough bread",
    "quarterly earnings report Q3 revenue",
    "stock market crash 2008 causes",
])
def test_off_topic_queries_return_nothing_under_the_coverage_floor(retriever, query):
    """Negative eval cases for the coverage floor (audit F3 asked for
    these alongside the positive RELEVANCE_CASES). None of these share
    any real topic with the maintenance knowledge base; some still
    produce a nonzero BM25 score on incidental single-word overlap, but
    all must land below the coverage floor."""
    assert retriever.search(query, top_k=5) == []


def test_coverage_floor_has_a_known_residual_false_positive(retriever):
    """Documented limitation, not silently accepted: "M-17 quarterly
    earnings" still returns one hit. Unlike the other negative cases,
    this one isn't pure coincidence — INC-500 genuinely uses "quarterly"
    in an unrelated context ("twice the quarterly-check limit"), which
    also happens to share the letter "m" with the query's "M-17". Two
    real (if unrelated) term matches out of four query terms clears the
    0.3 coverage floor. A small lexical corpus with real recurring
    vocabulary can't be fully disambiguated by term overlap alone; that
    is what the dense (embedding) backend is for. This test exists so a
    future change to the corpus or the floor has to touch this
    assertion deliberately, instead of the gap being silently plugged or
    silently widened.
    """
    hits = retriever.search("M-17 quarterly earnings", top_k=5)
    assert [h.doc_id for h in hits] == ["INC-500"]


def test_retrieval_surfaces_the_right_document_under_the_coverage_floor(retriever):
    """The calibrated floor must not cost any of the positive eval cases
    their required rank — this is what the floor's default was chosen
    against (see Settings.rag_min_term_coverage_lexical)."""
    report = evaluate_retriever(retriever, top_k=5)
    assert report.strict_pass_rate == 1.0, [
        (r.case.query, r.rank, r.top_doc_id) for r in report.failures
    ]


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("   ", top_k=5) == []


def test_doc_type_filter_restricts_results(retriever):
    hits = retriever.search("isolation before maintenance work", top_k=5, filters={
                            "doc_type": "safety"})
    assert hits
    assert {hit.doc_id for hit in hits} == {"SAF-300"}


def test_failure_mode_filter_restricts_results(retriever):
    hits = retriever.search("what should I inspect",
                            top_k=10, filters={"failure_mode": "HDF"})
    assert hits
    # SAF-300 declares no failure modes, so it must be filtered out.
    assert "SAF-300" not in {hit.doc_id for hit in hits}


def test_doc_id_filter_restricts_results(retriever):
    hits = retriever.search("torque", top_k=10, filters={"doc_id": "MAN-200"})
    assert hits
    assert {hit.doc_id for hit in hits} == {"MAN-200"}


def test_retrieved_chunk_reports_its_backend(retriever):
    hit = retriever.search("tool wear", top_k=1)[0]
    assert hit.backend == "local-bm25"
    assert isinstance(hit, RetrievedChunk)


# ---------------------------------------------------------------------------
# Tool contract used by the Phase 5 agent
# ---------------------------------------------------------------------------


def test_search_maintenance_documents_passes_filters_through(retriever):
    hits = search_maintenance_documents(
        "isolation procedure", top_k=3, doc_type="safety", retriever=retriever
    )
    assert hits and {hit.doc_id for hit in hits} == {"SAF-300"}


def test_search_maintenance_documents_returns_citations(retriever):
    hits = search_maintenance_documents(
        "heat dissipation failure", top_k=2, retriever=retriever)
    assert all("§" in hit.citation() for hit in hits)


# ---------------------------------------------------------------------------
# Dense backend logic (fakes — no network)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    dimension = 768

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


class _FakeStore:
    def __init__(self):
        self.last_filter = None

    def query(self, vector, top_k=5, metadata_filter=None):
        self.last_filter = metadata_filter
        return [
            {
                "id": "SOP-102--003",
                "score": 0.87,
                "metadata": {
                    "doc_id": "SOP-102",
                    "title": "Heat Dissipation and Cooling System Procedure",
                    "doc_type": "sop",
                    "section": "3. Heat dissipation failure (HDF) condition",
                    "text": "Both conditions must hold at the same time.",
                    "data_class": "SYNTHETIC",
                    "source_path": "SOP-102-heat-dissipation-cooling.md",
                },
            }
        ]


def test_dense_retriever_maps_matches_to_retrieved_chunks():
    hits = DenseRetriever(_FakeEmbedder(), _FakeStore()
                          ).search("cooling fault", top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "SOP-102"
    assert hits[0].score == pytest.approx(0.87)
    assert hits[0].backend == "pinecone-vertex"
    assert "synthetic" in hits[0].citation().lower()


def test_dense_retriever_translates_filters_to_pinecone_syntax():
    store = _FakeStore()
    DenseRetriever(_FakeEmbedder(), store).search(
        "isolation", top_k=1, filters={"doc_type": "safety", "failure_mode": "hdf"}
    )
    assert store.last_filter == {
        "doc_type": {"$eq": "safety"},
        "failure_modes": {"$in": ["HDF"]},
    }


def test_no_filters_means_no_pinecone_filter_expression():
    store = _FakeStore()
    DenseRetriever(_FakeEmbedder(), store).search("anything", top_k=1)
    assert store.last_filter is None


# ---------------------------------------------------------------------------
# Vertex embedder (fake client — no network, no credentials)
# ---------------------------------------------------------------------------


class _FakeGenAIModels:
    def __init__(self, vectors_per_call=None):
        self.calls: list[dict] = []
        self._vectors_per_call = vectors_per_call

    def embed_content(self, *, model, contents, config):
        from types import SimpleNamespace

        self.calls.append({"model": model, "n": len(
            contents), "task_type": config.task_type})
        count = self._vectors_per_call if self._vectors_per_call is not None else len(
            contents)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 8) for _ in range(count)])


def _embedder_with_fake_client(batch_size=3, vectors_per_call=None):
    """Build a VertexEmbedder without running __init__ (which needs the SDK client)."""
    from app.rag.embeddings import VertexEmbedder

    embedder = VertexEmbedder.__new__(VertexEmbedder)
    models = _FakeGenAIModels(vectors_per_call)
    embedder._client = type("FakeClient", (), {"models": models})()
    embedder._dimension = 8
    embedder._batch_size = batch_size
    embedder.model_name = "test-embedding-model"
    return embedder, models


def test_embedder_batches_documents_and_uses_document_task_type():
    pytest.importorskip("google.genai")
    embedder, models = _embedder_with_fake_client(batch_size=3)
    vectors = embedder.embed_documents([f"chunk {i}" for i in range(7)])
    assert len(vectors) == 7
    assert [call["n"] for call in models.calls] == [3, 3, 1]
    assert {call["task_type"]
            for call in models.calls} == {"RETRIEVAL_DOCUMENT"}


def test_embedder_uses_query_task_type_for_queries():
    """Asymmetric task types are a silent-quality-loss bug if they regress."""
    pytest.importorskip("google.genai")
    embedder, models = _embedder_with_fake_client()
    vector = embedder.embed_query("why is M-17 underperforming")
    assert len(vector) == 8
    assert models.calls[0]["task_type"] == "RETRIEVAL_QUERY"


def test_embedder_rejects_response_length_mismatch():
    """Misaligned vectors would mis-attribute every citation after the gap."""
    pytest.importorskip("google.genai")
    embedder, _ = _embedder_with_fake_client(batch_size=3, vectors_per_call=2)
    with pytest.raises(RuntimeError, match="embeddings for"):
        embedder.embed_documents(["a", "b", "c"])


def test_embedder_handles_empty_input_without_calling_the_api():
    pytest.importorskip("google.genai")
    embedder, models = _embedder_with_fake_client()
    assert embedder.embed_documents([]) == []
    assert models.calls == []


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_auto_backend_resolves_to_local_without_credentials(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("RAG_BACKEND", "auto")
    assert Settings().resolved_backend() == "local"


def test_auto_backend_resolves_to_pinecone_with_credentials(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("RAG_BACKEND", "auto")
    assert Settings().resolved_backend() == "pinecone"


def test_pinecone_backend_falls_back_to_lexical_when_unavailable(monkeypatch, chunks, tmp_path):
    """A broken dense path must degrade to a working retriever, observably."""
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)

    monkeypatch.setenv("RAG_BACKEND", "pinecone")
    monkeypatch.setenv("PINECONE_API_KEY", "definitely-not-a-real-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "definitely-not-a-real-project")
    monkeypatch.setenv("CHUNKS_PATH", str(path))

    retriever = build_retriever(Settings())
    assert retriever.backend_name == "local-bm25"
    assert retriever.search("tool wear", top_k=1)
