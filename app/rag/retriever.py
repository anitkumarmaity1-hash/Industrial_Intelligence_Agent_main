"""
Retrieval interface for the agent.

One contract — `DocumentRetriever.search()` — with two implementations
behind it:

  * `DenseRetriever`   — Vertex embeddings + Pinecone (production path)
  * `LexicalRetriever` — BM25 over the chunk file (offline/default path)

Both return `RetrievedChunk` objects carrying a citation, so nothing
downstream (the Phase 4 endpoint, the Phase 5 LangGraph tool, the Phase 6
prompt) knows or cares which backend answered. That is the point of the
abstraction: the backend becomes a config value rather than a code change,
and the agent's evidence format stays fixed.

Tenant isolation (production-readiness fix 2): a retriever is built for
exactly one tenant. The lexical backend reads that tenant's own chunk file
(`Settings.chunks_path_for`), the dense backend queries that tenant's own
Pinecone namespace (`Settings.pinecone_namespace_for`). There is no
cross-tenant fallback: a tenant with no ingested documents gets an EMPTY
retriever — "no documentation found", which the agent already handles as a
valid outcome — never another tenant's manuals.

`search_maintenance_documents()` at the bottom is the function the
LangGraph tool will wrap in Phase 5. The tool contract is defined here,
not in the agent, so it can be tested before an agent exists.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.config import Settings, get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id
from app.rag.chunking import Chunk, load_chunks
from app.rag.lexical import BM25Index

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieval hit: the passage, where it came from, and how it scored."""

    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    section: str
    text: str
    score: float
    backend: str
    data_class: str = "SYNTHETIC"
    source_path: str = ""
    failure_modes: list[str] = field(default_factory=list)

    def citation(self) -> str:
        """Source reference for the agent's supporting-evidence list."""
        suffix = " (synthetic document)" if self.data_class.upper(
        ) == "SYNTHETIC" else ""
        return f"{self.doc_id} § {self.section} — {self.title}{suffix}"

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float, backend: str) -> "RetrievedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            doc_type=chunk.doc_type,
            section=chunk.section,
            text=chunk.text,
            score=score,
            backend=backend,
            data_class=chunk.data_class,
            source_path=chunk.source_path,
            failure_modes=list(chunk.failure_modes),
        )


class DocumentRetriever(Protocol):
    """What every retrieval backend must provide."""

    backend_name: str

    def search(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...


class LexicalRetriever:
    """BM25 retrieval over the chunk file. No network, no credentials."""

    backend_name = "local-bm25"

    def __init__(self, chunks: list[Chunk], min_coverage: float = 0.0) -> None:
        """
        Args:
            min_coverage: Term-coverage floor below which a hit is
                dropped — see Settings.rag_min_term_coverage_lexical for
                where the default comes from and why it's a coverage
                ratio rather than a raw score cutoff. Zero (the dataclass
                default) disables the floor, which only happens if this
                class is built directly instead of through
                build_retriever().
        """
        self._index = BM25Index(chunks)
        self._min_coverage = min_coverage

    def search(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        hits = self._index.search(
            query, top_k=top_k, filters=filters, min_coverage=self._min_coverage)
        logger.info(
            "lexical retrieval: query=%r hits=%d (min_coverage=%.2f) latency=%.3fs",
            query, len(hits), self._min_coverage, time.perf_counter() - started
        )
        return [RetrievedChunk.from_chunk(chunk, score, self.backend_name) for chunk, score in hits]


class DenseRetriever:
    """Vertex embeddings + Pinecone similarity search."""

    backend_name = "pinecone-vertex"

    def __init__(self, embedder: Any, store: Any, min_score: float = 0.0) -> None:
        """
        Args:
            embedder: An `Embedder` (embed_query must match the index dimension).
            store: A connected `PineconeVectorStore`.
            min_score: Cosine-similarity floor below which a match is
                dropped — see Settings.rag_min_score_dense. Unverified
                against live traffic (no Pinecone credentials here); see
                that setting's docstring before trusting the default.
        """
        self._embedder = embedder
        self._store = store
        self._min_score = min_score

    @property
    def embedder(self) -> Any:
        """The embedding client, exposed so callers can verify its dimension."""
        return self._embedder

    @staticmethod
    def _to_pinecone_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """Translate the shared filter keys into Pinecone's filter syntax.

        Kept here rather than pushed onto callers so a caller writes the
        same `{"doc_type": "safety"}` regardless of backend.
        """
        if not filters:
            return None
        expression: dict[str, Any] = {}
        if filters.get("doc_type"):
            expression["doc_type"] = {"$eq": filters["doc_type"]}
        if filters.get("doc_id"):
            expression["doc_id"] = {"$eq": filters["doc_id"]}
        if filters.get("failure_mode"):
            expression["failure_modes"] = {
                "$in": [filters["failure_mode"].upper()]}
        return expression or None

    def search(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        vector = self._embedder.embed_query(query)
        matches = self._store.query(
            vector, top_k=top_k, metadata_filter=self._to_pinecone_filter(filters))

        results: list[RetrievedChunk] = []
        below_floor = 0
        for match in matches:
            score = float(match.get("score", 0.0))
            if score < self._min_score:
                below_floor += 1
                continue
            metadata = match.get("metadata", {}) or {}
            failure_modes = metadata.get("failure_modes", []) or []
            if isinstance(failure_modes, str):
                failure_modes = [failure_modes]
            results.append(
                RetrievedChunk(
                    chunk_id=str(match.get("id", "")),
                    doc_id=str(metadata.get("doc_id", "")),
                    title=str(metadata.get("title", "")),
                    doc_type=str(metadata.get("doc_type", "")),
                    section=str(metadata.get("section", "")),
                    text=str(metadata.get("text", "")),
                    score=score,
                    backend=self.backend_name,
                    data_class=str(metadata.get("data_class", "SYNTHETIC")),
                    source_path=str(metadata.get("source_path", "")),
                    failure_modes=[str(mode) for mode in failure_modes],
                )
            )
        logger.info(
            "dense retrieval: query=%r hits=%d (dropped %d below score floor %.2f) latency=%.3fs",
            query, len(
                results), below_floor, self._min_score, time.perf_counter() - started
        )
        return results


def build_retriever(
    settings: Settings | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
    min_coverage_override: float | None = None,
) -> DocumentRetriever:
    """Construct the retriever the configuration asks for, for one tenant.

    Falls back to the lexical backend if the dense path is requested but
    cannot be constructed (missing package, bad credentials, unreachable
    index). A degraded-but-working retriever is better here than a failed
    startup: the alternative is an agent that cannot answer any question
    because one of its evidence sources is unavailable. The fallback is
    logged at WARNING and reported through `backend_name`, so a
    silently-degraded system is still an *observably* degraded one.

    min_coverage_override: this tenant's calibrated BM25 relevance floor
    (production-readiness fix 11), or None to use
    Settings.rag_min_term_coverage_lexical — see
    app.core.tenant_settings.load_tenant_calibration. Has no effect on the
    dense backend, which is unaffected by this fix (only the lexical
    coverage floor was ever tuned to the demo corpus).
    """
    settings = settings or get_settings()
    validate_tenant_id(tenant_id)
    backend = settings.resolved_backend()

    if backend == "pinecone":
        try:
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
                namespace=settings.pinecone_namespace_for(tenant_id),
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            )
            store.ensure_index(dimension=settings.embedding_dimension)
            logger.info("Retrieval backend: pinecone-vertex (tenant=%s, namespace=%s)",
                        tenant_id, settings.pinecone_namespace_for(tenant_id))
            return DenseRetriever(embedder, store, min_score=settings.rag_min_score_dense)
        except Exception as exc:  # noqa: BLE001 - any failure degrades to lexical
            logger.warning("Dense retrieval unavailable (%s: %s); falling back to BM25.", type(
                exc).__name__, exc)

    chunks_path = settings.chunks_path_for(tenant_id)
    if tenant_id != DEFAULT_TENANT_ID and not chunks_path.exists():
        # A real tenant that hasn't ingested documents yet. Empty index,
        # never a fallback to the demo tenant's (or anyone else's) chunks.
        logger.info(
            "No chunk file for tenant=%s (%s) — serving an empty index.", tenant_id, chunks_path)
        chunks: list[Chunk] = []
    else:
        chunks = load_chunks(chunks_path)
    logger.info("Retrieval backend: local-bm25 (tenant=%s, %d chunks)", tenant_id, len(chunks))
    coverage = min_coverage_override if min_coverage_override is not None else settings.rag_min_term_coverage_lexical
    return LexicalRetriever(chunks, min_coverage=coverage)


# One retriever per tenant, built lazily. Bounded so a deployment with many
# tenants doesn't hold every BM25 index in memory forever; least recently
# used is evicted (it is simply rebuilt on next use).
MAX_CACHED_RETRIEVERS = 32
_retrievers: "OrderedDict[str, DocumentRetriever]" = OrderedDict()
# The min_coverage_override each cache entry was last built with — so a
# tenant whose calibration changes gets rebuilt instead of silently
# keeping the old floor until the process restarts (fix 11).
_retriever_overrides: dict[str, float | None] = {}


def get_retriever(
    refresh: bool = False,
    tenant_id: str = DEFAULT_TENANT_ID,
    min_coverage_override: float | None = None,
) -> DocumentRetriever:
    """Process-wide retriever for one tenant, built once and cached.

    Building is expensive on both paths (BM25 indexing; Vertex init plus a
    Pinecone describe_index round trip), and neither has per-request state.
    `refresh=True` rebuilds just this tenant's retriever; so does passing a
    `min_coverage_override` that differs from what's cached.
    """
    validate_tenant_id(tenant_id)
    stale_override = _retriever_overrides.get(tenant_id) != min_coverage_override
    if refresh or tenant_id not in _retrievers or stale_override:
        _retrievers[tenant_id] = build_retriever(
            tenant_id=tenant_id, min_coverage_override=min_coverage_override)
        _retriever_overrides[tenant_id] = min_coverage_override
    _retrievers.move_to_end(tenant_id)
    while len(_retrievers) > MAX_CACHED_RETRIEVERS:
        evicted, _ = _retrievers.popitem(last=False)
        _retriever_overrides.pop(evicted, None)
    return _retrievers[tenant_id]


def search_maintenance_documents(
    query: str,
    top_k: int | None = None,
    doc_type: str | None = None,
    failure_mode: str | None = None,
    retriever: DocumentRetriever | None = None,
) -> list[RetrievedChunk]:
    """Search the maintenance knowledge base.

    This is the tool contract the LangGraph agent will call in Phase 5.

    Args:
        query: Natural-language description of what the agent needs to know.
        top_k: Number of passages to return (defaults to configured RAG_TOP_K).
        doc_type: Restrict to one category — sop, manual, safety,
            troubleshooting, incident_report.
        failure_mode: Restrict to documents covering a mode — TWF, HDF,
            PWF, OSF, RNF.
        retriever: Injectable backend; defaults to the process-wide one.

    Returns:
        Ranked passages, each with a citation. Empty when nothing relevant
        is found — an empty result is a valid, reportable outcome, and the
        agent must say it found no documentation rather than fill the gap
        from the model's own knowledge.
    """
    retriever = retriever or get_retriever()
    top_k = top_k or get_settings().retrieval_top_k
    filters = {"doc_type": doc_type, "failure_mode": failure_mode}
    filters = {key: value for key, value in filters.items() if value}
    results = retriever.search(query, top_k=top_k, filters=filters or None)
    if not results:
        logger.info(
            "No maintenance documentation matched query=%r filters=%s", query, filters)
    else:
        logger.info("Retrieval succeeded: %d chunk(s) for query=%r filters=%s",
                    len(results), query, filters)
    return results
