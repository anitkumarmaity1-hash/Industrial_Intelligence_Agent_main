"""
Pinecone vector store for maintenance-document chunks.

Thin deliberately: create/verify the index, upsert chunk vectors with
metadata, run a filtered similarity query. No caching, no query
rewriting, no hybrid fusion — those belong above this layer if they are
ever justified.

Design notes:

  * Chunk text is stored in Pinecone metadata alongside the vector. It
    costs storage, but it means a retrieval result is self-contained
    evidence and the agent never has to re-open files to build a
    citation.
  * `chunk_id` is used as the vector ID. Because chunk IDs are
    deterministic, re-running ingestion overwrites rather than duplicates.
  * Metric is cosine, matching what Vertex embeddings are normalised for.

The Pinecone SDK is imported lazily so the package imports cleanly
without it installed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.rag.chunking import Chunk

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100


class PineconeVectorStore:
    """Wrapper around a single Pinecone serverless index and namespace."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "maintenance-docs",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        """Connect to Pinecone.

        Raises:
            ImportError: if the pinecone client is not installed.
            ValueError: if no API key is configured.
        """
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set.")

        try:
            from pinecone import Pinecone
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "pinecone is required for the Pinecone backend. "
                "Install it, or set RAG_BACKEND=local to use the offline retriever."
            ) from exc

        self._client = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._namespace = namespace
        self._cloud = cloud
        self._region = region
        self._index: Any | None = None

    def ensure_index(self, dimension: int, metric: str = "cosine") -> None:
        """Create the index if absent, and verify its dimension if present.

        Raises:
            ValueError: if an existing index has a different dimension —
                silently writing 768-d vectors into a 1536-d index is not
                possible, and silently creating a second index is worse.
        """
        from pinecone import ServerlessSpec

        if not self._client.has_index(self._index_name):
            logger.info("Creating Pinecone index %s (dim=%d, metric=%s)", self._index_name, dimension, metric)
            # create_index blocks until the serverless index is ready, so no
            # polling loop is needed (client >= 5.x).
            self._client.create_index(
                name=self._index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
        else:
            existing_dimension = getattr(self._client.describe_index(self._index_name), "dimension", None)
            if existing_dimension is not None and existing_dimension != dimension:
                raise ValueError(
                    f"Pinecone index {self._index_name!r} has dimension {existing_dimension}, "
                    f"but the configured embedding dimension is {dimension}."
                )
        self._index = self._client.Index(self._index_name)

    @property
    def index(self) -> Any:
        """The connected index handle."""
        if self._index is None:
            self._index = self._client.Index(self._index_name)
        return self._index

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, Any]:
        """Metadata payload stored with each vector.

        Only fields used for filtering or citation are stored — metadata
        is queried on every search and there is no reason to carry
        anything a result will not use.
        """
        return {
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "doc_type": chunk.doc_type,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "source_path": chunk.source_path,
            "effective_date": chunk.effective_date,
            "data_class": chunk.data_class,
            "failure_modes": chunk.failure_modes,
            "text": chunk.text,
        }

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Upsert chunk vectors in batches.

        Args:
            chunks: Chunks being indexed.
            vectors: Embeddings, positionally aligned with `chunks`.

        Returns:
            Number of vectors upserted.

        Raises:
            ValueError: if the two inputs are not the same length.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors — refusing to upsert.")

        payload = [
            {"id": chunk.chunk_id, "values": vector, "metadata": self._metadata(chunk)}
            for chunk, vector in zip(chunks, vectors)
        ]

        started = time.perf_counter()
        for offset in range(0, len(payload), UPSERT_BATCH_SIZE):
            self.index.upsert(vectors=payload[offset : offset + UPSERT_BATCH_SIZE], namespace=self._namespace)
        logger.info(
            "Upserted %d vectors to %s/%s in %.2fs",
            len(payload),
            self._index_name,
            self._namespace,
            time.perf_counter() - started,
        )
        return len(payload)

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a similarity search.

        Args:
            vector: Query embedding.
            top_k: Number of matches to return.
            metadata_filter: Pinecone filter expression, e.g. {"doc_type": {"$eq": "safety"}}.

        Returns:
            Match dicts with `id`, `score` and `metadata`, in ranked order.
        """
        started = time.perf_counter()
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=self._namespace,
            include_metadata=True,
            filter=metadata_filter,
        )
        matches = getattr(response, "matches", None)
        if matches is None:
            matches = response.get("matches", []) if isinstance(response, dict) else []
        logger.info("Pinecone query returned %d match(es) in %.3fs", len(matches), time.perf_counter() - started)
        # Each match is a `ScoredVector` (a msgspec Struct). Its `__iter__`
        # yields field *names*, not (key, value) pairs — that mixin exists so
        # `"field" in match` works, not so `dict(match)` does. dict(match)
        # instead unpacks each field-name string as if it were a length-2
        # key/value pair, which happens to produce garbage silently for
        # "id" (2 characters) and raise outright on "score" (5 characters).
        # Read the typed attributes explicitly instead.
        return [
            {"id": match.id, "score": match.score, "metadata": match.metadata or {}}
            for match in matches
        ]

    def delete_all(self) -> None:
        """Delete every vector in the namespace. Used by `--recreate` during ingestion.

        A namespace that has never been written to does not exist as far
        as Pinecone is concerned, and `delete(delete_all=True, ...)` on it
        raises a 404 rather than treating it as already empty. That is the
        expected state on the very first ingestion against a brand-new
        index, so it is treated as a no-op instead of an error — the
        caller asked for an empty namespace, and it already is one.
        """
        from pinecone import NotFoundError

        try:
            self.index.delete(delete_all=True, namespace=self._namespace)
            logger.info("Cleared namespace %s in index %s", self._namespace, self._index_name)
        except NotFoundError:
            logger.info(
                "Namespace %s does not exist yet in index %s — nothing to clear.",
                self._namespace,
                self._index_name,
            )
