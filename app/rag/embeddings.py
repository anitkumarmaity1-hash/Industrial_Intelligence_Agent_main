"""
Embedding generation via Google Vertex AI.

SDK choice, because it is not obvious and it bit this project already:
the older `vertexai.language_models.TextEmbeddingModel` path (from
`google-cloud-aiplatform`) is deprecated as of 2025-06-24 with a stated
removal date of 2026-06-24 — the SDK emits that warning at import time.
This module therefore uses the **Google Gen AI SDK** (`google-genai`)
with `vertexai=True`, which is the supported path and the same client
Phase 6 will use for Gemini. One SDK, one auth path, no deprecated
surface in a portfolio project.

Model choice: `gemini-embedding-001`, which Google positions as the
unified replacement for the earlier specialised models (text-embedding-005,
text-multilingual-embedding-002). Those are legacy — text-embedding-004 is
already end-of-life — so a new project should not start on them. The
trade-off is that gemini-embedding-001 accepts exactly **one input text per
request**, which is why batching is clamped rather than assumed.

Two details that matter for retrieval quality:

  * Vertex embedding models are *asymmetric*: passages must be embedded
    with task type RETRIEVAL_DOCUMENT and queries with RETRIEVAL_QUERY.
    Mixing them degrades ranking silently — nothing errors, results just
    get worse.
  * `output_dimensionality` is set explicitly so vectors always match the
    dimension the Pinecone index was created with. A mismatch then fails
    loudly at upsert instead of quietly creating a second index.

Authentication is Application Default Credentials (`gcloud auth
application-default login`, or GOOGLE_APPLICATION_CREDENTIALS). There is
deliberately no API-key parameter anywhere in this module.

The SDK is imported lazily inside the constructor so the rest of the RAG
package — loading, chunking, local search, the whole test suite — runs in
environments with no Google credentials and no cloud SDK installed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from app.core.genai_client import build_genai_client

logger = logging.getLogger(__name__)

# Vertex accepts up to 250 instances per request for most embedding models;
# 32 keeps individual requests small enough to retry cheaply on a transient
# failure.
DEFAULT_BATCH_SIZE = 32

# Models that accept exactly one input text per request. Sending a batch to
# one of these fails the call outright, so the batch size is forced to 1
# rather than left as a trap for whoever edits the model name in .env.
SINGLE_INSTANCE_MODEL_PREFIXES = ("gemini-embedding",)

# Vertex default for gemini-embedding-001 is 3072. 768 is requested instead:
# it is a valid Matryoshka truncation, it is what the Pinecone index is
# created with, and it cuts vector storage by a factor of four for a corpus
# this small with no measurable retrieval loss at 62 chunks.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSION = 768


def max_batch_size_for(model_name: str, requested: int) -> int:
    """Clamp the batch size to what the model actually accepts."""
    if model_name.startswith(SINGLE_INSTANCE_MODEL_PREFIXES):
        return 1
    return max(1, requested)


class Embedder(Protocol):
    """Minimal contract the retrieval layer needs from an embedding provider."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VertexEmbedder:
    """Embedding client backed by Vertex AI, via the Google Gen AI SDK."""

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Create the Vertex client.

        Args:
            project: Google Cloud project ID (from config; never hardcoded).
            location: Vertex region, e.g. "us-central1", or "global".
            model_name: Embedding model ID. Defaults to gemini-embedding-001,
                which supersedes the now-legacy text-embedding-005.
            dimension: Output dimensionality; must match the vector index.
            batch_size: Texts per API request. Silently clamped to 1 for
                models that accept a single instance per request.

        Raises:
            ImportError: if `google-genai` is not installed.
            ValueError: if no project is configured.
        """
        if not project:
            raise ValueError("A Google Cloud project is required for Vertex embeddings.")

        try:
            from google import genai  # noqa: F401 - import check only; the client is built in app.core.genai_client
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "google-genai is required for Vertex embeddings "
                "(pip install -r requirements-cloud.txt). "
                "Alternatively set RAG_BACKEND=local to use the offline retriever."
            ) from exc

        self._client = build_genai_client(project, location)
        self._dimension = dimension
        self._batch_size = max_batch_size_for(model_name, batch_size)
        self.model_name = model_name
        if self._batch_size != batch_size:
            logger.info(
                "%s accepts one input per request; batch size clamped %d -> %d.",
                model_name,
                batch_size,
                self._batch_size,
            )
        logger.info(
            "Vertex embedder ready: model=%s dim=%d region=%s batch=%d",
            model_name,
            dimension,
            location,
            self._batch_size,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed texts in batches with the given Vertex task type.

        Raises:
            RuntimeError: if the API returns a different number of vectors
                than texts sent. Positional alignment is what maps a vector
                back to its chunk, so a length mismatch must not pass
                silently — it would mis-attribute every citation after it.
        """
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimension,
        )

        vectors: list[list[float]] = []
        started = time.perf_counter()
        for offset in range(0, len(texts), self._batch_size):
            batch = texts[offset : offset + self._batch_size]
            response: Any = self._client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=config,
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Vertex returned {len(embeddings)} embeddings for {len(batch)} texts "
                    f"(model={self.model_name}, task={task_type})."
                )
            vectors.extend(list(embedding.values) for embedding in embeddings)

        logger.info(
            "Embedded %d text(s) task=%s in %.2fs",
            len(texts),
            task_type,
            time.perf_counter() - started,
        )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing (task type RETRIEVAL_DOCUMENT)."""
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query (task type RETRIEVAL_QUERY)."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]
