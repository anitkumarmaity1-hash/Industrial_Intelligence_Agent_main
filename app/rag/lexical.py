"""
BM25 lexical index over document chunks.

Why this exists: the dense path (Vertex embeddings + Pinecone) needs
network access, a GCP project and two API keys. That makes it untestable
in CI and unrunnable for anyone cloning the repo without credentials. A
retrieval layer that cannot be tested without a paid account is a
retrieval layer whose regressions are found in production.

So the same chunks are also searchable with BM25 — about 80 lines, no
dependencies, deterministic. It is genuinely weaker than dense retrieval
on paraphrased queries ("machine runs hot" will not match "heat
dissipation" the way an embedding does), and it is not proposed as a
replacement. It is the offline/default path and the one the retrieval
tests assert against.

BM25 is Robertson/Sparck-Jones ranking: term frequency saturated by k1,
length-normalised by b, weighted by inverse document frequency.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from app.rag.chunking import Chunk

logger = logging.getLogger(__name__)

K1 = 1.5   # term-frequency saturation: further repeats of a term add less
B = 0.75   # length normalisation strength

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Deliberately short. Aggressive stopword removal hurts on technical text
# where words like "not", "below" and "above" carry real meaning
# ("power below 3500 W" vs "power above 9000 W").
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or
    that the their then there these this to was were what when which will with""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


class BM25Index:
    """In-memory BM25 index over chunks, with metadata filtering."""

    def __init__(self, chunks: list[Chunk]) -> None:
        """Build the index.

        Args:
            chunks: Chunks to index. `embedding_text` is indexed rather than
                raw text, so the doc ID, title and section path are searchable
                too — the same context the dense path encodes into its vectors.
        """
        self.chunks = chunks
        self._token_counts: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for chunk in chunks:
            tokens = tokenize(chunk.embedding_text)
            counts = Counter(tokens)
            self._token_counts.append(counts)
            self._lengths.append(len(tokens))
            document_frequency.update(counts.keys())

        self._n = len(chunks)
        self._avg_length = (sum(self._lengths) / self._n) if self._n else 0.0
        # Robertson/Sparck-Jones IDF with the +1 smoothing that keeps very
        # common terms at a small positive weight instead of a negative one.
        self._idf = {
            term: math.log(1 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }
        logger.info("Built BM25 index over %d chunks (%d unique terms)",
                    self._n, len(self._idf))

    def __len__(self) -> int:
        return self._n

    def _score(self, query_tokens: list[str], position: int) -> float:
        """BM25 score of one indexed chunk against the query tokens."""
        counts = self._token_counts[position]
        length = self._lengths[position]
        norm = K1 * \
            (1 - B + B * (length / self._avg_length if self._avg_length else 1.0))
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if frequency:
                score += self._idf.get(token, 0.0) * \
                    (frequency * (K1 + 1)) / (frequency + norm)
        return score

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
        """Apply the same filter keys the Pinecone path supports."""
        if not filters:
            return True
        doc_type = filters.get("doc_type")
        if doc_type and chunk.doc_type != doc_type:
            return False
        doc_id = filters.get("doc_id")
        if doc_id and chunk.doc_id != doc_id:
            return False
        failure_mode = filters.get("failure_mode")
        if failure_mode and failure_mode.upper() not in chunk.failure_modes:
            return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_coverage: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks by BM25 score, highest first.

        Chunks scoring zero (no query term present) are dropped rather than
        padded into the result — returning irrelevant text to an LLM that
        is instructed to ground its answer in evidence is actively harmful.

        `min_coverage` additionally drops chunks matching fewer than that
        fraction of the query's *distinct* tokens, even if they score
        above zero. This is deliberately a coverage ratio, not a raw BM25
        score floor — an absolute score cutoff was tried and rejected
        (see Settings.rag_min_term_coverage_lexical for the numbers): BM25
        score scales with query length and per-term IDF, so a legitimate
        short query like "tool wear" (2 terms, both present) can score
        *lower* than a long off-topic query that coincidentally shares one
        common word with a chunk. Coverage ratio is normalized against the
        query's own length instead, so it doesn't have that problem: a
        2-of-2-term match beats a 1-of-4-term match regardless of either
        one's raw score.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        unique_query_tokens = set(query_tokens)

        candidates: list[tuple[Chunk, float]] = []
        for position, chunk in enumerate(self.chunks):
            if not self._matches_filters(chunk, filters):
                continue
            score = self._score(query_tokens, position)
            if score <= 0:
                continue
            counts = self._token_counts[position]
            matched = sum(
                1 for t in unique_query_tokens if counts.get(t, 0) > 0)
            coverage = matched / len(unique_query_tokens)
            if coverage < min_coverage:
                continue
            candidates.append((chunk, score))

        ranked = sorted(
            candidates,
            # chunk_id breaks ties deterministically
            key=lambda pair: (-pair[1], pair[0].chunk_id),
        )
        return ranked[:top_k]
