"""
Retrieval evaluation.

A RAG system without a fixed evaluation set is one where nobody can tell
whether a change helped. Chunk size, overlap, the embedding model, the
`embedding_text` prefix format, the choice of backend — each of these
changes ranking, and none of them announces when it makes ranking worse.

`RELEVANCE_CASES` is the ground truth: a question a maintenance engineer
would plausibly ask, and the document that should answer it. Cases were
written from the knowledge base's own contents, then checked against
actual BM25 output — the `within_top` values record measured behaviour,
not aspiration. Two cases rank at 2 and 3 rather than 1 for honest
reasons noted inline.

Metrics are the standard retrieval pair:

  * **hit rate @ k** — fraction of queries whose correct document appears
    in the top k. This is what matters for a RAG system: the agent reads
    everything it is given, so a correct passage at rank 3 is still
    usable evidence.
  * **MRR** — mean reciprocal rank, which does care about position.
    Useful for comparing two backends that have the same hit rate.

This module holds no test framework imports so it can be used from
scripts, tests and (later) a CI job alike.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RelevanceCase:
    """One evaluation query and the document that should answer it."""

    query: str
    expected_doc_id: str
    within_top: int
    note: str = ""


RELEVANCE_CASES: list[RelevanceCase] = [
    RelevanceCase(
        "what is the tool wear replacement interval",
        "SOP-101",
        3,
        note=(
            "MAN-200's maintenance-interval table legitimately outranks SOP-101 on "
            "lexical match — 'interval' is literally its column heading. A dense "
            "backend should pull SOP-101 up; that is the improvement to look for."
        ),
    ),
    RelevanceCase("overstrain limit wear times torque", "SOP-101", 1),
    RelevanceCase(
        "process temperature difference below 8.6 K and low rotational speed",
        "SOP-102",
        2,
        note=(
            "TRB-400 section 5 restates the same HDF condition in prose "
            "('narrow difference... low speed... documented heat-dissipation "
            "condition') without the exact numbers, and explicitly points the "
            "reader to SOP-102 section 4. Verified live against Pinecone + "
            "gemini-embedding-001: dense retrieval ranks TRB-400 first on this "
            "paraphrase-heavy query and SOP-102 second, which is a defensible "
            "call on a genuine near-duplicate, not a retrieval error — BM25 "
            "ranks SOP-102 first because it matches '8.6 K' verbatim, which "
            "TRB-400 does not restate."
        ),
    ),
    RelevanceCase("coolant pump inspection steps", "SOP-102", 1),
    RelevanceCase(
        "delivered power outside 3500 to 9000 W what should I check", "SOP-103", 1),
    RelevanceCase(
        "lockout tagout before opening the coolant loop", "SAF-300", 1),
    RelevanceCase("defect rate rising possible causes", "TRB-400", 1),
    RelevanceCase(
        "single isolated anomaly with no supporting trend", "TRB-400", 1),
    RelevanceCase(
        "has a coolant restriction been misdiagnosed before", "INC-500", 1),
    RelevanceCase(
        "nominal operating envelope rotational speed range",
        "MAN-200",
        2,
        note="SOP-103 shares most of this vocabulary; MAN-200 lands second on BM25.",
    ),
]


class _Searchable(Protocol):
    backend_name: str

    def search(self, query: str, top_k: int = 5,
               filters: dict | None = None) -> list[Any]: ...


@dataclass
class CaseResult:
    """Outcome of one evaluation case against one backend."""

    case: RelevanceCase
    rank: int | None           # 1-based rank of the expected document, None if absent
    top_doc_id: str | None
    latency_s: float

    @property
    def passed(self) -> bool:
        """Whether the expected document appeared within its allowed rank."""
        return self.rank is not None and self.rank <= self.case.within_top

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0


@dataclass
class EvaluationReport:
    """Aggregate metrics across all cases for one backend."""

    backend: str
    results: list[CaseResult]
    top_k: int

    @property
    def hit_rate(self) -> float:
        """Fraction of cases where the expected document appeared anywhere in top_k."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.rank is not None) / len(self.results)

    @property
    def strict_pass_rate(self) -> float:
        """Fraction of cases meeting their per-case rank requirement."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def mean_latency_s(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_s for r in self.results) / len(self.results)

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]


def evaluate_retriever(
    retriever: _Searchable,
    cases: list[RelevanceCase] | None = None,
    top_k: int = 5,
) -> EvaluationReport:
    """Run every relevance case against a retriever and collect metrics.

    Args:
        retriever: Any backend implementing `search()`.
        cases: Evaluation set; defaults to RELEVANCE_CASES.
        top_k: Passages requested per query.

    Returns:
        An EvaluationReport. Never raises on a bad result — a failing case
        is data, not an error.
    """
    cases = cases if cases is not None else RELEVANCE_CASES
    results: list[CaseResult] = []

    for case in cases:
        started = time.perf_counter()
        hits = retriever.search(case.query, top_k=top_k)
        latency = time.perf_counter() - started

        doc_ids = [hit.doc_id for hit in hits]
        rank = doc_ids.index(case.expected_doc_id) + \
            1 if case.expected_doc_id in doc_ids else None
        results.append(
            CaseResult(
                case=case,
                rank=rank,
                top_doc_id=doc_ids[0] if doc_ids else None,
                latency_s=latency,
            )
        )

    return EvaluationReport(backend=retriever.backend_name, results=results, top_k=top_k)
