"""
Run the retrieval evaluation set against both backends and compare them.

This is the script to run immediately after the first Pinecone ingestion.
It answers the only question that matters at that point: is the dense
backend actually better than the free baseline, or is something
misconfigured?

    python scripts/verify_dense_retrieval.py

Exit codes:
    0 — dense meets or beats the lexical baseline on every case
    1 — dense is available but ranks worse somewhere (details printed)
    2 — dense backend could not be constructed (credentials, SDK, index)

A non-zero exit is a configuration signal, not a verdict on embeddings.
The usual causes, in order of frequency: ingestion never ran against
Pinecone, the index dimension does not match EMBEDDING_DIMENSION, or the
query and document task types are mismatched.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import Settings  # noqa: E402
from app.rag.chunking import load_chunks  # noqa: E402
from app.rag.evaluation import EvaluationReport, evaluate_retriever  # noqa: E402
from app.rag.retriever import DenseRetriever, LexicalRetriever  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s: %(message)s")

TOP_K = 5


def build_dense(settings: Settings) -> DenseRetriever:
    """Construct the dense retriever, letting failures propagate with context."""
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


def print_comparison(dense: EvaluationReport, lexical: EvaluationReport) -> None:
    """Print a per-case rank comparison and the aggregate metrics."""
    print()
    print(f"{'query':<52} {'expect':<8} {'lex':>5} {'dense':>6}  verdict")
    print("-" * 88)

    for dense_result, lexical_result in zip(dense.results, lexical.results):
        case = dense_result.case
        lex_rank = str(lexical_result.rank) if lexical_result.rank else "-"
        dense_rank = str(dense_result.rank) if dense_result.rank else "-"

        if dense_result.passed and not lexical_result.passed:
            verdict = "dense wins"
        elif not dense_result.passed and lexical_result.passed:
            verdict = "DENSE LOSES"
        elif not dense_result.passed:
            verdict = "both fail"
        elif (dense_result.rank or 99) < (lexical_result.rank or 99):
            verdict = "dense better"
        elif (dense_result.rank or 99) > (lexical_result.rank or 99):
            verdict = "lexical better"
        else:
            verdict = "tie"

        print(f"{case.query[:52]:<52} {case.expected_doc_id:<8} {lex_rank:>5} {dense_rank:>6}  {verdict}")

    print("-" * 88)
    print(f"{'metric':<52} {'lexical':>12} {'dense':>12}")
    print(f"{'hit rate @' + str(TOP_K):<52} {lexical.hit_rate:>12.2f} {dense.hit_rate:>12.2f}")
    print(f"{'strict pass rate (per-case rank)':<52} "
          f"{lexical.strict_pass_rate:>12.2f} {dense.strict_pass_rate:>12.2f}")
    print(f"{'MRR':<52} {lexical.mrr:>12.3f} {dense.mrr:>12.3f}")
    print(f"{'mean latency (s)':<52} {lexical.mean_latency_s:>12.3f} {dense.mean_latency_s:>12.3f}")
    print()


def main() -> int:
    settings = Settings()

    if not settings.pinecone_configured:
        print(
            "Dense backend not configured.\n"
            "  PINECONE_API_KEY     : " + ("set" if settings.pinecone_api_key else "MISSING") + "\n"
            "  GOOGLE_CLOUD_PROJECT : " + (settings.gcp_project or "MISSING") + "\n"
            "Set both in .env, then re-run."
        )
        return 2

    print(f"project={settings.gcp_project} location={settings.gcp_location}")
    print(f"model={settings.embedding_model} dim={settings.embedding_dimension}")
    print(f"index={settings.pinecone_index} namespace={settings.pinecone_namespace}")

    try:
        dense_retriever = build_dense(settings)
    except Exception as exc:  # noqa: BLE001 - reported to the operator with guidance
        print(f"\nCould not construct the dense backend: {type(exc).__name__}: {exc}")
        print(
            "\nCheck, in order:\n"
            "  1. pip install -r requirements-cloud.txt\n"
            "  2. gcloud auth application-default login\n"
            "  3. Vertex AI API enabled on the project\n"
            "  4. PINECONE_API_KEY valid and the index reachable"
        )
        return 2

    lexical_retriever = LexicalRetriever(load_chunks(settings.chunks_path))

    dense_report = evaluate_retriever(dense_retriever, top_k=TOP_K)
    lexical_report = evaluate_retriever(lexical_retriever, top_k=TOP_K)

    if dense_report.hit_rate == 0.0:
        print("\nDense retrieval returned nothing for every query.")
        print("The index is almost certainly empty — run:")
        print("  python scripts/ingest_documents.py --backend pinecone --recreate")
        return 1

    print_comparison(dense_report, lexical_report)

    regressions = [
        result
        for result, baseline in zip(dense_report.results, lexical_report.results)
        if baseline.passed and not result.passed
    ]
    if regressions:
        print(f"{len(regressions)} case(s) the lexical baseline passes and dense does not:\n")
        for result in regressions:
            print(f"  query    : {result.case.query}")
            print(f"  expected : {result.case.expected_doc_id} within top {result.case.within_top}")
            print(f"  dense top: {result.top_doc_id} (expected doc at rank {result.rank})")
            if result.case.note:
                print(f"  note     : {result.case.note}")
            print()
        print("Likely causes: task type mismatch (RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT),")
        print("dimension mismatch, or a stale index from an earlier ingestion.")
        return 1

    print("Dense retrieval meets the lexical baseline on all cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
