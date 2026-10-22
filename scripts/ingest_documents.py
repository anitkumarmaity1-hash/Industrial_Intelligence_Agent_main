"""
Phase 3 ingestion: documents -> validated -> chunked -> JSONL (-> Pinecone).

Always writes the chunk JSONL, because that file is what the offline
lexical backend serves from and what the tests assert against. Embedding
and upserting to Pinecone is the optional second stage, run only when the
dense backend is requested and credentials exist.

Usage:
    python scripts/ingest_documents.py                  # chunk only (offline path)
    python scripts/ingest_documents.py --backend pinecone
    python scripts/ingest_documents.py --backend pinecone --recreate
    python scripts/ingest_documents.py --tenant acme    # a company's own documents

`--tenant` (default "default", the demo tenant) selects whose documents are
ingested and where they go: documents are read from that tenant's directory
(Settings.documents_dir_for), the chunk file is that tenant's own
(Settings.chunks_path_for), and vectors go to that tenant's own Pinecone
namespace (Settings.pinecone_namespace_for). Tenants never share an index.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter

# Same convention as scripts/run_pipeline.py: make the repo root importable
# so this runs as `python scripts/ingest_documents.py` from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings  # noqa: E402
from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id  # noqa: E402
from app.rag.chunking import chunk_documents, save_chunks  # noqa: E402
from app.rag.documents import DocumentValidationError, load_documents  # noqa: E402
from app.rag.manifest import (  # noqa: E402
    diff_against_manifest, find_duplicate_content, load_manifest, log_diff, save_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_documents")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest maintenance documents into the RAG index.")
    parser.add_argument(
        "--backend",
        choices=["local", "pinecone", "auto"],
        default=None,
        help="Override RAG_BACKEND for this run.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Clear the Pinecone namespace before upserting (Pinecone backend only).",
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT_ID,
        help="Tenant whose documents to ingest (default: the demo tenant).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ingest anyway even if a document's version looks like it went "
            "backwards vs. the last ingestion (production-readiness fix 18). "
            "Without this flag, a detected version regression aborts before "
            "any chunk file is written."
        ),
    )
    args = parser.parse_args()

    try:
        tenant_id = validate_tenant_id(args.tenant)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    settings = get_settings()
    backend = args.backend or settings.resolved_backend()
    if backend == "auto":
        backend = "pinecone" if settings.pinecone_configured else "local"

    started = time.perf_counter()

    try:
        documents = load_documents(settings.documents_dir_for(tenant_id))
    except (FileNotFoundError, DocumentValidationError) as exc:
        logger.error("Document loading failed: %s", exc)
        return 1

    by_type = Counter(doc.doc_type for doc in documents)
    logger.info("Documents by type: %s", dict(by_type))

    # Production-readiness fix 18: compare against the last successful
    # ingestion before writing anything. A version regression aborts
    # (unless --force); duplicate content and additions/removals are
    # logged but never block ingestion on their own.
    manifest_path = settings.chunks_path_for(tenant_id).with_suffix(".manifest.json")
    previous_manifest = load_manifest(manifest_path)
    diff = diff_against_manifest(documents, previous_manifest)
    duplicates = find_duplicate_content(documents)
    log_diff(diff, duplicates)
    if diff.has_problems and not args.force:
        logger.error("Aborting ingestion (pass --force to override).")
        return 1

    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    save_chunks(chunks, settings.chunks_path_for(tenant_id))
    save_manifest(documents, manifest_path)

    sizes = sorted(len(chunk.text) for chunk in chunks)
    logger.info(
        "Chunk stats: count=%d min=%d median=%d max=%d chars",
        len(sizes), sizes[0], sizes[len(sizes) // 2], sizes[-1],
    )

    if backend == "pinecone":
        if not settings.pinecone_configured:
            logger.error(
                "Pinecone backend requested but PINECONE_API_KEY and/or GOOGLE_CLOUD_PROJECT are unset. "
                "Chunks were written; skipping embedding/upsert."
            )
            return 1
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
            if args.recreate:
                store.delete_all()

            vectors = embedder.embed_documents([chunk.embedding_text for chunk in chunks])
            store.upsert_chunks(chunks, vectors)
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator, not swallowed
            logger.error("Pinecone ingestion failed (%s): %s", type(exc).__name__, exc)
            return 1
    else:
        logger.info("Local backend: chunks written; BM25 index is built at query time.")

    logger.info("Ingestion complete in %.2fs (tenant=%s, backend=%s)",
                time.perf_counter() - started, tenant_id, backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
