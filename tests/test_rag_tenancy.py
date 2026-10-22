"""
Per-tenant document isolation (production-readiness fix 2): a tenant's
retriever must only ever see that tenant's documents, on both backends.
No credentials or network needed — the Pinecone side is checked with a
fake store that records the namespace it was built with.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id
from app.rag import retriever as retriever_module
from app.rag.chunking import Chunk, save_chunks
from app.rag.retriever import build_retriever, get_retriever
from tests.helpers import env


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-0", doc_id=doc_id, title=f"{doc_id} manual", doc_type="manual",
        section="1. Overview", text=text, chunk_index=0, source_path=f"{doc_id}.md",
        effective_date="2026-01-01", data_class="SYNTHETIC", failure_modes=["TWF"],
    )


@pytest.fixture()
def two_tenants(tmp_path, monkeypatch):
    """Demo chunks + acme chunks on disk, RAG_BACKEND=local."""
    base = tmp_path / "document_chunks.jsonl"
    save_chunks([_chunk("DEMO-1", "spindle bearing lubrication interval is quarterly")], base)
    acme_path = tmp_path / "tenants" / "acme" / "document_chunks.jsonl"
    save_chunks([_chunk("ACME-9", "conveyor gearbox overheating shutdown procedure")], acme_path)
    retriever_module._retrievers.clear()
    with env(CHUNKS_PATH=str(base), RAG_BACKEND="local", PINECONE_API_KEY=None,
             GOOGLE_CLOUD_PROJECT=None):
        yield
    retriever_module._retrievers.clear()


def test_each_tenant_retrieves_only_its_own_documents(two_tenants):
    demo = get_retriever(tenant_id=DEFAULT_TENANT_ID)
    acme = get_retriever(tenant_id="acme")
    assert {h.doc_id for h in demo.search("bearing lubrication interval")} == {"DEMO-1"}
    assert {h.doc_id for h in acme.search("gearbox overheating shutdown")} == {"ACME-9"}
    # And neither can reach the other's text, even with the other's exact wording.
    assert acme.search("bearing lubrication interval") == []
    assert demo.search("gearbox overheating shutdown") == []


def test_tenant_without_documents_gets_an_empty_index_not_the_demo_one(two_tenants):
    empty = get_retriever(tenant_id="newco")
    assert empty.search("bearing lubrication interval") == []
    assert empty.search("gearbox overheating shutdown") == []


def test_demo_tenant_still_fails_loudly_when_its_chunk_file_is_missing(tmp_path):
    with env(CHUNKS_PATH=str(tmp_path / "missing.jsonl"), RAG_BACKEND="local",
             PINECONE_API_KEY=None, GOOGLE_CLOUD_PROJECT=None):
        with pytest.raises(FileNotFoundError):
            build_retriever(tenant_id=DEFAULT_TENANT_ID)


def test_retrievers_are_cached_per_tenant_and_bounded(two_tenants, monkeypatch):
    assert get_retriever(tenant_id="acme") is get_retriever(tenant_id="acme")
    assert get_retriever(tenant_id="acme") is not get_retriever(tenant_id=DEFAULT_TENANT_ID)
    monkeypatch.setattr(retriever_module, "MAX_CACHED_RETRIEVERS", 2)
    for tenant in ("t1", "t2", "t3"):
        get_retriever(tenant_id=tenant)
    assert len(retriever_module._retrievers) == 2


def test_namespaces_are_injective_and_keep_the_legacy_one_for_the_demo_tenant():
    s = Settings()
    assert s.pinecone_namespace_for(DEFAULT_TENANT_ID) == s.pinecone_namespace  # existing vectors stay valid
    names = {s.pinecone_namespace_for(t) for t in ("acme", "globex", "acme-2", DEFAULT_TENANT_ID)}
    assert len(names) == 4
    assert s.pinecone_namespace_for("acme") == f"acme:{s.pinecone_namespace}"


@pytest.mark.parametrize("bad", ["", "../etc", "Acme", "a/b", "a:b", "-x", "a" * 49, "a b", "a.b"])
def test_tenant_ids_that_could_escape_a_path_or_namespace_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_tenant_id(bad)
    with pytest.raises(ValueError):
        Settings().chunks_path_for(bad)
    with pytest.raises(ValueError):
        Settings().pinecone_namespace_for(bad)


def test_dense_retriever_is_built_on_the_tenants_own_namespace(monkeypatch, tmp_path):
    seen: dict[str, str] = {}

    class FakeStore:
        def __init__(self, api_key, index_name, namespace, cloud, region):
            seen["namespace"] = namespace

        def ensure_index(self, dimension):
            pass

    class FakeEmbedder:
        def __init__(self, **_kw):
            pass

    monkeypatch.setitem(sys.modules, "app.rag.vector_store",
                        types.SimpleNamespace(PineconeVectorStore=FakeStore))
    monkeypatch.setitem(sys.modules, "app.rag.embeddings",
                        types.SimpleNamespace(VertexEmbedder=FakeEmbedder))
    with env(RAG_BACKEND="pinecone", PINECONE_API_KEY="k", GOOGLE_CLOUD_PROJECT="p"):
        build_retriever(tenant_id="acme")
        assert seen["namespace"] == "acme:maintenance-docs"
        build_retriever(tenant_id=DEFAULT_TENANT_ID)
        assert seen["namespace"] == "maintenance-docs"
