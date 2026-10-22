"""Document versioning/dedup for RAG ingestion (production-readiness fix
18) — see app/rag/manifest.py. Pure unit tests, no filesystem/DB needed
except for the two save/load round-trip cases."""

from __future__ import annotations

from pathlib import Path

from app.rag.documents import IndustrialDocument
from app.rag.manifest import (
    ManifestEntry,
    content_hash,
    diff_against_manifest,
    find_duplicate_content,
    load_manifest,
    save_manifest,
)


def _doc(doc_id: str, body: str, version: str = "1.0") -> IndustrialDocument:
    return IndustrialDocument(
        doc_id=doc_id, title=f"Title {doc_id}", doc_type="sop", version=version,
        effective_date="2025-01-01", data_class="SYNTHETIC", body=body,
        source_path=Path(f"{doc_id}.md"),
    )


def test_content_hash_is_stable_and_body_specific():
    a = _doc("SOP-1", "identical body")
    b = _doc("SOP-2", "identical body")
    c = _doc("SOP-3", "different body")
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(c)


def test_find_duplicate_content_reports_each_pair_once():
    docs = [_doc("A", "same"), _doc("B", "same"), _doc("C", "same"), _doc("D", "unique")]
    pairs = find_duplicate_content(docs)
    assert pairs == [("A", "B"), ("A", "C")]


def test_find_duplicate_content_empty_when_all_unique():
    docs = [_doc("A", "one"), _doc("B", "two")]
    assert find_duplicate_content(docs) == []


def test_diff_classifies_new_unchanged_updated_and_removed():
    previous = {
        "SOP-1": ManifestEntry(content_hash=content_hash(_doc("SOP-1", "v1 body"), ), version="1.0"),
        "SOP-2": ManifestEntry(content_hash="deadbeef", version="1.0"),
    }
    documents = [
        _doc("SOP-1", "v1 body", version="1.0"),   # unchanged
        _doc("SOP-3", "brand new", version="1.0"),  # new
        # SOP-2 is absent -> removed
    ]
    diff = diff_against_manifest(documents, previous)
    assert diff.unchanged == ["SOP-1"]
    assert diff.new == ["SOP-3"]
    assert diff.removed == ["SOP-2"]
    assert diff.updated == []
    assert diff.stale_version == []
    assert not diff.has_problems


def test_diff_flags_a_genuine_version_regression():
    previous = {"SOP-1": ManifestEntry(content_hash="old-hash", version="3.1")}
    documents = [_doc("SOP-1", "edited body", version="2.0")]
    diff = diff_against_manifest(documents, previous)
    assert diff.stale_version == ["SOP-1"]
    assert diff.has_problems


def test_diff_treats_a_forward_version_bump_as_a_normal_update():
    previous = {"SOP-1": ManifestEntry(content_hash="old-hash", version="1.0")}
    documents = [_doc("SOP-1", "edited body", version="2.0")]
    diff = diff_against_manifest(documents, previous)
    assert diff.updated == ["SOP-1"]
    assert diff.stale_version == []


def test_diff_does_not_flag_regression_when_versions_are_not_comparable():
    """Non-numeric versions ("draft" -> "final") can't be ordered, so a
    content change there is an ordinary update, never a false regression."""
    previous = {"SOP-1": ManifestEntry(content_hash="old-hash", version="draft")}
    documents = [_doc("SOP-1", "edited body", version="final")]
    diff = diff_against_manifest(documents, previous)
    assert diff.updated == ["SOP-1"]
    assert diff.stale_version == []


def test_manifest_round_trips_through_disk(tmp_path):
    documents = [_doc("SOP-1", "body one", version="1.0"),
                 _doc("SOP-2", "body two", version="2.3")]
    path = tmp_path / "chunks.manifest.json"

    save_manifest(documents, path)
    loaded = load_manifest(path)

    assert set(loaded) == {"SOP-1", "SOP-2"}
    assert loaded["SOP-1"].version == "1.0"
    assert loaded["SOP-2"].content_hash == content_hash(documents[1])


def test_load_manifest_missing_file_returns_empty_dict(tmp_path):
    assert load_manifest(tmp_path / "does_not_exist.json") == {}
