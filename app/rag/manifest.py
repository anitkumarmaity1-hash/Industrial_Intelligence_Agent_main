"""
Document versioning and dedup for RAG ingestion — production-readiness
fix 18.

Two things the audit found missing from app/rag/documents.py +
scripts/ingest_documents.py:

  * **Dedup**: nothing stopped two differently-named documents (or one
    document re-saved under a new doc_id) from carrying byte-identical
    content into the index, doubling retrieval weight for one real
    passage and letting it crowd out other evidence.
  * **Versioning**: re-ingesting a knowledge base over an existing one
    had no concept of "this used to be newer" — a stale copy of a
    document (reverted file, wrong branch, accidental downgrade) would
    silently replace a more current version with no warning.

This module keeps a small on-disk manifest (one JSON file per tenant,
next to that tenant's chunk file) recording each doc_id's content hash
and declared `version` from the last successful ingestion, and diffs a
freshly loaded document set against it. It holds no chunking/retrieval
logic itself — scripts/ingest_documents.py is the only caller.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.rag.documents import IndustrialDocument

logger = logging.getLogger(__name__)


def content_hash(document: IndustrialDocument) -> str:
    """Stable fingerprint of a document's body (not its metadata — a
    title/date-only edit is a real change the manifest should still
    treat as "updated", but two documents with the same body under
    different frontmatter are still duplicate content for retrieval
    purposes, which is what `find_duplicate_content` below checks)."""
    return hashlib.sha256(document.body.encode("utf-8")).hexdigest()[:16]


def find_duplicate_content(documents: list[IndustrialDocument]) -> list[tuple[str, str]]:
    """Pairs of doc_ids whose bodies are byte-identical. Each pair is
    reported once (earlier doc_id first) regardless of how many
    documents share that content."""
    by_hash: dict[str, list[str]] = {}
    for doc in documents:
        by_hash.setdefault(content_hash(doc), []).append(doc.doc_id)

    pairs: list[tuple[str, str]] = []
    for doc_ids in by_hash.values():
        if len(doc_ids) > 1:
            ordered = sorted(doc_ids)
            pairs.extend((ordered[0], other) for other in ordered[1:])
    return pairs


def _parse_version(version: str) -> tuple[int, ...] | None:
    """"2.1" -> (2, 1); "v3" -> (3,); anything with no digits -> None (not
    comparable, so a change there is reported as a plain update, never a
    regression — we only flag going backwards when we can actually tell)."""
    parts = re.findall(r"\d+", version)
    return tuple(int(p) for p in parts) if parts else None


@dataclass(frozen=True)
class ManifestEntry:
    content_hash: str
    version: str


@dataclass(frozen=True)
class ManifestDiff:
    new: list[str]
    unchanged: list[str]
    updated: list[str]            # content changed, version moved forward (or not comparable)
    stale_version: list[str]      # content changed, but version looks like it went BACKWARDS
    removed: list[str]            # was in the manifest, missing from this load

    @property
    def has_problems(self) -> bool:
        return bool(self.stale_version)


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {doc_id: ManifestEntry(**entry) for doc_id, entry in raw.items()}


def save_manifest(documents: list[IndustrialDocument], path: Path) -> None:
    manifest = {
        doc.doc_id: asdict(ManifestEntry(content_hash=content_hash(doc), version=doc.version))
        for doc in documents
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def diff_against_manifest(
    documents: list[IndustrialDocument], previous: dict[str, ManifestEntry]
) -> ManifestDiff:
    """Compare a freshly loaded document set to the last successful
    ingestion's manifest. Pure/offline — takes the already-loaded
    manifest dict, does no file I/O itself, so it's trivially unit-
    testable without a filesystem fixture."""
    new, unchanged, updated, stale = [], [], [], []
    current_ids = {doc.doc_id for doc in documents}

    for doc in documents:
        prior = previous.get(doc.doc_id)
        new_hash = content_hash(doc)
        if prior is None:
            new.append(doc.doc_id)
            continue
        if prior.content_hash == new_hash:
            unchanged.append(doc.doc_id)
            continue
        # Content changed. Only call it "stale" when both versions parse
        # as comparable version numbers AND the new one is strictly lower
        # — anything else (non-numeric versions, equal version left
        # unbumped, a genuinely higher version) is an ordinary update.
        old_v, new_v = _parse_version(prior.version), _parse_version(doc.version)
        if old_v is not None and new_v is not None and new_v < old_v:
            stale.append(doc.doc_id)
        else:
            updated.append(doc.doc_id)

    removed = sorted(set(previous) - current_ids)
    return ManifestDiff(
        new=sorted(new), unchanged=sorted(unchanged),
        updated=sorted(updated), stale_version=sorted(stale), removed=removed,
    )


def log_diff(diff: ManifestDiff, duplicates: list[tuple[str, str]]) -> None:
    if diff.new:
        logger.info("New documents: %s", diff.new)
    if diff.updated:
        logger.info("Updated documents: %s", diff.updated)
    if diff.removed:
        logger.warning(
            "Documents present in the previous ingestion but missing now "
            "(removed, renamed, or doc_id changed): %s", diff.removed)
    if diff.stale_version:
        logger.error(
            "Version regression: content changed but the declared version "
            "went backwards for %s — refusing to ingest (see --force).",
            diff.stale_version)
    for first, second in duplicates:
        logger.warning(
            "Duplicate content: %r and %r have byte-identical bodies — "
            "retrieval will weight this passage twice.", first, second)
