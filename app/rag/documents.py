"""
Loading and validation of the industrial knowledge base.

Each document in `documents/` is Markdown with a small YAML-style
frontmatter block. The frontmatter is the metadata that later becomes
chunk metadata in the vector store, which is what makes filtered
retrieval ("only safety documents", "only documents covering HDF")
possible without a second index.

Frontmatter is parsed with a ~20-line reader rather than PyYAML: the
format is a flat `key: value` block that we control, and adding a
dependency for that would fail the project's own dependency rule.
Validation is strict — a missing or unknown field raises at load time,
because a chunk with wrong metadata produces a wrong citation, and a
wrong citation is worse than no citation in a decision-support system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Fields every knowledge-base document must declare.
REQUIRED_FIELDS = {"doc_id", "title", "doc_type", "version", "effective_date", "data_class"}

# Fields that are parsed as comma-separated lists rather than scalars.
LIST_FIELDS = {"failure_modes"}

# Document categories the retriever is allowed to filter on. Constrained
# so a typo in a frontmatter block fails loudly instead of silently
# creating a category nothing will ever match.
ALLOWED_DOC_TYPES = {"sop", "manual", "safety", "troubleshooting", "incident_report"}


class DocumentValidationError(ValueError):
    """Raised when a knowledge-base document has missing or invalid metadata."""


@dataclass(frozen=True)
class IndustrialDocument:
    """One knowledge-base document: validated metadata plus Markdown body."""

    doc_id: str
    title: str
    doc_type: str
    version: str
    effective_date: str
    data_class: str
    body: str
    source_path: Path
    equipment_class: str = ""
    failure_modes: list[str] = field(default_factory=list)

    @property
    def is_synthetic(self) -> bool:
        """Whether this document is authored demo content rather than real-world documentation."""
        return self.data_class.upper() == "SYNTHETIC"


def parse_frontmatter(raw: str) -> tuple[dict[str, str | list[str]], str]:
    """Split a document into its frontmatter mapping and its Markdown body.

    Args:
        raw: Full file contents, expected to open with a `---` delimited block.

    Returns:
        A (metadata, body) tuple. Metadata values are strings, except for
        keys in LIST_FIELDS which become lists of stripped strings.

    Raises:
        DocumentValidationError: if the frontmatter block is absent or unterminated.
    """
    text = raw.lstrip("\ufeff").lstrip()
    if not text.startswith("---"):
        raise DocumentValidationError("document does not start with a '---' frontmatter block")

    parts = text.split("---", 2)
    if len(parts) < 3:
        raise DocumentValidationError("frontmatter block is not terminated by a closing '---'")

    _, block, body = parts
    metadata: dict[str, str | list[str]] = {}
    for line_no, line in enumerate(block.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise DocumentValidationError(f"frontmatter line {line_no} is not 'key: value': {stripped!r}")
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if key in LIST_FIELDS:
            metadata[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            metadata[key] = value
    return metadata, body.strip()


def load_document(path: Path) -> IndustrialDocument:
    """Load and validate a single knowledge-base document.

    Raises:
        DocumentValidationError: on missing required fields or an unknown doc_type.
    """
    metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))

    missing = REQUIRED_FIELDS - metadata.keys()
    if missing:
        raise DocumentValidationError(f"{path.name}: missing frontmatter field(s): {sorted(missing)}")

    doc_type = str(metadata["doc_type"])
    if doc_type not in ALLOWED_DOC_TYPES:
        raise DocumentValidationError(
            f"{path.name}: doc_type {doc_type!r} not in {sorted(ALLOWED_DOC_TYPES)}"
        )
    if not body:
        raise DocumentValidationError(f"{path.name}: document body is empty")

    failure_modes = metadata.get("failure_modes", [])
    if isinstance(failure_modes, str):  # defensive: LIST_FIELDS should have handled this
        failure_modes = [failure_modes]

    return IndustrialDocument(
        doc_id=str(metadata["doc_id"]),
        title=str(metadata["title"]),
        doc_type=doc_type,
        version=str(metadata["version"]),
        effective_date=str(metadata["effective_date"]),
        data_class=str(metadata["data_class"]),
        equipment_class=str(metadata.get("equipment_class", "")),
        failure_modes=[mode.upper() for mode in failure_modes],
        body=body,
        source_path=path,
    )


def load_documents(documents_dir: Path) -> list[IndustrialDocument]:
    """Load every `.md` document in a directory, sorted by filename.

    Sorted so that chunk IDs are stable across runs — an unstable ID would
    mean re-ingestion creates duplicate vectors instead of overwriting.

    Raises:
        FileNotFoundError: if the directory does not exist or holds no documents.
        DocumentValidationError: if any document fails validation, or two share a doc_id.
    """
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"documents directory not found: {documents_dir}")

    paths = sorted(documents_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md documents found in {documents_dir}")

    documents = [load_document(path) for path in paths]

    seen: dict[str, Path] = {}
    for doc in documents:
        if doc.doc_id in seen:
            raise DocumentValidationError(
                f"duplicate doc_id {doc.doc_id!r} in {doc.source_path.name} and {seen[doc.doc_id].name}"
            )
        seen[doc.doc_id] = doc.source_path

    logger.info("Loaded %d knowledge-base documents from %s", len(documents), documents_dir)
    return documents
