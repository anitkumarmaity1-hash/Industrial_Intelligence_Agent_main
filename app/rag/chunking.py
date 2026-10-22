"""
Chunking of knowledge-base documents.

Two-stage split, using LangChain's text splitters:

  1. `MarkdownHeaderTextSplitter` cuts on `#`/`##`/`###`. Maintenance
     documents are already organised into semantically complete sections
     ("3.2 Tool wear failure band"), so the author's own structure is a
     better chunk boundary than any character count we could pick.
  2. `RecursiveCharacterTextSplitter` then splits any section that is
     still too long for a useful retrieval unit, with overlap so a
     threshold sentence is never orphaned from the rule it belongs to.

LangChain is used here because these two splitters are exactly the kind
of fiddly, well-tested utility worth taking a dependency for — the header
splitter in particular correctly tracks header nesting and skips fenced
code blocks. It is not used to wrap the retriever itself; the agent tools
in Phase 5 call our own functions, which keeps the tool contract explicit.

Retained header path is important: a chunk reading "replace at 150
minutes" is ambiguous, while the same chunk tagged
"SOP-101 > 3. Replacement thresholds > 3.1 Routine preventive
replacement" is citable evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.rag.documents import IndustrialDocument

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of documentation, carrying everything a citation needs."""

    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    section: str
    text: str
    chunk_index: int
    source_path: str
    effective_date: str
    data_class: str
    failure_modes: list[str] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """Text actually sent to the embedding model.

        The document ID, title and section path are prepended so the
        vector encodes *where* the passage comes from, not just its
        wording. Without this, a bare paragraph of thresholds embeds
        almost identically across SOP-101/102/103.
        """
        return f"[{self.doc_id} | {self.title} | {self.section}]\n{self.text}"

    def citation(self) -> str:
        """Human-readable source reference for the agent's evidence list."""
        suffix = " (synthetic document)" if self.data_class.upper() == "SYNTHETIC" else ""
        return f"{self.doc_id} § {self.section} — {self.title}{suffix}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Chunk":
        return cls(**payload)


def _section_path(header_metadata: dict[str, str]) -> str:
    """Join the header hierarchy into a single readable path.

    The document's h1 is dropped when deeper headers exist: it repeats the
    title, which is already stored separately, and spending citation
    characters on it twice helps nobody.
    """
    deeper = [header_metadata[key] for key in ("h2", "h3") if header_metadata.get(key)]
    if deeper:
        return " > ".join(deeper)
    return header_metadata.get("h1", "Document")


def chunk_document(
    document: IndustrialDocument,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split one document into retrievable chunks.

    Args:
        document: A validated knowledge-base document.
        chunk_size: Maximum characters per chunk after the secondary split.
        chunk_overlap: Characters repeated between adjacent sub-chunks.

    Returns:
        Chunks in document order, each with a deterministic `chunk_id`.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for section in header_splitter.split_text(document.body):
        section_name = _section_path(section.metadata)
        for piece in size_splitter.split_text(section.page_content):
            text = piece.strip()
            if not text:
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}--{index:03d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    doc_type=document.doc_type,
                    section=section_name,
                    text=text,
                    chunk_index=index,
                    source_path=document.source_path.name,
                    effective_date=document.effective_date,
                    data_class=document.data_class,
                    failure_modes=list(document.failure_modes),
                )
            )

    logger.info("Chunked %s into %d chunks", document.doc_id, len(chunks))
    return chunks


def chunk_documents(
    documents: list[IndustrialDocument],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    logger.info("Produced %d chunks from %d documents", len(chunks), len(documents))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks to JSONL.

    This file is the single shared artifact both retrieval backends read:
    the Pinecone path embeds it, the local path searches it directly. One
    source of truth means the two backends can never drift onto different
    text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks to %s", len(chunks), path)


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks back from JSONL.

    Raises:
        FileNotFoundError: if ingestion has not been run yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"chunk file not found: {path}. Run `python scripts/ingest_documents.py` first."
        )
    with path.open(encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]
