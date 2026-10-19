"""Text Chunking Service for Crawlbot.

Splits extracted document content into smaller, overlapping chunks suitable
for vector embedding and RAG (Retrieval-Augmented Generation) ingestion.

Supported strategies:
- ``paragraph``:      Split on double newlines; natural paragraph boundaries.
- ``sentence``:       Split on sentence-ending punctuation (. ! ?).
- ``fixed_size``:     Fixed character window with configurable overlap.
- ``semantic_heading``:  Split at Markdown headings (# ## ###), preserving context.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from ..schemas.chunks import ChunkingResult, DocumentChunk
from ..schemas.documents import ExtractedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper – chunk ID generation
# ---------------------------------------------------------------------------

def _make_chunk_id(source_url: str, chunk_index: int) -> str:
    """Generate a stable, unique chunk identifier."""
    raw = f"{source_url}::{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _split_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text by paragraph boundaries (double newlines).

    Paragraphs shorter than ``chunk_size`` chars are merged together until
    the limit is reached, then a new chunk starts.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _merge_segments(paragraphs, chunk_size, overlap)


def _split_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into sentences then group them into chunks."""
    # Simple sentence splitter using punctuation boundaries
    sentence_pattern = re.compile(r"(?<=[.!?؟])\s+")
    sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]
    if not sentences:
        sentences = [text.strip()]
    return _merge_segments(sentences, chunk_size, overlap)


def _split_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size character windows with overlap."""
    if not text:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _split_semantic_heading(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split Markdown text at heading boundaries (# / ## / ###).

    Each heading begins a new chunk. Long sections are further split by
    the paragraph strategy if they exceed ``chunk_size``.
    """
    heading_pattern = re.compile(r"^(#{1,3})\s+.+", re.MULTILINE)
    positions = [m.start() for m in heading_pattern.finditer(text)]

    if not positions:
        # Fall back to paragraph splitting when no headings found
        return _split_paragraph(text, chunk_size, overlap)

    sections: list[str] = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[pos:end].strip()
        if section:
            sections.append(section)

    # Any leading content before the first heading
    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            sections.insert(0, preamble)

    # Sub-split sections that are still too long
    result: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            result.append(section)
        else:
            result.extend(_split_paragraph(section, chunk_size, overlap))
    return result


def _merge_segments(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Merge short segments into chunks of approximately ``chunk_size`` characters.

    When a chunk is full it is saved and the next chunk starts with the last
    ``overlap`` characters from the previous chunk to preserve context.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg)
        if current_len + seg_len + 1 > chunk_size and current_parts:
            # Save current chunk
            chunk_text = "\n\n".join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # Start new chunk with overlap from previous
            if overlap > 0 and current_parts:
                overlap_text = ("\n\n".join(current_parts))[-overlap:]
                current_parts = [overlap_text, seg]
                current_len = len(overlap_text) + seg_len + 1
            else:
                current_parts = [seg]
                current_len = seg_len
        else:
            current_parts.append(seg)
            current_len += seg_len + 1

    if current_parts:
        final_chunk = "\n\n".join(current_parts).strip()
        if final_chunk:
            chunks.append(final_chunk)

    return chunks


# ---------------------------------------------------------------------------
# Heading context tracker
# ---------------------------------------------------------------------------

def _extract_heading_contexts(text: str) -> list[tuple[int, str]]:
    """Return a list of (char_position, heading_text) tuples from Markdown text."""
    pattern = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)
    return [(m.start(), m.group(2).strip()) for m in pattern.finditer(text)]


def _find_heading_context(char_pos: int, heading_contexts: list[tuple[int, str]]) -> Optional[str]:
    """Return the nearest heading that precedes ``char_pos``."""
    context = None
    for pos, heading in heading_contexts:
        if pos <= char_pos:
            context = heading
        else:
            break
    return context


# ---------------------------------------------------------------------------
# Public API – ChunkerService
# ---------------------------------------------------------------------------

class ChunkerService:
    """Splits extracted documents into structured chunks for RAG pipelines."""

    STRATEGIES = {"paragraph", "sentence", "fixed_size", "semantic_heading"}

    def __init__(
        self,
        strategy: str = "paragraph",
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None:
        """Initialize the chunker.

        Args:
            strategy: Splitting strategy name.
            chunk_size: Target chunk size in characters.
            overlap: Number of characters to repeat at the start of each new chunk.

        Raises:
            ValueError: If an unknown strategy is given.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Allowed: {sorted(self.STRATEGIES)}"
            )
        self.strategy = strategy
        self.chunk_size = max(50, chunk_size)
        self.overlap = max(0, min(overlap, self.chunk_size // 2))

    # ------------------------------------------------------------------
    # Document-level chunking
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        doc: ExtractedDocument,
        category: Optional[str] = None,
    ) -> ChunkingResult:
        """Chunk a single ExtractedDocument into DocumentChunk objects.

        Args:
            doc: Source extracted document.
            category: Optional classification tag.

        Returns:
            ChunkingResult with produced chunks.
        """
        # Prefer markdown content (richer structure for heading-based splitting)
        text = doc.content_markdown or doc.content_text or ""
        text = text.strip()

        if not text:
            logger.debug("Document %s has no content to chunk.", doc.url)
            return ChunkingResult(
                source_url=doc.url,
                total_chunks=0,
                strategy=self.strategy,
                chunks=[],
            )

        # Compute heading positions for context tagging
        heading_contexts = _extract_heading_contexts(text)

        # Select splitter
        raw_chunks = self._split(text)

        # Build DocumentChunk objects
        total = len(raw_chunks)
        doc_chunks: list[DocumentChunk] = []
        running_pos = 0

        for idx, chunk_text in enumerate(raw_chunks):
            # Estimate position of chunk in original text for heading context
            chunk_start = text.find(chunk_text[:50].strip(), running_pos)
            if chunk_start == -1:
                chunk_start = running_pos
            running_pos = chunk_start + len(chunk_text)

            heading_ctx = _find_heading_context(chunk_start, heading_contexts)

            word_count = len(chunk_text.split())
            char_count = len(chunk_text)

            dc = DocumentChunk(
                chunk_id=_make_chunk_id(doc.url, idx),
                source_url=doc.url,
                chunk_index=idx,
                total_chunks=total,
                text=chunk_text,
                char_count=char_count,
                word_count=word_count,
                heading_context=heading_ctx,
                title=doc.title,
                source_title=doc.title,
                language=doc.language,
                category=category,
                metadata={
                    "content_hash": doc.content_hash,
                    "http_status": doc.http_status,
                    "fetched_at": doc.fetched_at.isoformat() if doc.fetched_at else None,
                },
            )
            doc_chunks.append(dc)

        logger.info(
            "Chunked %s into %d chunks (strategy=%s, size=%d, overlap=%d).",
            doc.url,
            total,
            self.strategy,
            self.chunk_size,
            self.overlap,
        )

        return ChunkingResult(
            source_url=doc.url,
            total_chunks=total,
            strategy=self.strategy,
            chunks=doc_chunks,
        )

    def chunk_documents(
        self,
        docs: list[ExtractedDocument],
        category: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Chunk multiple documents and return a flat list of chunks.

        Args:
            docs: List of extracted documents.
            category: Optional classification tag.

        Returns:
            Flat list of DocumentChunk objects.
        """
        all_chunks: list[DocumentChunk] = []
        for doc in docs:
            result = self.chunk_document(doc, category=category)
            all_chunks.extend(result.chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        """Dispatch to the configured strategy."""
        if self.strategy == "paragraph":
            return _split_paragraph(text, self.chunk_size, self.overlap)
        elif self.strategy == "sentence":
            return _split_sentence(text, self.chunk_size, self.overlap)
        elif self.strategy == "fixed_size":
            return _split_fixed(text, self.chunk_size, self.overlap)
        elif self.strategy == "semantic_heading":
            return _split_semantic_heading(text, self.chunk_size, self.overlap)
        else:
            return _split_paragraph(text, self.chunk_size, self.overlap)
