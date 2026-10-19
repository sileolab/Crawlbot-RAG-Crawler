"""Text Chunking Service for Crawlbot.

Splits extracted document content into smaller, overlapping chunks suitable
for vector embedding and RAG (Retrieval-Augmented Generation) ingestion.

Supported strategies:
- ``auto``:             Intelligently evaluates document properties (headings, markdown quality,
                        paragraph length, page type, language) to select the optimal strategy.
- ``semantic_heading``: Hierarchical splitting at Markdown headings (H1-H6), preserving parent-child
                        context stack and preventing orphan headings.
- ``paragraph``:        Split on paragraph boundaries (double newlines) with sub-splitting of oversized paragraphs.
- ``sentence``:         Multilingual sentence-boundary splitting protecting decimals (3.14), URLs, and abbreviations.
- ``fixed_size``:       Character window splitting conscious of word boundaries and Persian half-spaces (\\u200c).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from ..schemas.chunks import ChunkingResult, ChunkingStrategy, DocumentChunk
from ..schemas.documents import ExtractedDocument
from .normalizer import PersianNormalizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns for multilingual protection and normalization
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
_ABBR_RE = re.compile(
    r"\b(?:e\.g\.|i\.e\.|etc\.|vs\.|al\.|Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|approx\.|dept\.|est\.|fig\.|inc\.|ltd\.|no\.)",
    re.IGNORECASE,
)
_PERSIAN_ABBR_RE = re.compile(r"(?:\b|\s)(?:ص|ش|ق|م|هـ|د)\.\s*")

_UI_NOISE_RE = re.compile(
    r"(?:استفاده از کوکی|پذیرش کوکی|کوکی‌ها|cookie policy|use cookies|accept cookies|"
    r"تمامی حقوق محفوظ است|کلیه حقوق محفوظ|all rights reserved|"
    r"ورود / ثبت‌نام|ورود یا عضویت|رمز عبور خود را فراموش|log in / sign up|sign in to your account|"
    r"صفحه نخست >|خانه >|home >)",
    re.IGNORECASE,
)


def detect_chunk_quality_flags(text: str, heading_ctx: Optional[str] = None) -> list[str]:
    """Detect non-destructive quality flags such as UI noise, boilerplate, or low-information content."""
    flags: list[str] = []
    clean = text.strip()
    words = clean.split()

    if len(words) < 5 or len(clean) < 25:
        flags.append("low_info")

    if _UI_NOISE_RE.search(clean):
        flags.append("ui_noise")
        flags.append("boilerplate")
    elif heading_ctx and any(k in heading_ctx.lower() for k in ["footer", "فوتر", "منو", "navigation", "nav"]):
        flags.append("ui_noise")

    return flags


# ---------------------------------------------------------------------------
# Helper – chunk ID generation
# ---------------------------------------------------------------------------

def _make_chunk_id(source_url: str, chunk_index: int, content_hash: Optional[str] = None) -> str:
    """Generate a stable, unique chunk identifier incorporating URL, index, and content hash."""
    raw = f"{source_url}::{chunk_index}"
    if content_hash:
        raw += f"::{content_hash[:8]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Protection helpers for sentence splitting
# ---------------------------------------------------------------------------

def _protect_text_for_sentence_split(text: str) -> tuple[str, dict[str, str]]:
    """Mask URLs, decimals, and abbreviations so sentence splitting never breaks them."""
    placeholders: dict[str, str] = {}
    counter = 0

    def repl(m: re.Match) -> str:
        nonlocal counter
        token = f"«__PRTK_{counter}__»"
        placeholders[token] = m.group(0)
        counter += 1
        return token

    # Order: URLs first, then decimals, then abbreviations
    text = _URL_RE.sub(repl, text)
    text = _DECIMAL_RE.sub(repl, text)
    text = _ABBR_RE.sub(repl, text)
    text = _PERSIAN_ABBR_RE.sub(repl, text)

    return text, placeholders


def _restore_protected_text(text: str, placeholders: dict[str, str]) -> str:
    """Restore masked tokens back to their original text."""
    for token, orig in placeholders.items():
        text = text.replace(token, orig)
    return text


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences while protecting decimals, abbreviations, and URLs."""
    if not text or not text.strip():
        return []
    protected, placeholders = _protect_text_for_sentence_split(text)
    pattern = re.compile(r"(?<=[.!?؟])\s+")
    parts = [p.strip() for p in pattern.split(protected) if p.strip()]
    if not parts:
        parts = [protected.strip()]
    return [_restore_protected_text(p, placeholders) for p in parts if p.strip()]


def _extract_overlap_tail(text: str, overlap: int) -> str:
    """Extract a clean tail of text for chunk overlap, preserving word boundaries."""
    if not text or overlap <= 0:
        return ""
    clean = text.strip()
    if len(clean) <= overlap:
        return clean
    tail = clean[-overlap:]
    # Snap to the first space in tail so we don't start with a severed word
    space_idx = tail.find(" ")
    if space_idx != -1 and space_idx < len(tail) // 2:
        tail = tail[space_idx + 1:].strip()
    # Strip any leading \u200c
    tail = tail.lstrip("\u200c")
    return tail


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _merge_segments(
    segments: list[str],
    chunk_size: int,
    overlap: int,
    is_sentence_level: bool = False,
) -> list[str]:
    """Merge short segments into chunks of approximately ``chunk_size`` characters.

    Sub-splits individual oversized segments using sentence and fixed-size strategies.
    Preserves context continuity with clean overlap between adjacent chunks.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for raw_seg in segments:
        seg = raw_seg.strip()
        if not seg:
            continue
        seg_len = len(seg)

        # If a single segment is longer than chunk_size, sub-split it
        if seg_len > chunk_size:
            # Flush accumulated parts first
            if current_parts:
                chunk_text = "\n\n".join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_parts = []
                current_len = 0

            # Sub-split oversized segment
            if is_sentence_level:
                # Already at sentence level, split using fixed window
                chunks.extend(_split_fixed(seg, chunk_size, overlap))
            else:
                # Paragraph level: sub-split by sentence
                sub_chunks = _split_sentence(seg, chunk_size, overlap)
                for sc in sub_chunks:
                    if len(sc) > chunk_size:
                        chunks.extend(_split_fixed(sc, chunk_size, overlap))
                    else:
                        chunks.append(sc)
            continue

        if current_len + seg_len + (2 if current_parts else 0) > chunk_size and current_parts:
            # Save current chunk
            chunk_text = "\n\n".join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Start new chunk with overlap from previous
            if overlap > 0 and current_parts:
                prev_text = "\n\n".join(current_parts)
                overlap_text = _extract_overlap_tail(prev_text, overlap)
                if overlap_text:
                    current_parts = [overlap_text, seg]
                    current_len = len(overlap_text) + seg_len + 2
                else:
                    current_parts = [seg]
                    current_len = seg_len
            else:
                current_parts = [seg]
                current_len = seg_len
        else:
            current_parts.append(seg)
            current_len += seg_len + (2 if len(current_parts) > 1 else 0)

    if current_parts:
        final_chunk = "\n\n".join(current_parts).strip()
        if final_chunk:
            chunks.append(final_chunk)

    return chunks


def _split_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text by paragraph boundaries (double newlines).

    Paragraphs shorter than ``chunk_size`` chars are merged together until
    the limit is reached. Oversized paragraphs are sub-split into sentences.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []
    return _merge_segments(paragraphs, chunk_size, overlap, is_sentence_level=False)


def _split_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into sentences then group them into chunks."""
    sentences = _extract_sentences(text)
    if not sentences:
        return []
    return _merge_segments(sentences, chunk_size, overlap, is_sentence_level=True)


def _split_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size character windows with overlap.

    Word-boundary conscious: avoids cleaving words or breaking Persian half-spaces (\\u200c)
    when a suitable boundary exists near the target window edge.
    """
    if not text:
        return []

    text_len = len(text)
    if text_len <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < text_len:
        target_end = min(text_len, start + chunk_size)

        if target_end >= text_len:
            actual_end = text_len
        else:
            # Look for boundary in slack window (last 15% of chunk_size)
            slack = max(10, int(chunk_size * 0.15))
            search_start = max(start + 1, target_end - slack)
            candidate = -1

            # 1. Look for paragraph break
            p_idx = text.rfind("\n\n", search_start, target_end)
            if p_idx != -1:
                candidate = p_idx
            else:
                # 2. Look for sentence break [.!?؟] followed by space or newline
                for m in re.finditer(r"[.!?؟][\s\r\n]", text[search_start:target_end]):
                    candidate = search_start + m.end()

                if candidate == -1:
                    # 3. Look for word space
                    s_idx = text.rfind(" ", search_start, target_end)
                    if s_idx != -1:
                        candidate = s_idx

            if candidate != -1:
                actual_end = candidate
            else:
                actual_end = target_end

        # Protect Persian half-space (\u200c)
        if actual_end < text_len:
            if text[actual_end] == "\u200c":
                actual_end += 1
            elif actual_end > 0 and text[actual_end - 1] == "\u200c":
                space_after = text.find(" ", actual_end)
                if space_after != -1 and space_after - actual_end < 15:
                    actual_end = space_after
                else:
                    actual_end = max(start + 1, actual_end - 2)

        chunk = text[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)

        if actual_end >= text_len:
            break

        # Calculate next start with overlap
        if overlap > 0 and actual_end > start:
            next_start = max(start + 1, actual_end - overlap)
            if next_start < actual_end:
                space_pos = text.find(" ", next_start, min(text_len, next_start + 20))
                if space_pos != -1:
                    next_start = space_pos + 1
        else:
            next_start = actual_end

        if next_start <= start:
            next_start = start + step

        start = next_start

    return chunks


def _split_semantic_heading_with_context(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, Optional[str]]]:
    """Split Markdown text at heading boundaries (H1 to H6), preserving context and hierarchy.

    Returns a list of (chunk_text, hierarchical_heading_context) tuples.
    Prevents orphan headings and sub-splits oversized sections.
    """
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        # Fall back to paragraph splitting when no headings found
        raw = _split_paragraph(text, chunk_size, overlap)
        return [(c, None) for c in raw]

    chunks_with_ctx: list[tuple[str, Optional[str]]] = []

    # 1. Preamble: Content before the first heading
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            if len(preamble) <= chunk_size:
                chunks_with_ctx.append((preamble, None))
            else:
                for sub in _split_paragraph(preamble, chunk_size, overlap):
                    chunks_with_ctx.append((sub, None))

    # 2. Iterate headings and maintain hierarchical stack
    heading_stack: list[tuple[int, str]] = []  # (level, title)

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        # Maintain stack hierarchy: pop elements at or deeper than current level
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        current_context = " > ".join(t for _, t in heading_stack)

        # Slice section content until next heading or EOF
        start_pos = m.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_raw = text[start_pos:end_pos].strip()

        # Check if section has substantive body text beyond the heading line itself
        heading_line = m.group(0).strip()
        if section_raw.startswith(heading_line):
            body_text = section_raw[len(heading_line):].strip()
        else:
            body_text = section_raw.replace(heading_line, "", 1).strip()

        if not body_text:
            # Heading has no body content before the next heading.
            # Prevent orphan heading chunk! The title is preserved in heading_stack for child sections.
            continue

        # Section has substantive content
        if len(section_raw) <= chunk_size:
            chunks_with_ctx.append((section_raw, current_context))
        else:
            # Sub-split oversized section
            sub_chunks = _split_paragraph(section_raw, chunk_size, overlap)
            for sc in sub_chunks:
                chunks_with_ctx.append((sc, current_context))

    # Fallback if document only had empty headings and no chunks were produced
    if not chunks_with_ctx and text.strip():
        raw = _split_paragraph(text, chunk_size, overlap)
        return [(c, None) for c in raw]

    return chunks_with_ctx


def _split_semantic_heading(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split Markdown text at heading boundaries, returning chunk text strings only."""
    tuples = _split_semantic_heading_with_context(text, chunk_size, overlap)
    return [chunk for chunk, _ in tuples]


# ---------------------------------------------------------------------------
# Heading context tracker for non-heading strategies
# ---------------------------------------------------------------------------

def _extract_heading_contexts(text: str) -> list[tuple[int, str]]:
    """Return a list of (char_position, hierarchical_heading_context) tuples from Markdown text."""
    pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    stack: list[tuple[int, str]] = []
    contexts: list[tuple[int, str]] = []

    for m in pattern.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        full_ctx = " > ".join(t for _, t in stack)
        contexts.append((m.start(), full_ctx))

    return contexts


def _find_heading_context(char_pos: int, heading_contexts: list[tuple[int, str]]) -> Optional[str]:
    """Return the nearest heading context that precedes ``char_pos``."""
    context = None
    for pos, heading in heading_contexts:
        if pos <= char_pos:
            context = heading
        else:
            break
    return context


# ---------------------------------------------------------------------------
# Auto strategy selector
# ---------------------------------------------------------------------------

def select_auto_strategy(doc: ExtractedDocument, text: str) -> tuple[str, str]:
    """Deterministically analyze document characteristics and select the best strategy.

    Evaluates:
    - Page type classification (faq, policy, product, article, etc.)
    - Structured Markdown heading hierarchy (H1-H6)
    - Paragraph counts and distribution
    - Content length and density
    - Multilingual punctuation structure

    Returns:
        (strategy_name, selection_reason)
    """
    clean_text = text.strip()
    if not clean_text:
        return "paragraph", "Empty document defaulted to paragraph strategy"

    # 1. Page Type heuristics
    ptype = (doc.page_type or "").lower()

    # Markdown headings count (H1 to H6)
    heading_pattern = re.compile(r"^(#{1,6})\s+\S+", re.MULTILINE)
    md_headings = heading_pattern.findall(clean_text)
    heading_count = len(md_headings)
    if heading_count == 0 and doc.headings:
        heading_count = len(doc.headings)

    if ptype == "faq":
        if heading_count >= 2:
            return "semantic_heading", f"FAQ page with {heading_count} structured Q&A headings"
        return "paragraph", "FAQ page without headings; paragraph boundaries preserve question-answer pairs"

    if ptype in {"policy", "terms", "legal", "privacy"}:
        if heading_count >= 3:
            return "semantic_heading", f"Policy/legal document with {heading_count} structured section headings"
        return "sentence", "Dense policy/legal content segmented by granular sentence boundaries"

    if ptype in {"product", "pricing"}:
        if heading_count >= 2:
            return "semantic_heading", f"Product/pricing page with {heading_count} structured feature headings"
        return "paragraph", "Product specifications and features grouped into paragraph blocks"

    # 2. Check Markdown Headings
    if heading_count >= 2:
        return (
            "semantic_heading",
            f"Document contains {heading_count} structured Markdown headings (H1-H6) forming a clear hierarchy",
        )

    # 3. Paragraph structure analysis
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_text) if p.strip()]
    if len(paragraphs) >= 2:
        avg_para_len = sum(len(p) for p in paragraphs) / len(paragraphs)
        if avg_para_len >= 60:
            return (
                "paragraph",
                f"Document has {len(paragraphs)} well-formed paragraphs with average length of {int(avg_para_len)} characters",
            )

    # 4. Dense / single-block / fragmented text
    raw_sentences = _extract_sentences(clean_text)
    if len(raw_sentences) >= 2:
        return (
            "sentence",
            f"Content lacks distinct paragraph breaks; segmented across {len(raw_sentences)} natural sentences",
        )

    # 5. Fallback
    return "paragraph", "Default fallback to paragraph strategy for standard prose flow"


# ---------------------------------------------------------------------------
# Public API – ChunkerService
# ---------------------------------------------------------------------------

class ChunkerService:
    """Splits extracted documents into structured chunks for RAG pipelines."""

    STRATEGIES = {"auto", "paragraph", "sentence", "fixed_size", "semantic_heading"}

    def __init__(
        self,
        strategy: str = "auto",
        chunk_size: int = 1000,
        overlap: int = 150,
        normalize: bool = False,
        normalizer: Optional[PersianNormalizer] = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            strategy: Splitting strategy name ('auto', 'semantic_heading', 'paragraph', 'sentence', 'fixed_size').
            chunk_size: Target chunk size in characters (min 50).
            overlap: Number of characters to repeat at the start of each new chunk.
            normalize: Whether to apply Persian & Unicode normalization prior to chunking.
            normalizer: Optional custom PersianNormalizer instance.

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
        self.normalize = normalize
        self.normalizer = normalizer

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
            ChunkingResult with produced chunks and execution metadata.
        """
        text = doc.content_markdown or doc.content_text or ""
        text = text.strip()

        norm_applied = (
            bool(doc.metadata.get("normalized", False))
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict)
            else False
        )

        if self.normalize:
            active_norm = self.normalizer or PersianNormalizer()
            norm_text, norm_rep = active_norm.normalize_with_report(text)
            if norm_rep.get("changed", False):
                text = norm_text
                norm_applied = True

        if not text:
            logger.debug("Document %s has no content to chunk.", doc.url)
            return ChunkingResult(
                source_url=doc.url,
                total_chunks=0,
                strategy=self.strategy,
                applied_strategy=self.strategy,
                chunks=[],
                metadata={"reason": "Document has no content to chunk"},
            )

        # Resolve strategy if auto
        if self.strategy == "auto":
            applied_strategy, reason = select_auto_strategy(doc, text)
        else:
            applied_strategy = self.strategy
            reason = f"Explicitly requested strategy: {self.strategy}"

        # Split text and capture heading contexts
        if applied_strategy == "semantic_heading":
            raw_tuples = _split_semantic_heading_with_context(text, self.chunk_size, self.overlap)
        else:
            if applied_strategy == "paragraph":
                raw_texts = _split_paragraph(text, self.chunk_size, self.overlap)
            elif applied_strategy == "sentence":
                raw_texts = _split_sentence(text, self.chunk_size, self.overlap)
            elif applied_strategy == "fixed_size":
                raw_texts = _split_fixed(text, self.chunk_size, self.overlap)
            else:
                raw_texts = _split_paragraph(text, self.chunk_size, self.overlap)

            # Associate heading context for non-semantic splitters using character position
            heading_contexts = _extract_heading_contexts(text)
            raw_tuples = []
            running_pos = 0
            for ct in raw_texts:
                chunk_start = text.find(ct[:50].strip(), running_pos)
                if chunk_start == -1:
                    chunk_start = running_pos
                running_pos = chunk_start + len(ct)
                ctx = _find_heading_context(chunk_start, heading_contexts)
                raw_tuples.append((ct, ctx))

        # Filter out empty chunks
        raw_tuples = [(t.strip(), ctx) for t, ctx in raw_tuples if t and t.strip()]

        total = len(raw_tuples)
        doc_chunks: list[DocumentChunk] = []

        for idx, (chunk_text, heading_ctx) in enumerate(raw_tuples):
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunk_id = _make_chunk_id(doc.url, idx, chunk_hash)

            word_count = len(chunk_text.split())
            char_count = len(chunk_text)

            # Fallback category to ensure non-null, meaningful categorization
            resolved_category = (
                category
                or (heading_ctx.split(" > ")[0].strip() if heading_ctx else None)
                or getattr(doc, "page_type", None)
                or "general"
            )

            q_flags = detect_chunk_quality_flags(chunk_text, heading_ctx)

            dc = DocumentChunk(
                chunk_id=chunk_id,
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
                category=resolved_category,
                content_hash=chunk_hash,
                source_doc_id=getattr(doc, "document_id", None),
                source_doc_hash=getattr(doc, "content_hash", None),
                processed_at=datetime.now(timezone.utc).isoformat(),
                quality_flags=q_flags,
                normalization_applied=norm_applied,
                metadata={
                    "source_doc_hash": doc.content_hash,
                    "content_hash": chunk_hash,
                    "http_status": doc.http_status,
                    "fetched_at": doc.fetched_at.isoformat() if doc.fetched_at else None,
                    "strategy": applied_strategy,
                    "selection_reason": reason if self.strategy == "auto" else None,
                    "quality_flags": q_flags,
                },
            )
            doc_chunks.append(dc)

        logger.info(
            "Chunked %s into %d chunks (strategy=%s, applied=%s, size=%d, overlap=%d).",
            doc.url,
            total,
            self.strategy,
            applied_strategy,
            self.chunk_size,
            self.overlap,
        )

        return ChunkingResult(
            source_url=doc.url,
            total_chunks=total,
            strategy=self.strategy,
            applied_strategy=applied_strategy,
            chunks=doc_chunks,
            metadata={
                "selected_strategy": applied_strategy,
                "selection_reason": reason,
                "requested_strategy": self.strategy,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
            },
        )

    def chunk_documents(
        self,
        docs: list[ExtractedDocument],
        category: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Chunk multiple documents and return a flat list of chunks."""
        all_chunks: list[DocumentChunk] = []
        for doc in docs:
            result = self.chunk_document(doc, category=category)
            all_chunks.extend(result.chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        """Dispatch to the configured strategy (returns string chunks)."""
        if self.strategy == "paragraph":
            return _split_paragraph(text, self.chunk_size, self.overlap)
        elif self.strategy == "sentence":
            return _split_sentence(text, self.chunk_size, self.overlap)
        elif self.strategy == "fixed_size":
            return _split_fixed(text, self.chunk_size, self.overlap)
        elif self.strategy == "semantic_heading":
            return _split_semantic_heading(text, self.chunk_size, self.overlap)
        elif self.strategy == "auto":
            # For raw string splitting in auto mode without doc metadata, evaluate headings
            heading_pattern = re.compile(r"^(#{1,6})\s+\S+", re.MULTILINE)
            if len(heading_pattern.findall(text)) >= 2:
                return _split_semantic_heading(text, self.chunk_size, self.overlap)
            return _split_paragraph(text, self.chunk_size, self.overlap)
        else:
            return _split_paragraph(text, self.chunk_size, self.overlap)
