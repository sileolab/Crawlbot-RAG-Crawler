"""Content Quality Assessment Service for Crawlbot.

This module evaluates extracted document quality, scores it from 0 to 100,
detects empty/sparse documents, calculates text density, and identifies duplicate
content using cryptographic hashing.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from ..schemas.documents import ExtractedDocument, QualityReport, QualityStatus

logger = logging.getLogger(__name__)


class QualityChecker:
    """Evaluates content quality, completeness, and deduplication for extracted documents."""

    def __init__(
        self,
        min_word_count: int = 20,
        min_char_count: int = 100,
        min_quality_score: int = 40,
    ) -> None:
        """Initialize QualityChecker with minimum threshold parameters.

        Args:
            min_word_count: Minimum acceptable word count for quality classification.
            min_char_count: Minimum acceptable character count for quality classification.
            min_quality_score: Minimum acceptable quality score (0-100).
        """
        self.min_word_count: int = min_word_count
        self.min_char_count: int = min_char_count
        self.min_quality_score: int = min_quality_score
        self._content_hashes: dict[str, str] = {}  # SHA-256 hash -> original URL

    def check(self, document: ExtractedDocument) -> ExtractedDocument:
        """Run all quality checks on an extracted document, updating its quality_status and quality_report.

        Args:
            document: The ExtractedDocument to inspect and classify.

        Returns:
            The updated ExtractedDocument instance.
        """
        # If extraction already failed or was blocked by security, preserve status
        if document.quality_status in (QualityStatus.FAILED, QualityStatus.BLOCKED):
            if not document.quality_report:
                document.quality_report = QualityReport(
                    score=0,
                    passed=False,
                    reasons=[f"Status is {document.quality_status.value}"],
                )
            return document

        # 1. Compute Quality Report (0 - 100)
        report = self.evaluate(document)
        document.quality_report = report

        # 2. Check for empty content
        if self.is_empty(document):
            document.quality_status = QualityStatus.EMPTY
            warning = "Document content is empty or contains only whitespace"
            if warning not in document.warnings:
                document.warnings.append(warning)
            report.passed = False
            return document

        # 3. Check for duplicate content
        is_dup, original_url = self.is_duplicate(document)
        if is_dup:
            document.quality_status = QualityStatus.DUPLICATE
            warning = f"Duplicate content of previously indexed page: {original_url}"
            if warning not in document.warnings:
                document.warnings.append(warning)
            report.duplicate_percentage = 100.0
            report.score = max(0, report.score - 50)
            report.passed = False
            report.reasons.append(f"Duplicate content matches {original_url}")
            return document

        # Record hash for subsequent deduplication
        doc_hash = document.content_hash or self._compute_content_hash(document)
        if doc_hash:
            self._content_hashes[doc_hash] = str(document.url)

        # 4. Check for low quality (sparse text / below threshold or score < min_quality_score)
        if self.is_low_quality(document) or report.score < self.min_quality_score:
            document.quality_status = QualityStatus.LOW_QUALITY
            warning = (
                f"Low quality content: {document.word_count} words (min {self.min_word_count}), "
                f"score {report.score}/100 (min {self.min_quality_score})"
            )
            if warning not in document.warnings:
                document.warnings.append(warning)
            report.passed = False
            return document

        # Document passed all quality checks
        document.quality_status = QualityStatus.SUCCESS
        report.passed = True
        return document

    def evaluate(self, document: ExtractedDocument, raw_html: Optional[str] = None) -> QualityReport:
        """Calculate comprehensive 0-100 quality score and metrics.

        Args:
            document: ExtractedDocument to assess.
            raw_html: Optional raw HTML string used to calculate text density.

        Returns:
            QualityReport populated with detailed scores, metrics, and reasons.
        """
        text = (document.content_text or "").strip()
        markdown = (document.content_markdown or "").strip()
        content = text or markdown

        if not content:
            return QualityReport(
                score=0,
                char_count=0,
                word_count=0,
                token_count_approx=0,
                text_density=0.0,
                passed=False,
                reasons=["Empty content"],
            )

        words = re.findall(r"\b\w+\b", content)
        word_count = len(words)
        char_count = len(content)
        token_count = int(word_count * 1.3)

        reasons: list[str] = []
        score = 0

        # A. Word count score (max 30 points)
        if word_count < 15:
            score += 0
            reasons.append("Severe penalty: less than 15 words")
        elif word_count < 50:
            score += 10
            reasons.append("Low word count (under 50 words)")
        elif word_count < 150:
            score += 20
            reasons.append("Moderate word count (50-150 words)")
        else:
            score += 30
            reasons.append(f"Substantial word count ({word_count} words)")

        # B. Heading structure (max 20 points)
        headings = document.headings or []
        heading_count = len(headings)
        has_h1 = any(h.level == 1 for h in headings) or bool(document.title)
        has_subheadings = any(h.level in (2, 3) for h in headings)

        if has_h1:
            score += 10
        if has_subheadings:
            score += 5
        if heading_count >= 3:
            score += 5
            reasons.append(f"Well-structured with {heading_count} headings")
        elif heading_count == 0:
            reasons.append("No headings found")

        # C. Text density (max 20 points)
        html_len = len(raw_html) if raw_html else (char_count * 3)
        text_density = min(1.0, round(char_count / max(1, html_len), 4))
        if text_density >= 0.20:
            score += 20
            reasons.append(f"High text density ({text_density:.1%})")
        elif text_density >= 0.08:
            score += 14
            reasons.append(f"Acceptable text density ({text_density:.1%})")
        elif text_density >= 0.03:
            score += 8
        else:
            score += 2
            reasons.append(f"Very low text density ({text_density:.1%})")

        # D. Paragraph structure (max 15 points)
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
        paragraph_count = len(paragraphs)
        if paragraph_count >= 3:
            score += 15
            reasons.append(f"Good reading flow ({paragraph_count} paragraphs)")
        elif paragraph_count >= 1:
            score += 8
        else:
            score += 3

        # E. Vocabulary diversity (max 15 points)
        unique_words = set(w.lower() for w in words)
        diversity = len(unique_words) / max(1, word_count)
        if diversity >= 0.50:
            score += 15
        elif diversity >= 0.35:
            score += 10
        else:
            score += 4
            reasons.append("Low lexical diversity (repetitive text)")

        score = max(0, min(100, score))
        passed = score >= self.min_quality_score

        has_title = bool(document.title and document.title.strip())
        has_meta_desc = bool(
            document.metadata.get("description")
            or document.metadata.get("og:description")
        )
        internal_links = sum(1 for link in (document.links or []) if link.is_internal)

        return QualityReport(
            score=score,
            char_count=char_count,
            word_count=word_count,
            token_count_approx=token_count,
            text_density=text_density,
            heading_count=heading_count,
            paragraph_count=paragraph_count,
            has_title=has_title,
            has_meta_description=has_meta_desc,
            internal_link_count=internal_links,
            duplicate_percentage=0.0,
            noise_blocks_count=0,
            warnings=list(document.warnings),
            confidence=round(min(1.0, score / 85.0), 2),
            reasons=reasons,
            passed=passed,
        )

    def is_empty(self, document: ExtractedDocument) -> bool:
        """Check if content is empty or effectively contains only whitespace."""
        has_text = bool(document.content_text and document.content_text.strip())
        has_markdown = bool(document.content_markdown and document.content_markdown.strip())

        if not has_text and not has_markdown:
            return True

        if document.word_count == 0 and document.character_count == 0:
            return True

        return False

    def is_low_quality(self, document: ExtractedDocument) -> bool:
        """Check if document content falls below minimum quality thresholds."""
        if self.is_empty(document):
            return False

        return (
            document.word_count < self.min_word_count
            or document.character_count < self.min_char_count
        )

    def is_duplicate(self, document: ExtractedDocument) -> tuple[bool, Optional[str]]:
        """Check if content hash matches a previously seen document in this session."""
        doc_hash = document.content_hash or self._compute_content_hash(document)
        if not doc_hash:
            return False, None

        if doc_hash in self._content_hashes:
            original_url = self._content_hashes[doc_hash]
            if original_url != str(document.url):
                return True, original_url

        return False, None

    def _compute_content_hash(self, document: ExtractedDocument) -> Optional[str]:
        """Compute SHA-256 fingerprint from available text or markdown content."""
        raw_content = document.content_text or document.content_markdown or ""
        stripped = raw_content.strip()
        if not stripped:
            return None
        return hashlib.sha256(stripped.encode("utf-8")).hexdigest()

    def reset(self) -> None:
        """Clear cached content hashes and reset state for a new crawl run."""
        self._content_hashes.clear()
