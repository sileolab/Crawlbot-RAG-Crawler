"""Document schemas for Crawlbot.

Defines Pydantic models representing extracted web documents, headings, links,
and quality assessment statuses for content extraction and RAG ingestion.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_validator,
)


class QualityStatus(str, Enum):
    """Quality classification status for extracted documents."""

    SUCCESS = "success"
    LOW_QUALITY = "low_quality"
    EMPTY = "empty"
    DUPLICATE = "duplicate"
    FAILED = "failed"
    BLOCKED = "blocked"


class Heading(BaseModel):
    """Represents a structured heading extracted from HTML."""

    model_config = ConfigDict(populate_by_name=True)

    level: int = Field(ge=1, le=6, description="Heading level from 1 to 6 (H1-H6).")
    text: str = Field(description="Heading text content.")

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        """Strip surrounding whitespace from heading text."""
        return v.strip() if isinstance(v, str) else v


class LinkInfo(BaseModel):
    """Metadata for hyperlinks discovered within extracted documents."""

    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(description="Target URL of the link.")
    text: str = Field(default="", description="Anchor text or description of the link.")
    is_internal: bool = Field(
        default=False,
        description="Whether the hyperlink points to the same domain/host.",
    )

    @field_validator("url")
    @classmethod
    def clean_url(cls, v: str) -> str:
        """Strip surrounding whitespace from URL string."""
        return v.strip() if isinstance(v, str) else v

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        """Strip surrounding whitespace from anchor text."""
        return v.strip() if isinstance(v, str) else v


class ExtractionAttempt(BaseModel):
    """Record of an extraction attempt with an engine."""

    model_config = ConfigDict(populate_by_name=True)

    engine: str = Field(description="Engine name: 'trafilatura', 'beautifulsoup', 'crawl4ai'")
    status: str = Field(default="pending", description="'success', 'failed', 'fallback', 'skipped'")
    latency_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    char_count: int = Field(default=0, description="Character count extracted")
    word_count: int = Field(default=0, description="Word count extracted")
    quality_score: int = Field(default=0, description="Quality score (0-100)")
    reason: Optional[str] = Field(default=None, description="Reason for selection or fallback")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class QualityReport(BaseModel):
    """Detailed quality assessment report for an extracted document."""

    model_config = ConfigDict(populate_by_name=True)

    score: int = Field(default=0, ge=0, le=100, description="Overall quality score from 0 to 100")
    char_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    token_count_approx: int = Field(default=0, ge=0)
    text_density: float = Field(default=0.0, ge=0.0, le=1.0, description="Useful text to HTML ratio")
    heading_count: int = Field(default=0, ge=0)
    paragraph_count: int = Field(default=0, ge=0)
    has_title: bool = Field(default=False)
    has_meta_description: bool = Field(default=False)
    internal_link_count: int = Field(default=0, ge=0)
    duplicate_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    noise_blocks_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    passed: bool = Field(default=True)


class ExtractedDocument(BaseModel):
    """Core Pydantic model for an extracted web document.

    Encapsulates raw text, markdown content, structured metadata, links,
    headings, and extraction quality metrics.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    document_id: str = Field(
        default_factory=lambda: hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16],
        description="Stable unique document identifier.",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Final canonical or target URL of the source page.",
    )

    url: str = Field(description="Original URL where document was fetched.")
    final_url: Optional[str] = Field(
        default=None,
        description="Final destination URL after resolving redirects.",
    )
    canonical_url: Optional[str] = Field(
        default=None,
        description="Canonical URL if specified by document or rel=canonical tag.",
    )
    title: Optional[str] = Field(default=None, description="Extracted HTML page title.")
    description: Optional[str] = Field(
        default=None,
        description="Extracted meta description or abstract.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected or declared document language code (e.g. 'en', 'fa').",
    )
    headings: list[Heading] = Field(
        default_factory=list,
        description="List of extracted hierarchical headings (H1-H6).",
    )
    content_markdown: Optional[str] = Field(
        default=None,
        description="Main article/page content formatted in Markdown.",
    )
    content_text: Optional[str] = Field(
        default=None,
        description="Cleaned plain-text content without formatting tags.",
    )
    links: list[LinkInfo] = Field(
        default_factory=list,
        description="Collection of outbound or discovered hyperlinks within the content.",
    )
    http_status: Optional[int] = Field(
        default=None,
        ge=100,
        le=599,
        description="HTTP response status code received during fetch.",
    )
    fetched_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp indicating when the document was downloaded.",
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 fingerprint hash of the extracted textual content.",
    )
    word_count: int = Field(
        default=0,
        ge=0,
        description="Total word count of the primary extracted text.",
    )
    character_count: int = Field(
        default=0,
        ge=0,
        description="Total character count of the primary extracted text.",
    )
    extraction_method: str = Field(
        default="unknown",
        description="Extraction library or strategy utilized (e.g. 'trafilatura', 'bs4').",
    )
    quality_status: QualityStatus = Field(
        default=QualityStatus.SUCCESS,
        description="Assessment status of the extraction result quality.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of warnings or alerts recorded during parsing/extraction.",
    )
    page_type: str = Field(
        default="unknown",
        description="Classified page category (article, docs, policy, product, faq, about, contact, spa, dashboard, file, unknown).",
    )
    extractor_used: str = Field(
        default="unknown",
        description="Final extraction engine utilized ('trafilatura', 'beautifulsoup', 'crawl4ai').",
    )
    extraction_attempts: list[ExtractionAttempt] = Field(
        default_factory=list,
        description="History of extraction attempts and fallback decisions.",
    )
    quality_report: Optional[QualityReport] = Field(
        default=None,
        description="Detailed quality report with metric scores and explanations.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured page metadata, OpenGraph, JSON-LD, and HTTP headers.",
    )
    chunks: list[Any] = Field(
        default_factory=list,
        description="RAG semantic chunks produced from this document.",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL scheme and structure."""
        if not isinstance(v, str):
            raise ValueError("URL must be a string")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("URL cannot be empty")
        parsed = urlsplit(cleaned)
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(
                f"Invalid URL scheme '{parsed.scheme}': only http and https are supported"
            )
        if not parsed.netloc:
            raise ValueError(f"Invalid URL '{cleaned}': missing network domain/host")
        return cleaned

    @field_validator("canonical_url")
    @classmethod
    def validate_canonical_url(cls, v: Optional[str]) -> Optional[str]:
        """Normalize canonical URL, stripping whitespace and treating empty as None."""
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None

    @model_validator(mode="after")
    def compute_metrics_and_defaults(self) -> ExtractedDocument:
        """Calculate word count, character count, content hash, and timestamp defaults."""
        # Calculate character and word count if not explicitly set
        if self.content_text:
            if self.character_count == 0:
                self.character_count = len(self.content_text)
            if self.word_count == 0:
                self.word_count = len(self.content_text.split())
        elif self.content_markdown:
            if self.character_count == 0:
                self.character_count = len(self.content_markdown)
            if self.word_count == 0:
                self.word_count = len(self.content_markdown.split())

        # Auto-compute SHA-256 content hash if not provided
        if self.content_hash is None:
            raw_content = self.content_text or self.content_markdown or ""
            if raw_content.strip():
                self.content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

        # Set default fetch timestamp to current UTC if not specified
        if self.fetched_at is None:
            self.fetched_at = datetime.now(timezone.utc)

        # Set source_url default
        if self.source_url is None:
            self.source_url = self.final_url or self.url

        # Sync extractor_used and extraction_method
        if self.extractor_used == "unknown" and self.extraction_method != "unknown":
            self.extractor_used = self.extraction_method
        elif self.extraction_method == "unknown" and self.extractor_used != "unknown":
            self.extraction_method = self.extractor_used

        return self

    @computed_field
    @property
    def status(self) -> str:
        """Quality status string value for frontend and API compatibility."""
        return self.quality_status.value

    @computed_field
    @property
    def quality(self) -> str:
        """Quality string representation for UI badge rendering."""
        return self.quality_status.value

    @computed_field
    @property
    def crawl_timestamp(self) -> Optional[str]:
        """ISO 8601 crawl timestamp string for RAG ingestion pipelines."""
        return self.fetched_at.isoformat() if self.fetched_at else None

    @property
    def is_success(self) -> bool:
        """Return True if document was successfully extracted with acceptable quality."""
        return self.quality_status == QualityStatus.SUCCESS

    @property
    def has_content(self) -> bool:
        """Return True if document contains non-empty text or markdown."""
        return bool(
            (self.content_text and self.content_text.strip())
            or (self.content_markdown and self.content_markdown.strip())
        )
