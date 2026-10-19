"""Base definitions and result schemas for extraction engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.schemas.documents import Heading, LinkInfo


@dataclass
class EngineResult:
    """Standardized result produced by any extraction engine."""

    engine: str
    success: bool
    text: Optional[str] = None
    markdown: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    canonical_url: Optional[str] = None
    headings: list[Heading] = field(default_factory=list)
    links: list[LinkInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    char_count: int = 0
    word_count: int = 0
    latency_ms: float = 0.0
    browser_used: bool = False
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        """Check if substantive text or markdown was extracted."""
        return bool((self.text and self.text.strip()) or (self.markdown and self.markdown.strip()))

    def compute_counts(self) -> None:
        """Calculate word and character counts from text or markdown."""
        target = (self.text or "").strip() or (self.markdown or "").strip()
        self.char_count = len(target)
        self.word_count = len(target.split()) if target else 0
