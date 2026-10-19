"""Request schemas for Crawlbot.

Defines Pydantic models for website discovery, content extraction,
and data export endpoints, including URL structure validation, pattern compilation checks,
and domain resolution.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def validate_url(url: str) -> str:
    """Validate and sanitize an HTTP or HTTPS URL string.

    Args:
        url: Raw URL string.

    Returns:
        Stripped and normalized URL string.

    Raises:
        ValueError: If the URL is not a valid HTTP/HTTPS URL with host.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("URL cannot be empty")

    parsed = urlsplit(cleaned)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"Invalid URL scheme '{parsed.scheme}': only 'http' and 'https' are supported."
        )
    if not parsed.netloc or not parsed.hostname:
        raise ValueError(f"Invalid URL '{cleaned}': missing network domain/host.")

    return cleaned


class ExtractionStrategy(str, Enum):
    """Supported content extraction engines and modes."""

    TRAFILATURA = "trafilatura"
    BEAUTIFUL_SOUP = "beautifulsoup"
    BS4 = "bs4"  # Alias for backward compatibility
    CRAWL4AI = "crawl4ai"
    AUTO = "auto"
    COMPARE = "compare"


class CrawlMode(str, Enum):
    """Crawling and capture execution modes."""

    SINGLE_PAGE = "single_page"
    SITE_CRAWL = "site_crawl"
    SPA_CAPTURE = "spa_capture"
    COMPARE = "compare"


class ExportFormat(str, Enum):
    """Supported file formats for document export."""

    JSON = "json"
    JSONL = "jsonl"
    MARKDOWN = "markdown"
    TEXT = "text"


class DiscoveryRequest(BaseModel):
    """Request payload for starting a website discovery crawl."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    root_url: str = Field(description="Starting root URL for the website discovery crawl.")
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum crawl depth relative to the root URL.",
    )
    max_pages: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum number of pages to discover before terminating.",
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="HTTP request timeout in seconds.",
    )
    max_concurrency: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent HTTP requests during crawling.",
    )
    delay_between_requests: float = Field(
        default=1.0,
        ge=0.0,
        le=30.0,
        description="Polite delay between consecutive requests in seconds.",
    )
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="List of allowed domain names. If empty, auto-derived from root_url.",
    )
    include_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns; only URLs matching these patterns will be crawled.",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns; URLs matching these patterns will be skipped.",
    )
    allow_subdomains: bool = Field(
        default=False,
        description="Whether to follow links pointing to subdomains of allowed domains.",
    )
    keep_query_params: bool = Field(
        default=False,
        description="Whether to preserve URL query parameters when normalizing URLs.",
    )
    use_sitemap: bool = Field(
        default=True,
        description="Whether to attempt discovery using the site's robots.txt / sitemap.xml.",
    )
    path_prefix: Optional[str] = Field(
        default=None,
        description="Optional URL path prefix restriction (e.g. '/docs' or '/blog').",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        """Support field name aliases commonly passed by frontend clients."""
        if isinstance(data, dict):
            if "timeout" in data and "request_timeout" not in data:
                data["request_timeout"] = data["timeout"]
            if "concurrency" in data and "max_concurrency" not in data:
                data["max_concurrency"] = data["concurrency"]
            if "delay" in data and "delay_between_requests" not in data:
                data["delay_between_requests"] = data["delay"]
        return data

    @field_validator("root_url")
    @classmethod
    def validate_root_url(cls, v: str) -> str:
        """Validate that root_url is a well-formed HTTP/HTTPS URL."""
        return validate_url(v)

    @field_validator("allowed_domains", mode="before")
    @classmethod
    def normalize_allowed_domains(cls, v: Any) -> list[str]:
        """Normalize domain list: strip spaces, lowercase, and extract hostname if URL provided."""
        if not v:
            return []
        cleaned: list[str] = []
        for item in v:
            if isinstance(item, str):
                d = item.strip().lower()
                if not d:
                    continue
                if "://" in d:
                    parsed = urlsplit(d)
                    d = parsed.hostname or d
                d = d.split("/")[0].split(":")[0]
                if d and d not in cleaned:
                    cleaned.append(d)
        return cleaned

    @field_validator("path_prefix")
    @classmethod
    def normalize_path_prefix(cls, v: Optional[str]) -> Optional[str]:
        """Normalize URL path prefix to start with a forward slash."""
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned or cleaned == "/":
            return None
        if not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"
        return cleaned

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def validate_regex_patterns(cls, patterns: list[str]) -> list[str]:
        """Ensure all provided patterns are valid compilable regular expressions."""
        cleaned: list[str] = []
        for pattern in patterns:
            pattern_str = pattern.strip()
            if not pattern_str:
                continue
            try:
                re.compile(pattern_str)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern_str}': {e}")
            cleaned.append(pattern_str)
        return cleaned

    @model_validator(mode="after")
    def auto_derive_allowed_domains(self) -> DiscoveryRequest:
        """Auto-populate allowed_domains from root_url hostname if not explicitly provided."""
        if not self.allowed_domains and self.root_url:
            parsed = urlsplit(self.root_url)
            if parsed.hostname:
                self.allowed_domains = [parsed.hostname.lower()]
        return self


class ExtractionRequest(BaseModel):
    """Request payload for extracting structured content from a list of URLs."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    urls: list[str] = Field(
        min_length=1,
        description="List of target URLs to extract content from (at least 1 required).",
    )
    extraction_strategy: str = Field(
        default=ExtractionStrategy.TRAFILATURA.value,
        description="Extraction engine strategy: 'auto', 'trafilatura', 'beautifulsoup', 'crawl4ai', 'compare'.",
    )
    crawl_mode: str = Field(
        default=CrawlMode.SINGLE_PAGE.value,
        description="Crawl mode: 'single_page', 'site_crawl', 'spa_capture', 'compare'.",
    )
    render_mode: str = Field(
        default="auto",
        description="Browser rendering mode: 'auto', 'never', 'always'.",
    )
    min_quality_score: int = Field(
        default=40,
        ge=0,
        le=100,
        description="Minimum acceptable quality score before fallback triggering.",
    )
    llm_cleanup: bool = Field(
        default=False,
        description="Whether to run AI structuring & cleanup after extraction.",
    )
    chunking_mode: Optional[str] = Field(
        default=None,
        description="Chunking strategy to apply ('semantic', 'sentence', 'paragraph', 'fixed').",
    )
    include_links: bool = Field(
        default=True,
        description="Whether to extract hyperlinks found within the document body.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional tag/category to classify the extracted documents.",
    )
    max_concurrency: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrency for extraction workers.",
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Request timeout per page in seconds.",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        """Support field name aliases commonly passed by frontend clients."""
        if isinstance(data, dict):
            if "extractor" in data and "extraction_strategy" not in data:
                data["extraction_strategy"] = data["extractor"]
            if "strategy" in data and "extraction_strategy" not in data:
                data["extraction_strategy"] = data["strategy"]
            if "mode" in data and "crawl_mode" not in data:
                data["crawl_mode"] = data["mode"]
            if "concurrency" in data and "max_concurrency" not in data:
                data["max_concurrency"] = data["concurrency"]
            if "timeout" in data and "request_timeout" not in data:
                data["request_timeout"] = data["timeout"]
        return data

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, urls: list[str]) -> list[str]:
        """Validate and deduplicate URL list, ensuring valid HTTP/HTTPS schemes."""
        if not urls:
            raise ValueError("URL list cannot be empty")
        validated: list[str] = []
        seen: set[str] = set()
        for u in urls:
            valid_url = validate_url(u)
            if valid_url not in seen:
                seen.add(valid_url)
                validated.append(valid_url)
        if not validated:
            raise ValueError("At least one valid URL must be provided")
        return validated

    @field_validator("extraction_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate extraction strategy against allowed engine types."""
        cleaned = v.strip().lower()
        allowed = {s.value for s in ExtractionStrategy}
        if cleaned not in allowed:
            valid_vals = ", ".join(f"'{s}'" for s in sorted(allowed))
            raise ValueError(
                f"Invalid extraction_strategy '{v}'. Allowed strategies are: {valid_vals}"
            )
        return cleaned


class ExportRequest(BaseModel):
    """Request payload to export extracted documents of a given job into a file."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    job_id: str = Field(
        min_length=1,
        description="Unique identifier of the extraction or crawl job to export.",
    )
    format: str = Field(
        default=ExportFormat.JSON.value,
        description="Target export format: 'json', 'jsonl', 'markdown', or 'text'.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional filter by document category.",
    )

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """Ensure job_id is non-empty after stripping whitespace."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("job_id cannot be empty")
        return cleaned

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate and normalize export format to lowercase."""
        cleaned = v.strip().lower()
        allowed = {f.value for f in ExportFormat}
        if cleaned not in allowed:
            valid_vals = ", ".join(f"'{f}'" for f in sorted(allowed))
            raise ValueError(
                f"Invalid export format '{v}'. Allowed formats are: {valid_vals}"
            )
        return cleaned


class SingleExtractRequest(BaseModel):
    """Request payload for extracting a single URL on-demand."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    url: str = Field(description="URL to extract.")
    extraction_strategy: str = Field(
        default="auto",
        description="Engine strategy: 'auto', 'trafilatura', 'beautifulsoup', 'crawl4ai', 'compare'.",
    )
    crawl_mode: str = Field(
        default="single_page",
        description="Mode: 'single_page', 'spa_capture', 'compare'.",
    )
    render_mode: str = Field(
        default="auto",
        description="Browser render mode: 'auto', 'never', 'always'.",
    )
    min_quality_score: int = Field(default=40, ge=0, le=100)
    timeout: float = Field(default=30.0, ge=1.0, le=120.0)

    @field_validator("url")
    @classmethod
    def check_url(cls, v: str) -> str:
        return validate_url(v)


class CompareRequest(BaseModel):
    """Request payload for comparing all 3 extraction engines on a single URL."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    url: str = Field(description="URL to benchmark across Trafilatura, BeautifulSoup, and Crawl4AI.")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0)

    @field_validator("url")
    @classmethod
    def check_url(cls, v: str) -> str:
        return validate_url(v)
