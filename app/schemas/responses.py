"""Response schemas for Crawlbot.

Defines Pydantic response models for discovered pages, background job status tracking,
generic API responses, and export file details.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)


class PageStatus(str, Enum):
    """Lifecycle and crawling status of a discovered page."""

    DISCOVERED = "discovered"
    CRAWLED = "crawled"
    BLOCKED = "blocked"
    FAILED = "failed"


class PageSource(str, Enum):
    """Source origin where a page was identified."""

    SITEMAP = "sitemap"
    CRAWL = "crawl"
    MANUAL = "manual"


class JobStatusCode(str, Enum):
    """Execution status for asynchronous background jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JobType(str, Enum):
    """Type of job executed by the crawler engine."""

    DISCOVERY = "discovery"
    EXTRACTION = "extraction"


class DiscoveredPage(BaseModel):
    """Represents a discovered web page during the crawling phase."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    url: str = Field(description="Normalized URL of the discovered page.")
    final_url: Optional[str] = Field(
        default=None,
        description="Final destination URL after resolving redirects.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Extracted page title if available during discovery.",
    )
    status: str = Field(
        default=PageStatus.DISCOVERED.value,
        description="Current crawl status: 'discovered', 'crawled', 'blocked', 'failed'.",
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Crawl depth level relative to the root URL.",
    )
    source: str = Field(
        default=PageSource.CRAWL.value,
        description="Origin source of discovery: 'sitemap', 'crawl', or 'manual'.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Detailed error message if the page crawl or fetch encountered an issue.",
    )


class JobStatus(BaseModel):
    """Comprehensive status model for monitoring ongoing and completed jobs."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    job_id: str = Field(description="Unique identifier of the job.")
    status: str = Field(
        description="Job state: 'pending', 'running', 'completed', 'cancelled', 'failed'.",
    )
    job_type: str = Field(
        description="Category of job: 'discovery' or 'extraction'.",
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total items to process (pages discovered or URLs queued).",
    )
    completed: int = Field(
        default=0,
        ge=0,
        description="Number of items successfully processed.",
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Number of items that failed during processing.",
    )
    pages: list[DiscoveredPage] = Field(
        default_factory=list,
        description="Discovered pages list (populated for discovery jobs).",
    )
    documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Summary information of extracted documents (populated for extraction jobs).",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of recorded error messages during job execution.",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp indicating when job execution began.",
    )
    finished_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp indicating when job execution terminated or completed.",
    )
    progress_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Execution progress percentage (0.0 to 100.0).",
    )

    @model_validator(mode="after")
    def compute_progress_if_unset(self) -> JobStatus:
        """Automatically calculate progress_percent from completed/failed counts if total is set."""
        if self.total > 0 and self.progress_percent == 0.0:
            self.progress_percent = min(
                100.0,
                round(((self.completed + self.failed) / self.total) * 100.0, 2),
            )
        return self

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Elapsed execution time in seconds, calculated safely across timezone differences."""
        if self.started_at is None:
            return None
        end_time = self.finished_at
        if end_time is None:
            end_time = (
                datetime.now(self.started_at.tzinfo)
                if self.started_at.tzinfo
                else datetime.utcnow()
            )
        elif self.started_at.tzinfo and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=self.started_at.tzinfo)
        elif not self.started_at.tzinfo and end_time.tzinfo:
            end_time = end_time.replace(tzinfo=None)
        return max(0.0, round((end_time - self.started_at).total_seconds(), 2))

    @computed_field
    @property
    def is_active(self) -> bool:
        """Return True if the job is actively pending or running."""
        return self.status.lower() in {JobStatusCode.PENDING.value, JobStatusCode.RUNNING.value}

    @computed_field
    @property
    def is_finished(self) -> bool:
        """Return True if the job has terminated (completed, cancelled, or failed)."""
        return self.status.lower() in {
            JobStatusCode.COMPLETED.value,
            JobStatusCode.CANCELLED.value,
            JobStatusCode.FAILED.value,
        }

    @computed_field
    @property
    def progress(self) -> float:
        """Progress percentage alias for frontend compatibility."""
        return self.progress_percent

    @computed_field
    @property
    def discovered(self) -> int:
        """Discovered pages count alias for frontend compatibility."""
        return len(self.pages) if self.pages else self.total

    @computed_field
    @property
    def total_discovered(self) -> int:
        """Total discovered pages count alias for frontend compatibility."""
        return self.discovered

    @computed_field
    @property
    def crawled(self) -> int:
        """Crawled pages count alias for frontend compatibility."""
        return self.completed

    @computed_field
    @property
    def total_crawled(self) -> int:
        """Total crawled pages count alias for frontend compatibility."""
        return self.completed

    @computed_field
    @property
    def total_failed(self) -> int:
        """Total failed pages count alias for frontend compatibility."""
        return self.failed


class ApiResponse(BaseModel):
    """Standardized API envelope for endpoint responses."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    success: bool = Field(description="Whether the requested API operation succeeded.")
    message: str = Field(default="", description="Human-readable status or error message.")
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional payload data dictionary.",
    )

    @classmethod
    def ok(
        cls,
        data: Optional[dict[str, Any]] = None,
        message: str = "Operation completed successfully",
    ) -> ApiResponse:
        """Convenience factory method for successful responses."""
        return cls(success=True, message=message, data=data)

    @classmethod
    def error(
        cls,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> ApiResponse:
        """Convenience factory method for error responses."""
        return cls(success=False, message=message, data=data)


class ExportResponse(BaseModel):
    """Response returned upon initiating or completing document export."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    success: bool = Field(
        default=True,
        description="Whether the export file was generated successfully.",
    )
    file_path: str = Field(
        default="",
        description="Local filesystem path or download URL for the exported file.",
    )
    file_size: int = Field(
        default=0,
        ge=0,
        description="File size in bytes.",
    )
    document_count: int = Field(
        default=0,
        ge=0,
        description="Total number of documents included in the export.",
    )
    format: str = Field(
        default="",
        description="Export format: 'json', 'jsonl', 'markdown', or 'text'.",
    )
