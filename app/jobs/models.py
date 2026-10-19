"""Data models for background crawling and extraction jobs.

Defines lifecycle enumerations (JobType, JobState) and the primary JobInfo model
used by the JobManager to track execution progress, metrics, and results.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


class JobType(str, Enum):
    """Types of asynchronous jobs supported by Crawlbot."""

    DISCOVERY = "discovery"
    EXTRACTION = "extraction"


class JobState(str, Enum):
    """Lifecycle states of background jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JobInfo(BaseModel):
    """Comprehensive state tracking model for a crawler or extraction background job."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID identifier for this job.",
    )
    job_type: JobType = Field(
        description="Type of operation performed by this job (discovery or extraction).",
    )
    state: JobState = Field(
        default=JobState.PENDING,
        description="Current lifecycle state of the job.",
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
    pages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Discovered pages list (url, title, status, depth, source, error).",
    )
    documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted document records serialized as dictionaries.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages accumulated during job execution.",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp indicating when job execution began.",
    )
    finished_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp indicating when job execution completed or terminated.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters and options used to launch this job.",
    )

    @computed_field
    @property
    def progress_percent(self) -> float:
        """Calculate execution completion percentage (0.0 - 100.0)."""
        if self.total == 0:
            return 0.0
        return round(min(100.0, (self.completed + self.failed) / self.total * 100), 1)

    @computed_field
    @property
    def status(self) -> str:
        """Alias returning the string representation of state for API compatibility."""
        return self.state.value

    @property
    def is_finished(self) -> bool:
        """Return True if job is completed, cancelled, or failed."""
        return self.state in {JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED}

    def mark_running(self) -> None:
        """Transition job state to RUNNING and record started_at timestamp."""
        self.state = JobState.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        """Transition job state to COMPLETED and record finished_at timestamp."""
        self.state = JobState.COMPLETED
        self.finished_at = datetime.now(timezone.utc)

    def mark_cancelled(self) -> None:
        """Transition job state to CANCELLED and record finished_at timestamp."""
        self.state = JobState.CANCELLED
        self.finished_at = datetime.now(timezone.utc)

    def mark_failed(self, error: Optional[str] = None) -> None:
        """Transition job state to FAILED and record error message."""
        self.state = JobState.FAILED
        self.finished_at = datetime.now(timezone.utc)
        if error:
            self.errors.append(error)

    def add_error(self, message: str) -> None:
        """Append an error message to the job's error log."""
        if message:
            self.errors.append(message)

    def to_status_dict(self) -> dict[str, Any]:
        """Convert job state into a dictionary matching API response formats."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "job_type": self.job_type.value,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "progress_percent": self.progress_percent,
            "pages": self.pages,
            "documents": self.documents,
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "config": self.config,
        }
