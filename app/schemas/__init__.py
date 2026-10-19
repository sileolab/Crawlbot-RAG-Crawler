"""Schemas package for Crawlbot.

Provides core Pydantic data models for extracted documents, API requests,
and structured API responses.
"""

from .documents import (
    ExtractedDocument,
    Heading,
    LinkInfo,
    QualityStatus,
)
from .requests import (
    DiscoveryRequest,
    ExportFormat,
    ExportRequest,
    ExtractionRequest,
    ExtractionStrategy,
)
from .responses import (
    ApiResponse,
    DiscoveredPage,
    ExportResponse,
    JobStatus,
    JobStatusCode,
    JobType,
    PageSource,
    PageStatus,
)

__all__ = [
    "ApiResponse",
    "DiscoveredPage",
    "DiscoveryRequest",
    "ExportFormat",
    "ExportRequest",
    "ExportResponse",
    "ExtractedDocument",
    "ExtractionRequest",
    "ExtractionStrategy",
    "Heading",
    "JobStatus",
    "JobStatusCode",
    "JobType",
    "LinkInfo",
    "PageSource",
    "PageStatus",
    "QualityStatus",
]
