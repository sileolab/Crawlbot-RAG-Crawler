"""In-memory job management service for Crawlbot.

Coordinates background execution of discovery and extraction tasks, manages
lifecycle states, supports real-time progress tracking, cancellation events,
and stores in-memory extracted document artifacts.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.logging_config import get_logger
from app.jobs.models import JobInfo, JobState, JobType
from app.schemas.documents import ExtractedDocument, QualityStatus
from app.schemas.responses import DiscoveredPage, JobStatus
from app.services.crawler import CrawlerService, DiscoveryResult
from app.services.extractor import ContentExtractor
from app.services.quality_checker import QualityChecker

logger = get_logger(__name__)


class JobManager:
    """Manages creation, execution, cancellation, and retrieval of crawl and extraction jobs."""

    def __init__(self) -> None:
        """Initialize the in-memory JobManager state storage."""
        self._jobs: dict[str, JobInfo] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._documents: dict[str, list[ExtractedDocument]] = {}
        self._chunks: dict[str, list] = {}       # job_id -> list[DocumentChunk]
        self._rag_metadata: dict[str, list] = {}  # job_id -> list[dict]


    def create_discovery_job(self, config: dict[str, Any]) -> JobInfo:
        """Create and queue a new website discovery job.

        Args:
            config: Discovery configuration parameters (root_url, max_depth, max_pages, etc.).

        Returns:
            JobInfo instance for the newly registered discovery job.
        """
        root_url = str(config.get("root_url", "")).strip()
        max_pages = int(config.get("max_pages", 500))

        job = JobInfo(
            job_type=JobType.DISCOVERY,
            state=JobState.PENDING,
            total=max_pages,
            config=dict(config),
        )

        self._jobs[job.job_id] = job
        self._cancel_events[job.job_id] = asyncio.Event()
        self._documents[job.job_id] = []

        logger.info("Created discovery job %s for root URL: %s", job.job_id, root_url)

        # Launch background task if loop running and auto_start requested
        try:
            loop = asyncio.get_running_loop()
            if config.get("auto_start", False):
                self._tasks[job.job_id] = loop.create_task(self.run_discovery(job.job_id))
        except RuntimeError:
            pass

        return job

    def create_extraction_job(self, urls: list[str], config: dict[str, Any]) -> JobInfo:
        """Create and queue a new content extraction job.

        Args:
            urls: Target URLs to extract.
            config: Extraction options and configurations.

        Returns:
            JobInfo instance for the newly registered extraction job.
        """
        clean_urls = [str(u).strip() for u in urls if u and isinstance(u, str) and str(u).strip()]
        job_config = dict(config)
        job_config["urls"] = clean_urls

        job = JobInfo(
            job_type=JobType.EXTRACTION,
            state=JobState.PENDING,
            total=len(clean_urls),
            config=job_config,
        )

        self._jobs[job.job_id] = job
        self._cancel_events[job.job_id] = asyncio.Event()
        self._documents[job.job_id] = []

        logger.info("Created extraction job %s for %d URLs", job.job_id, len(clean_urls))

        # Launch background task if loop running and auto_start requested
        try:
            loop = asyncio.get_running_loop()
            if config.get("auto_start", False):
                self._tasks[job.job_id] = loop.create_task(self.run_extraction(job.job_id))
        except RuntimeError:
            pass

        return job

    async def run_discovery(self, job_id: str) -> None:
        """Run discovery job in background.

        Args:
            job_id: ID of the discovery job to execute.
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.error("Discovery job %s not found in manager", job_id)
            return

        cancel_event = self._cancel_events.setdefault(job_id, asyncio.Event())
        if cancel_event.is_set() or job.state == JobState.CANCELLED:
            job.mark_cancelled()
            return

        job.mark_running()
        logger.info("Running discovery job %s", job_id)

        async def on_progress(result: DiscoveryResult) -> None:
            """Update job state as progress updates are emitted by the crawler."""
            job.completed = len(result.crawled)
            job.failed = len(result.failed)
            # Update total to reflect discovered pages count or target ceiling
            job.total = max(len(result.discovered), 1)
            job.pages = [
                {
                    "url": url,
                    "title": meta.get("title"),
                    "status": meta.get("status", 0),
                    "depth": meta.get("depth", 0),
                    "source": meta.get("source", "crawl"),
                    "error": meta.get("error"),
                    "final_url": meta.get("final_url"),
                }
                for url, meta in result.discovered.items()
            ]

        try:
            root_url = str(job.config.get("root_url", "")).strip()
            max_depth = int(job.config.get("max_depth", 3))
            max_pages = int(job.config.get("max_pages", 500))
            request_timeout = float(job.config.get("request_timeout", 30.0))
            max_concurrency = int(job.config.get("max_concurrency", 5))
            delay = float(job.config.get("delay", job.config.get("delay_between_requests", 1.0)))
            allowed_domains = job.config.get("allowed_domains")
            include_patterns = job.config.get("include_patterns")
            exclude_patterns = job.config.get("exclude_patterns")
            allow_subdomains = bool(job.config.get("allow_subdomains", False))
            keep_query_params = bool(job.config.get("keep_query_params", False))
            use_sitemap = bool(job.config.get("use_sitemap", True))
            path_prefix = job.config.get("path_prefix")
            max_response_size = int(job.config.get("max_response_size", 10 * 1024 * 1024))

            crawler = CrawlerService(
                root_url=root_url,
                max_depth=max_depth,
                max_pages=max_pages,
                request_timeout=request_timeout,
                max_concurrency=max_concurrency,
                delay=delay,
                allowed_domains=allowed_domains,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                allow_subdomains=allow_subdomains,
                keep_query_params=keep_query_params,
                use_sitemap=use_sitemap,
                path_prefix=path_prefix,
                max_response_size=max_response_size,
                progress_callback=on_progress,
                cancel_event=cancel_event,
            )

            result = await crawler.discover()

            # Record final outcome
            job.completed = len(result.crawled)
            job.failed = len(result.failed)
            job.total = len(result.discovered)
            job.pages = [
                {
                    "url": url,
                    "title": meta.get("title"),
                    "status": meta.get("status", 0),
                    "depth": meta.get("depth", 0),
                    "source": meta.get("source", "crawl"),
                    "error": meta.get("error"),
                }
                for url, meta in result.discovered.items()
            ]

            if cancel_event.is_set():
                job.mark_cancelled()
                logger.info("Discovery job %s cancelled", job_id)
            else:
                job.mark_completed()
                logger.info("Discovery job %s completed successfully", job_id)

        except asyncio.CancelledError:
            logger.info("Discovery job %s cancelled via task cancellation", job_id)
            job.mark_cancelled()
        except Exception as exc:
            logger.error("Discovery job %s failed with exception: %s", job_id, exc, exc_info=True)
            job.mark_failed(str(exc))
        finally:
            self._cleanup_task(job_id)

    async def run_extraction(self, job_id: str) -> None:
        """Run extraction job in background.

        Args:
            job_id: ID of the extraction job to execute.
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.error("Extraction job %s not found in manager", job_id)
            return

        cancel_event = self._cancel_events.setdefault(job_id, asyncio.Event())
        if cancel_event.is_set() or job.state == JobState.CANCELLED:
            job.mark_cancelled()
            return

        job.mark_running()
        logger.info("Running extraction job %s", job_id)

        urls: list[str] = job.config.get("urls", [])
        strategy = str(job.config.get("extraction_strategy", job.config.get("extractor", "auto")))
        crawl_mode = str(job.config.get("crawl_mode", "single_page"))
        render_mode = str(job.config.get("render_mode", "auto"))
        min_quality_score = int(job.config.get("min_quality_score", 40))
        max_concurrency = int(job.config.get("max_concurrency", 3))
        request_timeout = float(job.config.get("request_timeout", 30.0))
        min_word_count = int(job.config.get("min_word_count", 20))
        min_char_count = int(job.config.get("min_char_count", 100))

        job.total = len(urls)
        job.completed = 0
        job.failed = 0

        extractor = ContentExtractor(
            strategy=strategy,
            min_quality_score=min_quality_score,
            render_mode=render_mode,
        )
        quality_checker = QualityChecker(
            min_word_count=min_word_count,
            min_char_count=min_char_count,
            min_quality_score=min_quality_score,
        )

        semaphore = asyncio.Semaphore(max_concurrency)
        lock = asyncio.Lock()

        async def process_single_url(url: str) -> None:
            if cancel_event.is_set():
                return

            async with semaphore:
                if cancel_event.is_set():
                    return

                try:
                    doc = await extractor.extract(
                        url,
                        timeout=request_timeout,
                        crawl_mode=crawl_mode,
                    )
                    doc = quality_checker.check(doc)

                    async with lock:
                        self._documents.setdefault(job_id, []).append(doc)
                        job.documents.append(doc.model_dump(mode="json"))

                        if doc.quality_status in (QualityStatus.FAILED, QualityStatus.BLOCKED):
                            job.failed += 1
                        else:
                            job.completed += 1

                except Exception as exc:
                    logger.warning("Extraction failed for %s: %s", url, exc)
                    failed_doc = ExtractedDocument(
                        url=url,
                        extraction_method=strategy,
                        quality_status=QualityStatus.FAILED,
                        warnings=[f"Extraction error: {str(exc)}"],
                    )
                    async with lock:
                        self._documents.setdefault(job_id, []).append(failed_doc)
                        job.documents.append(failed_doc.model_dump(mode="json"))
                        job.failed += 1
                        job.add_error(f"{url}: {str(exc)}")

        try:
            tasks = [asyncio.create_task(process_single_url(url)) for url in urls]
            await asyncio.gather(*tasks, return_exceptions=True)

            if cancel_event.is_set():
                job.mark_cancelled()
                logger.info("Extraction job %s cancelled", job_id)
            else:
                job.mark_completed()
                logger.info("Extraction job %s completed successfully", job_id)

        except asyncio.CancelledError:
            logger.info("Extraction job %s cancelled via task cancellation", job_id)
            job.mark_cancelled()
        except Exception as exc:
            logger.error("Extraction job %s failed with exception: %s", job_id, exc, exc_info=True)
            job.mark_failed(str(exc))
        finally:
            self._cleanup_task(job_id)

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job status.

        Args:
            job_id: The job ID to look up.

        Returns:
            JobInfo if found, None otherwise.
        """
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job.

        Args:
            job_id: ID of the job to cancel.

        Returns:
            True if job was cancelled, False if job not found or already terminated.
        """
        event = self._cancel_events.get(job_id)
        if event:
            event.set()

        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.state in (JobState.PENDING, JobState.RUNNING):
            job.mark_cancelled()
            task = self._tasks.get(job_id)
            if task and not task.done():
                task.cancel()
            logger.info("Cancelled job %s", job_id)
            return True

        return False

    def get_documents(self, job_id: str) -> list[ExtractedDocument]:
        """Get extracted documents for a job.

        Args:
            job_id: Target extraction job ID.

        Returns:
            List of ExtractedDocument instances.
        """
        return list(self._documents.get(job_id, []))

    def set_documents(self, job_id: str, documents: list[ExtractedDocument]) -> None:
        """Set or update extracted documents for a job.

        Args:
            job_id: Target extraction job ID.
            documents: List of ExtractedDocument instances.
        """
        self._documents[job_id] = list(documents)

    def list_jobs(self) -> list[JobInfo]:
        """List all tracked jobs.

        Returns:
            List of all JobInfo instances sorted by creation/started time.
        """
        return list(self._jobs.values())

    def to_job_status_response(self, job: JobInfo) -> JobStatus:
        """Convert internal JobInfo to API JobStatus response model."""
        parsed_pages: list[DiscoveredPage] = []
        for p in job.pages:
            raw_status = p.get("status", "discovered")
            if isinstance(raw_status, int):
                if 200 <= raw_status < 300:
                    status_str = "crawled"
                elif raw_status in (401, 403):
                    status_str = "blocked"
                elif raw_status == 311:
                    status_str = "redirect_loop"
                elif raw_status == 310:
                    status_str = "too_many_redirects"
                else:
                    status_str = "failed"
            else:
                status_str = str(raw_status) if raw_status else "discovered"

            parsed_pages.append(
                DiscoveredPage(
                    url=str(p.get("url", "")),
                    final_url=p.get("final_url"),
                    title=p.get("title"),
                    status=status_str,
                    depth=int(p.get("depth", 0)),
                    source=str(p.get("source", "crawl")),
                    error=p.get("error"),
                )
            )

        status_val = job.state.value if hasattr(job.state, "value") else str(job.state)
        type_val = job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type)

        return JobStatus(
            job_id=job.job_id,
            status=status_val,
            job_type=type_val,
            total=job.total,
            completed=job.completed,
            failed=job.failed,
            pages=parsed_pages,
            documents=job.documents,
            errors=job.errors,
            started_at=job.started_at,
            finished_at=job.finished_at,
            progress_percent=job.progress_percent,
        )

    def _cleanup_task(self, job_id: str) -> None:
        """Clean up background task reference on completion."""
        self._tasks.pop(job_id, None)

    # ------------------------------------------------------------------
    # AI Feature: Chunk storage
    # ------------------------------------------------------------------

    def store_chunks(self, job_id: str, chunks: list) -> None:
        """Store document chunks for a job (replaces any previously stored chunks).

        Args:
            job_id: Job identifier.
            chunks: List of DocumentChunk objects.
        """
        self._chunks[job_id] = list(chunks)
        logger.debug("Stored %d chunks for job %s.", len(chunks), job_id)

    def get_chunks(self, job_id: str) -> list:
        """Retrieve stored chunks for a job.

        Args:
            job_id: Job identifier.

        Returns:
            List of DocumentChunk objects, or empty list if none stored.
        """
        return self._chunks.get(job_id, [])

    # ------------------------------------------------------------------
    # AI Feature: RAG metadata storage
    # ------------------------------------------------------------------

    def store_rag_metadata(self, job_id: str, metadata: list) -> None:
        """Store RAG enhancement metadata for a job.

        Args:
            job_id: Job identifier.
            metadata: List of enrichment dicts from RAGEnhancer.
        """
        self._rag_metadata[job_id] = list(metadata)
        logger.debug("Stored RAG metadata for %d docs in job %s.", len(metadata), job_id)

    def get_rag_metadata(self, job_id: str) -> list:
        """Retrieve stored RAG metadata for a job.

        Args:
            job_id: Job identifier.

        Returns:
            List of enrichment dicts, or empty list if none stored.
        """
        return self._rag_metadata.get(job_id, [])



# Global singleton instance
_global_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Retrieve or create the global JobManager singleton instance.

    Returns:
        JobManager instance.
    """
    global _global_job_manager
    if _global_job_manager is None:
        _global_job_manager = JobManager()
    return _global_job_manager
