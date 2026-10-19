"""API endpoints for content extraction."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.jobs.manager import JobManager
from app.schemas.requests import CompareRequest, ExtractionRequest, SingleExtractRequest
from app.schemas.responses import ApiResponse, JobStatus

router = APIRouter(prefix="/api/extraction", tags=["extraction"])

job_manager: Optional[JobManager] = None


def set_job_manager(manager: JobManager) -> None:
    """Inject JobManager dependency."""
    global job_manager
    job_manager = manager


def get_manager() -> JobManager:
    """Ensure JobManager is initialized."""
    if job_manager is None:
        raise HTTPException(status_code=500, detail="Job manager not initialized")
    return job_manager


@router.post("/start", response_model=ApiResponse)
async def start_extraction(request: ExtractionRequest, background_tasks: BackgroundTasks) -> ApiResponse:
    """Start extraction for selected URLs in the background."""
    manager = get_manager()

    config = request.model_dump()
    job = manager.create_extraction_job(urls=request.urls, config=config)

    background_tasks.add_task(manager.run_extraction, job.job_id)

    return ApiResponse.ok(
        data={"job_id": job.job_id, "total": len(request.urls)},
        message="فرآیند استخراج صفحات با موفقیت آغاز شد.",
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_extraction_status(job_id: str) -> JobStatus:
    """Get extraction job progress and document summaries."""
    manager = get_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="شناسه کار استخراج پیدا نشد.")

    return manager.to_job_status_response(job)


@router.post("/cancel/{job_id}", response_model=ApiResponse)
async def cancel_extraction(job_id: str) -> ApiResponse:
    """Cancel a running extraction job."""
    manager = get_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="شناسه کار پیدا نشد.")

    if job.is_finished:
        return ApiResponse.ok(message="فرآیند استخراج از قبل به پایان رسیده است.")

    manager.cancel_job(job_id)
    return ApiResponse.ok(message="دستور لغو فرآیند استخراج صادر شد.")


@router.post("/retry/{job_id}", response_model=ApiResponse)
async def retry_failed(job_id: str, background_tasks: BackgroundTasks) -> ApiResponse:
    """Retry extraction for URLs that failed or were blocked in a previous job."""
    manager = get_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="کار قبلی پیدا نشد.")

    failed_urls: list[str] = []
    for doc in job.documents:
        if doc.get("status") in {"failed", "blocked"} or doc.get("quality") in {"failed", "blocked"}:
            failed_urls.append(doc.get("url", ""))

    if not failed_urls:
        return ApiResponse.ok(
            data={"job_id": job_id},
            message="هیچ صفحه‌ای با وضعیت خطا برای تلاش مجدد وجود ندارد.",
        )

    new_config = dict(job.config)
    new_job = manager.create_extraction_job(urls=failed_urls, config=new_config)
    background_tasks.add_task(manager.run_extraction, new_job.job_id)

    return ApiResponse.ok(
        data={"job_id": new_job.job_id, "retry_count": len(failed_urls)},
        message=f"فرآیند تلاش مجدد برای {len(failed_urls)} صفحه آغاز گردید.",
    )


@router.post("/single", response_model=ApiResponse)
async def extract_single_url(request: SingleExtractRequest) -> ApiResponse:
    """Extract content from a single URL synchronously using specified engine or auto mode."""
    from app.services.extractor import ContentExtractor

    extractor = ContentExtractor(
        strategy=request.extraction_strategy,
        min_quality_score=request.min_quality_score,
        render_mode=request.render_mode,
    )
    doc = await extractor.extract(
        url=request.url,
        timeout=request.timeout,
        crawl_mode=request.crawl_mode,
    )
    return ApiResponse.ok(
        data=doc.model_dump(mode="json"),
        message="استخراج صفحه با موفقیت انجام شد.",
    )


@router.post("/compare", response_model=ApiResponse)
async def compare_engines(request: CompareRequest) -> ApiResponse:
    """Benchmark all 3 extraction engines (Trafilatura, BeautifulSoup, Crawl4AI) on a single URL."""
    from app.services.extractor import ContentExtractor

    extractor = ContentExtractor()
    doc = await extractor.extract(
        url=request.url,
        timeout=request.timeout,
        crawl_mode="compare",
    )
    return ApiResponse.ok(
        data=doc.model_dump(mode="json"),
        message="مقایسه هر ۳ موتور استخراج با موفقیت انجام شد.",
    )
