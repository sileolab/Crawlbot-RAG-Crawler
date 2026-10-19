"""API endpoints for website discovery."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.jobs.manager import JobManager
from app.schemas.requests import DiscoveryRequest
from app.schemas.responses import ApiResponse, JobStatus

router = APIRouter(prefix="/api/discovery", tags=["discovery"])

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
async def start_discovery(request: DiscoveryRequest, background_tasks: BackgroundTasks) -> ApiResponse:
    """Start a new discovery job in the background."""
    manager = get_manager()

    config = request.model_dump()
    job = manager.create_discovery_job(config)

    # Launch background task
    background_tasks.add_task(manager.run_discovery, job.job_id)

    return ApiResponse.ok(
        data={"job_id": job.job_id},
        message="فرآیند کشف وب‌سایت با موفقیت آغاز شد.",
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_discovery_status(job_id: str) -> JobStatus:
    """Get the live status and discovered pages of a discovery job."""
    manager = get_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="شناسه کار پیدا نشد.")

    return manager.to_job_status_response(job)


@router.post("/cancel/{job_id}", response_model=ApiResponse)
async def cancel_discovery(job_id: str) -> ApiResponse:
    """Cancel a running discovery job."""
    manager = get_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="شناسه کار پیدا نشد.")

    if job.is_finished:
        return ApiResponse.ok(message="فرآیند کشف از قبل به پایان رسیده است.")

    manager.cancel_job(job_id)
    return ApiResponse.ok(message="دستور لغو فرآیند کشف صادر شد.")
