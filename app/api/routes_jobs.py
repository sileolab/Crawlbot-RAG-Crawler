"""API endpoints for managing and inspecting background jobs."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from app.jobs.manager import JobManager
from app.schemas.responses import JobStatus

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

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


@router.get("/list", response_model=list[JobStatus])
async def list_jobs() -> list[JobStatus]:
    """List all registered jobs."""
    manager = get_manager()
    jobs = manager.list_jobs()
    return [manager.to_job_status_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Get status and details of a specific job."""
    manager = get_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="کار مورد نظر پیدا نشد.")

    return manager.to_job_status_response(job)
