"""Universal Website Crawler & RAG Data Extraction Studio - Main Application.

Exposes REST API endpoints and serves the local RTL-friendly GUI.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_discovery, routes_exports, routes_extraction, routes_jobs, routes_ai, routes_files
from app.core.config import get_settings
from app.core.logging_config import get_logger, setup_logging
from app.jobs.manager import JobManager

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Application factory for Crawlbot."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description="سامانه هوشمند کشف، پاک‌سازی و استخراج داده‌های وب جهت سامانه‌های RAG",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Allow local CORS for testing and development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize shared JobManager singleton
    manager = JobManager()

    # Inject dependency into router modules
    routes_discovery.set_job_manager(manager)
    routes_extraction.set_job_manager(manager)
    routes_jobs.set_job_manager(manager)
    routes_exports.set_job_manager(manager)
    routes_ai.set_job_manager(manager)

    # Register routers
    app.include_router(routes_discovery.router)
    app.include_router(routes_extraction.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_exports.router)
    app.include_router(routes_ai.router)
    app.include_router(routes_files.router)

    # Static files setup
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the local GUI."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Crawlbot API is running, but GUI index.html was not found."}

    @app.get("/health", tags=["system"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "app": settings.app_name, "env": settings.app_env}

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
