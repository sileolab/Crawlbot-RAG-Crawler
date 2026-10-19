"""API Endpoints for File & Data Processing Studio.

Supports uploading, processing, transforming, chunking, and downloading
Markdown, Text, JSON, JSONL, and CSV files in Crawlbot.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas.responses import ApiResponse
from app.services.file_processor import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    FileProcessorService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])

_file_service = FileProcessorService()


@router.post("/upload", response_model=ApiResponse)
async def upload_file(file: UploadFile = File(...)) -> ApiResponse:
    """Upload a file (.md, .txt, .json, .jsonl, .csv) for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="نام فایل ارسالی خالی است.")

    try:
        content_bytes = await file.read()
        record = _file_service.upload_file(file.filename, content_bytes)
        logger.info("Uploaded file %s (ID: %s, %d bytes)", record["filename"], record["file_id"], record["size_bytes"])
        return ApiResponse.ok(
            data=record,
            message=f"فایل '{record['filename']}' با موفقیت بارگذاری شد.",
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.exception("Upload failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"خطا در ذخیره‌سازی فایل: {str(exc)}")


@router.post("/process", response_model=ApiResponse)
async def process_file(request: dict[str, Any]) -> ApiResponse:
    """Execute selected processing pipeline on an uploaded file.

    Body parameters:
        file_id (str): Unique file identifier from upload.
        operations (list[str]): List of operations to apply.
        options (dict, optional): Configuration options.
    """
    file_id = request.get("file_id", "").strip()
    if not file_id:
        raise HTTPException(status_code=400, detail="شناسه فایل (file_id) الزامی است.")

    operations = request.get("operations", [])
    if not operations or not isinstance(operations, list):
        raise HTTPException(status_code=400, detail="حداقل یک عملیات پردازش باید انتخاب شود.")

    options = request.get("options", {})

    try:
        result = await _file_service.process(file_id, operations, options)
        logger.info("Processed file %s with operations %s", file_id, operations)
        return ApiResponse.ok(
            data=result,
            message="عملیات پردازش فایل با موفقیت پایان یافت.",
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as exc:
        logger.exception("Processing failed for file %s", file_id)
        raise HTTPException(status_code=500, detail=f"خطا در پردازش فایل: {str(exc)}")


@router.get("/preview/{artifact_id}", response_model=ApiResponse)
async def preview_artifact(artifact_id: str) -> ApiResponse:
    """Retrieve metadata and preview for a processed artifact."""
    artifact = _file_service.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="نتیجه پردازش (artifact) یافت نشد.")

    path = Path(artifact["stored_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="فایل فیزیکی خروجی یافت نشد.")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content_preview = f.read(3000)

    return ApiResponse.ok(
        data={**artifact, "content_preview": content_preview},
        message="پیش‌نمایش خروجی دریافت شد.",
    )


@router.get("/download/{artifact_id}")
async def download_artifact(artifact_id: str):
    """Download the processed output file."""
    artifact = _file_service.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="نتیجه پردازش جهت دانلود یافت نشد.")

    path = Path(artifact["stored_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="فایل فیزیکی خروجی وجود ندارد.")

    content_type, _ = mimetypes.guess_type(artifact["filename"])
    content_type = content_type or "application/octet-stream"

    return FileResponse(
        path=str(path),
        filename=artifact["filename"],
        media_type=content_type,
    )


@router.get("/info/{file_id}", response_model=ApiResponse)
async def file_info(file_id: str) -> ApiResponse:
    """Retrieve metadata for an uploaded file."""
    info = _file_service.get_file(file_id)
    if not info:
        raise HTTPException(status_code=404, detail="فایل مورد نظر یافت نشد.")
    return ApiResponse.ok(data=info, message="اطلاعات فایل دریافت شد.")
