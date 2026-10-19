"""API endpoints for AI-powered features: Chunking, Embedding, and RAG Enhancement.

All endpoints are optional — they activate only when CRAWLBOT_OPENAI_API_KEY is set.
Chunking works without an API key; Embedding and RAG Enhancement require one.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.jobs.manager import JobManager
from app.schemas.responses import ApiResponse
from app.services.chunker import ChunkerService
from app.services.embedding_service import EmbeddingService
from app.services.rag_enhancer import RAGEnhancer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai"])

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


def _require_api_key() -> str:
    """Return the OpenAI API key or raise HTTP 400 if not configured."""
    settings = get_settings()
    key = settings.openai_api_key.strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail=(
                "کلید API هوش مصنوعی (CRAWLBOT_OPENAI_API_KEY) در فایل .env تنظیم نشده است. "
                "لطفاً کلید OpenAI خود را وارد و برنامه را مجدداً اجرا کنید."
            ),
        )
    return key


# ---------------------------------------------------------------------------
# GET /api/ai/status
# ---------------------------------------------------------------------------

@router.get("/status", response_model=ApiResponse)
async def get_ai_status() -> ApiResponse:
    """Return AI service availability and configuration status."""
    settings = get_settings()
    api_key_set = bool(settings.openai_api_key.strip())
    return ApiResponse.ok(
        data={
            "chunking_available": True,
            "embedding_available": api_key_set,
            "rag_enhancement_available": api_key_set,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "api_key_configured": api_key_set,
            "base_url": settings.openai_base_url or None,
        },
        message="وضعیت سرویس‌های هوش مصنوعی",
    )


# ---------------------------------------------------------------------------
# POST /api/ai/chunk
# ---------------------------------------------------------------------------

@router.post("/chunk", response_model=ApiResponse)
async def chunk_documents(request: dict[str, Any]) -> ApiResponse:
    """Chunk the extracted documents of a job.

    Request body:
        job_id (str): Extraction job identifier.
        strategy (str): One of paragraph|sentence|fixed_size|semantic_heading.
        chunk_size (int): Target chunk size in characters (default 1000).
        overlap (int): Overlap in characters (default 150).
        category (str|null): Optional document category tag.
    """
    manager = get_manager()

    job_id = request.get("job_id", "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id الزامی است.")

    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="کار (job) با این شناسه پیدا نشد.")

    documents = manager.get_documents(job_id)
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="این کار هنوز سند استخراج‌شده‌ای ندارد. ابتدا عملیات استخراج را انجام دهید.",
        )

    strategy = str(request.get("strategy", "auto")).strip().lower()
    try:
        chunk_size = int(request.get("chunk_size", 1000))
        overlap = int(request.get("overlap", 150))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="مقادیر chunk_size و overlap باید اعداد صحیح معتبر باشند.",
        )
    category = request.get("category") or None

    # Validate strategy
    allowed = {"auto", "paragraph", "sentence", "fixed_size", "semantic_heading"}
    if strategy not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"استراتژی '{strategy}' نامعتبر است. موارد مجاز: {', '.join(sorted(allowed))}",
        )

    # Validate chunk size bounds
    if chunk_size < 100 or chunk_size > 8000:
        raise HTTPException(
            status_code=400,
            detail=f"اندازه قطعه (chunk_size) باید بین ۱۰۰ تا ۸۰۰۰ کاراکتر باشد. مقدار ارسالی: {chunk_size}",
        )

    # Validate overlap bounds
    if overlap < 0 or overlap > 500:
        raise HTTPException(
            status_code=400,
            detail=f"میزان همپوشانی (overlap) باید بین ۰ تا ۵۰۰ کاراکتر باشد. مقدار ارسالی: {overlap}",
        )

    if overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail=f"میزان همپوشانی ({overlap}) نمی‌تواند بزرگتر یا مساوی اندازه قطعه ({chunk_size}) باشد.",
        )

    try:
        normalize = bool(request.get("normalize", False))
        chunker = ChunkerService(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            normalize=normalize,
        )
        # Filter only successfully extracted docs with content
        eligible = [d for d in documents if d.has_content]
        if not eligible:
            raise HTTPException(
                status_code=400,
                detail="هیچ سند موفق و دارای محتوایی برای تقطیع پیدا نشد.",
            )

        all_chunks = chunker.chunk_documents(eligible, category=category)

        # Store chunks on the job for later embedding/export
        manager.store_chunks(job_id, all_chunks)

        applied_set = list({c.metadata.get("strategy") for c in all_chunks if c.metadata.get("strategy")})
        applied_strategy = applied_set[0] if len(applied_set) == 1 else (", ".join(applied_set) if applied_set else strategy)

        chunk_stats = {
            "total_chunks": len(all_chunks),
            "total_docs_chunked": len(eligible),
            "strategy": strategy,
            "applied_strategy": applied_strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "avg_chunk_chars": (
                int(sum(c.char_count for c in all_chunks) / len(all_chunks))
                if all_chunks else 0
            ),
        }
        logger.info(
            "Chunked job %s: %d chunks from %d docs (requested=%s, applied=%s).",
            job_id,
            len(all_chunks),
            len(eligible),
            strategy,
            applied_strategy,
        )

        return ApiResponse.ok(
            data={"job_id": job_id, **chunk_stats},
            message=f"تقطیع با موفقیت انجام شد. {len(all_chunks)} قطعه از {len(eligible)} سند تولید شد (استراتژی: {applied_strategy}).",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chunking failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"خطا در عملیات تقطیع: {str(exc)}")


# ---------------------------------------------------------------------------
# POST /api/ai/embed
# ---------------------------------------------------------------------------

@router.post("/embed", response_model=ApiResponse)
async def embed_chunks(request: dict[str, Any]) -> ApiResponse:
    """Generate embeddings for the chunks of a job using OpenAI.

    Requires CRAWLBOT_OPENAI_API_KEY to be set.

    Request body:
        job_id (str): Job identifier (must have chunks stored first via /chunk).
    """
    api_key = _require_api_key()
    manager = get_manager()
    settings = get_settings()

    job_id = request.get("job_id", "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id الزامی است.")

    chunks = manager.get_chunks(job_id)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                "این کار هنوز قطعه‌ای ندارد. "
                "ابتدا عملیات تقطیع (/api/ai/chunk) را انجام دهید."
            ),
        )

    try:
        embedder = EmbeddingService(
            api_key=api_key,
            base_url=settings.openai_base_url or None,
            model=settings.embedding_model,
        )
        stats = await embedder.embed_chunks(chunks)

        # Persist updated chunks (with embeddings) back into manager
        manager.store_chunks(job_id, chunks)

        return ApiResponse.ok(
            data={
                "job_id": job_id,
                "total_chunks": stats.total_chunks,
                "embedded_chunks": stats.embedded_chunks,
                "failed_chunks": stats.failed_chunks,
                "model": stats.model,
                "total_tokens": stats.total_tokens,
            },
            message=f"جاسازی بردار با موفقیت انجام شد: {stats.embedded_chunks} قطعه از {stats.total_chunks}.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Embedding failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"خطا در عملیات جاسازی بردار: {str(exc)}")


# ---------------------------------------------------------------------------
# POST /api/ai/enhance
# ---------------------------------------------------------------------------

@router.post("/enhance", response_model=ApiResponse)
async def enhance_documents(request: dict[str, Any]) -> ApiResponse:
    """Enrich extracted documents with LLM-generated RAG metadata.

    Requires CRAWLBOT_OPENAI_API_KEY to be set.

    Request body:
        job_id (str): Extraction job identifier.
    """
    api_key = _require_api_key()
    manager = get_manager()
    settings = get_settings()

    job_id = request.get("job_id", "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id الزامی است.")

    documents = manager.get_documents(job_id)
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="این کار هنوز سند استخراج‌شده‌ای ندارد. ابتدا عملیات استخراج را انجام دهید.",
        )

    eligible = [d for d in documents if d.has_content]
    if not eligible:
        raise HTTPException(
            status_code=400,
            detail="هیچ سند موفق و دارای محتوایی برای بهبود RAG پیدا نشد.",
        )

    try:
        enhancer = RAGEnhancer(
            api_key=api_key,
            base_url=settings.openai_base_url or None,
            model=settings.llm_model,
            concurrency=settings.ai_concurrency,
        )
        results = await enhancer.enhance_documents(eligible)

        # Store RAG metadata on the job for export
        manager.store_rag_metadata(job_id, results)

        success_count = sum(1 for r in results if r.get("success"))
        failed_count = len(results) - success_count

        return ApiResponse.ok(
            data={
                "job_id": job_id,
                "total_docs": len(eligible),
                "enhanced_docs": success_count,
                "failed_docs": failed_count,
                "model": settings.llm_model,
            },
            message=f"بهبود RAG با موفقیت انجام شد: {success_count} سند از {len(eligible)} پردازش شد.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG enhancement failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"خطا در بهبود RAG: {str(exc)}")


# ---------------------------------------------------------------------------
# POST /api/ai/export
# ---------------------------------------------------------------------------

@router.post("/export")
async def export_ai_data(request: dict[str, Any]):
    """Export chunks and/or RAG-enhanced data for a job.

    Request body:
        job_id (str): Job identifier.
        format (str): 'jsonl_chunks' | 'json_chunks' | 'json_enhanced'
    """
    import io
    import json as json_lib
    from fastapi.responses import Response

    manager = get_manager()

    job_id = request.get("job_id", "").strip()
    fmt = request.get("format", "jsonl_chunks").strip().lower()
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id الزامی است.")

    if fmt in {"jsonl_chunks", "json_chunks"}:
        chunks = manager.get_chunks(job_id)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="قطعه‌ای برای این کار وجود ندارد. ابتدا تقطیع انجام دهید.",
            )
        records = [c.model_dump() for c in chunks]

        if fmt == "jsonl_chunks":
            content = "\n".join(json_lib.dumps(r, ensure_ascii=False) for r in records)
            media_type = "application/x-ndjson"
            filename = f"crawlbot_chunks_{job_id[:8]}.jsonl"
        else:
            content = json_lib.dumps(records, ensure_ascii=False, indent=2)
            media_type = "application/json"
            filename = f"crawlbot_chunks_{job_id[:8]}.json"

        return Response(
            content=content.encode("utf-8"),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    elif fmt in {"json_enhanced", "jsonl_enhanced"}:
        rag_data = manager.get_rag_metadata(job_id)
        if not rag_data:
            raise HTTPException(
                status_code=400,
                detail="داده RAG برای این کار وجود ندارد. ابتدا عملیات بهبود RAG را انجام دهید.",
            )
        documents = manager.get_documents(job_id)
        rag_by_url = {r.get("source_url"): r for r in rag_data if r.get("source_url")}
        
        enhanced_docs = []
        for doc in documents:
            meta = rag_by_url.get(doc.url, {})
            enhanced_docs.append({
                "url": doc.url,
                "final_url": doc.final_url,
                "title": doc.title,
                "summary": meta.get("summary"),
                "keywords": meta.get("keywords", []),
                "topics": meta.get("topics", []),
                "qa_pairs": meta.get("qa_pairs", []),
                "verification_summary": meta.get("verification_summary", {}),
                "content_markdown": doc.content_markdown,
                "content_text": doc.content_text,
                "word_count": doc.word_count,
                "character_count": doc.character_count,
                "language": doc.language,
                "fetched_at": doc.fetched_at.isoformat() if doc.fetched_at else None,
                "rag_enhanced": bool(meta.get("success")),
            })

        if fmt == "jsonl_enhanced":
            content = "\n".join(json_lib.dumps(d, ensure_ascii=False) for d in enhanced_docs)
            media_type = "application/x-ndjson"
            filename = f"crawlbot_rag_enhanced_{job_id[:8]}.jsonl"
        else:
            content = json_lib.dumps(enhanced_docs, ensure_ascii=False, indent=2)
            media_type = "application/json"
            filename = f"crawlbot_rag_enhanced_{job_id[:8]}.json"

        return Response(
            content=content.encode("utf-8"),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    elif fmt in {"markdown_enhanced", "md_enhanced"}:
        import zipfile
        from urllib.parse import urlparse
        rag_data = manager.get_rag_metadata(job_id)
        if not rag_data:
            raise HTTPException(
                status_code=400,
                detail="داده RAG برای این کار وجود ندارد. ابتدا عملیات بهبود RAG را انجام دهید.",
            )
        documents = manager.get_documents(job_id)
        rag_by_url = {r.get("source_url"): r for r in rag_data if r.get("source_url")}

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, doc in enumerate(documents):
                meta = rag_by_url.get(doc.url, {})
                md_content = _build_enhanced_markdown(doc, meta)
                try:
                    host = urlparse(doc.url).netloc.replace(":", "_")
                    path_slug = urlparse(doc.url).path.strip("/").replace("/", "_")
                    base = f"{host}_{path_slug}" if path_slug else f"{host}_page"
                except Exception:
                    base = f"doc_{idx}"
                clean_name = "".join(c for c in base if c.isalnum() or c in "._-")[:60]
                zf.writestr(f"{clean_name}_{idx}.md", md_content)

        zip_buffer.seek(0)
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=crawlbot_rag_markdown_{job_id[:8]}.zip"
            },
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"فرمت '{fmt}' نامعتبر است. موارد مجاز: jsonl_chunks, json_chunks, json_enhanced, jsonl_enhanced, markdown_enhanced",
        )


def _build_enhanced_markdown(doc, rag_meta: dict) -> str:
    """Build a structured Markdown document with YAML frontmatter and LLM RAG sections."""
    lines = ["---"]
    lines.append(f'url: "{doc.url}"')
    if doc.final_url:
        lines.append(f'final_url: "{doc.final_url}"')
    if doc.title:
        escaped_title = doc.title.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'title: "{escaped_title}"')
    if rag_meta.get("summary"):
        escaped_sum = rag_meta["summary"].replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'summary: "{escaped_sum}"')
    if rag_meta.get("keywords"):
        kws = ", ".join(f'"{k}"' for k in rag_meta["keywords"])
        lines.append(f'keywords: [{kws}]')
    if rag_meta.get("topics"):
        tps = ", ".join(f'"{t}"' for t in rag_meta["topics"])
        lines.append(f'topics: [{tps}]')
    lines.append(f'word_count: {doc.word_count}')
    lines.append("---")
    lines.append("")

    if doc.title:
        lines.append(f"# {doc.title}\n")
    if rag_meta.get("summary"):
        lines.append(f"> **خلاصه محتوا:** {rag_meta['summary']}\n")
    if rag_meta.get("topics"):
        lines.append(f"> **موضوعات:** {', '.join(rag_meta['topics'])}\n")
    if rag_meta.get("keywords"):
        lines.append(f"> **کلیدواژه‌ها:** {', '.join(rag_meta['keywords'])}\n")

    lines.append("\n## متن مستند\n")
    lines.append(doc.content_markdown or doc.content_text or "")

    qa_pairs = rag_meta.get("qa_pairs", [])
    if qa_pairs:
        lines.append("\n\n## پرسش و پاسخ‌های تحلیلی راستی‌آزمایی‌شده (Verified RAG Q&A)\n")
        for i, pair in enumerate(qa_pairs, 1):
            q = pair.get("question", "")
            a = pair.get("answer", "")
            ev = pair.get("evidence", "")
            ver = pair.get("verification", {})
            status_tag = "✅ راستی‌آزمایی شده با متن منبع" if ver.get("status") == "verified" else "⚠️ نیازمند بررسی"
            if q and a:
                lines.append(f"### {i}. {q}\n")
                lines.append(f"{a}\n")
                if ev:
                    lines.append(f"> 🔍 **مستند منبع (Evidence):** «{ev}»\n")
                lines.append(f"> **وضعیت راستی‌آزمایی:** {status_tag}\n")

    return "\n".join(lines)
