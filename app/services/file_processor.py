"""File Processing Service for Crawlbot.

Provides secure, non-destructive file processing, transformation, normalization,
and chunking pipelines for RAG knowledge bases.

Supported file formats:
- Markdown (.md)
- Plain Text (.txt)
- JSON (.json)
- JSON Lines (.jsonl)
- Comma/Semicolon/Tab Separated Values (.csv)

Supported operations:
1. normalize: Unicode NFC, Arabic character unification, and Persian ZWNJ affix fixing.
2. markdown_to_json: Structural extraction of headings, sections, lists, tables, and code.
3. json_validate: Strict syntax, type, and recursive depth validation.
4. json_cleanup: Value normalization, whitespace trimming, and duplicate record elimination.
5. llm_json_optimize: Optional LLM-assisted schema structuring with strict grounding.
6. chunk: Smart chunking using ChunkerService (auto, semantic_heading, paragraph, sentence, fixed_size).
7. metadata_enrich: Ingestion metadata (hashes, word/character statistics, category, timestamps).
8. embed: Vector embedding for produced chunks via EmbeddingService.
9. pipeline: Chained multi-operation pipeline execution.
10. export: Serialization to JSON, JSONL, Markdown, Text, or CSV.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional

from ..core.config import get_settings
from ..core.security import sanitize_filename
from ..schemas.chunks import ChunkingResult, DocumentChunk
from ..schemas.documents import ExtractedDocument
from .chunker import ChunkerService
from .embedding_service import EmbeddingService
from .normalizer import PersianNormalizer
from .rag_enhancer import RAGEnhancer

logger = logging.getLogger(__name__)

# Constants and limits
MAX_FILE_SIZE_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_EXTENSIONS: Final[set[str]] = {".md", ".txt", ".json", ".jsonl", ".csv"}
MAX_JSON_DEPTH: Final[int] = 25


def _check_json_depth(obj: Any, current_depth: int = 1) -> int:
    """Recursively check maximum nesting depth of a JSON object."""
    if current_depth > MAX_JSON_DEPTH:
        raise ValueError(f"JSON nesting depth exceeds maximum limit of {MAX_JSON_DEPTH}.")
    if isinstance(obj, dict):
        return max([_check_json_depth(v, current_depth + 1) for v in obj.values()] or [current_depth])
    elif isinstance(obj, list):
        return max([_check_json_depth(item, current_depth + 1) for item in obj] or [current_depth])
    return current_depth


class FileProcessorService:
    """Manages file uploads, transformation pipelines, and export artifacts."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        """Initialize file processor with workspace storage directories."""
        settings = get_settings()
        root = base_dir or getattr(settings, "output_dir", "output")
        self.storage_dir = Path(root) / "file_processing"
        self.uploads_dir = self.storage_dir / "uploads"
        self.artifacts_dir = self.storage_dir / "artifacts"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._normalizer = PersianNormalizer()
        self._memory_files: dict[str, dict[str, Any]] = {}
        self._memory_artifacts: dict[str, dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # File Storage & Validation
    # -----------------------------------------------------------------------

    def upload_file(self, filename: str, content_bytes: bytes) -> dict[str, Any]:
        """Save and inspect an uploaded file in the secure sandbox.

        Args:
            filename: Original filename from the client.
            content_bytes: Raw binary bytes of the file.

        Returns:
            Dictionary with file metadata, preview snippet, and record counts.

        Raises:
            ValueError: If file size exceeds limit or extension is not allowed.
        """
        if len(content_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"حجم فایل ({len(content_bytes)} بایت) از سقف مجاز "
                f"({MAX_FILE_SIZE_BYTES} بایت / 10MB) بیشتر است."
            )

        safe_name = sanitize_filename(filename)
        _, ext = os.path.splitext(safe_name.lower())

        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"فرمت '{ext}' مجاز نیست. فرمت‌های پشتیبانی‌شده: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        file_id = uuid.uuid4().hex[:12]
        dest_filename = f"{file_id}_{safe_name}"
        dest_path = self.uploads_dir / dest_filename

        with open(dest_path, "wb") as f:
            f.write(content_bytes)

        # Decode content with encoding fallback
        text_content, encoding_used = self._decode_bytes(content_bytes)

        line_count = len(text_content.splitlines())
        snippet = text_content[:1000]

        record = {
            "file_id": file_id,
            "filename": safe_name,
            "format": ext.lstrip("."),
            "size_bytes": len(content_bytes),
            "line_count": line_count,
            "encoding": encoding_used,
            "preview_snippet": snippet,
            "stored_path": str(dest_path),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._memory_files[file_id] = record
        return record

    def get_file(self, file_id: str) -> Optional[dict[str, Any]]:
        """Retrieve stored file record by ID."""
        return self._memory_files.get(file_id)

    def get_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        """Retrieve stored artifact record by ID."""
        return self._memory_artifacts.get(artifact_id)

    # -----------------------------------------------------------------------
    # Core Pipeline & Execution
    # -----------------------------------------------------------------------

    async def process(
        self,
        file_id: str,
        operations: list[str],
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute selected processing pipeline on an uploaded file.

        Args:
            file_id: Identifier of the uploaded file.
            operations: List of operation names in order of execution.
            options: Configuration parameters for operations.

        Returns:
            Dictionary containing process execution report, diff, and output artifact.
        """
        start_time = datetime.now(timezone.utc)
        file_info = self.get_file(file_id)
        if not file_info:
            raise ValueError(f"فایل با شناسه '{file_id}' یافت نشد.")

        dest_path = Path(file_info["stored_path"])
        if not dest_path.exists():
            raise FileNotFoundError(f"فایل در مسیر فیزیکی '{dest_path}' یافت نشد.")

        with open(dest_path, "rb") as f:
            raw_bytes = f.read()

        text_content, _ = self._decode_bytes(raw_bytes)
        file_format = file_info["format"]
        opts = options or {}

        # Pipeline state tracking
        current_data: Any = text_content
        current_format: str = file_format
        telemetry: dict[str, Any] = {
            "records_in": file_info["line_count"],
            "records_out": file_info["line_count"],
            "chunks_produced": 0,
            "embedded_chunks": 0,
            "normalization_fixes": 0,
            "warnings": [],
            "operations_applied": [],
        }

        # Sequential pipeline execution
        for op in operations:
            op_name = op.strip().lower()

            if op_name == "normalize":
                current_data, stats = self._op_normalize(current_data, opts)
                telemetry["normalization_fixes"] += stats.get("zwnj_fixes", 0) + stats.get("arabic_unifications", 0)
                telemetry["operations_applied"].append("normalize")

            elif op_name == "markdown_to_json":
                current_data, stats = self._op_markdown_to_json(current_data, opts)
                current_format = "json"
                telemetry["operations_applied"].append("markdown_to_json")

            elif op_name == "json_validate":
                current_data, stats = self._op_json_validate(current_data, opts)
                if stats.get("warnings"):
                    telemetry["warnings"].extend(stats["warnings"])
                telemetry["operations_applied"].append("json_validate")

            elif op_name == "json_cleanup":
                current_data, stats = self._op_json_cleanup(current_data, opts)
                telemetry["operations_applied"].append("json_cleanup")

            elif op_name == "llm_json_optimize":
                current_data, stats = await self._op_llm_json_optimize(
                    current_data, current_format, opts, file_info["filename"]
                )
                current_format = "json"
                if stats.get("warnings"):
                    telemetry["warnings"].extend(stats["warnings"])
                telemetry["operations_applied"].append("llm_json_optimize")

            elif op_name == "chunk":
                current_data, stats = self._op_chunk(
                    current_data, current_format, opts, file_info["filename"]
                )
                current_format = "json"
                telemetry["chunks_produced"] = stats.get("total_chunks", 0)
                telemetry["records_out"] = stats.get("total_chunks", 0)
                telemetry["operations_applied"].append("chunk")

            elif op_name == "embed":
                current_data, stats = await self._op_embed(
                    current_data, current_format, opts, file_info["filename"]
                )
                current_format = "json"
                telemetry["embedded_chunks"] = stats.get("embedded_chunks", 0)
                if stats.get("warnings"):
                    telemetry["warnings"].extend(stats["warnings"])
                telemetry["operations_applied"].append("embed")

            elif op_name == "metadata_enrich":
                current_data, stats = self._op_metadata_enrich(current_data, opts)
                current_format = "json"
                telemetry["operations_applied"].append("metadata_enrich")

        # Prepare output artifact
        artifact_id = uuid.uuid4().hex[:12]
        output_format = opts.get("output_format") or current_format
        output_str = self._serialize_output(current_data, output_format)

        artifact_filename = f"artifact_{artifact_id}_{file_info['filename']}.{output_format}"
        artifact_path = self.artifacts_dir / artifact_filename
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(output_str)

        # Generate Diff if applicable (textual comparison)
        diff_str = self._generate_diff(text_content, output_str)

        duration_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
        artifact_record = {
            "artifact_id": artifact_id,
            "file_id": file_id,
            "filename": artifact_filename,
            "format": output_format,
            "stored_path": str(artifact_path),
            "size_bytes": len(output_str.encode("utf-8")),
            "telemetry": telemetry,
            "diff_summary": diff_str[:5000],
            "duration_seconds": duration_sec,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._memory_artifacts[artifact_id] = artifact_record

        return {
            "status": "success",
            "file_id": file_id,
            "operations_executed": operations,
            "artifact_id": artifact_id,
            "filename": artifact_filename,
            "format": output_format,
            "download_url": f"/api/files/download/{artifact_id}",
            "telemetry": telemetry,
            "diff": diff_str[:5000],
            "output_preview": output_str[:2500],
            "duration_seconds": duration_sec,
        }

    def process_sync(
        self,
        file_id: str,
        operations: list[str],
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper for process."""
        import asyncio
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.process(file_id, operations, options)).result()
            return loop.run_until_complete(self.process(file_id, operations, options))
        except RuntimeError:
            return asyncio.run(self.process(file_id, operations, options))

    # -----------------------------------------------------------------------
    # Operation Implementations
    # -----------------------------------------------------------------------

    def _op_normalize(self, data: Any, opts: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Apply Persian and Unicode normalization recursively."""
        norm = PersianNormalizer(
            nfc=opts.get("nfc", True),
            unify_arabic=opts.get("unify_arabic", True),
            clean_controls=opts.get("clean_controls", True),
            fix_spacing=opts.get("fix_spacing", True),
            fix_zwnj_affixes=opts.get("fix_zwnj_affixes", True),
        )

        total_stats = {"zwnj_fixes": 0, "arabic_unifications": 0}

        def _norm_recursive(item: Any) -> Any:
            if isinstance(item, str):
                res, rep = norm.normalize_with_report(item)
                total_stats["zwnj_fixes"] += rep.get("zwnj_fixes", 0)
                total_stats["arabic_unifications"] += rep.get("arabic_unifications", 0)
                return res
            elif isinstance(item, dict):
                return {k: _norm_recursive(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [_norm_recursive(elem) for elem in item]
            return item

        normalized_data = _norm_recursive(data)
        return normalized_data, total_stats

    def _op_markdown_to_json(self, text: str, opts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Parse Markdown structure into a hierarchical JSON schema."""
        if not isinstance(text, str):
            text = str(text)

        lines = text.splitlines()
        title = None
        sections: list[dict[str, Any]] = []
        current_section = {"heading": "Overview", "level": 1, "content": []}

        heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                h_text = m.group(2).strip()
                if not title and level == 1:
                    title = h_text
                # Flush previous section if it has content
                if current_section["content"]:
                    sections.append({
                        "heading": current_section["heading"],
                        "level": current_section["level"],
                        "text": "\n".join(current_section["content"]).strip(),
                    })
                current_section = {"heading": h_text, "level": level, "content": []}
            else:
                current_section["content"].append(line)

        if current_section["content"]:
            sections.append({
                "heading": current_section["heading"],
                "level": current_section["level"],
                "text": "\n".join(current_section["content"]).strip(),
            })

        result = {
            "title": title or "Document",
            "total_sections": len(sections),
            "sections": sections,
            "metadata": {
                "char_count": len(text),
                "word_count": len(text.split()),
            },
        }
        return result, {"sections_extracted": len(sections)}

    def _op_json_validate(self, data: Any, opts: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Validate JSON syntax, depth, and structural integrity."""
        warnings: list[str] = []

        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as err:
                raise ValueError(f"خطای اعتبارسنجی ساختار JSON: {err.msg} (خط {err.lineno}, ستون {err.colno})")
        else:
            parsed = data

        try:
            depth = _check_json_depth(parsed)
            if depth > 15:
                warnings.append(f"عمق تودرتویی JSON بالا است ({depth} سطح).")
        except ValueError as e:
            warnings.append(str(e))

        return parsed, {"warnings": warnings}

    def _op_json_cleanup(self, data: Any, opts: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Clean JSON values by trimming strings and removing nulls if configured."""
        drop_nulls = opts.get("drop_nulls", False)

        def _clean_node(node: Any) -> Any:
            if isinstance(node, dict):
                cleaned_dict = {}
                for k, v in node.items():
                    if drop_nulls and v is None:
                        continue
                    cleaned_dict[k] = _clean_node(v)
                return cleaned_dict
            elif isinstance(node, list):
                return [_clean_node(item) for item in node]
            elif isinstance(node, str):
                return node.strip()
            return node

        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                cleaned = _clean_node(parsed)
            except Exception:
                cleaned = data.strip()
        else:
            cleaned = _clean_node(data)

        return cleaned, {"status": "cleaned"}

    def _to_extracted_documents(
        self,
        data: Any,
        fmt: str,
        opts: dict[str, Any],
        fallback_title: str = "Uploaded Document",
    ) -> list[ExtractedDocument]:
        """Convert input data (str, dict, list) into ExtractedDocument instances."""
        docs: list[ExtractedDocument] = []

        if isinstance(data, str):
            if fmt == "json" or data.strip().startswith(("{", "[")):
                try:
                    parsed = json.loads(data)
                    return self._to_extracted_documents(parsed, "json", opts, fallback_title)
                except Exception:
                    pass

            doc = ExtractedDocument(
                document_id=uuid.uuid4().hex[:12],
                url=f"https://crawlbot.local/file/{fallback_title}",
                source_url=f"https://crawlbot.local/file/{fallback_title}",
                title=opts.get("doc_title") or fallback_title,
                content_markdown=data if fmt == "md" else None,
                content_text=data,
                word_count=len(data.split()),
                character_count=len(data),
            )
            return [doc]

        elif isinstance(data, dict):
            text = (
                data.get("content_text")
                or data.get("text")
                or data.get("content")
                or data.get("body")
                or data.get("description")
            )
            if text and isinstance(text, str):
                md = data.get("content_markdown") or (text if fmt == "md" else None)
                title = data.get("title") or opts.get("doc_title") or fallback_title
                doc = ExtractedDocument(
                    document_id=str(data.get("document_id") or uuid.uuid4().hex[:12]),
                    url=str(data.get("url") or data.get("source_url") or f"https://crawlbot.local/file/{fallback_title}"),
                    source_url=str(data.get("source_url") or data.get("url") or f"https://crawlbot.local/file/{fallback_title}"),
                    title=title,
                    content_markdown=md,
                    content_text=text,
                    word_count=len(text.split()),
                    character_count=len(text),
                )
                return [doc]
            else:
                dumped = json.dumps(data, ensure_ascii=False, indent=2)
                title = data.get("title") or opts.get("doc_title") or fallback_title
                doc = ExtractedDocument(
                    document_id=uuid.uuid4().hex[:12],
                    url=f"https://crawlbot.local/file/{fallback_title}",
                    source_url=f"https://crawlbot.local/file/{fallback_title}",
                    title=title,
                    content_text=dumped,
                    word_count=len(dumped.split()),
                    character_count=len(dumped),
                )
                return [doc]

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text = (
                        item.get("content_text")
                        or item.get("text")
                        or item.get("content")
                        or item.get("body")
                    )
                    if text and isinstance(text, str):
                        md = item.get("content_markdown") or (text if fmt == "md" else None)
                        title = item.get("title") or f"{fallback_title} - رکورد {i+1}"
                        docs.append(
                            ExtractedDocument(
                                document_id=str(item.get("document_id") or uuid.uuid4().hex[:12]),
                                url=str(item.get("url") or item.get("source_url") or f"https://crawlbot.local/file/{fallback_title}/{i+1}"),
                                source_url=str(item.get("source_url") or item.get("url") or f"https://crawlbot.local/file/{fallback_title}/{i+1}"),
                                title=title,
                                content_markdown=md,
                                content_text=text,
                                word_count=len(text.split()),
                                character_count=len(text),
                            )
                        )
                    else:
                        dumped = json.dumps(item, ensure_ascii=False)
                        docs.append(
                            ExtractedDocument(
                                document_id=uuid.uuid4().hex[:12],
                                url=f"https://crawlbot.local/file/{fallback_title}/{i+1}",
                                source_url=f"https://crawlbot.local/file/{fallback_title}/{i+1}",
                                title=item.get("title") or f"{fallback_title} - رکورد {i+1}",
                                content_text=dumped,
                                word_count=len(dumped.split()),
                                character_count=len(dumped),
                            )
                        )
                elif isinstance(item, str) and item.strip():
                    docs.append(
                        ExtractedDocument(
                            document_id=uuid.uuid4().hex[:12],
                            url=f"https://crawlbot.local/file/{fallback_title}/{i+1}",
                            source_url=f"https://crawlbot.local/file/{fallback_title}/{i+1}",
                            title=f"{fallback_title} - بخش {i+1}",
                            content_text=item,
                            word_count=len(item.split()),
                            character_count=len(item),
                        )
                    )

            if not docs:
                text_to_chunk = "\n\n".join(
                    [json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x) for x in data]
                )
                docs.append(
                    ExtractedDocument(
                        document_id=uuid.uuid4().hex[:12],
                        url=f"https://crawlbot.local/file/{fallback_title}",
                        source_url=f"https://crawlbot.local/file/{fallback_title}",
                        title=fallback_title,
                        content_text=text_to_chunk,
                        word_count=len(text_to_chunk.split()),
                        character_count=len(text_to_chunk),
                    )
                )
            return docs

        dumped = str(data)
        return [
            ExtractedDocument(
                document_id=uuid.uuid4().hex[:12],
                url=f"https://crawlbot.local/file/{fallback_title}",
                source_url=f"https://crawlbot.local/file/{fallback_title}",
                title=fallback_title,
                content_text=dumped,
                word_count=len(dumped.split()),
                character_count=len(dumped),
            )
        ]

    def _op_chunk(
        self,
        data: Any,
        fmt: str,
        opts: dict[str, Any],
        fallback_title: str = "Uploaded Document",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Extract text content and apply ChunkerService to produce DocumentChunks."""
        strategy = opts.get("chunk_strategy") or opts.get("strategy") or "auto"
        chunk_size = int(opts.get("chunk_size", 1000))
        overlap = int(opts.get("overlap", 150))
        category = opts.get("category") or None
        normalize = bool(opts.get("normalize", False))

        docs = self._to_extracted_documents(data, fmt, opts, fallback_title)
        eligible = [d for d in docs if d.has_content]
        if not eligible:
            eligible = docs

        chunker = ChunkerService(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            normalize=normalize,
        )

        all_chunks = chunker.chunk_documents(eligible, category=category)
        serialized_chunks = [c.model_dump() for c in all_chunks]

        applied_strategies = list({c.metadata.get("strategy") for c in all_chunks if c.metadata.get("strategy")})
        applied_str = applied_strategies[0] if len(applied_strategies) == 1 else (", ".join(applied_strategies) if applied_strategies else strategy)

        return serialized_chunks, {
            "total_chunks": len(all_chunks),
            "strategy": strategy,
            "applied_strategy": applied_str,
            "total_docs_chunked": len(eligible),
        }

    def _op_metadata_enrich(self, data: Any, opts: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Add hashes, category inference, timestamps, and quality flags."""
        timestamp = datetime.now(timezone.utc).isoformat()

        if isinstance(data, list):
            enriched = []
            for item in data:
                if isinstance(item, dict):
                    text_val = item.get("text") or json.dumps(item, ensure_ascii=False)
                    c_hash = hashlib.sha256(text_val.encode("utf-8")).hexdigest()
                    item_copy = dict(item)
                    item_copy.setdefault("content_hash", c_hash)
                    item_copy.setdefault("processed_at", timestamp)
                    item_copy.setdefault("category", opts.get("category", "unknown"))
                    enriched.append(item_copy)
                else:
                    enriched.append(item)
            return enriched, {"enriched_items": len(enriched)}

        elif isinstance(data, dict):
            copy_dict = dict(data)
            dumped = json.dumps(data, ensure_ascii=False)
            copy_dict["content_hash"] = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
            copy_dict["processed_at"] = timestamp
            return copy_dict, {"status": "enriched"}

        elif isinstance(data, str):
            c_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
            enriched_obj = {
                "text": data,
                "content_hash": c_hash,
                "processed_at": timestamp,
                "category": opts.get("category", "general"),
                "metadata": opts,
            }
            return enriched_obj, {"status": "enriched"}

        return data, {"status": "skipped"}

    async def _op_llm_json_optimize(
        self,
        data: Any,
        fmt: str = "json",
        opts: Optional[dict[str, Any]] = None,
        fallback_title: str = "Uploaded Document",
    ) -> tuple[Any, dict[str, Any]]:
        """AI RAG Enhancement with strict grounding and verifiable citations."""
        opts = opts or {}
        settings = get_settings()
        api_key = (
            opts.get("api_key")
            or getattr(settings, "openai_api_key", None)
            or os.getenv("CRAWLBOT_OPENAI_API_KEY", "")
        )

        if not api_key or not api_key.strip():
            return data, {
                "warnings": [
                    "کلید API هوش مصنوعی (CRAWLBOT_OPENAI_API_KEY) تنظیم نشده است؛ عملیات بهینه‌سازی LLM رد شد."
                ]
            }

        base_url = opts.get("base_url") or settings.openai_base_url or None
        model = opts.get("llm_model") or settings.llm_model or "gpt-4o-mini"
        concurrency = opts.get("ai_concurrency") or getattr(settings, "ai_concurrency", 3)

        try:
            enhancer = RAGEnhancer(
                api_key=api_key.strip(),
                base_url=base_url,
                model=model,
                concurrency=concurrency,
            )
            docs = self._to_extracted_documents(data, fmt, opts, fallback_title)
            eligible = [d for d in docs if d.has_content]
            if not eligible:
                return data, {"warnings": ["محتوای متنی کافی برای غنی‌سازی هوش مصنوعی یافت نشد."]}

            results = await enhancer.enhance_documents(eligible)

            if isinstance(data, str) and (fmt != "json" or not data.strip().startswith(("{", "["))):
                res0 = results[0] if results else {}
                enriched_data = {
                    "title": eligible[0].title,
                    "summary": res0.get("summary", ""),
                    "topics": res0.get("topics", []),
                    "keywords": res0.get("keywords", []),
                    "qa_pairs": res0.get("qa_pairs", []),
                    "page_type": res0.get("page_type", ""),
                    "dynamic_content": res0.get("dynamic_content", False),
                    "content": data,
                }
                return enriched_data, {
                    "status": "llm_optimized",
                    "enhanced_docs": len(results),
                    "qa_pairs_count": sum(len(r.get("qa_pairs", [])) for r in results),
                }

            if isinstance(data, dict):
                enriched_data = dict(data)
                res0 = results[0] if results else {}
                enriched_data["rag_enhancement"] = {
                    "summary": res0.get("summary", ""),
                    "topics": res0.get("topics", []),
                    "keywords": res0.get("keywords", []),
                    "qa_pairs": res0.get("qa_pairs", []),
                    "page_type": res0.get("page_type", ""),
                }
                return enriched_data, {
                    "status": "llm_optimized",
                    "enhanced_docs": 1,
                    "qa_pairs_count": len(res0.get("qa_pairs", [])),
                }

            if isinstance(data, list):
                enriched_list = []
                for idx, item in enumerate(data):
                    res_item = results[idx] if idx < len(results) else {}
                    if isinstance(item, dict):
                        item_copy = dict(item)
                        item_copy["rag_enhancement"] = {
                            "summary": res_item.get("summary", ""),
                            "topics": res_item.get("topics", []),
                            "keywords": res_item.get("keywords", []),
                            "qa_pairs": res_item.get("qa_pairs", []),
                            "page_type": res_item.get("page_type", ""),
                        }
                        enriched_list.append(item_copy)
                    else:
                        enriched_list.append(item)
                return enriched_list, {
                    "status": "llm_optimized",
                    "enhanced_docs": len(results),
                    "qa_pairs_count": sum(len(r.get("qa_pairs", [])) for r in results),
                }

            return data, {"status": "llm_optimized", "enhanced_docs": len(results)}

        except Exception as e:
            logger.warning("LLM optimization failed: %s", e)
            return data, {"warnings": [f"خطا در بهینه‌سازی هوش مصنوعی: {str(e)}"]}

    async def _op_embed(
        self,
        data: Any,
        fmt: str = "json",
        opts: Optional[dict[str, Any]] = None,
        fallback_title: str = "Uploaded Document",
    ) -> tuple[Any, dict[str, Any]]:
        """Generate dense vector embeddings using OpenAI Embedding API."""
        opts = opts or {}
        settings = get_settings()
        api_key = (
            opts.get("api_key")
            or getattr(settings, "openai_api_key", None)
            or os.getenv("CRAWLBOT_OPENAI_API_KEY", "")
        )

        if not api_key or not api_key.strip():
            return data, {
                "warnings": [
                    "کلید API هوش مصنوعی (CRAWLBOT_OPENAI_API_KEY) برای تولید وکتور Embedding تنظیم نشده است."
                ]
            }

        base_url = opts.get("base_url") or settings.openai_base_url or None
        model = opts.get("embedding_model") or settings.embedding_model or "text-embedding-3-small"

        try:
            embedder = EmbeddingService(
                api_key=api_key.strip(),
                base_url=base_url,
                model=model,
            )

            is_chunks = (
                isinstance(data, list)
                and len(data) > 0
                and isinstance(data[0], dict)
                and ("chunk_id" in data[0] or ("text" in data[0] and "chunk_index" in data[0]))
            )

            if is_chunks:
                chunks = [DocumentChunk(**item) for item in data]
            else:
                chunks_data, _ = self._op_chunk(data, fmt, opts, fallback_title)
                chunks = [DocumentChunk(**item) for item in chunks_data]

            if not chunks:
                return data, {"warnings": ["هیچ قطعه‌ای برای تولید وکتور Embedding یافت نشد."]}

            stats = await embedder.embed_chunks(chunks)
            serialized = [c.model_dump() for c in chunks]

            return serialized, {
                "total_chunks": stats.total_chunks,
                "embedded_chunks": stats.embedded_chunks,
                "failed_chunks": stats.failed_chunks,
                "model": stats.model,
                "total_tokens": stats.total_tokens,
            }

        except Exception as e:
            logger.warning("Embedding operation failed: %s", e)
            return data, {"warnings": [f"خطا در ایجاد بردارهای Embedding: {str(e)}"]}

    # -----------------------------------------------------------------------
    # Helper Utilities
    # -----------------------------------------------------------------------

    def _decode_bytes(self, b: bytes) -> tuple[str, str]:
        """Decode raw bytes trying UTF-8, UTF-8-sig, CP1256, and Latin-1."""
        for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
            try:
                return b.decode(enc), enc
            except (UnicodeDecodeError, LookupError):
                continue
        return b.decode("utf-8", errors="replace"), "utf-8-replace"

    def _serialize_output(self, data: Any, fmt: str) -> str:
        """Serialize data to specified format."""
        fmt = fmt.lower().strip()
        if fmt == "json":
            return json.dumps(data, ensure_ascii=False, indent=2)
        elif fmt == "jsonl":
            if isinstance(data, list):
                return "\n".join(json.dumps(x, ensure_ascii=False) for x in data)
            return json.dumps(data, ensure_ascii=False)
        elif fmt == "csv":
            if isinstance(data, list) and data and isinstance(data[0], dict):
                output = io.StringIO()
                fieldnames = list(data[0].keys())
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    row_clean = {}
                    for k, v in row.items():
                        if isinstance(v, (dict, list)):
                            row_clean[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            row_clean[k] = v
                    writer.writerow(row_clean)
                return output.getvalue()
            return str(data)
        elif fmt == "md":
            if isinstance(data, str):
                return data
            if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
                parts = []
                for i, c in enumerate(data, 1):
                    header = f"### {c.get('heading_context') or c.get('title') or f'قطعه شماره {i}'}\n\n"
                    parts.append(f"{header}{c.get('text', '')}")
                return "\n\n---\n\n".join(parts)
            return json.dumps(data, ensure_ascii=False, indent=2)
        elif fmt == "txt":
            if isinstance(data, str):
                return data
            if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
                return "\n\n".join(c.get("text", "") for c in data)
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)

    def _generate_diff(self, original: str, modified: str) -> str:
        """Generate unified textual diff between original and modified strings."""
        orig_lines = original.splitlines(keepends=True)[:200]
        mod_lines = modified.splitlines(keepends=True)[:200]
        diff = difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile="original",
            tofile="processed",
            n=2,
        )
        return "".join(diff)


# Singleton service instance
file_processor = FileProcessorService()
