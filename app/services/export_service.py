"""Export Service for Crawlbot.

This module provides the ExportService to persist extracted web documents
into JSON, JSONL, Markdown (with YAML frontmatter), and plain text formats.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from ..core.config import get_settings
from ..core.security import sanitize_filename
from ..schemas.documents import ExtractedDocument

logger = logging.getLogger(__name__)


class ExportService:
    """Service to export extracted crawl documents into structured file formats."""

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        """Initialize the ExportService.

        Args:
            output_dir: Base directory for storing export files.
                        Defaults to configured application output_dir.
        """
        base_dir = output_dir or get_settings().output_dir
        self.output_dir: Path = Path(base_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        documents: list[ExtractedDocument],
        job_id: str,
        category: Optional[str] = None,
    ) -> Path:
        """Export documents as a single JSON array file.

        Args:
            documents: List of ExtractedDocument models to serialize.
            job_id: Crawl job identifier used for directory scoping.
            category: Optional classification category for filename differentiation.

        Returns:
            Path to the written JSON file.
        """
        filename = (
            f"documents_{sanitize_filename(category)}.json"
            if category
            else "documents.json"
        )
        out_path = self._get_output_path(job_id, filename)

        serialized_data = [doc.model_dump(mode="json") for doc in documents]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=2)

        logger.info("Exported %d documents to JSON: %s", len(documents), out_path)
        return out_path

    def export_jsonl(
        self,
        documents: list[ExtractedDocument],
        job_id: str,
        category: Optional[str] = None,
    ) -> Path:
        """Export documents as JSONL (one serialized JSON object per line).

        Args:
            documents: List of ExtractedDocument models to serialize.
            job_id: Crawl job identifier used for directory scoping.
            category: Optional classification category for filename differentiation.

        Returns:
            Path to the written JSONL file.
        """
        filename = (
            f"documents_{sanitize_filename(category)}.jsonl"
            if category
            else "documents.jsonl"
        )
        out_path = self._get_output_path(job_id, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            for doc in documents:
                line = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")

        logger.info("Exported %d documents to JSONL: %s", len(documents), out_path)
        return out_path

    def export_markdown(
        self,
        documents: list[ExtractedDocument],
        job_id: str,
        category: Optional[str] = None,
    ) -> Path:
        """Export each document as a separate markdown file with YAML frontmatter in a folder.

        Args:
            documents: List of ExtractedDocument models to serialize.
            job_id: Crawl job identifier used for directory scoping.
            category: Optional category name used for folder naming.

        Returns:
            Path to the folder containing the exported markdown files.
        """
        safe_job_id = sanitize_filename(job_id)
        folder_name = (
            f"markdown_{sanitize_filename(category)}" if category else "markdown"
        )
        folder_path = (self.output_dir / safe_job_id / folder_name).resolve()
        folder_path.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            filename = f"{self._make_filename_from_url(doc.url)}.md"
            file_path = folder_path / filename

            frontmatter = self._build_markdown_frontmatter(doc)
            body = doc.content_markdown or doc.content_text or ""
            full_content = f"{frontmatter}\n\n{body}\n"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_content)

        logger.info(
            "Exported %d markdown files to directory: %s", len(documents), folder_path
        )
        return folder_path

    def export_text(
        self,
        documents: list[ExtractedDocument],
        job_id: str,
        category: Optional[str] = None,
    ) -> Path:
        """Export documents as individual plain text files in a dedicated folder.

        Args:
            documents: List of ExtractedDocument models to serialize.
            job_id: Crawl job identifier used for directory scoping.
            category: Optional category name used for folder naming.

        Returns:
            Path to the folder containing the exported text files.
        """
        safe_job_id = sanitize_filename(job_id)
        folder_name = f"text_{sanitize_filename(category)}" if category else "text"
        folder_path = (self.output_dir / safe_job_id / folder_name).resolve()
        folder_path.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            filename = f"{self._make_filename_from_url(doc.url)}.txt"
            file_path = folder_path / filename

            header_lines = [
                f"URL: {doc.url}",
                f"Title: {doc.title or ''}",
                f"Fetched At: {doc.fetched_at.isoformat() if doc.fetched_at else ''}",
                f"Word Count: {doc.word_count}",
                "=" * 80,
                "",
            ]
            body = doc.content_text or doc.content_markdown or ""
            full_content = "\n".join(header_lines) + body + "\n"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_content)

        logger.info("Exported %d text files to directory: %s", len(documents), folder_path)
        return folder_path

    def _get_output_path(self, job_id: str, filename: str) -> Path:
        """Resolve and validate a secure output file path preventing path traversal.

        Args:
            job_id: Crawl job identifier.
            filename: Target filename.

        Returns:
            Absolute resolved Path instance.

        Raises:
            ValueError: If path traversal outside base output directory is detected.
        """
        safe_job_id = sanitize_filename(job_id)
        safe_filename = sanitize_filename(filename)

        job_dir = (self.output_dir / safe_job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        target_path = (job_dir / safe_filename).resolve()

        # Security boundary check
        if not str(target_path).startswith(str(self.output_dir.resolve())):
            raise ValueError(f"Path traversal detected: {target_path}")

        return target_path

    def _make_filename_from_url(self, url: str) -> str:
        """Generate a safe, collision-resistant filename from a URL.

        Combines normalized hostname and path elements with a short SHA-256 hash.

        Args:
            url: The URL to derive filename from.

        Returns:
            Safe sanitized filename string without extension.
        """
        if not url:
            return "document"

        try:
            parsed = urlparse(url)
            host = parsed.netloc.replace(":", "_")
            path = parsed.path.strip("/").replace("/", "_")
            raw_base = f"{host}_{path}" if path else f"{host}_index"
        except Exception:
            raw_base = "page"

        safe_base = sanitize_filename(raw_base)
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
        truncated_base = safe_base[:120].rstrip("._-")

        return f"{truncated_base}_{url_hash}"

    def _build_markdown_frontmatter(self, doc: ExtractedDocument) -> str:
        """Construct standard YAML frontmatter for a document.

        Args:
            doc: ExtractedDocument containing document metadata.

        Returns:
            Formatted YAML frontmatter block string.
        """
        lines = ["---"]
        lines.append(f'url: "{doc.url}"')
        if doc.final_url:
            lines.append(f'final_url: "{doc.final_url}"')
        if doc.canonical_url:
            lines.append(f'canonical_url: "{doc.canonical_url}"')
        if doc.title:
            escaped_title = doc.title.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'title: "{escaped_title}"')
        if doc.description:
            escaped_desc = doc.description.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'description: "{escaped_desc}"')
        if doc.language:
            lines.append(f'language: "{doc.language}"')
        if doc.fetched_at:
            lines.append(f'fetched_at: "{doc.fetched_at.isoformat()}"')
        lines.append(f"word_count: {doc.word_count}")
        lines.append(f"character_count: {doc.character_count}")

        status_val = (
            doc.quality_status.value
            if hasattr(doc.quality_status, "value")
            else str(doc.quality_status)
        )
        lines.append(f'quality_status: "{status_val}"')
        if doc.content_hash:
            lines.append(f'content_hash: "{doc.content_hash}"')
        lines.append("---")
        return "\n".join(lines)
