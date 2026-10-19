"""Engine A: Trafilatura Extraction Engine.

Specialized for main content extraction from articles, documentation,
blogs, policy documents, and structured textual publications.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import trafilatura

from app.schemas.documents import Heading, LinkInfo
from .base import EngineResult

logger = logging.getLogger(__name__)


class TrafilaturaEngine:
    """Engine A implementation using Trafilatura."""

    ENGINE_NAME: str = "trafilatura"

    def __init__(self, favor_recall: bool = True, include_tables: bool = True) -> None:
        self.favor_recall = favor_recall
        self.include_tables = include_tables

    def extract(self, html: str, url: str) -> EngineResult:
        """Extract content and metadata using Trafilatura.

        Args:
            html: Raw HTML content.
            url: Source URL of the webpage.

        Returns:
            Standardized EngineResult.
        """
        start_time = time.perf_counter()
        result = EngineResult(engine=self.ENGINE_NAME, success=False)

        if not html or not html.strip():
            result.error = "HTML content is empty"
            result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return result

        try:
            # 1. Extract markdown
            md = trafilatura.extract(
                html,
                url=url,
                output_format="markdown",
                include_tables=self.include_tables,
                include_links=True,
                include_formatting=True,
                favor_recall=self.favor_recall,
            )

            # 2. Extract plain text
            txt = trafilatura.extract(
                html,
                url=url,
                output_format="txt",
                include_tables=self.include_tables,
                include_links=True,
                favor_recall=self.favor_recall,
            )

            # 3. Extract metadata dict via bare_extraction
            meta_dict = {}
            try:
                extracted_data = trafilatura.bare_extraction(
                    html,
                    url=url,
                    include_tables=self.include_tables,
                    include_links=True,
                )
                if extracted_data:
                    # extracted_data is a dict or Document object depending on version
                    if isinstance(extracted_data, dict):
                        meta_dict = extracted_data
                    elif hasattr(extracted_data, "as_dict"):
                        meta_dict = extracted_data.as_dict()
                    elif hasattr(extracted_data, "__dict__"):
                        meta_dict = vars(extracted_data)
            except Exception as meta_exc:
                logger.debug("Trafilatura metadata extraction error for %s: %s", url, meta_exc)

            title = meta_dict.get("title")
            description = meta_dict.get("description")
            language = meta_dict.get("language")

            # Fallback if text extracted but markdown is missing
            if txt and not md:
                md = txt
            elif md and not txt:
                txt = md

            result.markdown = md.strip() if md else None
            result.text = txt.strip() if txt else None
            result.title = title
            result.description = description
            result.language = language
            safe_metadata = {}
            for k, v in meta_dict.items():
                if k in ("text", "raw_text", "comments", "body") or v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    safe_metadata[k] = v
                elif isinstance(v, (list, tuple)):
                    safe_metadata[k] = [str(x) for x in v if not hasattr(x, "xpath")]
                elif isinstance(v, dict):
                    safe_metadata[k] = {str(dk): str(dv) for dk, dv in v.items()}
                elif not hasattr(v, "xpath"):
                    safe_metadata[k] = str(v)

            result.metadata = safe_metadata

            result.compute_counts()
            result.success = result.has_content
            if not result.success:
                result.warnings.append("Trafilatura returned empty textual content")

        except Exception as exc:
            logger.warning("Trafilatura extraction failed for %s: %s", url, exc)
            result.error = f"Trafilatura error: {str(exc)}"
            result.success = False

        result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return result
