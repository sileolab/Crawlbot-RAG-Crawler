"""Content extraction pipeline and unified engine coordinator for Crawlbot.

Coordinates three independent extraction engines:
- Engine A: Trafilatura (optimized for articles, blogs, policy, and structured text)
- Engine B: BeautifulSoup (DOM control, tables, JSON-LD, OpenGraph, microdata)
- Engine C: Crawl4AI (headless browser rendering for JavaScript-heavy and SPA pages)

Supports 'auto' mode with intelligent page classification and controlled fallback,
as well as 'compare' mode for benchmarking all three engines.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Final, Optional
from urllib.parse import urlparse, urlsplit

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from ..core.security import validate_url_safety
from ..schemas.documents import (
    ExtractedDocument,
    ExtractionAttempt,
    Heading,
    LinkInfo,
    QualityReport,
    QualityStatus,
)
from .content_cleaner import ContentCleaner
from .extractors.base import EngineResult
from .extractors.bs4_engine import BeautifulSoupEngine
from .extractors.crawl4ai_engine import Crawl4AIEngine
from .extractors.playwright_engine import PlaywrightEngine
from .extractors.trafilatura_engine import TrafilaturaEngine
from .link_discovery import LinkDiscovery
from .page_classifier import (
    PAGE_TYPE_ARTICLE,
    PAGE_TYPE_DASHBOARD,
    PAGE_TYPE_DOCS,
    PAGE_TYPE_FAQ,
    PAGE_TYPE_POLICY,
    PAGE_TYPE_PRODUCT,
    PAGE_TYPE_SPA,
    PAGE_TYPE_UNKNOWN,
    PageClassifier,
)
from .loading_detector import has_loading_dom, is_loading_text
from .quality_checker import QualityChecker
from .url_normalizer import URLNormalizer

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Full extraction coordinator supporting Trafilatura, BeautifulSoup, and Crawl4AI."""

    def __init__(
        self,
        strategy: str = "auto",
        client: Optional[httpx.AsyncClient] = None,
        min_quality_score: int = 40,
        render_mode: str = "auto",
    ) -> None:
        """Initialize the extraction coordinator.

        Args:
            strategy: Extraction strategy: 'trafilatura', 'beautifulsoup'/'bs4', 'crawl4ai', 'auto', 'compare'.
            client: Optional shared httpx.AsyncClient for HTTP fetch operations.
            min_quality_score: Minimum quality score (0-100) before auto-fallback triggers.
            render_mode: Browser rendering mode ('auto', 'always', 'never').
        """
        self.strategy: str = strategy.lower().strip()
        self.client: Optional[httpx.AsyncClient] = client
        self.min_quality_score: int = min_quality_score
        self.render_mode: str = render_mode.lower().strip()

        self.cleaner: ContentCleaner = ContentCleaner()
        self.normalizer: URLNormalizer = URLNormalizer()
        self.link_discovery: LinkDiscovery = LinkDiscovery(self.normalizer)
        self.quality_checker: QualityChecker = QualityChecker(min_quality_score=min_quality_score)
        self.page_classifier: PageClassifier = PageClassifier()

        # Initialize engines
        self.trafilatura_engine = TrafilaturaEngine()
        self.bs4_engine = BeautifulSoupEngine()
        self.crawl4ai_engine = Crawl4AIEngine()
        self.playwright_engine = PlaywrightEngine()

    async def extract(
        self,
        url: str,
        timeout: float = 30.0,
        crawl_mode: str = "single_page",
        override_strategy: Optional[str] = None,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
    ) -> ExtractedDocument:
        """Execute extraction pipeline for a single URL using configured or auto strategy.

        Args:
            url: Target URL to extract.
            timeout: Request timeout in seconds.
            crawl_mode: Mode ('single_page', 'site_crawl', 'spa_capture', 'compare').
            override_strategy: Optional per-request extraction strategy override.
            loading_delay: Manual delay in seconds to wait for initial loading screens.
            wait_for_selector: Optional CSS/XPath selector to wait for before extraction.

        Returns:
            ExtractedDocument containing unified result, metadata, and quality report.
        """
        active_strategy = (override_strategy or self.strategy).lower().strip()
        if active_strategy in ("playwrite", "pw", "playwright_engine"):
            active_strategy = "playwright"
        if crawl_mode == "compare":
            active_strategy = "compare"
        elif crawl_mode == "spa_capture" and active_strategy == "auto":
            active_strategy = "playwright"

        # 1. Validate URL safety against SSRF and private network ranges
        is_safe, safety_reason = validate_url_safety(url)
        if not is_safe:
            logger.warning("Blocked unsafe URL %s: %s", url, safety_reason)
            return ExtractedDocument(
                url=url,
                source_url=url,
                extraction_method=active_strategy,
                extractor_used=active_strategy,
                quality_status=QualityStatus.BLOCKED,
                warnings=[f"URL safety validation failed: {safety_reason}"],
                quality_report=QualityReport(
                    score=0,
                    passed=False,
                    reasons=[f"Blocked URL safety check: {safety_reason}"],
                ),
            )

        # 2. If strategy is 'compare', run comparison pipeline across engines
        if active_strategy in ("compare", "compare_all"):
            return await self._run_compare_pipeline(
                url=url,
                timeout=timeout,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
                include_playwright=(active_strategy == "compare_all"),
            )

        # 3. Handle pure browser modes directly if browser rendering is unconditionally forced
        if active_strategy == "playwright" and self.render_mode != "never":
            return await self._extract_via_playwright_direct(
                url=url,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )

        if active_strategy == "crawl4ai" and self.render_mode != "never":
            return await self._extract_via_crawl4ai_direct(
                url=url,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )

        # 4. Fetch page HTML safely via HTTP GET
        fetch_res = await self._safe_fetch_html(url=url, timeout=timeout)
        if fetch_res.get("failed"):
            # If standard HTTP fetch failed and strategy was playwright, crawl4ai or auto, try browser fallback
            if active_strategy in ("playwright", "crawl4ai", "auto"):
                logger.info("HTTP fetch failed for %s, attempting browser fallback", url)
                if active_strategy == "playwright":
                    return await self._extract_via_playwright_direct(
                        url=url,
                        loading_delay=loading_delay,
                        wait_for_selector=wait_for_selector,
                    )
                return await self._extract_via_crawl4ai_direct(
                    url=url,
                    loading_delay=loading_delay,
                    wait_for_selector=wait_for_selector,
                )

            return ExtractedDocument(
                url=url,
                source_url=url,
                final_url=fetch_res.get("final_url", url),
                http_status=fetch_res.get("status_code"),
                fetched_at=fetch_res.get("fetched_at"),
                extraction_method=active_strategy,
                extractor_used=active_strategy,
                quality_status=QualityStatus.FAILED,
                warnings=[fetch_res.get("error", "Fetch failed")],
                quality_report=QualityReport(
                    score=0,
                    passed=False,
                    reasons=[fetch_res.get("error", "Fetch failed")],
                ),
            )

        html: str = fetch_res["html"]
        final_url: str = fetch_res["final_url"]
        http_status: Optional[int] = fetch_res.get("status_code")
        fetched_at: datetime = fetch_res.get("fetched_at") or datetime.now(timezone.utc)

        # 5. Classify Page Type
        page_type = self.page_classifier.classify(url=final_url, html=html)

        # 6. Execute extraction based on strategy
        attempts: list[ExtractionAttempt] = []
        best_result: Optional[EngineResult] = None
        best_score: int = -1

        if active_strategy in ("trafilatura",):
            res = self.trafilatura_engine.extract(html, final_url)
            score = self._score_engine_result(res, html)
            att = ExtractionAttempt(
                engine="trafilatura",
                status="success" if res.success else "failed",
                latency_ms=res.latency_ms,
                char_count=res.char_count,
                word_count=res.word_count,
                quality_score=score,
                reason="Single engine execution",
                error=res.error,
            )
            attempts.append(att)
            best_result = res
            best_score = score

            # If static engine captured only a loading screen, auto-promote to Crawl4AI
            if self.render_mode != "never" and (
                is_loading_text(res.text, title=res.title)
                or (not res.has_content and has_loading_dom(html))
            ):
                logger.info("Trafilatura encountered a loading screen for %s. Auto-promoting to Crawl4AI browser capture...", final_url)
                res_c4a = await self.crawl4ai_engine.extract(
                    final_url,
                    html=html,
                    render_js=True,
                    loading_delay=loading_delay,
                    wait_for_selector=wait_for_selector,
                )
                score_c4a = self._score_engine_result(res_c4a, html)
                attempts.append(
                    ExtractionAttempt(
                        engine="crawl4ai",
                        status="success" if res_c4a.success else "failed",
                        latency_ms=res_c4a.latency_ms,
                        char_count=res_c4a.char_count,
                        word_count=res_c4a.word_count,
                        quality_score=score_c4a,
                        reason="Bypassed loading screen via Crawl4AI",
                        error=res_c4a.error,
                    )
                )
                if score_c4a > best_score:
                    best_result = res_c4a
                    best_score = score_c4a

        elif active_strategy in ("beautifulsoup", "bs4"):
            res = self.bs4_engine.extract(html, final_url)
            score = self._score_engine_result(res, html)
            att = ExtractionAttempt(
                engine="beautifulsoup",
                status="success" if res.success else "failed",
                latency_ms=res.latency_ms,
                char_count=res.char_count,
                word_count=res.word_count,
                quality_score=score,
                reason="Single engine execution",
                error=res.error,
            )
            attempts.append(att)
            best_result = res
            best_score = score

            # If static engine captured only a loading screen, auto-promote to Crawl4AI
            if self.render_mode != "never" and (
                is_loading_text(res.text, title=res.title)
                or (not res.has_content and has_loading_dom(html))
            ):
                logger.info("BeautifulSoup encountered a loading screen for %s. Auto-promoting to Crawl4AI browser capture...", final_url)
                res_c4a = await self.crawl4ai_engine.extract(
                    final_url,
                    html=html,
                    render_js=True,
                    loading_delay=loading_delay,
                    wait_for_selector=wait_for_selector,
                )
                score_c4a = self._score_engine_result(res_c4a, html)
                attempts.append(
                    ExtractionAttempt(
                        engine="crawl4ai",
                        status="success" if res_c4a.success else "failed",
                        latency_ms=res_c4a.latency_ms,
                        char_count=res_c4a.char_count,
                        word_count=res_c4a.word_count,
                        quality_score=score_c4a,
                        reason="Bypassed loading screen via Crawl4AI",
                        error=res_c4a.error,
                    )
                )
                if score_c4a > best_score:
                    best_result = res_c4a
                    best_score = score_c4a

        elif active_strategy in ("crawl4ai",):
            res = await self.crawl4ai_engine.extract(
                final_url,
                html=html,
                render_js=True,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )
            score = self._score_engine_result(res, html)
            att = ExtractionAttempt(
                engine="crawl4ai",
                status="success" if res.success else "failed",
                latency_ms=res.latency_ms,
                char_count=res.char_count,
                word_count=res.word_count,
                quality_score=score,
                reason="Single engine execution",
                error=res.error,
            )
            attempts.append(att)
            best_result = res
            best_score = score

        elif active_strategy in ("playwright", "playwrite", "pw"):
            res = await self.playwright_engine.extract(
                final_url,
                html=html,
                render_js=True,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )
            score = self._score_engine_result(res, html)
            att = ExtractionAttempt(
                engine="playwright",
                status="success" if res.success else "failed",
                latency_ms=res.latency_ms,
                char_count=res.char_count,
                word_count=res.word_count,
                quality_score=score,
                reason="Single engine execution",
                error=res.error,
            )
            attempts.append(att)
            best_result = res
            best_score = score

        else:
            # AUTO MODE with intelligent selection and fallback
            best_result, best_score, attempts = await self._run_auto_pipeline(
                url=final_url,
                html=html,
                page_type=page_type,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )

        # 7. Construct final ExtractedDocument
        document = self._build_document(
            url=url,
            final_url=final_url,
            engine_result=best_result,
            raw_html=html,
            http_status=http_status,
            fetched_at=fetched_at,
            page_type=page_type,
            attempts=attempts,
        )

        # Inspect SPA routes and public APIs if in spa_capture mode
        if crawl_mode == "spa_capture" or page_type == PAGE_TYPE_SPA:
            spa_routes, public_apis = self._discover_spa_endpoints(html)
            if spa_routes:
                document.metadata["spa_routes"] = spa_routes
            if public_apis:
                document.metadata["public_apis"] = public_apis

        # Run final quality check and deduplication assessment
        document = self.quality_checker.check(document)
        return document

    async def _run_auto_pipeline(
        self,
        url: str,
        html: str,
        page_type: str,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
    ) -> tuple[EngineResult, int, list[ExtractionAttempt]]:
        """Execute Auto Mode: selects primary engine and manages controlled fallbacks."""
        attempts: list[ExtractionAttempt] = []

        # Step A: Choose primary engine based on page type classification or loader detection
        if page_type in (PAGE_TYPE_SPA, PAGE_TYPE_DASHBOARD) or has_loading_dom(html):
            primary_engine_name = "crawl4ai"
        elif page_type in (PAGE_TYPE_PRODUCT, PAGE_TYPE_FAQ):
            primary_engine_name = "beautifulsoup"
        else:
            primary_engine_name = "trafilatura"

        # Step B: Run primary engine
        primary_result = await self._run_single_engine(
            primary_engine_name,
            url,
            html,
            loading_delay=loading_delay,
            wait_for_selector=wait_for_selector,
        )
        primary_score = self._score_engine_result(primary_result, html)

        attempts.append(
            ExtractionAttempt(
                engine=primary_engine_name,
                status="success" if (primary_result.success and primary_score >= self.min_quality_score) else "fallback",
                latency_ms=primary_result.latency_ms,
                char_count=primary_result.char_count,
                word_count=primary_result.word_count,
                quality_score=primary_score,
                reason=f"Initial engine selected for {page_type}",
                error=primary_result.error,
            )
        )

        best_result = primary_result
        best_score = primary_score

        # Step C: If primary score is sufficient, accept it
        if primary_result.success and primary_score >= self.min_quality_score:
            return best_result, best_score, attempts

        logger.info(
            "Auto mode fallback triggered for %s: %s scored %d (threshold %d)",
            url,
            primary_engine_name,
            primary_score,
            self.min_quality_score,
        )

        # Step D: Determine fallback chain
        fallback_order = [e for e in ["beautifulsoup", "trafilatura", "playwright", "crawl4ai"] if e != primary_engine_name]

        # If page has SPA indicators, loading screen, or very low text, prioritize playwright / crawl4ai
        if (
            "root" in html[:2000]
            or ("<script" in html.lower() and len(primary_result.text or "") < 100)
            or is_loading_text(primary_result.text, title=primary_result.title)
            or has_loading_dom(html)
        ):
            if "playwright" in fallback_order:
                fallback_order.remove("playwright")
                fallback_order.insert(0, "playwright")
            elif "crawl4ai" in fallback_order:
                fallback_order.remove("crawl4ai")
                fallback_order.insert(0, "crawl4ai")

        for fb_engine in fallback_order:
            fb_result = await self._run_single_engine(
                fb_engine,
                url,
                html,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )
            fb_score = self._score_engine_result(fb_result, html)

            passed = fb_result.success and fb_score >= self.min_quality_score
            attempts.append(
                ExtractionAttempt(
                    engine=fb_engine,
                    status="success" if passed else "fallback",
                    latency_ms=fb_result.latency_ms,
                    char_count=fb_result.char_count,
                    word_count=fb_result.word_count,
                    quality_score=fb_score,
                    reason=f"Fallback from {primary_engine_name}",
                    error=fb_result.error,
                )
            )

            if fb_score > best_score:
                best_result = fb_result
                best_score = fb_score

            if passed:
                logger.info("Fallback %s succeeded with score %d for %s", fb_engine, fb_score, url)
                break

        return best_result, best_score, attempts

    async def _run_compare_pipeline(
        self,
        url: str,
        timeout: float,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
        include_playwright: bool = False,
    ) -> ExtractedDocument:
        """Run all three (or four) engines on the same page and produce a side-by-side comparison report."""
        start_time = time.perf_counter()
        fetch_res = await self._safe_fetch_html(url=url, timeout=timeout)
        html = fetch_res.get("html", "")
        final_url = fetch_res.get("final_url", url)
        http_status = fetch_res.get("status_code")
        fetched_at = fetch_res.get("fetched_at") or datetime.now(timezone.utc)
        page_type = self.page_classifier.classify(url=final_url, html=html)

        attempts: list[ExtractionAttempt] = []
        results: dict[str, EngineResult] = {}
        scores: dict[str, int] = {}

        # 1. Engine A: Trafilatura
        res_traf = self.trafilatura_engine.extract(html, final_url)
        score_traf = self._score_engine_result(res_traf, html)
        results["trafilatura"] = res_traf
        scores["trafilatura"] = score_traf
        attempts.append(
            ExtractionAttempt(
                engine="trafilatura",
                status="success" if res_traf.success else "failed",
                latency_ms=res_traf.latency_ms,
                char_count=res_traf.char_count,
                word_count=res_traf.word_count,
                quality_score=score_traf,
                reason="Compare mode benchmark",
                error=res_traf.error,
            )
        )

        # 2. Engine B: BeautifulSoup
        res_bs4 = self.bs4_engine.extract(html, final_url)
        score_bs4 = self._score_engine_result(res_bs4, html)
        results["beautifulsoup"] = res_bs4
        scores["beautifulsoup"] = score_bs4
        attempts.append(
            ExtractionAttempt(
                engine="beautifulsoup",
                status="success" if res_bs4.success else "failed",
                latency_ms=res_bs4.latency_ms,
                char_count=res_bs4.char_count,
                word_count=res_bs4.word_count,
                quality_score=score_bs4,
                reason="Compare mode benchmark",
                error=res_bs4.error,
            )
        )

        # 3. Engine C: Crawl4AI
        res_c4a = await self.crawl4ai_engine.extract(
            final_url,
            html=html,
            render_js=True,
            loading_delay=loading_delay,
            wait_for_selector=wait_for_selector,
        )
        score_c4a = self._score_engine_result(res_c4a, html)
        results["crawl4ai"] = res_c4a
        scores["crawl4ai"] = score_c4a
        attempts.append(
            ExtractionAttempt(
                engine="crawl4ai",
                status="success" if res_c4a.success else "failed",
                latency_ms=res_c4a.latency_ms,
                char_count=res_c4a.char_count,
                word_count=res_c4a.word_count,
                quality_score=score_c4a,
                reason="Compare mode benchmark",
                error=res_c4a.error,
            )
        )

        # 4. Engine D: Playwright (included when requested or benchmark_all)
        if include_playwright:
            res_pw = await self.playwright_engine.extract(
                final_url,
                html=html,
                render_js=True,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )
            score_pw = self._score_engine_result(res_pw, html)
            results["playwright"] = res_pw
            scores["playwright"] = score_pw
            attempts.append(
                ExtractionAttempt(
                    engine="playwright",
                    status="success" if res_pw.success else "failed",
                    latency_ms=res_pw.latency_ms,
                    char_count=res_pw.char_count,
                    word_count=res_pw.word_count,
                    quality_score=score_pw,
                    reason="Compare mode benchmark",
                    error=res_pw.error,
                )
            )

        # Pick winning engine with highest quality score
        winner_engine = max(scores, key=lambda k: scores[k])
        winner_result = results[winner_engine]

        document = self._build_document(
            url=url,
            final_url=final_url,
            engine_result=winner_result,
            raw_html=html,
            http_status=http_status,
            fetched_at=fetched_at,
            page_type=page_type,
            attempts=attempts,
        )

        document.metadata["comparison_report"] = {
            "winner": winner_engine,
            "total_benchmark_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "engines": {
                eng: {
                    "score": scores[eng],
                    "char_count": results[eng].char_count,
                    "word_count": results[eng].word_count,
                    "latency_ms": results[eng].latency_ms,
                    "success": results[eng].success,
                    "error": results[eng].error,
                }
                for eng in results
            },
        }

        return self.quality_checker.check(document)

    async def _run_single_engine(
        self,
        engine_name: str,
        url: str,
        html: str,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
    ) -> EngineResult:
        """Execute a specific engine safely and return its EngineResult."""
        if engine_name == "trafilatura":
            return self.trafilatura_engine.extract(html, url)
        elif engine_name in ("beautifulsoup", "bs4"):
            return self.bs4_engine.extract(html, url)
        elif engine_name == "crawl4ai":
            return await self.crawl4ai_engine.extract(
                url,
                html=html,
                render_js=True,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )
        elif engine_name in ("playwright", "playwrite", "pw"):
            return await self.playwright_engine.extract(
                url,
                html=html,
                render_js=True,
                loading_delay=loading_delay,
                wait_for_selector=wait_for_selector,
            )
        else:
            return EngineResult(
                engine=engine_name,
                success=False,
                error=f"Unknown engine: {engine_name}",
            )

    async def _extract_via_playwright_direct(
        self,
        url: str,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
    ) -> ExtractedDocument:
        """Direct extraction via Playwright Chromium browser without preliminary httpx fetch."""
        start_time = time.perf_counter()
        res = await self.playwright_engine.extract(
            url=url,
            render_js=True,
            loading_delay=loading_delay,
            wait_for_selector=wait_for_selector,
        )
        score = self._score_engine_result(res, raw_html="")

        attempts = [
            ExtractionAttempt(
                engine="playwright",
                status="success" if res.success else "failed",
                latency_ms=res.latency_ms,
                char_count=res.char_count,
                word_count=res.word_count,
                quality_score=score,
                reason="Direct Playwright Chromium capture",
                error=res.error,
            )
        ]

        page_type = self.page_classifier.classify(url=url, html=res.text or "")
        doc = self._build_document(
            url=url,
            final_url=res.canonical_url or url,
            engine_result=res,
            raw_html="",
            http_status=200 if res.success else 500,
            fetched_at=datetime.now(timezone.utc),
            page_type=page_type,
            attempts=attempts,
        )
        return self.quality_checker.check(doc)

    def _score_engine_result(self, res: EngineResult, raw_html: str) -> int:
        """Calculate quality score for an EngineResult."""
        if not res.has_content:
            return 0

        # Check if the extracted content is merely a loading indicator or splash screen
        if is_loading_text(res.text, title=res.title):
            res.warnings.append("محتوای استخراج‌شده به عنوان پیام یا صفحه لودینگ اولیه تشخیص داده شد.")
            return 0

        # Construct temporary lightweight document to score
        temp_doc = ExtractedDocument(
            url="http://temp",
            content_text=res.text,
            content_markdown=res.markdown,
            headings=res.headings,
            links=res.links,
            title=res.title,
            metadata=res.metadata,
        )
        report = self.quality_checker.evaluate(temp_doc, raw_html=raw_html)
        return report.score

    def _build_document(
        self,
        url: str,
        final_url: str,
        engine_result: Optional[EngineResult],
        raw_html: str,
        http_status: Optional[int],
        fetched_at: datetime,
        page_type: str,
        attempts: list[ExtractionAttempt],
    ) -> ExtractedDocument:
        """Assemble standardized ExtractedDocument from an EngineResult."""
        if not engine_result:
            return ExtractedDocument(
                url=url,
                source_url=final_url,
                final_url=final_url,
                http_status=http_status,
                fetched_at=fetched_at,
                page_type=page_type,
                quality_status=QualityStatus.FAILED,
                extraction_attempts=attempts,
            )

        cleaned_text = self.cleaner.normalize_whitespace(engine_result.text or "")
        cleaned_markdown = self.cleaner.normalize_whitespace(engine_result.markdown or "")

        content_hash = (
            self._compute_hash(cleaned_text or cleaned_markdown)
            if (cleaned_text or cleaned_markdown)
            else None
        )

        word_count = len(cleaned_text.split()) if cleaned_text else 0
        char_count = len(cleaned_text) if cleaned_text else 0

        # Extract headings & links if missing from engine_result
        headings = engine_result.headings
        if not headings and raw_html:
            headings = self._extract_headings(raw_html)

        links = engine_result.links
        if not links and raw_html:
            base_domain = self.normalizer.extract_domain(final_url)
            links = self._extract_links(raw_html, final_url, base_domain)

        title = engine_result.title
        if not title and raw_html:
            title = self.link_discovery.extract_title(raw_html)

        description = engine_result.description
        if not description and raw_html:
            description = self.link_discovery.extract_meta_description(raw_html)

        canonical = engine_result.canonical_url
        if not canonical and raw_html:
            canonical = self.link_discovery.extract_canonical_url(raw_html)

        language = engine_result.language
        if not language and raw_html:
            language = self.link_discovery.extract_language(raw_html)

        all_warnings = list(engine_result.warnings)
        if engine_result.error and engine_result.error not in all_warnings:
            all_warnings.append(engine_result.error)

        doc = ExtractedDocument(
            document_id=hashlib.sha256(f"{final_url}:{time.time()}".encode()).hexdigest()[:16],
            url=url,
            source_url=final_url,
            final_url=final_url,
            canonical_url=canonical,
            title=title,
            description=description,
            language=language,
            headings=headings,
            content_markdown=cleaned_markdown or None,
            content_text=cleaned_text or None,
            links=links,
            http_status=http_status,
            fetched_at=fetched_at,
            content_hash=content_hash,
            word_count=word_count,
            character_count=char_count,
            extraction_method=engine_result.engine,
            extractor_used=engine_result.engine,
            quality_status=QualityStatus.SUCCESS if cleaned_text else QualityStatus.EMPTY,
            warnings=all_warnings,
            page_type=page_type,
            extraction_attempts=attempts,
            metadata=dict(engine_result.metadata),
        )

        # Calculate QualityReport
        report = self.quality_checker.evaluate(doc, raw_html=raw_html)
        doc.quality_report = report
        return doc

    async def _extract_via_crawl4ai_direct(
        self,
        url: str,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
    ) -> ExtractedDocument:
        """Direct extraction via Crawl4AI browser without preliminary httpx fetch."""
        start_time = time.perf_counter()
        res = await self.crawl4ai_engine.extract(
            url=url,
            render_js=True,
            loading_delay=loading_delay,
            wait_for_selector=wait_for_selector,
        )
        score = self._score_engine_result(res, raw_html="")

        attempts = [
            ExtractionAttempt(
                engine="crawl4ai",
                status="success" if res.success else "failed",
                latency_ms=res.latency_ms,
                char_count=res.char_count,
                word_count=res.word_count,
                quality_score=score,
                reason="Direct Crawl4AI browser capture",
                error=res.error,
            )
        ]

        page_type = self.page_classifier.classify(url=url, html=res.text or "")
        doc = self._build_document(
            url=url,
            final_url=res.canonical_url or url,
            engine_result=res,
            raw_html="",
            http_status=200 if res.success else 500,
            fetched_at=datetime.now(timezone.utc),
            page_type=page_type,
            attempts=attempts,
        )
        return self.quality_checker.check(doc)

    async def _safe_fetch_html(self, url: str, timeout: float = 30.0) -> dict[str, Any]:
        """Fetch page content with manual redirect following and SSRF validation."""
        fetched_at = datetime.now(timezone.utc)
        current_url: str = url
        redirect_count: int = 0
        max_redirects: int = 5
        visited_set: set[str] = {current_url}

        client_to_use = self.client
        own_client: Optional[httpx.AsyncClient] = None
        if client_to_use is None:
            own_client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                headers={"User-Agent": "Crawlbot/1.0 (+https://github.com/crawlbot)"},
            )
            client_to_use = own_client

        try:
            while redirect_count <= max_redirects:
                is_safe, safety_reason = validate_url_safety(current_url)
                if not is_safe:
                    return {
                        "failed": True,
                        "error": f"URL safety validation failed at {current_url}: {safety_reason}",
                        "final_url": current_url,
                        "status_code": 403,
                        "fetched_at": fetched_at,
                    }

                response = await client_to_use.get(current_url, follow_redirects=False)
                http_status = response.status_code
                fetched_at = datetime.now(timezone.utc)

                if http_status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        break
                    has_slash = location.rstrip().endswith("/") or (
                        urlparse(location).path.endswith("/") and urlparse(location).path != "/"
                    )
                    next_url = self.normalizer.normalize(
                        location,
                        base_url=current_url,
                        preserve_trailing_slash=has_slash,
                    )
                    if not next_url:
                        break
                    if next_url == current_url or next_url in visited_set:
                        return {
                            "failed": True,
                            "error": f"Redirect loop detected: {current_url} -> {next_url}",
                            "final_url": current_url,
                            "status_code": http_status,
                            "fetched_at": fetched_at,
                        }
                    redirect_count += 1
                    if redirect_count > max_redirects:
                        return {
                            "failed": True,
                            "error": f"Too many redirects encountered ({redirect_count})",
                            "final_url": current_url,
                            "status_code": http_status,
                            "fetched_at": fetched_at,
                        }
                    visited_set.add(next_url)
                    current_url = next_url
                    continue

                if response.status_code >= 400:
                    return {
                        "failed": True,
                        "error": f"HTTP error status {response.status_code}",
                        "final_url": current_url,
                        "status_code": response.status_code,
                        "fetched_at": fetched_at,
                    }

                return {
                    "failed": False,
                    "html": response.text,
                    "final_url": current_url,
                    "status_code": http_status,
                    "fetched_at": fetched_at,
                }

            return {
                "failed": False,
                "html": "",
                "final_url": current_url,
                "status_code": 200,
                "fetched_at": fetched_at,
            }
        except httpx.TimeoutException:
            return {
                "failed": True,
                "error": f"Request timed out after {timeout} seconds",
                "final_url": current_url,
                "status_code": 408,
                "fetched_at": fetched_at,
            }
        except Exception as exc:
            return {
                "failed": True,
                "error": f"Fetch failed: {str(exc)}",
                "final_url": current_url,
                "status_code": 500,
                "fetched_at": fetched_at,
            }
        finally:
            if own_client is not None:
                await own_client.aclose()

    def _discover_spa_endpoints(self, html: str) -> tuple[list[str], list[str]]:
        """Identify client-side routes and public API endpoints from script tags and initial state."""
        routes: set[str] = set()
        apis: set[str] = set()

        if not html:
            return [], []

        # Find API URLs mentioned in scripts or data attributes
        api_pattern = re.compile(r'["\'](/(?:api|v[1-9]|graphql|data)/[^"\'\s>]+)["\']', re.I)
        for match in api_pattern.findall(html):
            apis.add(match)

        # Find client-side route paths like path: '/docs/:id' or "path": "/docs"
        route_pattern = re.compile(r'(?:["\']?(?:path|route)["\']?)\s*:\s*["\'](/[a-zA-Z0-9_\-/:]+)["\']', re.I)
        for match in route_pattern.findall(html):
            routes.add(match)

        return sorted(list(routes)), sorted(list(apis))

    # --- Backward compatibility methods ---

    def _extract_with_trafilatura(self, html: str, url: str) -> tuple[str | None, str | None]:
        res = self.trafilatura_engine.extract(html, url)
        return res.markdown, res.text

    def _extract_with_bs4(self, html: str, url: str) -> tuple[str | None, str | None]:
        res = self.bs4_engine.extract(html, url)
        return res.markdown, res.text

    def _extract_headings(self, html: str) -> list[Heading]:
        return self.bs4_engine._extract_headings(BeautifulSoup(html or "", "html.parser"))

    def _compute_hash(self, content: str) -> str:
        if not content:
            return ""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _extract_links(self, html: str, page_url: str, base_domain: str) -> list[LinkInfo]:
        return self.bs4_engine._extract_links(BeautifulSoup(html or "", "html.parser"), page_url, base_domain)

    def _html_to_markdown(self, soup: BeautifulSoup) -> str:
        return self.bs4_engine._html_to_markdown(soup)


# Service alias
ExtractorService = ContentExtractor
