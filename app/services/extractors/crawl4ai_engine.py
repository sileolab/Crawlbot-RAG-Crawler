"""Engine C: Crawl4AI Extraction Engine.

Standalone extraction engine utilizing Crawl4AI and Playwright for
headless browser rendering, client-side SPA execution, and markdown generation.
Includes automated loading screen detection, smart delays, overlay removal,
and retry mechanisms for dynamic websites.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional
from urllib.parse import urlsplit

from app.schemas.documents import Heading, LinkInfo
from app.services.loading_detector import get_loader_remover_js, is_loading_text
from .base import EngineResult

logger = logging.getLogger(__name__)


class Crawl4AIEngine:
    """Engine C implementation using Crawl4AI AsyncWebCrawler."""

    ENGINE_NAME: str = "crawl4ai"

    def __init__(
        self,
        headless: bool = True,
        page_timeout: int = 30000,
        wait_until: str = "domcontentloaded",
        default_loading_delay: float = 0.0,
    ) -> None:
        self.headless = headless
        self.page_timeout = page_timeout
        self.wait_until = wait_until
        self.default_loading_delay = default_loading_delay

    async def extract(
        self,
        url: str,
        html: Optional[str] = None,
        render_js: bool = True,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
    ) -> EngineResult:
        """Extract web content using Crawl4AI AsyncWebCrawler.

        Args:
            url: Target URL to crawl and render.
            html: Optional raw HTML string (used with raw: prefix if render_js is False).
            render_js: Whether to launch headless browser for dynamic JS rendering.
            loading_delay: Manual delay in seconds to wait for initial loading screens.
            wait_for_selector: Optional CSS/XPath selector to wait for before extraction.

        Returns:
            Standardized EngineResult.
        """
        start_time = time.perf_counter()
        result = EngineResult(engine=self.ENGINE_NAME, success=False)

        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
            from crawl4ai.async_configs import CacheMode
        except ImportError as imp_err:
            logger.error("Crawl4AI is not installed or import failed: %s", imp_err)
            result.error = f"Crawl4AI not installed: {str(imp_err)}"
            result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return result

        browser_config = BrowserConfig(
            headless=self.headless,
            verbose=False,
            java_script_enabled=True,
            ignore_https_errors=True,
        )

        effective_delay = loading_delay if loading_delay > 0 else self.default_loading_delay
        wait_until_val = "networkidle" if effective_delay >= 2.0 else self.wait_until

        run_config = CrawlerRunConfig(
            wait_until=wait_until_val,
            page_timeout=self.page_timeout,
            cache_mode=CacheMode.BYPASS,
            scan_full_page=False,
            process_iframes=False,
            verbose=False,
            remove_overlay_elements=True,
            simulate_user=True,
            js_code=get_loader_remover_js(),
        )

        if effective_delay > 0:
            run_config.delay_before_return_html = effective_delay

        if wait_for_selector:
            run_config.wait_for = wait_for_selector

        target_url = url
        browser_used = True

        # If render_js is disabled and html is supplied, we can bypass network
        if not render_js and html and html.strip():
            target_url = f"raw:{html}"
            browser_used = False

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                crawl_res = await crawler.arun(url=target_url, config=run_config)

                if not crawl_res.success:
                    result.error = crawl_res.error_message or "Crawl4AI run failed"
                    result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    return result

                result = self._parse_crawl_result(crawl_res, url, browser_used)

                # Check if the extracted content is an initial loading screen or splash
                if browser_used and not target_url.startswith("raw:"):
                    is_loader = is_loading_text(result.text, title=result.title)
                    is_empty = not result.text or len(result.text.strip()) < 80

                    if is_loader or (is_empty and effective_delay < 2.0):
                        logger.info(
                            "Loading screen detected on %s (is_loader=%s, is_empty=%s). "
                            "Triggering automatic second-chance retry with extended wait...",
                            url,
                            is_loader,
                            is_empty,
                        )
                        retry_delay = max(effective_delay + 3.0, 4.0)
                        retry_config = CrawlerRunConfig(
                            wait_until="networkidle",
                            wait_for_timeout=15000,
                            delay_before_return_html=retry_delay,
                            remove_overlay_elements=True,
                            simulate_user=True,
                            js_code=get_loader_remover_js(),
                            page_timeout=self.page_timeout + 10000,
                            cache_mode=CacheMode.BYPASS,
                            scan_full_page=True,
                            verbose=False,
                        )
                        if wait_for_selector:
                            retry_config.wait_for = wait_for_selector

                        retry_res = await crawler.arun(url=target_url, config=retry_config)
                        if retry_res.success:
                            candidate_result = self._parse_crawl_result(retry_res, url, browser_used=True)
                            candidate_is_loader = is_loading_text(candidate_result.text, title=candidate_result.title)
                            if candidate_result.text and not candidate_is_loader:
                                candidate_result.warnings.append("محتوای پویا پس از عبور خودکار از صفحه لودینگ اولیه استخراج شد.")
                                result = candidate_result
                            elif (candidate_result.char_count or 0) > (result.char_count or 0):
                                result = candidate_result

        except Exception as exc:
            logger.warning("Crawl4AI extraction error for %s: %s", url, exc)
            result.error = f"Crawl4AI error: {str(exc)}"
            result.success = False

        result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return result

    def _parse_crawl_result(
        self,
        crawl_res: Any,
        url: str,
        browser_used: bool,
    ) -> EngineResult:
        """Parse Crawl4AI crawl result into a standardized EngineResult."""
        result = EngineResult(engine=self.ENGINE_NAME, success=False)

        # Extract markdown string
        md_content = ""
        if crawl_res.markdown:
            md_content = str(crawl_res.markdown).strip()

        # Extract cleaned text from cleaned_html or markdown
        cleaned_html = crawl_res.cleaned_html or crawl_res.html or ""
        text_content = ""
        if cleaned_html:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(cleaned_html, "html.parser")
            text_content = soup.get_text(separator="\n", strip=True)
        elif md_content:
            text_content = re.sub(r"[#*`_\[\]()>-]", " ", md_content)
            text_content = re.sub(r"\s+", " ", text_content).strip()

        # Extract metadata
        meta = crawl_res.metadata or {}
        title = meta.get("title")
        description = meta.get("description") or meta.get("og:description")
        canonical = crawl_res.redirected_url or meta.get("canonical_url")

        # Extract headings from markdown or HTML
        headings: list[Heading] = []
        if md_content:
            for line in md_content.splitlines():
                match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
                if match:
                    level = len(match.group(1))
                    htext = match.group(2).strip()
                    if htext:
                        headings.append(Heading(level=level, text=htext))
        elif cleaned_html:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(cleaned_html, "html.parser")
            for h_tag in soup.find_all(re.compile(r"^h[1-6]$")):
                try:
                    level = int(h_tag.name[1])
                    htext = h_tag.get_text(strip=True)
                    if htext:
                        headings.append(Heading(level=level, text=htext))
                except (ValueError, IndexError):
                    continue

        # Extract links from crawl_res.links
        links: list[LinkInfo] = []
        source_domain = urlsplit(url).netloc.lower()
        raw_links = crawl_res.links or {}

        def process_link_item(item: Any, default_internal: bool) -> None:
            if isinstance(item, dict):
                l_url = item.get("href") or item.get("url")
                l_text = item.get("text") or item.get("title") or ""
                if l_url:
                    l_domain = urlsplit(l_url).netloc.lower()
                    is_int = (l_domain == source_domain) or (not l_domain) or default_internal
                    links.append(LinkInfo(url=str(l_url), text=str(l_text), is_internal=is_int))
            elif hasattr(item, "href"):
                l_url = getattr(item, "href", "")
                l_text = getattr(item, "text", "") or ""
                if l_url:
                    l_domain = urlsplit(l_url).netloc.lower()
                    is_int = (l_domain == source_domain) or (not l_domain) or default_internal
                    links.append(LinkInfo(url=str(l_url), text=str(l_text), is_internal=is_int))

        if isinstance(raw_links, dict):
            for item in raw_links.get("internal", []):
                process_link_item(item, default_internal=True)
            for item in raw_links.get("external", []):
                process_link_item(item, default_internal=False)
        elif isinstance(raw_links, list):
            for item in raw_links:
                process_link_item(item, default_internal=False)

        result.markdown = md_content or None
        result.text = text_content or None
        result.title = title
        result.description = description
        result.canonical_url = canonical
        result.headings = headings
        result.links = links
        result.metadata = meta
        result.browser_used = browser_used
        result.compute_counts()
        result.success = result.has_content

        if not result.success:
            result.warnings.append("Crawl4AI returned empty textual content")

        return result
