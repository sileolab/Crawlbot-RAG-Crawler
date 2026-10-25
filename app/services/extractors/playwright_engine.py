"""Engine D: Playwright Chromium Extraction Engine.

Standalone extraction engine utilizing native Playwright and Chromium
for high-fidelity headless browser rendering, client-side SPA execution,
dynamic JS injection extraction, and clean Markdown generation.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag

from app.schemas.documents import Heading, LinkInfo
from app.services.loading_detector import get_loader_remover_js, is_loading_text
from .base import EngineResult

logger = logging.getLogger(__name__)

# Ensure Windows ProactorEventLoop is active for subprocess spawning
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass


class PlaywrightEngine:
    """Engine D implementation using native Playwright Chromium."""

    ENGINE_NAME: str = "playwright"

    def __init__(
        self,
        headless: bool = True,
        page_timeout: int = 35000,
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
        """Extract web content using native Playwright Chromium browser.

        Seamlessly supports Windows environments where the main event loop
        is a SelectorEventLoop by delegating execution to a dedicated Proactor worker thread.
        """
        # Fast path: If rendering is disabled and HTML is supplied, parse statically
        if not render_js and html and html.strip():
            start_time = time.perf_counter()
            result = EngineResult(engine=self.ENGINE_NAME, success=False)
            self._parse_static_html(html, url, result)
            result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            result.browser_used = False
            return result

        # On Windows, check if the current event loop can run subprocesses.
        # If running inside a SelectorEventLoop (common in Uvicorn on Windows),
        # async_playwright fails with NotImplementedError. We route to a dedicated Proactor thread.
        if sys.platform == "win32":
            is_proactor = False
            try:
                loop = asyncio.get_running_loop()
                is_proactor = isinstance(loop, getattr(asyncio, "ProactorEventLoop", ()))
            except RuntimeError:
                pass

            if not is_proactor:
                logger.info(
                    "Current loop does not support Windows subprocesses. "
                    "Routing Playwright to dedicated Proactor thread for %s",
                    url,
                )
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    self._extract_in_dedicated_proactor_thread,
                    url,
                    html,
                    render_js,
                    loading_delay,
                    wait_for_selector,
                )

        return await self._extract_internal(
            url=url,
            html=html,
            render_js=render_js,
            loading_delay=loading_delay,
            wait_for_selector=wait_for_selector,
        )

    def _extract_in_dedicated_proactor_thread(
        self,
        url: str,
        html: Optional[str] = None,
        render_js: bool = True,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
    ) -> EngineResult:
        """Run Playwright extraction inside an isolated worker thread with dedicated ProactorEventLoop."""
        thread_loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(thread_loop)
        try:
            return thread_loop.run_until_complete(
                self._extract_internal(
                    url=url,
                    html=html,
                    render_js=render_js,
                    loading_delay=loading_delay,
                    wait_for_selector=wait_for_selector,
                )
            )
        finally:
            try:
                thread_loop.close()
            except Exception:
                pass

    async def _extract_internal(
        self,
        url: str,
        html: Optional[str] = None,
        render_js: bool = True,
        loading_delay: float = 0.0,
        wait_for_selector: Optional[str] = None,
    ) -> EngineResult:
        """Core internal extraction routine utilizing Playwright async API."""
        start_time = time.perf_counter()
        result = EngineResult(engine=self.ENGINE_NAME, success=False)

        # Fast path: If rendering is disabled and HTML is supplied, parse statically
        if not render_js and html and html.strip():
            self._parse_static_html(html, url, result)
            result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            result.browser_used = False
            return result

        try:
            from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
        except ImportError as imp_err:
            logger.error("Playwright is not installed or import failed: %s", imp_err)
            result.error = f"Playwright not installed: {str(imp_err)}"
            result.warnings.append(f"پلی‌رایت روی سیستم نصب نشده است: {str(imp_err)}")
            result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return result

        effective_delay = loading_delay if loading_delay > 0 else self.default_loading_delay
        # Prefer domcontentloaded for fast reliable navigation without hanging on analytics / streaming
        nav_wait = "domcontentloaded" if self.wait_until in ("domcontentloaded", "load", "networkidle") else self.wait_until

        browser = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-accelerated-2d-canvas",
                        "--no-first-run",
                        "--no-zygote",
                    ],
                )

                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1366, "height": 768},
                    locale="fa-IR,en-US,en;q=0.9",
                    ignore_https_errors=True,
                )

                page = await context.new_page()

                # 1. Navigate to target URL with fallback
                response = None
                try:
                    response = await page.goto(url, wait_until=nav_wait, timeout=self.page_timeout)
                except Exception as nav_err:
                    logger.info("Playwright initial %s navigation for %s hit: %s; retrying with commit", nav_wait, url, nav_err)
                    try:
                        response = await page.goto(url, wait_until="commit", timeout=15000)
                    except Exception as retry_err:
                        result.error = f"Navigation failed: {str(retry_err)}"
                        result.warnings.append(f"خطای بارگذاری آدرس: {str(retry_err)}")
                        result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                        return result

                http_status = response.status if response else 200

                # 2. Wait for client-side hydration (React/Vue/vanilla SPA mount)
                hydration_ms = int(effective_delay * 1000) if effective_delay > 0 else 1800
                await page.wait_for_timeout(hydration_ms)

                # Non-blocking networkidle wait: give pending requests up to 3s to settle without blocking
                try:
                    await page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass

                # 3. Wait for specific selector if requested
                if wait_for_selector:
                    try:
                        await page.wait_for_selector(wait_for_selector, timeout=12000)
                    except Exception as sel_err:
                        logger.warning("Wait for selector '%s' on %s timed out: %s", wait_for_selector, url, sel_err)
                        result.warnings.append(f"سلکتور انتظار '{wait_for_selector}' در زمان مشخص یافت نشد.")

                # 4. Remove loader overlays and preloader screens
                try:
                    await page.evaluate(get_loader_remover_js())
                except Exception as js_err:
                    logger.debug("Loader remover JS evaluation error on %s: %s", url, js_err)

                # 5. Smooth auto-scroll to trigger lazy loading / IntersectionObservers
                try:
                    await page.evaluate("""async () => {
                        await new Promise((resolve) => {
                            let totalHeight = 0;
                            let distance = 350;
                            let timer = setInterval(() => {
                                let scrollHeight = document.body ? document.body.scrollHeight : 0;
                                window.scrollBy(0, distance);
                                totalHeight += distance;
                                if (totalHeight >= scrollHeight || totalHeight > 12000) {
                                    clearInterval(timer);
                                    window.scrollTo(0, 0);
                                    resolve();
                                }
                            }, 50);
                        });
                    }""")
                    await page.wait_for_timeout(500)
                except Exception as scroll_err:
                    logger.debug("Auto-scroll evaluation error on %s: %s", url, scroll_err)

                # 6. Extract DOM and metadata
                title = await page.title()
                rendered_html = await page.content()
                final_page_url = page.url or url

                self._parse_rendered_dom(
                    html=rendered_html,
                    url=final_page_url,
                    title=title,
                    http_status=http_status,
                    result=result,
                )

                # 7. Check if extracted content is sparse or a loading screen; retry with adaptive wait
                if not result.has_content or is_loading_text(result.text, title=result.title):
                    logger.info("Playwright initial capture on %s was sparse or a loading screen. Running adaptive wait...", url)
                    await page.wait_for_timeout(2500)
                    try:
                        await page.evaluate(get_loader_remover_js())
                    except Exception:
                        pass
                    new_html = await page.content()
                    new_title = await page.title()
                    candidate = EngineResult(engine=self.ENGINE_NAME, success=False)
                    self._parse_rendered_dom(
                        html=new_html,
                        url=final_page_url,
                        title=new_title,
                        http_status=http_status,
                        result=candidate,
                    )
                    if candidate.has_content and not is_loading_text(candidate.text, title=candidate.title):
                        candidate.warnings.append("محتوای پویا پس از عبور خودکار از صفحه لودینگ اولیه با Playwright استخراج شد.")
                        result = candidate
                    elif (candidate.char_count or 0) > (result.char_count or 0):
                        result = candidate

                result.browser_used = True

        except NotImplementedError as nie:
            # If NotImplementedError happened inside this process, fallback to dedicated thread
            if sys.platform == "win32":
                logger.info("Caught NotImplementedError on Windows; falling back to dedicated Proactor thread for %s", url)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    self._extract_in_dedicated_proactor_thread,
                    url,
                    html,
                    render_js,
                    loading_delay,
                    wait_for_selector,
                )
            result.error = f"Subprocess not implemented: {str(nie)}"
            result.success = False

        except Exception as exc:
            logger.warning("Playwright extraction error for %s: %s", url, exc)
            result.error = f"Playwright error: {str(exc)}"
            result.warnings.append(f"خطای اجرای Playwright: {str(exc)}")
            result.success = False

        result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return result

    def _parse_static_html(self, html: str, url: str, result: EngineResult) -> None:
        """Parse raw HTML without launching a browser."""
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None
        self._parse_rendered_dom(html=html, url=url, title=title, http_status=200, result=result)

    def _parse_rendered_dom(
        self,
        html: str,
        url: str,
        title: Optional[str],
        http_status: int,
        result: EngineResult,
    ) -> None:
        """Parse rendered HTML string into structured markdown, headings, links, and metadata."""
        soup = BeautifulSoup(html, "html.parser")

        # 1. Extract Headings in document order
        headings: list[Heading] = []
        for h_tag in soup.find_all(re.compile(r"^h[1-6]$", re.IGNORECASE)):
            try:
                level = int(h_tag.name[1])
                htext = h_tag.get_text(strip=True)
                if htext:
                    headings.append(Heading(level=level, text=htext))
            except (ValueError, IndexError):
                continue

        # 2. Extract Hyperlinks
        links: list[LinkInfo] = []
        source_domain = urlsplit(url).netloc.lower()
        seen_links: set[str] = set()

        for a_tag in soup.find_all("a", href=True):
            raw_href = a_tag.get("href", "").strip()
            if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            full_url = urljoin(url, raw_href)
            if full_url in seen_links:
                continue
            seen_links.add(full_url)

            link_text = a_tag.get_text(strip=True)
            link_domain = urlsplit(full_url).netloc.lower()
            is_internal = (link_domain == source_domain) or (not link_domain)
            links.append(LinkInfo(url=full_url, text=link_text, is_internal=is_internal))

        # 3. Extract Meta tags
        meta_dict: dict[str, Any] = {"http_status": http_status}
        description: Optional[str] = None
        canonical_url: Optional[str] = None

        meta_desc = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"].strip()

        og_desc = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
        if not description and og_desc and og_desc.get("content"):
            description = og_desc["content"].strip()

        link_canon = soup.find("link", attrs={"rel": re.compile(r"^canonical$", re.I)})
        if link_canon and link_canon.get("href"):
            canonical_url = urljoin(url, link_canon["href"].strip())

        # Collect OpenGraph metadata
        og_data: dict[str, str] = {}
        for og_tag in soup.find_all("meta", attrs={"property": re.compile(r"^og:", re.I)}):
            prop = og_tag.get("property", "")
            val = og_tag.get("content", "")
            if prop and val:
                og_data[prop] = val.strip()
        if og_data:
            meta_dict["open_graph"] = og_data

        # Detect Language
        html_tag = soup.find("html")
        language: Optional[str] = None
        if html_tag and html_tag.get("lang"):
            language = html_tag["lang"].strip().lower()

        # 4. Remove noise elements before generating Markdown and text
        for tag in soup.find_all(["script", "style", "noscript", "svg", "canvas", "iframe", "link"]):
            tag.decompose()

        # Generate Structured Markdown
        markdown_content = self._dom_to_markdown(soup)
        clean_markdown = re.sub(r"[ \t]+", " ", markdown_content)
        clean_markdown = re.sub(r"\n{3,}", "\n\n", clean_markdown).strip()

        # Generate Clean Text
        plain_text = soup.get_text(separator="\n", strip=True)
        clean_text = re.sub(r"[ \t]+", " ", plain_text)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

        # Populate result fields
        result.title = title or (headings[0].text if headings else None)
        result.description = description
        result.canonical_url = canonical_url
        result.language = language
        result.headings = headings
        result.links = links
        result.metadata = meta_dict
        result.markdown = clean_markdown or None
        result.text = clean_text or None
        result.compute_counts()
        result.success = result.has_content

        if not result.success:
            result.warnings.append("Playwright returned empty textual content")

    def _dom_to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert cleaned DOM tree into clean structured Markdown."""
        container = soup.find("main") or soup.find(id=lambda i: i in ("root", "app", "__next", "content", "main")) or soup.body or soup

        def _render_node(node: Any) -> str:
            if isinstance(node, NavigableString):
                return str(node)
            if not isinstance(node, Tag):
                return ""

            name = node.name.lower() if node.name else ""

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                inner = "".join(_render_node(c) for c in node.children).strip()
                return f"\n\n{'#' * level} {inner}\n\n" if inner else ""

            if name == "p":
                inner = "".join(_render_node(c) for c in node.children).strip()
                return f"\n\n{inner}\n\n" if inner else ""

            if name in ("strong", "b"):
                inner = "".join(_render_node(c) for c in node.children).strip()
                return f"**{inner}**" if inner else ""

            if name in ("em", "i"):
                inner = "".join(_render_node(c) for c in node.children).strip()
                return f"*{inner}*" if inner else ""

            if name == "code":
                if node.parent and node.parent.name == "pre":
                    return node.get_text()
                inner = node.get_text().strip()
                return f"`{inner}`" if inner else ""

            if name == "pre":
                code_text = node.get_text().strip()
                return f"\n\n```\n{code_text}\n```\n\n" if code_text else ""

            if name == "blockquote":
                inner = "".join(_render_node(c) for c in node.children).strip()
                quoted = "\n".join(f"> {line}" if line else ">" for line in inner.split("\n"))
                return f"\n\n{quoted}\n\n" if inner else ""

            if name == "a":
                href = node.get("href", "").strip()
                inner = "".join(_render_node(c) for c in node.children).strip()
                if href and inner:
                    return f"[{inner}]({href})"
                return inner or href

            if name in ("ul", "ol"):
                items: list[str] = []
                idx = 1
                for li in node.find_all("li", recursive=False):
                    li_text = "".join(_render_node(c) for c in li.children).strip()
                    if li_text:
                        bullet = f"{idx}. " if name == "ol" else "- "
                        items.append(f"{bullet}{li_text}")
                        idx += 1
                return ("\n\n" + "\n".join(items) + "\n\n") if items else ""

            if name == "table":
                return self._render_table(node)

            if name in ("br", "hr"):
                return "\n\n---\n\n" if name == "hr" else "\n"

            # Generic container tags: traverse children
            return "".join(_render_node(c) for c in node.children)

        return _render_node(container)

    def _render_table(self, table: Tag) -> str:
        """Render HTML table element to Markdown table."""
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if not rows:
            return ""

        max_cols = max(len(r) for r in rows)
        padded = [r + [""] * (max_cols - len(r)) for r in rows]
        header = padded[0]

        lines = ["\n\n| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * max_cols) + " |"]
        for r in padded[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines) + "\n\n"
