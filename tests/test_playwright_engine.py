"""Unit and Integration Tests for Playwright Chromium Extraction Engine.

Verifies:
1. PlaywrightEngine initialization and attributes.
2. Fast static HTML parsing when render_js is False.
3. Markdown table, list, code block, and heading generation.
4. Request schema compliance and 'playwrite'/'pw' alias mapping.
5. ContentExtractor integration with strategy="playwright".
6. Browser lifecycle, timeout handling, and graceful error recovery.
7. Auto Mode fallback to Playwright on SPA shells.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.schemas.documents import ExtractedDocument, Heading, LinkInfo, QualityStatus
from app.schemas.requests import ExtractionRequest, ExtractionStrategy, SingleExtractRequest
from app.services.extractor import ContentExtractor
from app.services.extractors.base import EngineResult
from app.services.extractors.playwright_engine import PlaywrightEngine


# ---------------------------------------------------------------------------
# 1. Engine Initialization & Attributes
# ---------------------------------------------------------------------------
def test_playwright_engine_initialization():
    engine = PlaywrightEngine(headless=True, page_timeout=25000, wait_until="networkidle")
    assert engine.ENGINE_NAME == "playwright"
    assert engine.headless is True
    assert engine.page_timeout == 25000
    assert engine.wait_until == "networkidle"


# ---------------------------------------------------------------------------
# 2. Static HTML Parsing (render_js=False)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_playwright_static_html_parsing():
    engine = PlaywrightEngine()
    html = """
    <!DOCTYPE html>
    <html lang="fa">
    <head>
        <title>صفحه آزمایشی پلی‌رایت</title>
        <meta name="description" content="توضیحات متای صفحه آزمون">
        <link rel="canonical" href="https://example.com/test">
    </head>
    <body>
        <h1>عنوان اصلی مقاله</h1>
        <p>این یک پاراگراف آزمایشی است.</p>
        <h2>بخش دوم</h2>
        <ul>
            <li>مورد اول</li>
            <li>مورد دوم</li>
        </ul>
        <a href="/about">درباره ما</a>
        <a href="https://external.com/docs">مستندات خارجی</a>
    </body>
    </html>
    """
    res = await engine.extract(url="https://example.com/test", html=html, render_js=False)
    assert res.success is True
    assert res.browser_used is False
    assert res.title == "صفحه آزمایشی پلی‌رایت"
    assert res.description == "توضیحات متای صفحه آزمون"
    assert res.canonical_url == "https://example.com/test"
    assert res.language == "fa"
    assert len(res.headings) == 2
    assert res.headings[0].text == "عنوان اصلی مقاله"
    assert res.headings[0].level == 1
    assert len(res.links) == 2
    assert res.links[0].is_internal is True
    assert res.links[1].is_internal is False
    assert res.word_count > 0
    assert "# عنوان اصلی مقاله" in (res.markdown or "")


# ---------------------------------------------------------------------------
# 3. DOM to Markdown Generation (Tables, Lists, Code)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_playwright_markdown_generation_features():
    engine = PlaywrightEngine()
    html = """
    <div>
        <h2>جدول داده‌ها</h2>
        <table>
            <tr><th>نام</th><th>نقش</th></tr>
            <tr><td>یونس</td><td>توسعه‌دهنده هوش مصنوعی</td></tr>
            <tr><td>رضا</td><td>مهندس داده</td></tr>
        </table>
        <pre><code>def hello():\n    return "world"</code></pre>
        <blockquote>نقل قول مهم</blockquote>
    </div>
    """
    res = await engine.extract(url="https://example.com", html=html, render_js=False)
    assert res.markdown is not None
    assert "## جدول داده‌ها" in res.markdown
    assert "| نام | نقش |" in res.markdown
    assert "---" in res.markdown
    assert "```" in res.markdown
    assert "> نقل قول مهم" in res.markdown


# ---------------------------------------------------------------------------
# 4. Request Schema & Alias Mapping ('playwrite', 'pw' -> 'playwright')
# ---------------------------------------------------------------------------
def test_playwright_schema_and_aliases():
    assert ExtractionStrategy.PLAYWRIGHT.value == "playwright"

    # Exact name
    req = ExtractionRequest(urls=["https://example.com"], extraction_strategy="playwright")
    assert req.extraction_strategy == "playwright"

    # Alias: 'playwrite' (as written in user prompt)
    req_alias1 = ExtractionRequest.model_validate({
        "urls": ["https://example.com"],
        "strategy": "playwrite",
    })
    assert req_alias1.extraction_strategy == "playwright"

    # Alias: 'pw'
    req_alias2 = ExtractionRequest.model_validate({
        "urls": ["https://example.com"],
        "extractor": "pw",
    })
    assert req_alias2.extraction_strategy == "playwright"

    # SingleExtractRequest
    single_req = SingleExtractRequest(
        url="https://example.com",
        extraction_strategy="playwright",
    )
    assert single_req.extraction_strategy == "playwright"


# ---------------------------------------------------------------------------
# 5. Mocked Playwright Browser Execution
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_playwright_engine_mocked_browser_flow():
    engine = PlaywrightEngine(headless=True)

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
    mock_page.title = AsyncMock(return_value="پورتفولیو یونس ابراهیمی")
    mock_page.content = AsyncMock(return_value="""
        <html>
            <body>
                <h1>یونس ابراهیمی</h1>
                <p>توسعه‌دهنده هوش مصنوعی و مهندسی نرم‌افزار</p>
            </body>
        </html>
    """)
    mock_page.url = "https://younesebrahimi.ir/"
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.evaluate = AsyncMock()

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    mock_pw = AsyncMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

    with patch("playwright.async_api.async_playwright") as mock_pw_ctx:
        mock_pw_ctx.return_value.__aenter__.return_value = mock_pw

        res = await engine.extract(
            url="https://younesebrahimi.ir/",
            render_js=True,
            loading_delay=1.5,
            wait_for_selector=".app-wrapper",
        )

        assert res.success is True
        assert res.browser_used is True
        assert res.title == "پورتفولیو یونس ابراهیمی"
        assert "یونس ابراهیمی" in (res.text or "")
        mock_page.goto.assert_called_once()
        mock_page.wait_for_selector.assert_called_once_with(".app-wrapper", timeout=12000)


# ---------------------------------------------------------------------------
# 6. Error & Timeout Handling in PlaywrightEngine
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_playwright_engine_timeout_handling():
    engine = PlaywrightEngine(headless=True)

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(side_effect=RuntimeError("Connection refused by target"))

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    mock_pw = AsyncMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

    with patch("playwright.async_api.async_playwright") as mock_pw_ctx:
        mock_pw_ctx.return_value.__aenter__.return_value = mock_pw

        res = await engine.extract(url="https://unreachable.example.com", render_js=True)
        assert res.success is False
        assert "Navigation failed" in (res.error or "")


# ---------------------------------------------------------------------------
# 7. ContentExtractor Integration with strategy="playwright"
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_content_extractor_with_playwright_strategy():
    extractor = ContentExtractor(strategy="playwright")

    mock_result = EngineResult(
        engine="playwright",
        success=True,
        text="این متن استخراج‌شده واقعی با مرورگر Chromium است. " * 8,
        markdown="# عنوان اصلی\n\nمتن استخراج شده.",
        title="تست استخراج پلی‌رایت",
        char_count=350,
        word_count=55,
    )

    with patch.object(extractor.playwright_engine, "extract", new=AsyncMock(return_value=mock_result)):
        doc = await extractor.extract("https://example.com/playwright-test")
        assert doc.extractor_used == "playwright"
        assert doc.quality_status == QualityStatus.SUCCESS
        assert doc.word_count > 50
        assert "Chromium" in (doc.content_text or "")


# ---------------------------------------------------------------------------
# 8. Auto Mode Fallback to Playwright on Empty Static SPA Shell
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_playwright_on_sparse_spa():
    extractor = ContentExtractor(strategy="auto", min_quality_score=40)

    # Empty SPA HTML shell with only bundle script
    spa_html = """
    <html>
        <head><script src="/bundle.js"></script></head>
        <body><div id="root"></div></body>
    </html>
    """

    mock_pw_result = EngineResult(
        engine="playwright",
        success=True,
        text="محتوای غنی و کامل پس از رندر کلاینت توسط پلی‌رایت. " * 10,
        markdown="# صفحه SPA رندر شده",
        title="داشبورد SPA",
        char_count=450,
        word_count=70,
    )

    with patch.object(extractor, "_safe_fetch_html", return_value={"failed": False, "html": spa_html, "final_url": "https://example.com/app", "status_code": 200}):
        # Mock Crawl4AI failing / low quality, Playwright succeeding
        failed_c4a = EngineResult(engine="crawl4ai", success=False, error="Crawl4AI timeout")
        with patch.object(extractor.crawl4ai_engine, "extract", new=AsyncMock(return_value=failed_c4a)):
            with patch.object(extractor.playwright_engine, "extract", new=AsyncMock(return_value=mock_pw_result)):
                doc = await extractor.extract("https://example.com/app")
                assert doc.extractor_used == "playwright"
                assert doc.quality_status == QualityStatus.SUCCESS
                assert "محتوای غنی و کامل" in (doc.content_text or "")
