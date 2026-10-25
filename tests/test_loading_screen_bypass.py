"""Tests for Loading Screen and Splash Screen Bypass in Crawlbot.

Verifies:
1. Persian and English loading text detection (is_loading_text).
2. DOM preloader, skeleton, and overlay detection (has_loading_dom).
3. PageClassifier automatically classifying loader shells as PAGE_TYPE_SPA.
4. ContentExtractor scoring loading screens as 0 and triggering fallback to Crawl4AI.
5. Static engine auto-promotion to Crawl4AI when loader text is captured.
6. Crawl4AI smart wait and second-chance retry handling.
7. Request schema validation and alias handling for loading_delay and wait_for_selector.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.schemas.documents import ExtractedDocument, Heading, LinkInfo
from app.schemas.requests import CompareRequest, ExtractionRequest, SingleExtractRequest
from app.services.extractor import ContentExtractor
from app.services.extractors.base import EngineResult
from app.services.extractors.crawl4ai_engine import Crawl4AIEngine
from app.services.loading_detector import (
    get_loader_remover_js,
    has_loading_dom,
    is_loading_screen,
    is_loading_text,
)
from app.services.page_classifier import PAGE_TYPE_SPA, PageClassifier


# ---------------------------------------------------------------------------
# 1. Detection of Persian Loading Screen Texts
# ---------------------------------------------------------------------------
def test_detect_persian_loading_texts():
    assert is_loading_text("در حال بارگذاری...") is True
    assert is_loading_text("لطفا کمی صبر کنید...") is True
    assert is_loading_text("لطفاً چند لحظه صبر کنید") is True
    assert is_loading_text("اطلاعات در حال آماده‌سازی است") is True
    assert is_loading_text("در حال دریافت اطلاعات از سرور") is True
    assert is_loading_text("صبور باشید، صفحه در حال بارگذاری است") is True

    # Real Persian article should NOT be classified as loading screen
    real_article = (
        "هوش مصنوعی و یادگیری ماشین دو حوزه بسیار جذاب در علم داده هستند. "
        "در این مقاله جامع به بررسی نحوه استفاده از وب اسکرپینگ در پایپ‌لاین‌های مدرن RAG می‌پردازیم."
    )
    assert is_loading_text(real_article) is False


# ---------------------------------------------------------------------------
# 2. Detection of English Loading Screen Texts
# ---------------------------------------------------------------------------
def test_detect_english_loading_texts():
    assert is_loading_text("Loading...") is True
    assert is_loading_text("Loading content, please wait...") is True
    assert is_loading_text("Please wait while we load your dashboard.") is True
    assert is_loading_text("Fetching data from server...") is True
    assert is_loading_text("Initializing application, hang tight...") is True
    assert is_loading_text("Getting things ready...") is True

    # Title match with sparse body
    assert is_loading_text("Please wait", title="Loading...") is True

    # Real English text should NOT be classified as loading screen
    real_text = (
        "Crawlbot is an enterprise-grade web scraping and RAG pipeline tool. "
        "It provides unified extraction across Trafilatura, BeautifulSoup, and Crawl4AI."
    )
    assert is_loading_text(real_text) is False


# ---------------------------------------------------------------------------
# 3. DOM Loader and Preloader Element Detection
# ---------------------------------------------------------------------------
def test_detect_loading_dom_elements():
    # Preloader with spinner
    html_preloader = """
    <html>
        <body>
            <div id="preloader">
                <div class="spinner"></div>
                <p>در حال بارگذاری سایت...</p>
            </div>
        </body>
    </html>
    """
    assert has_loading_dom(html_preloader) is True

    # Skeleton screen
    html_skeleton = """
    <html>
        <body>
            <div class="skeleton-loader">
                <div class="skeleton-card"></div>
            </div>
        </body>
    </html>
    """
    assert has_loading_dom(html_skeleton) is True

    # Normal HTML without loader
    html_normal = """
    <html>
        <body>
            <h1>مقاله آموزشی</h1>
            <p>این یک مقاله کامل و کاربردی درباره استخراج محتواست.</p>
        </body>
    </html>
    """
    assert has_loading_dom(html_normal) is False


# ---------------------------------------------------------------------------
# 4. Loader Remover JavaScript Generation
# ---------------------------------------------------------------------------
def test_loader_remover_js_validity():
    js = get_loader_remover_js()
    assert "preloader" in js
    assert "spinner" in js
    assert "skeleton" in js
    assert "style.setProperty('display', 'none'" in js
    assert "#root" in js


# ---------------------------------------------------------------------------
# 5. PageClassifier Recognizes Loader as SPA
# ---------------------------------------------------------------------------
def test_page_classifier_classifies_loader_as_spa():
    classifier = PageClassifier()
    loader_html = """
    <html>
        <head><script src="/app.js"></script></head>
        <body>
            <div id="preloader" class="loader">
                <span>لطفاً چند لحظه صبر کنید...</span>
            </div>
        </body>
    </html>
    """
    page_type = classifier.classify("https://example.com/portal", html=loader_html)
    assert page_type == PAGE_TYPE_SPA


# ---------------------------------------------------------------------------
# 6. Quality Scoring Penalizes Loading Screen (Score = 0)
# ---------------------------------------------------------------------------
def test_score_engine_result_penalizes_loading_screen():
    extractor = ContentExtractor()

    loader_result = EngineResult(
        engine="trafilatura",
        success=True,
        text="در حال بارگذاری... لطفاً صبر کنید",
        markdown="در حال بارگذاری...",
        title="Loading...",
    )
    score = extractor._score_engine_result(loader_result, raw_html="")
    assert score == 0
    assert any("لودینگ" in w for w in loader_result.warnings)


# ---------------------------------------------------------------------------
# 7. Auto Mode Bypasses Static Loader Screen to Crawl4AI
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_auto_mode_bypasses_loader_to_crawl4ai():
    extractor = ContentExtractor(min_quality_score=40)

    # Mock safe HTTP fetch returning a loader shell
    loader_html = """
    <html>
        <head><script src="/bundle.js"></script></head>
        <body>
            <div id="preloader">در حال بارگذاری اطلاعات سایت...</div>
        </body>
    </html>
    """

    async def mock_fetch(*args, **kwargs):
        return {
            "html": loader_html,
            "final_url": "https://example.com/spa",
            "status_code": 200,
            "fetched_at": None,
            "failed": False,
        }

    # Mock Crawl4AI engine returning the rendered content
    mock_c4a_result = EngineResult(
        engine="crawl4ai",
        success=True,
        text="این متن اصلی سایت است که بعد از پایان لودینگ رندر شده است. " * 10,
        markdown="# عنوان اصلی\n\nمتن استخراج شده.",
        title="صفحه اصلی سایت",
        char_count=500,
        word_count=80,
    )

    with patch("app.services.extractor.validate_url_safety", return_value=(True, "")):
        with patch.object(extractor, "_safe_fetch_html", side_effect=mock_fetch):
            with patch.object(extractor.crawl4ai_engine, "extract", new=AsyncMock(return_value=mock_c4a_result)):
                doc = await extractor.extract("https://example.com/spa")
                assert doc.extractor_used == "crawl4ai"
                assert "متن اصلی سایت" in (doc.content_text or "")
            assert doc.quality_report is not None
            assert doc.quality_report.passed is True


# ---------------------------------------------------------------------------
# 8. Single Engine Trafilatura Auto-Promotes to Crawl4AI on Loader Screen
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_trafilatura_promotes_to_crawl4ai_on_loader():
    extractor = ContentExtractor(strategy="trafilatura")

    loader_html = '<div id="loader">لطفاً صبر کنید... در حال بارگذاری</div>'

    async def mock_fetch(*args, **kwargs):
        return {
            "html": loader_html,
            "final_url": "https://example.com/loading-page",
            "status_code": 200,
            "fetched_at": None,
            "failed": False,
        }

    mock_c4a_result = EngineResult(
        engine="crawl4ai",
        success=True,
        text="محتوای واقعی صفحه بعد از لودینگ به طور کامل نمایش داده شد. " * 8,
        title="مستندات واقعی",
        char_count=400,
        word_count=60,
    )

    with patch.object(extractor, "_safe_fetch_html", side_effect=mock_fetch):
        with patch.object(extractor.crawl4ai_engine, "extract", new=AsyncMock(return_value=mock_c4a_result)):
            doc = await extractor.extract("https://example.com/loading-page")
            assert doc.extractor_used == "crawl4ai"
            assert "محتوای واقعی صفحه" in (doc.content_text or "")


# ---------------------------------------------------------------------------
# 9. Crawl4AIEngine Smart Retry when First Pass Catches Loader
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_crawl4ai_second_chance_retry_on_loader():
    engine = Crawl4AIEngine(headless=True)

    # First crawl result is a preloader
    crawl_res_1 = MagicMock()
    crawl_res_1.success = True
    crawl_res_1.markdown = "Loading please wait..."
    crawl_res_1.cleaned_html = "<p>Loading please wait...</p>"
    crawl_res_1.html = crawl_res_1.cleaned_html
    crawl_res_1.metadata = {"title": "Loading..."}
    crawl_res_1.links = {}

    # Second crawl result (after retry wait) is the actual content
    crawl_res_2 = MagicMock()
    crawl_res_2.success = True
    rich_text = "This is the actual dynamic client-rendered dashboard data. " * 10
    crawl_res_2.markdown = f"# Dashboard\n\n{rich_text}"
    crawl_res_2.cleaned_html = f"<div><h1>Dashboard</h1><p>{rich_text}</p></div>"
    crawl_res_2.html = crawl_res_2.cleaned_html
    crawl_res_2.metadata = {"title": "Dashboard"}
    crawl_res_2.links = {}

    mock_crawler = AsyncMock()
    mock_crawler.arun = AsyncMock(side_effect=[crawl_res_1, crawl_res_2])

    with patch("crawl4ai.AsyncWebCrawler") as mock_crawler_class:
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler

        result = await engine.extract("https://app.example.com", render_js=True)
        assert result.success is True
        assert "actual dynamic client-rendered" in (result.text or "")
        assert mock_crawler.arun.call_count == 2
        assert any("لودینگ" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 10. ExtractionRequest and Schemas Accept Loading Parameters
# ---------------------------------------------------------------------------
def test_extraction_request_loading_parameters():
    req = ExtractionRequest(
        urls=["https://example.com/app"],
        loading_delay=4.5,
        wait_for_selector="#app-container",
    )
    assert req.loading_delay == 4.5
    assert req.wait_for_selector == "#app-container"

    # Alias handling
    req_alias = ExtractionRequest.model_validate({
        "urls": ["https://example.com"],
        "loading_wait": 3.0,
        "wait_selector": ".main-content",
    })
    assert req_alias.loading_delay == 3.0
    assert req_alias.wait_for_selector == ".main-content"

    # SingleExtractRequest and CompareRequest
    single_req = SingleExtractRequest(
        url="https://example.com",
        loading_delay=2.0,
        wait_for_selector="#content",
    )
    assert single_req.loading_delay == 2.0
    assert single_req.wait_for_selector == "#content"

    comp_req = CompareRequest(
        url="https://example.com",
        loading_delay=5.0,
    )
    assert comp_req.loading_delay == 5.0
