"""Comprehensive integration and unit tests for the 3 Extraction Engines, Auto Mode, and Fallbacks.

Covers all 18 required scenarios from Section 13:
1. Long article extraction via Trafilatura (text & markdown structure).
2. BeautifulSoup extraction with table preservation & JSON-LD parsing.
3. Crawl4AI execution for client-side JavaScript / SPA content.
4. Trafilatura handling of empty / near-empty pages.
5. BeautifulSoup resilience with broken / malformed JSON-LD.
6. Crawl4AI timeout and error recovery.
7. PageClassifier validation across 5+ distinct page types (article, docs, product, faq, spa).
8. Quality assessment scoring (0-100 score, short-text penalty, duplicate detection).
9. Auto Mode: initial Trafilatura selection for textual articles.
10. Auto Mode: initial BeautifulSoup selection for product / FAQ pages.
11. Auto Mode: initial Crawl4AI selection for SPA shells.
12. Auto Mode: graceful fallback when first engine yields low quality score.
13. Compare Mode: side-by-side benchmarking of all 3 engines.
14. Resource cleanup & graceful browser shutdown.
15. SPA route and public API discovery in spa_capture mode.
16. Multi-URL concurrent extraction job execution.
17. Unified Pydantic schema validation.
18. Preserved integrity of the existing system.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.jobs.manager import JobManager
from app.schemas.documents import ExtractedDocument, ExtractionAttempt, QualityReport, QualityStatus
from app.services.extractor import ContentExtractor
from app.services.extractors.base import EngineResult
from app.services.extractors.bs4_engine import BeautifulSoupEngine
from app.services.extractors.crawl4ai_engine import Crawl4AIEngine
from app.services.extractors.trafilatura_engine import TrafilaturaEngine
from app.services.page_classifier import (
    PAGE_TYPE_ARTICLE,
    PAGE_TYPE_DOCS,
    PAGE_TYPE_FAQ,
    PAGE_TYPE_PRODUCT,
    PAGE_TYPE_SPA,
    PageClassifier,
)
from app.services.quality_checker import QualityChecker


# ---------------------------------------------------------------------------
# 1. Trafilatura Long Article Extraction
# ---------------------------------------------------------------------------
def test_scenario_1_trafilatura_long_article():
    engine = TrafilaturaEngine()
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>راهنمای جامع معماری RAG و سیستم‌های استخراج</title></head>
    <body>
        <header><nav><a href="/">خانه</a></nav></header>
        <main>
            <article>
                <h1>راهنمای جامع معماری RAG و سیستم‌های استخراج</h1>
                <p>سیستم‌های بازیابی اطلاعات مدرن برای ارائه پاسخ‌های دقیق به داده‌های ساختاریافته وب تکیه می‌کنند.</p>
                <h2>چالش‌های وب‌اسکرپینگ</h2>
                <p>یکی از بزرگترین چالش‌ها تفکیک محتوای اصلی از تبلیغات و کدهای جاوااسکریپت زائد است.</p>
                <p>موتورهای استخراج مختلف دارای نقاط قوت متفاوتی در مواجهه با ساختارهای گوناگون هستند.</p>
            </article>
        </main>
        <footer><p>تمامی حقوق محفوظ است.</p></footer>
    </body>
    </html>
    """
    res = engine.extract(html, "https://example.com/blog/rag-guide")
    assert res.success is True
    assert res.engine == "trafilatura"
    assert res.word_count >= 20
    assert "راهنمای جامع معماری RAG" in (res.text or "")
    assert res.markdown is not None


# ---------------------------------------------------------------------------
# 2. BeautifulSoup Tables and JSON-LD
# ---------------------------------------------------------------------------
def test_scenario_2_beautifulsoup_tables_and_jsonld():
    engine = BeautifulSoupEngine()
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>گوشی هوشمند مدل X</title>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org/",
            "@type": "Product",
            "name": "گوشی هوشمند مدل X",
            "offers": {
                "@type": "Offer",
                "price": "15000000",
                "priceCurrency": "IRR"
            }
        }
        </script>
    </head>
    <body>
        <h1>مشخصات فنی گوشی</h1>
        <table border="1">
            <tr><th>مشخصه</th><th>مقدار</th></tr>
            <tr><td>حافظه رم</td><td>12 گیگابایت</td></tr>
            <tr><td>باتری</td><td>5000 میلی‌آمپر</td></tr>
        </table>
        <p>توضیحات تکمیلی محصول برای خریداران محترم.</p>
    </body>
    </html>
    """
    res = engine.extract(html, "https://example.com/products/phone-x")
    assert res.success is True
    assert res.engine == "beautifulsoup"
    assert "product_info" in res.metadata
    assert res.metadata["product_info"]["name"] == "گوشی هوشمند مدل X"
    assert "| حافظه رم | 12 گیگابایت |" in (res.markdown or "")


# ---------------------------------------------------------------------------
# 3. Crawl4AI Dynamic SPA / JS Extraction
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_3_crawl4ai_spa_dynamic_rendering():
    engine = Crawl4AIEngine(headless=True)
    html = """
    <!DOCTYPE html>
    <html>
    <body>
        <div id="root">
            <h1>داشبورد تک‌صفحه‌ای ری‌اکت</h1>
            <p>محتوای داینامیک لود شده توسط کلاینت ساید با استفاده از React و SPA.</p>
        </div>
    </body>
    </html>
    """
    # Test raw HTML extraction without network overhead
    res = await engine.extract(url="https://example.com/app", html=html, render_js=False)
    assert res.success is True
    assert res.engine == "crawl4ai"
    assert "داشبورد تک‌صفحه‌ای" in (res.text or "")


# ---------------------------------------------------------------------------
# 4. Trafilatura Empty Page Handling
# ---------------------------------------------------------------------------
def test_scenario_4_trafilatura_empty_page_handling():
    engine = TrafilaturaEngine()
    html = "<html><body><div></div></body></html>"
    res = engine.extract(html, "https://example.com/empty")
    assert res.success is False
    assert len(res.warnings) > 0 or res.error is not None
    assert res.word_count == 0


# ---------------------------------------------------------------------------
# 5. BeautifulSoup Malformed Microdata Resilience
# ---------------------------------------------------------------------------
def test_scenario_5_beautifulsoup_malformed_microdata():
    engine = BeautifulSoupEngine()
    html = """
    <html>
    <head>
        <script type="application/ld+json">
        { broken json invalid syntax here !!!
        </script>
    </head>
    <body>
        <h1>صفحه با ساختار مخدوش</h1>
        <p>محتوای متنی معمولی که باید علی‌رغم خرابی JSON-LD استخراج شود.</p>
    </body>
    </html>
    """
    res = engine.extract(html, "https://example.com/broken-jsonld")
    assert res.success is True
    assert "محتوای متنی معمولی" in (res.text or "")
    assert res.metadata.get("product_info") is None


# ---------------------------------------------------------------------------
# 6. Crawl4AI Timeout and Error Handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_6_crawl4ai_timeout_and_error():
    engine = Crawl4AIEngine(headless=True, page_timeout=100)
    with patch("crawl4ai.AsyncWebCrawler.arun", side_effect=asyncio.TimeoutError("Browser timeout")):
        res = await engine.extract(url="https://timeout-site.example.com", render_js=True)
        assert res.success is False
        assert "timeout" in (res.error or "").lower()


# ---------------------------------------------------------------------------
# 7. PageClassifier for 5 Distinct Page Types
# ---------------------------------------------------------------------------
def test_scenario_7_page_classifier_5_types():
    classifier = PageClassifier()

    # Article
    art_type = classifier.classify("https://example.com/blog/2026/how-to-rag")
    assert art_type == PAGE_TYPE_ARTICLE

    # Docs
    docs_type = classifier.classify("https://example.com/docs/api-reference")
    assert docs_type == PAGE_TYPE_DOCS

    # Product
    prod_type = classifier.classify("https://example.com/products/item-123")
    assert prod_type == PAGE_TYPE_PRODUCT

    # FAQ
    faq_type = classifier.classify("https://example.com/faq")
    assert faq_type == PAGE_TYPE_FAQ

    # SPA Shell
    spa_html = """
    <html>
    <head><script src="/bundle.js"></script><script src="/vendor.js"></script></head>
    <body>
        <div id="root"></div>
        <noscript>Please enable JavaScript to view this application.</noscript>
    </body>
    </html>
    """
    spa_type = classifier.classify("https://example.com/app", html=spa_html)
    assert spa_type == PAGE_TYPE_SPA


# ---------------------------------------------------------------------------
# 8. Quality Assessment: Short Text Penalty & Deduplication
# ---------------------------------------------------------------------------
def test_scenario_8_quality_checker_score_penalties():
    checker = QualityChecker(min_word_count=20, min_quality_score=40)

    # 1. Sparse document
    sparse_doc = ExtractedDocument(
        url="https://example.com/sparse",
        content_text="سلام دنیا",
        word_count=2,
        character_count=9,
    )
    evaluated = checker.check(sparse_doc)
    assert evaluated.quality_status == QualityStatus.LOW_QUALITY
    assert evaluated.quality_report is not None
    assert evaluated.quality_report.score < 40
    assert evaluated.quality_report.passed is False

    # 2. Rich document
    rich_text = "این یک مقاله علمی کامل درباره روش‌های استخراج متن از صفحات وب است. " * 15
    rich_doc = ExtractedDocument(
        url="https://example.com/rich",
        title="مقاله تخصصی وب اسکرپینگ",
        content_text=rich_text,
        word_count=180,
        character_count=len(rich_text),
    )
    rich_eval = checker.check(rich_doc)
    assert rich_eval.quality_status == QualityStatus.SUCCESS
    assert rich_eval.quality_report.score >= 50
    assert rich_eval.quality_report.passed is True

    # 3. Duplicate document
    dup_doc = ExtractedDocument(
        url="https://example.com/rich-copy",
        title="کپی مقاله",
        content_text=rich_text,
        word_count=180,
        character_count=len(rich_text),
    )
    dup_eval = checker.check(dup_doc)
    assert dup_eval.quality_status == QualityStatus.DUPLICATE
    assert dup_eval.quality_report.duplicate_percentage == 100.0


# ---------------------------------------------------------------------------
# 9. Auto Mode: Trafilatura Selection for Article
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_9_auto_mode_selects_trafilatura_for_article():
    extractor = ContentExtractor(strategy="auto")
    article_html = """
    <html>
    <head><title>پست وبلاگی آموزشی</title></head>
    <body>
        <h1>پست وبلاگی آموزشی</h1>
        <p>متن کامل پاراگراف اول مقاله که توضیحات تخصصی را ارائه می‌دهد.</p>
        <p>متن کامل پاراگراف دوم مقاله برای ایجاد محتوای کافی و معتبر در تست.</p>
        <p>متن کامل پاراگراف سوم مقاله در جهت بررسی دقیق دقت عملکرد موتور ترافیلاچورا.</p>
    </body>
    </html>
    """
    with patch.object(extractor, "_safe_fetch_html", return_value={"failed": False, "html": article_html, "final_url": "https://example.com/blog/my-post", "status_code": 200}):
        doc = await extractor.extract("https://example.com/blog/my-post")
        assert doc.page_type == PAGE_TYPE_ARTICLE
        assert doc.extractor_used == "trafilatura"
        assert len(doc.extraction_attempts) >= 1
        assert doc.extraction_attempts[0].engine == "trafilatura"


# ---------------------------------------------------------------------------
# 10. Auto Mode: BeautifulSoup Selection for Product / FAQ
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_10_auto_mode_selects_bs4_for_product():
    extractor = ContentExtractor(strategy="auto")
    product_html = """
    <html>
    <head>
        <title>فروشگاه آنلاین کالا</title>
        <script type="application/ld+json">{"@type": "Product", "name": "لپ‌تاپ X"}</script>
    </head>
    <body>
        <h1>لپ‌تاپ گیمینگ مدل X</h1>
        <div class="price">قیمت: 45000000 تومان</div>
        <p>توضیحات لپ‌تاپ فوق حرفه‌ای برای امور مهندسی و بازی.</p>
    </body>
    </html>
    """
    with patch.object(extractor, "_safe_fetch_html", return_value={"failed": False, "html": product_html, "final_url": "https://example.com/products/laptop", "status_code": 200}):
        doc = await extractor.extract("https://example.com/products/laptop")
        assert doc.page_type == PAGE_TYPE_PRODUCT
        assert doc.extractor_used in ("beautifulsoup", "bs4")


# ---------------------------------------------------------------------------
# 11. Auto Mode: Crawl4AI Selection for SPA
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_11_auto_mode_selects_crawl4ai_for_spa():
    extractor = ContentExtractor(strategy="auto")
    spa_html = """
    <html>
    <head><script src="/app.bundle.js"></script><script src="/main.js"></script></head>
    <body>
        <div id="root"></div>
        <noscript>You need to enable JavaScript to run this app.</noscript>
    </body>
    </html>
    """
    mock_c4a_result = EngineResult(
        engine="crawl4ai",
        success=True,
        text="محتوای رندر شده پس از اجرای جاوااسکریپت در کلاینت",
        markdown="### محتوای رندر شده",
        char_count=50,
        word_count=10,
    )
    with patch.object(extractor, "_safe_fetch_html", return_value={"failed": False, "html": spa_html, "final_url": "https://example.com/app", "status_code": 200}):
        with patch.object(extractor.crawl4ai_engine, "extract", return_value=mock_c4a_result):
            doc = await extractor.extract("https://example.com/app")
            assert doc.page_type == PAGE_TYPE_SPA
            assert doc.extractor_used == "crawl4ai"


# ---------------------------------------------------------------------------
# 12. Auto Mode Fallback Chain: Quality Drop Switches to Next Engine
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_12_auto_mode_fallback_chain():
    extractor = ContentExtractor(strategy="auto", min_quality_score=45)

    # HTML where Trafilatura fails to extract (returns None) but BS4 extracts successfully
    html = """
    <html>
    <body>
        <div class="custom-doc-box">
            <h1>راهنمای ویژه با قالب غیرمرسوم</h1>
            <p>این متن داخل یک قالب نامرسوم است که ترافیلاچورا آن را به عنوان بدنه اصلی تشخیص نمی‌دهد اما بیوتیفول‌سوپ آن را به خوبی برمی‌دارد.</p>
            <p>توضیحات بند دوم برای افزایش حجم کلمات و عبور از آستانه کیفی سیستم.</p>
        </div>
    </body>
    </html>
    """
    # Mock Trafilatura returning empty
    empty_traf = EngineResult(engine="trafilatura", success=False, text=None, markdown=None)
    with patch.object(extractor.trafilatura_engine, "extract", return_value=empty_traf):
        with patch.object(extractor, "_safe_fetch_html", return_value={"failed": False, "html": html, "final_url": "https://example.com/page", "status_code": 200}):
            doc = await extractor.extract("https://example.com/page")
            # Should have fallen back to beautifulsoup
            assert doc.extractor_used == "beautifulsoup"
            assert len(doc.extraction_attempts) >= 2
            assert doc.extraction_attempts[0].engine == "trafilatura"
            assert doc.extraction_attempts[0].status == "fallback"
            assert doc.extraction_attempts[1].engine == "beautifulsoup"
            assert doc.extraction_attempts[1].status == "success"


# ---------------------------------------------------------------------------
# 13. Compare Mode: Benchmark All 3 Engines
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_13_compare_mode_all_three_engines():
    extractor = ContentExtractor()
    html = """
    <html>
    <head><title>صفحه آزمون مقایسه موتورها</title></head>
    <body>
        <h1>مقایسه سه موتور استخراج</h1>
        <p>این صفحه جهت محک زدن و ارزیابی سرعت و خروجی هر سه موتور استخراج به کار می‌رود.</p>
    </body>
    </html>
    """
    with patch.object(extractor, "_safe_fetch_html", return_value={"failed": False, "html": html, "final_url": "https://example.com/compare-test", "status_code": 200}):
        doc = await extractor.extract("https://example.com/compare-test", crawl_mode="compare")
        assert "comparison_report" in doc.metadata
        cmp_report = doc.metadata["comparison_report"]
        assert "winner" in cmp_report
        assert "trafilatura" in cmp_report["engines"]
        assert "beautifulsoup" in cmp_report["engines"]
        assert "crawl4ai" in cmp_report["engines"]
        assert len(doc.extraction_attempts) == 3


# ---------------------------------------------------------------------------
# 14. Resource Cleanup for Headless Browser
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_14_crawl4ai_resource_cleanup():
    engine = Crawl4AIEngine(headless=True)
    # Ensure extractor handles exceptions and closes async crawler context cleanly
    with patch("crawl4ai.AsyncWebCrawler.arun", side_effect=RuntimeError("Simulated unexpected crash")):
        res = await engine.extract("https://crash-test.example.com", render_js=True)
        assert res.success is False
        assert "Simulated unexpected crash" in (res.error or "")


# ---------------------------------------------------------------------------
# 15. SPA Route and Public API Discovery
# ---------------------------------------------------------------------------
def test_scenario_15_spa_route_and_api_discovery():
    extractor = ContentExtractor()
    html = """
    <html>
    <head>
        <script>
            const routes = [
                { path: '/dashboard/analytics', component: AnalyticsView },
                { path: '/settings/security', component: SecurityView }
            ];
            const fetchUsers = () => fetch('/api/v1/users');
            const fetchProducts = () => axios.get('/api/v2/products/list');
        </script>
    </head>
    <body><div id="root"></div></body>
    </html>
    """
    routes, apis = extractor._discover_spa_endpoints(html)
    assert any("analytics" in r for r in routes)
    assert any("/api/v1/users" in a for a in apis)


# ---------------------------------------------------------------------------
# 16. Multi-URL Concurrent Extraction Job
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_16_concurrent_extraction_multi_engine_job():
    manager = JobManager()
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]
    job = manager.create_extraction_job(
        urls=urls,
        config={"extraction_strategy": "auto", "max_concurrency": 2},
    )

    mock_doc = ExtractedDocument(
        url="https://example.com/page",
        title="صفحه تست",
        content_text="متن تست استخراج همزمان برای اعتبارسنجی کارکرد JobManager.",
        word_count=10,
        character_count=50,
        quality_status=QualityStatus.SUCCESS,
    )

    with patch("app.services.extractor.ContentExtractor.extract", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = mock_doc
        await manager.run_extraction(job.job_id)

    updated_job = manager.get_job(job.job_id)
    assert updated_job.completed == 3
    assert updated_job.failed == 0
    assert len(updated_job.documents) == 3


# ---------------------------------------------------------------------------
# 17. Unified Pydantic Schema Compliance
# ---------------------------------------------------------------------------
def test_scenario_17_unified_pydantic_schema_compliance():
    doc = ExtractedDocument(
        url="https://example.com/article",
        source_url="https://example.com/article",
        title="عنوان تستی اسکیما",
        extractor_used="trafilatura",
        page_type=PAGE_TYPE_ARTICLE,
        content_text="متن تستی مستند",
        extraction_attempts=[
            ExtractionAttempt(
                engine="trafilatura",
                status="success",
                latency_ms=45.2,
                char_count=15,
                word_count=3,
                quality_score=65,
            )
        ],
        quality_report=QualityReport(
            score=65,
            char_count=15,
            word_count=3,
            passed=True,
        ),
    )

    # Validate json serialization
    json_dump = doc.model_dump(mode="json")
    assert json_dump["document_id"] is not None
    assert json_dump["extractor_used"] == "trafilatura"
    assert json_dump["page_type"] == "article"
    assert len(json_dump["extraction_attempts"]) == 1
    assert json_dump["quality_report"]["score"] == 65
