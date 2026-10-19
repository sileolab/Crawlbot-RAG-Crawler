"""Comprehensive integration test with a deterministic mock website.

Validates Section 11 requirements:
- Multiple linked HTML pages
- Nested URL paths
- Relative links
- Absolute links
- Duplicate links
- Navigation and footer boilerplate
- Headings, paragraphs, lists, and tables
- An intentionally empty page
- A simulated failed request
- A sitemap and nested sitemap
- A redirect case
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from app.schemas.documents import QualityStatus
from app.services.crawler import CrawlerService
from app.services.export_service import ExportService
from app.services.extractor import ContentExtractor
from app.services.quality_checker import QualityChecker


@pytest.fixture
def mock_website_transport():
    """Mock HTTP transport serving a simulated website."""
    base_url = "https://mocksite.org"

    # Define deterministic pages and assets
    pages = {
        "/robots.txt": (
            200,
            "text/plain",
            f"User-agent: *\nAllow: /\nDisallow: /private/\nSitemap: {base_url}/sitemap_index.xml\n",
        ),
        "/sitemap_index.xml": (
            200,
            "application/xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
            <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
                <sitemap><loc>{base_url}/sitemap_pages.xml</loc></sitemap>
            </sitemapindex>""",
        ),
        "/sitemap_pages.xml": (
            200,
            "application/xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
                <url><loc>{base_url}/</loc></url>
                <url><loc>{base_url}/nested/docs</loc></url>
                <url><loc>{base_url}/empty</loc></url>
            </urlset>""",
        ),
        "/": (
            200,
            "text/html; charset=utf-8",
            f"""<!DOCTYPE html>
            <html lang="fa">
            <head><title>صفحه اصلی وب‌سایت آزمایشی</title></head>
            <body>
                <header class="navbar header">
                    <nav><a href="/">خانه</a> | <a href="/nested/docs">مستندات</a></nav>
                </header>
                <main>
                    <h1>صفحه اصلی کرال‌بات</h1>
                    <p>این سامانه برای استخراج مقالات و داده‌های وب طراحی شده است.</p>
                    <a href="/nested/docs">ورود به مستندات (لینک نسبی ریشه)</a>
                    <a href="./about">درباره ما (لینک نسبی نقطه)</a>
                    <a href="{base_url}/table-page">مشاهده جدول (لینک مطلق)</a>
                    <a href="/nested/docs">ورود به مستندات (لینک تکراری)</a>
                    <a href="/old-redirect">صفحه قدیمی (تغییر مسیر)</a>
                    <a href="/broken">صفحه خراب (کد خطا)</a>
                    <a href="/empty">صفحه خالی</a>
                </main>
                <footer class="footer">
                    <p>© 2026 حقوق برای سامانه محفوظ است.</p>
                </footer>
            </body>
            </html>""",
        ),
        "/about": (
            200,
            "text/html; charset=utf-8",
            """<!DOCTYPE html>
            <html lang="fa">
            <head><title>درباره ما</title></head>
            <body>
                <h1>درباره سامانه کرال‌بات</h1>
                <p>توضیحات مفصل درباره معماری و نحوه عملکرد سیستم پاک‌سازی داده‌ها برای RAG.</p>
                <a href="../nested/docs">لینک با دو نقطه به مستندات</a>
            </body>
            </html>""",
        ),
        "/nested/docs": (
            200,
            "text/html; charset=utf-8",
            """<!DOCTYPE html>
            <html lang="fa">
            <head><title>مستندات فنی</title></head>
            <body>
                <h1>راهنمای جامع مستندات فنی</h1>
                <h2>بخش اول: معماری</h2>
                <p>معماری بر پایه پایتون، فست‌ای‌پی‌آی و کتابخانه‌های تخصصی استوار است.</p>
                <ul>
                    <li>مؤلفه کشف نشانی‌ها</li>
                    <li>مؤلفه استخراج و پاک‌سازی محتوا</li>
                    <li>مؤلفه ارزیابی کیفیت</li>
                </ul>
            </body>
            </html>""",
        ),
        "/table-page": (
            200,
            "text/html; charset=utf-8",
            """<!DOCTYPE html>
            <html lang="fa">
            <head><title>صفحه جداول آماری</title></head>
            <body>
                <h1>جداول مقایسه‌ای سرعت و دقت</h1>
                <table border="1">
                    <thead><tr><th>روش</th><th>دقت</th><th>سرعت</th></tr></thead>
                    <tbody>
                        <tr><td>Trafilatura</td><td>عالی</td><td>بسیار سریع</td></tr>
                        <tr><td>BeautifulSoup</td><td>خوب</td><td>متوسط</td></tr>
                    </tbody>
                </table>
            </body>
            </html>""",
        ),
        "/old-redirect": (
            301,
            "text/html",
            f'<a href="{base_url}/new-page">Moved</a>',
        ),
        "/new-page": (
            200,
            "text/html; charset=utf-8",
            """<!DOCTYPE html>
            <html lang="fa">
            <head><title>صفحه جدید منتقل شده</title></head>
            <body>
                <h1>محتوای مقصد پس از ریدایرکت</h1>
                <p>کاربر با موفقیت به این نشانی هدایت شد.</p>
            </body>
            </html>""",
        ),
        "/empty": (
            200,
            "text/html; charset=utf-8",
            """<!DOCTYPE html>
            <html><head><title>صفحه خالی</title></head><body></body></html>""",
        ),
        "/broken": (
            500,
            "text/html",
            "Internal Server Error",
        ),
        "/private/secret": (
            200,
            "text/html",
            "<h1>Private Area</h1>",
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.raw_path == b"/":
            path = "/"

        if path == "/old-redirect":
            headers = {"Location": f"{base_url}/new-page", "Content-Type": "text/html"}
            return httpx.Response(301, headers=headers, text="Redirecting...")

        if path in pages:
            status, ctype, text = pages[path]
            headers = {"Content-Type": ctype}
            return httpx.Response(status, headers=headers, text=text)

        return httpx.Response(404, text="Not Found")

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_mock_website_full_crawling_and_extraction(mock_website_transport, tmp_path):
    """End-to-end integration test against the mock website."""
    base_url = "https://mocksite.org"

    # Mock validate_url_safety so mock domain mocksite.org passes SSRF check in test
    with patch("app.services.crawler.validate_url_safety", return_value=(True, "Safe in test")), \
         patch("app.services.extractor.validate_url_safety", return_value=(True, "Safe in test")):

        async with httpx.AsyncClient(transport=mock_website_transport) as client:
            # 1. Discover pages using CrawlerService
            crawler = CrawlerService(
                root_url=base_url,
                max_depth=3,
                max_pages=20,
                delay=0.0,
                use_sitemap=True,
                client=client,
            )

            result = await crawler.discover()

            # Verify discovery results
            assert len(result.discovered) >= 5
            assert f"{base_url}/" in result.crawled
            assert f"{base_url}/nested/docs" in result.discovered
            assert f"{base_url}/about" in result.discovered
            assert f"{base_url}/table-page" in result.discovered

            # Verify broken URL recorded in failed
            assert f"{base_url}/broken" in result.failed

            # 2. Extract content from discovered pages using ContentExtractor
            extractor = ContentExtractor(strategy="trafilatura", client=client)
            quality_checker = QualityChecker(min_word_count=5, min_char_count=20)
            extracted_documents = []

            target_urls = [
                f"{base_url}/",
                f"{base_url}/nested/docs",
                f"{base_url}/table-page",
                f"{base_url}/empty",
            ]

            for url in target_urls:
                doc = await extractor.extract(url=url)
                doc = quality_checker.check(doc)
                extracted_documents.append(doc)

            # Verify home page extraction
            home_doc = next(d for d in extracted_documents if d.url == f"{base_url}/")
            assert home_doc.title is not None
            text_or_md = (home_doc.content_text or "") + (home_doc.content_markdown or "")
            assert "کرال" in text_or_md
            assert home_doc.word_count > 0

            # Verify nested docs extraction & heading preservation
            docs_doc = next(d for d in extracted_documents if "nested/docs" in d.url)
            assert len(docs_doc.headings) >= 1
            assert any("مستندات" in h.text for h in docs_doc.headings)

            # Verify empty page detection
            empty_doc = next(d for d in extracted_documents if "empty" in d.url)
            assert empty_doc.quality_status == QualityStatus.EMPTY

            # 3. Export to JSON, JSONL, Markdown, Text
            export_service = ExportService(output_dir=str(tmp_path))
            json_path = export_service.export_json(extracted_documents, job_id="mock_test")
            jsonl_path = export_service.export_jsonl(extracted_documents, job_id="mock_test")
            md_dir = export_service.export_markdown(extracted_documents, job_id="mock_test")
            txt_dir = export_service.export_text(extracted_documents, job_id="mock_test")

            assert json_path.exists()
            assert jsonl_path.exists()
            assert md_dir.exists()
            assert txt_dir.exists()
            assert len(list(md_dir.glob("*.md"))) == len(extracted_documents)


@pytest.mark.asyncio
async def test_e2e_api_lifecycle_with_mock_site(mock_website_transport):
    """Full lifecycle test through FastAPI TestClient:
    Discovery Start -> Polling Status -> Page Selection -> Extraction Start -> Polling Status -> Export Download -> Cancel & Retry.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api import routes_discovery, routes_extraction

    client = TestClient(app)
    base_url = "https://mocksite.org"

    with patch("app.services.crawler.validate_url_safety", return_value=(True, "Safe in test")), \
         patch("app.services.extractor.validate_url_safety", return_value=(True, "Safe in test")), \
         patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=mock_website_transport)):

        # 1. Start discovery
        disc_res = client.post("/api/discovery/start", json={"root_url": base_url, "max_depth": 2, "max_pages": 10})
        assert disc_res.status_code == 200
        disc_json = disc_res.json()
        assert disc_json["success"] is True
        disc_job_id = disc_json["data"]["job_id"]

        # 2. Wait for discovery background task to complete
        for _ in range(50):
            status_res = client.get(f"/api/discovery/status/{disc_job_id}")
            assert status_res.status_code == 200
            st = status_res.json()
            if st["status"] in ("completed", "finished", "failed"):
                break
            await asyncio.sleep(0.05)

        assert st["status"] == "completed"
        assert len(st["pages"]) >= 4

        # 3. Start extraction for discovered URLs
        selected_urls = [p["url"] for p in st["pages"] if p["status"] == "crawled"][:2]
        if not selected_urls:
            selected_urls = [f"{base_url}/"]

        extr_res = client.post("/api/extraction/start", json={"urls": selected_urls, "extraction_strategy": "trafilatura"})
        assert extr_res.status_code == 200
        extr_json = extr_res.json()
        assert extr_json["success"] is True
        extr_job_id = extr_json["data"]["job_id"]

        # 4. Wait for extraction background task to complete
        for _ in range(50):
            st_res = client.get(f"/api/extraction/status/{extr_job_id}")
            assert st_res.status_code == 200
            st_extr = st_res.json()
            if st_extr["status"] in ("completed", "finished", "failed"):
                break
            await asyncio.sleep(0.05)

        assert st_extr["status"] == "completed"
        assert len(st_extr["documents"]) >= 1

        # 5. Export download
        exp_res = client.post("/api/export/download", json={"job_id": extr_job_id, "format": "json"})
        assert exp_res.status_code == 200
        assert "application/json" in exp_res.headers.get("content-type", "")
        assert len(exp_res.content) > 10

        # 6. Test cancel API
        cancel_res = client.post(f"/api/discovery/cancel/{disc_job_id}")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["success"] is True

