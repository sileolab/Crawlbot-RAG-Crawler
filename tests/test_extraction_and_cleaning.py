"""Tests for content cleaning, extraction pipeline, quality checking, and exports."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.schemas.documents import ExtractedDocument, Heading, LinkInfo, QualityStatus
from app.services.content_cleaner import ContentCleaner
from app.services.export_service import ExportService
from app.services.extractor import ContentExtractor
from app.services.quality_checker import QualityChecker


class TestContentCleaner:
    def test_remove_boilerplate_and_scripts(self):
        cleaner = ContentCleaner()
        dirty_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav class="navbar main-menu">
                <a href="/home">Home</a>
                <a href="/about">About</a>
            </nav>
            <script>console.log("analytics");</script>
            <style>.banner { color: red; }</style>
            <main>
                <h1>عنوان اصلی صفحه</h1>
                <p>این یک پاراگراف حاوی اطلاعات ارزشمند است.</p>
                <div class="cookie-banner" id="cookie-notice">ما از کوکی استفاده می‌کنیم.</div>
            </main>
            <footer class="footer">
                <p>حقوق نشر ۲۰۲۶</p>
            </footer>
        </body>
        </html>
        """
        cleaned_html = cleaner.clean(dirty_html)

        assert "analytics" not in cleaned_html
        assert "main-menu" not in cleaned_html
        assert "cookie-notice" not in cleaned_html
        assert "عنوان اصلی صفحه" in cleaned_html
        assert "اطلاعات ارزشمند" in cleaned_html

    def test_whitespace_normalization(self):
        cleaner = ContentCleaner()
        raw = "متن     دارای \n\n\n فاصله‌های    بسیار    زیاد است."
        normalized = cleaner.normalize_whitespace(raw)
        assert "    " not in normalized
        assert "\n\n\n" not in normalized


class TestQualityChecker:
    def test_empty_content_detection(self):
        checker = QualityChecker(min_word_count=10, min_char_count=50)
        doc = ExtractedDocument(
            url="https://example.com/empty",
            content_text="",
            content_markdown="",
        )
        checked = checker.check(doc)
        assert checked.quality_status == QualityStatus.EMPTY
        assert any("empty" in w.lower() for w in checked.warnings)

    def test_low_quality_detection(self):
        checker = QualityChecker(min_word_count=20, min_char_count=100)
        doc = ExtractedDocument(
            url="https://example.com/short",
            content_text="متن خیلی کوتاه",
            content_markdown="متن خیلی کوتاه",
            word_count=3,
            character_count=14,
        )
        checked = checker.check(doc)
        assert checked.quality_status == QualityStatus.LOW_QUALITY

    def test_duplicate_content_detection(self):
        checker = QualityChecker(min_word_count=5, min_char_count=20)
        doc1 = ExtractedDocument(
            url="https://example.com/page1",
            content_text="این یک محتوای تکراری مشترک برای آزمون سامانه است.",
            content_markdown="این یک محتوای تکراری مشترک برای آزمون سامانه است.",
            word_count=10,
            character_count=50,
        )
        checked1 = checker.check(doc1)
        assert checked1.quality_status == QualityStatus.SUCCESS

        doc2 = ExtractedDocument(
            url="https://example.com/page2",
            content_text="این یک محتوای تکراری مشترک برای آزمون سامانه است.",
            content_markdown="این یک محتوای تکراری مشترک برای آزمون سامانه است.",
            word_count=10,
            character_count=50,
        )
        checked2 = checker.check(doc2)
        assert checked2.quality_status == QualityStatus.DUPLICATE


class TestExportService:
    @pytest.fixture
    def sample_docs(self) -> list[ExtractedDocument]:
        return [
            ExtractedDocument(
                url="https://example.com/doc1",
                final_url="https://example.com/doc1-final",
                title="مستند اول",
                headings=[Heading(level=1, text="مستند اول")],
                content_markdown="# مستند اول\n\nاین محتوای مارک‌داون سند اول است.",
                content_text="این محتوای مارک‌داون سند اول است.",
                word_count=7,
                character_count=34,
                extraction_method="trafilatura",
                quality_status=QualityStatus.SUCCESS,
            ),
            ExtractedDocument(
                url="https://example.com/doc2",
                title="مستند دوم",
                headings=[Heading(level=2, text="بخش دوم")],
                content_markdown="## بخش دوم\n\nاین متن دومین سند است.",
                content_text="این متن دومین سند است.",
                word_count=5,
                character_count=22,
                extraction_method="bs4",
                quality_status=QualityStatus.SUCCESS,
            ),
        ]

    def test_export_json(self, sample_docs):
        with TemporaryDirectory() as tmp_dir:
            service = ExportService(output_dir=tmp_dir)
            out_file = service.export_json(sample_docs, job_id="test_job_1")

            assert out_file.exists()
            assert out_file.suffix == ".json"

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["url"] == "https://example.com/doc1"
            assert data[0]["title"] == "مستند اول"
            assert data[0]["headings"][0]["text"] == "مستند اول"

    def test_export_jsonl(self, sample_docs):
        with TemporaryDirectory() as tmp_dir:
            service = ExportService(output_dir=tmp_dir)
            out_file = service.export_jsonl(sample_docs, job_id="test_job_2")

            assert out_file.exists()
            assert out_file.suffix == ".jsonl"

            lines = out_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2

            obj1 = json.loads(lines[0])
            obj2 = json.loads(lines[1])
            assert obj1["url"] == "https://example.com/doc1"
            assert obj2["url"] == "https://example.com/doc2"

    def test_export_markdown(self, sample_docs):
        with TemporaryDirectory() as tmp_dir:
            service = ExportService(output_dir=tmp_dir)
            folder = service.export_markdown(sample_docs, job_id="test_job_3")

            assert folder.exists()
            assert folder.is_dir()

            md_files = list(folder.glob("*.md"))
            assert len(md_files) == 2

            doc1_file = next(f for f in md_files if "doc1" in f.name)
            content = doc1_file.read_text(encoding="utf-8")
            assert "---" in content
            assert 'url: "https://example.com/doc1"' in content
            assert 'final_url: "https://example.com/doc1-final"' in content

    def test_export_text(self, sample_docs):
        with TemporaryDirectory() as tmp_dir:
            service = ExportService(output_dir=tmp_dir)
            folder = service.export_text(sample_docs, job_id="test_job_4")

            assert folder.exists()
            assert folder.is_dir()

            txt_files = list(folder.glob("*.txt"))
            assert len(txt_files) == 2

            content = txt_files[0].read_text(encoding="utf-8")
            assert "URL:" in content
            assert "=" * 80 in content


class TestContentExtractorFallbacks:
    def test_html_to_markdown_preserves_structure(self):
        extractor = ContentExtractor(strategy="bs4")
        html = """
        <div>
            <h1>عنوان یک</h1>
            <h2>عنوان دو</h2>
            <p>این یک پاراگراف همراه با <a href="https://example.com">لینک مرجع</a> و <strong>متن برجسته</strong> است.</p>
            <ul>
                <li>مورد اول</li>
                <li>مورد دوم</li>
            </ul>
        </div>
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        md = extractor._html_to_markdown(soup)

        assert "# عنوان یک" in md
        assert "## عنوان دو" in md
        assert "[لینک مرجع](https://example.com)" in md
        assert "**متن برجسته**" in md
        assert "- مورد اول" in md
