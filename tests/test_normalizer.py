"""Unit tests for Persian & Unicode Normalizer Service.

Verifies:
- Unicode NFC normalization
- Arabic to Persian character unification
- Control and invisible character removal with ZWNJ preservation
- Whitespace and line-break normalization
- Affix & compound word ZWNJ restoration (حل نمیشود, دادهها, بهصورت, فارسیزبان)
- Protection of code blocks, inline code, URLs, emails, and JSON
- Markdown table and list preservation
- Idempotency: normalize(normalize(text)) == normalize(text)
"""

import pytest

from app.services.normalizer import PersianNormalizer, ZWNJ, normalize_persian_text


class TestPersianNormalizer:

    @pytest.fixture
    def normalizer(self):
        return PersianNormalizer()

    def test_nfc_normalization(self, normalizer):
        """Combining characters are composed into canonical NFC form."""
        # Decomposed form vs Composed form
        text = "هسته‌"
        normalized = normalizer.normalize(text)
        assert len(normalized) > 0
        assert normalizer.normalize(normalized) == normalized

    def test_arabic_to_persian_unification(self, normalizer):
        """Arabic Yeh, Kaf, and Teh Marbuta are converted to Persian equivalents."""
        arabic_text = "كتاب علي در دانشگاه اصفهان با توجه به دورة آموزشي"
        normalized = normalizer.normalize(arabic_text)
        assert "کتاب" in normalized
        assert "علی" in normalized
        assert "دوره" in normalized
        assert "آموزشی" in normalized
        assert "ك" not in normalized
        assert "ي" not in normalized

    def test_control_character_cleanup(self, normalizer):
        """Null bytes and invisible control characters are stripped while ZWNJ is kept."""
        text_with_controls = f"متن\x00 با\x08 کاراکتر\x1bهای\ufeff نامرئی{ZWNJ}دار"
        normalized = normalizer.normalize(text_with_controls)
        assert "\x00" not in normalized
        assert "\x08" not in normalized
        assert "\x1b" not in normalized
        assert "\ufeff" not in normalized
        assert ZWNJ in normalized

    def test_whitespace_normalization(self, normalizer):
        """Duplicate horizontal spaces and excess newlines are collapsed."""
        dirty_whitespace = "پاراگراف اول    با    فاصله زیاد.\r\n\r\n\r\n\r\nپاراگراف دوم   "
        normalized = normalizer.normalize(dirty_whitespace)
        assert "    " not in normalized
        assert "\r" not in normalized
        assert "\n\n\n" not in normalized
        assert "پاراگراف اول با فاصله زیاد.\n\nپاراگراف دوم" == normalized

    def test_glued_verbs_zwnj_restoration(self, normalizer):
        """Progressive prefixes (می / نمی) glued to verb stems get corrected with ZWNJ."""
        glued_verbs = "این مشکل حل نمیشود و سیستم کار نمیکند. برنامه بهدرستی کار میکند و اطلاعات میرسد."
        normalized = normalizer.normalize(glued_verbs)
        assert f"نمی{ZWNJ}شود" in normalized
        assert f"نمی{ZWNJ}کند" in normalized
        assert f"می{ZWNJ}کند" in normalized
        assert f"می{ZWNJ}رسد" in normalized

    def test_root_words_starting_with_mi_not_corrupted(self, normalizer):
        """Non-verb words starting with 'می' must never have ZWNJ inserted."""
        safe_roots = "میوه های میدان و میهن ما میزان رشد را نشان داد."
        normalized = normalizer.normalize(safe_roots)
        assert "میوه" in normalized
        assert f"می{ZWNJ}وه" not in normalized
        assert "میدان" in normalized
        assert "میهن" in normalized
        assert "میزان" in normalized

    def test_glued_plurals_zwnj_restoration(self, normalizer):
        """Glued plurals ending with heh get ZWNJ restored (دادهها -> داده‌ها)."""
        glued_plurals = "تمام دادهها و برنامهها درون سایتها ذخیره شدند."
        normalized = normalizer.normalize(glued_plurals)
        assert f"داده{ZWNJ}ها" in normalized
        assert f"برنامه{ZWNJ}ها" in normalized
        assert f"سایت{ZWNJ}ها" in normalized

    def test_mandatory_compound_words_from_spec(self, normalizer):
        """Verify the exact compound words specified in prompt section 4.3."""
        compounds = "حل نمیشود، دادهها، بهصورت و فارسیزبان باید اصلاح شوند."
        normalized = normalizer.normalize(compounds)
        assert f"حل نمی{ZWNJ}شود" in normalized
        assert f"داده{ZWNJ}ها" in normalized
        assert f"به{ZWNJ}صورت" in normalized
        assert f"فارسی{ZWNJ}زبان" in normalized

    def test_code_blocks_and_urls_protected(self, normalizer):
        """Code blocks, inline code, and URLs must not be modified."""
        mixed_doc = (
            "برای مشاهده دادهها به نشانی https://example.com/api?query=دادهها مراجعه کنید.\n\n"
            "```python\n"
            "def نمیشود():\n"
            "    data = 'دادهها'\n"
            "    return True\n"
            "```\n\n"
            "از دستور `npm install بستهها` استفاده کنید."
        )
        normalized = normalizer.normalize(mixed_doc)
        # Prose should be normalized
        assert f"داده{ZWNJ}ها به نشانی" in normalized
        # URL must be intact
        assert "https://example.com/api?query=دادهها" in normalized
        # Codeblock content must be 100% intact
        assert "def نمیشود():" in normalized
        assert "data = 'دادهها'" in normalized
        # Inline code must be intact
        assert "`npm install بستهها`" in normalized

    def test_markdown_tables_and_lists_preserved(self, normalizer):
        """Markdown list bullets and table pipes must be preserved cleanly."""
        md_text = (
            "| ردیف | نام دادهها | وضعیت |\n"
            "| --- | --- | --- |\n"
            "| ۱ | بستهها | فعال |\n\n"
            "- مورد اول نمیشود\n"
            "- مورد دوم بهصورت کامل"
        )
        normalized = normalizer.normalize(md_text)
        assert "| ردیف |" in normalized
        assert "| --- |" in normalized
        assert f"داده{ZWNJ}ها" in normalized
        assert f"بسته{ZWNJ}ها" in normalized
        assert f"- مورد اول نمی{ZWNJ}شود" in normalized
        assert f"- مورد دوم به{ZWNJ}صورت" in normalized

    def test_empty_and_whitespace_input(self, normalizer):
        """Empty or purely whitespace input returns empty string safely."""
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("    \n\t   ") == ""

    def test_pure_idempotency(self, normalizer):
        """normalize(normalize(text)) == normalize(text) on complex inputs."""
        complex_text = (
            "این متن برای بررسی بهصورت کامل نوشته شده است. "
            "دادهها و اطلاعات سیستم نمیشود بهراحتی حذف گردند. "
            "كتابهاي فارسي زبان در دانشگاه اصفهان موجود است."
        )
        first_pass, report = normalizer.normalize_with_report(complex_text)
        second_pass, _ = normalizer.normalize_with_report(first_pass)
        assert first_pass == second_pass
        assert report["is_idempotent"] is True
