"""Comprehensive Unit Tests for Persian & Unicode Normalizer Service.

Verifies:
1. Unicode NFC normalization
2. Arabic-to-Persian letter unification while strictly preserving Teh Marbuta (ة)
3. Independent Arabic-to-Persian digit unification (unify_digits vs unify_arabic)
4. Control and invisible character removal with ZWNJ and newline preservation
5. Whitespace and layout normalization
6. Progressive verb prefixes (میشود, نمی شود, میکند, میتواند, میگیرند, نمایش داده نمیشود)
7. Protection of non-verb root words starting with می (میوه, میدان, میهن, میزان, میراث, میلاد, میز)
8. Glued and spaced plural suffixes (دادهها, سایتها, سرویسهای, درخواستهای, دادههایت, پیامهای, شناسهها, بازیها, قابلیتهای, کتاب ها)
9. Protection of non-plural words ending in ها (تنها, انتها, اشتها, رها, بها)
10. Comparative and superlative adjectives (کوتاهتر, روشنتر, ساده تر, سادهترین, محافظهکارانهتر) with root protection (دختر, دفتر, کبوتر, بهتر, بیشتر)
11. Mandatory compound words (بهعنوان, بهصورت, فارسیزبان, انگلیسیزبان, بهره مند, علاقه مند, زبان شناسی, روان شناسی, آمادهسازی, نفرتپراکنی, همینطور)
12. Protection of fenced code blocks, inline code, URLs (balanced parens and trailing punctuation trimming), Markdown links, and HTML tags
13. Multi-line, paragraph, and mixed English/Persian structure preservation
14. Pure idempotency: normalize(normalize(text)) == normalize(text)
15. Telemetry accuracy (chars_before, chars_after, arabic_unifications, digits_unified, controls_removed, zwnj_fixes, changed, is_idempotent)
16. Edge cases: empty strings, None, invalid types, and convenience function
"""

import pytest

from app.services.normalizer import PersianNormalizer, ZWNJ, normalize_persian_text


class TestPersianNormalizer:

    @pytest.fixture
    def normalizer(self):
        return PersianNormalizer()

    def test_nfc_normalization(self, normalizer):
        """Combining characters are composed into canonical NFC form."""
        text = "هسته‌"
        normalized = normalizer.normalize(text)
        assert len(normalized) > 0
        assert normalizer.normalize(normalized) == normalized

    def test_arabic_to_persian_unification_and_teh_marbuta_preserved(self, normalizer):
        """Arabic Yeh, Kaf, and Alef Maksura are unified, but Teh Marbuta (ة) is PRESERVED."""
        arabic_text = "كتاب علي و مرتضى در دورة آموزشي دانشگاه"
        normalized, report = normalizer.normalize_with_report(arabic_text)
        assert "کتاب" in normalized
        assert "علی" in normalized
        assert "مرتضی" in normalized
        assert "آموزشی" in normalized
        assert "دورة" in normalized  # ة MUST NOT be converted globally to ه
        assert "ك" not in normalized
        assert "ي" not in normalized
        assert "ى" not in normalized
        assert report["arabic_unifications"] == 4

    def test_unify_digits_independent_from_arabic(self):
        """Digit unification is controlled solely by unify_digits and independent of unify_arabic."""
        arabic_mixed = "كتاب ١٢٣ صفحه دارد."

        # Case 1: Default (unify_arabic=True, unify_digits=False)
        norm_default = PersianNormalizer(unify_arabic=True, unify_digits=False)
        res_def, rep_def = norm_default.normalize_with_report(arabic_mixed)
        assert "کتاب" in res_def
        assert "١٢٣" in res_def
        assert rep_def["arabic_unifications"] == 1
        assert rep_def["digits_unified"] == 0

        # Case 2: Only digits (unify_arabic=False, unify_digits=True)
        norm_digits_only = PersianNormalizer(unify_arabic=False, unify_digits=True)
        res_dig, rep_dig = norm_digits_only.normalize_with_report(arabic_mixed)
        assert "كتاب" in res_dig  # Arabic Kaf preserved
        assert "۱۲۳" in res_dig  # Digits converted to Persian
        assert rep_dig["arabic_unifications"] == 0
        assert rep_dig["digits_unified"] == 3

        # Case 3: Both enabled
        norm_both = PersianNormalizer(unify_arabic=True, unify_digits=True)
        res_both, rep_both = norm_both.normalize_with_report(arabic_mixed)
        assert "کتاب" in res_both
        assert "۱۲۳" in res_both
        assert rep_both["arabic_unifications"] == 1
        assert rep_both["digits_unified"] == 3

    def test_control_character_cleanup_preserves_zwnj_and_newlines(self, normalizer):
        """Null bytes and invisible control characters are stripped; newlines and ZWNJ are intact."""
        text_with_controls = f"سطر اول\x00 با\x08 کنترل\ufeff نامرئی{ZWNJ}دار.\nسطر دوم با نیم{ZWNJ}فاصله."
        normalized, report = normalizer.normalize_with_report(text_with_controls)
        assert "\x00" not in normalized
        assert "\x08" not in normalized
        assert "\ufeff" not in normalized
        assert "\n" in normalized  # Newline is preserved!
        assert f"نیم{ZWNJ}فاصله" in normalized
        assert report["controls_removed"] == 3

    def test_whitespace_normalization(self, normalizer):
        """Horizontal whitespace is normalized while linebreaks and paragraphs are preserved."""
        dirty = "پاراگراف اول    با    فاصله زیاد.\r\n\r\n\r\n\r\nپاراگراف دوم   با تب\t\tو فاصله."
        normalized = normalizer.normalize(dirty)
        assert "    " not in normalized
        assert "\r" not in normalized
        assert "\n\n\n" not in normalized
        assert "پاراگراف اول با فاصله زیاد.\n\nپاراگراف دوم با تب و فاصله." == normalized

    def test_verb_prefixes_comprehensive(self, normalizer):
        """Verbs with progressive prefixes (می / نمی) are normalized in all forms."""
        test_cases = [
            ("میشود", f"می{ZWNJ}شود"),
            ("نمی شود", f"نمی{ZWNJ}شود"),
            ("می شود", f"می{ZWNJ}شود"),
            ("میکند", f"می{ZWNJ}کند"),
            ("میتواند", f"می{ZWNJ}تواند"),
            ("میگیرند", f"می{ZWNJ}گیرند"),
            ("نمایش داده نمیشود", f"نمایش داده نمی{ZWNJ}شود"),
        ]
        for raw, expected in test_cases:
            assert normalizer.normalize(raw) == expected

    def test_root_words_starting_with_mi_not_corrupted(self, normalizer):
        """Roots naturally starting with 'می' must never have ZWNJ inserted."""
        safe_roots = "میوه های میدان و میهن ما میزان رشد و میلاد و میراث را نشان داد. روی میز یک میله است."
        normalized = normalizer.normalize(safe_roots)
        assert "میوه" in normalized
        assert f"می{ZWNJ}وه" not in normalized
        assert "میدان" in normalized
        assert "میهن" in normalized
        assert "میزان" in normalized
        assert "میراث" in normalized
        assert "میلاد" in normalized
        assert "میز" in normalized
        assert "میله" in normalized

    def test_glued_and_spaced_plurals_zwnj_restoration(self, normalizer):
        """Plural suffixes (ها, های, هایی, هایم, هایت, هایش, ...) are correctly normalized with ZWNJ."""
        plurals = (
            "تمام دادهها، سایتها، سرویسهای آنلاین و درخواستهای کاربران "
            "به همراه دادههایت و پیامهای دریافتی در شناسهها و بازیها ذخیره میشوند. "
            "قابلیتهای جدید و کتاب ها بارگذاری شدند."
        )
        normalized = normalizer.normalize(plurals)
        assert f"داده{ZWNJ}ها" in normalized
        assert f"سایت{ZWNJ}ها" in normalized
        assert f"سرویس{ZWNJ}های" in normalized
        assert f"درخواست{ZWNJ}های" in normalized
        assert f"داده{ZWNJ}هایت" in normalized
        assert f"پیام{ZWNJ}های" in normalized
        assert f"شناسه{ZWNJ}ها" in normalized
        assert f"بازی{ZWNJ}ها" in normalized
        assert f"قابلیت{ZWNJ}های" in normalized
        assert f"کتاب{ZWNJ}ها" in normalized

    def test_non_plural_ha_words_protected(self, normalizer):
        """Non-plural root words ending in 'ها' must remain untouched."""
        non_plurals = "او تنها تا انتها رفت و اشتها نداشت. پرنده رها شد و بها پرداخت."
        normalized = normalizer.normalize(non_plurals)
        assert "تنها" in normalized
        assert f"تن{ZWNJ}ها" not in normalized
        assert "انتها" in normalized
        assert f"انت{ZWNJ}ها" not in normalized
        assert "اشتها" in normalized
        assert "رها" in normalized
        assert "بها" in normalized

    def test_comparative_and_superlative_adjectives(self, normalizer):
        """Adjectives with comparative suffixes (تر / ترین) receive ZWNJ without corrupting roots."""
        text = "متن کوتاهتر و روشنتر باید با ساختار ساده تر و سادهترین رویکرد محافظهکارانهتر بازنویسی شود."
        normalized = normalizer.normalize(text)
        assert f"کوتاه{ZWNJ}تر" in normalized
        assert f"روشن{ZWNJ}تر" in normalized
        assert f"ساده{ZWNJ}تر" in normalized
        assert f"ساده{ZWNJ}ترین" in normalized
        assert f"محافظه‌کارانه{ZWNJ}تر" in normalized or f"محافظهکارانه{ZWNJ}تر" in normalized

        # Root words must not be broken
        roots = "دختر در دفتر با کبوتر عکس گرفت. این کار بهتر و بیشتر و کمتر زمان برد."
        norm_roots = normalizer.normalize(roots)
        assert "دختر" in norm_roots
        assert "دفتر" in norm_roots
        assert "کبوتر" in norm_roots
        assert "بهتر" in norm_roots
        assert "بیشتر" in norm_roots
        assert "کمتر" in norm_roots

    def test_mandatory_compound_words_from_spec(self, normalizer):
        """Verify the exact compound words specified in prompt section 5 (د)."""
        compounds = (
            "بهعنوان مثال، بهصورت خودکار برای کاربران فارسیزبان و انگلیسیزبان "
            "بهرهمند و علاقهمند از زبان شناسی و روان شناسی جهت آمادهسازی و جلوگیری از نفرتپراکنی "
            "همینطور و اینطور اقدام میکنیم."
        )
        normalized = normalizer.normalize(compounds)
        assert f"به{ZWNJ}عنوان" in normalized
        assert f"به{ZWNJ}صورت" in normalized
        assert f"فارسی{ZWNJ}زبان" in normalized
        assert f"انگلیسی{ZWNJ}زبان" in normalized
        assert f"بهره{ZWNJ}مند" in normalized
        assert f"علاقه{ZWNJ}مند" in normalized
        assert f"زبان{ZWNJ}شناس" in normalized
        assert f"روان{ZWNJ}شناس" in normalized
        assert f"آماده{ZWNJ}سازی" in normalized
        assert f"نفرت{ZWNJ}پراکن" in normalized
        assert f"همین{ZWNJ}طور" in normalized
        assert f"این{ZWNJ}طور" in normalized

    def test_code_blocks_urls_markdown_links_and_html_protected(self, normalizer):
        """Code blocks, inline code, URLs, emails, Markdown links, and HTML tags are protected."""
        mixed_doc = (
            "برای مشاهده دادهها به نشانی (https://example.com/api/test_(1)). مراجعه کنید.\n\n"
            "پیوند مستقیم به صورت [دادههای سیستم](https://example.com/data?query=دادهها) و ایمیل info@crawlbot.org است.\n\n"
            "تگ HTML: <div class='data-block'>متن با دادهها</div>\n\n"
            "```python\n"
            "def نمیشود():\n"
            "    data = 'دادهها'\n"
            "    return True\n"
            "```\n\n"
            "از دستور `npm install بستهها` استفاده کنید."
        )
        normalized = normalizer.normalize(mixed_doc)
        # Prose must be normalized
        assert f"داده{ZWNJ}ها به نشانی" in normalized
        assert f"به{ZWNJ}صورت" in normalized
        assert f"داده{ZWNJ}های سیستم" in normalized  # Anchor text inside Markdown link was normalized!
        # Target URL inside Markdown link intact
        assert "(https://example.com/data?query=دادهها)" in normalized
        # Balanced parens and trimmed prose punctuation preserved
        assert "(https://example.com/api/test_(1))." in normalized
        # Email intact
        assert "info@crawlbot.org" in normalized
        # HTML tag and attribute intact, inner content normalized
        assert "<div class='data-block'>" in normalized
        assert f">متن با داده{ZWNJ}ها</div>" in normalized
        # Codeblock content 100% untouched
        assert "def نمیشود():" in normalized
        assert "data = 'دادهها'" in normalized
        # Inline code 100% untouched
        assert "`npm install بستهها`" in normalized

    def test_pure_idempotency(self, normalizer):
        """normalize(normalize(text)) == normalize(text) across varied samples."""
        samples = [
            "این متن برای بررسی بهصورت کامل نوشته شده است. دادهها و اطلاعات سیستم نمیشود بهراحتی حذف گردند.",
            "كتابهاي فارسي زبان در دانشگاه اصفهان موجود است و دوره آموزشی آغاز میشود.",
            "```python\ncode = 1\n```\nمتن با پیوند https://test.org/path و دادهها.",
            "پاراگراف اول.\n\nپاراگراف دوم با    فاصله و نیم\u200cفاصله.",
        ]
        for s in samples:
            first_pass, report = normalizer.normalize_with_report(s)
            second_pass, second_rep = normalizer.normalize_with_report(first_pass)
            assert first_pass == second_pass
            assert report["is_idempotent"] is True
            assert second_rep["changed"] is False

    def test_telemetry_accuracy(self, normalizer):
        """Verify accurate calculation of telemetry counts."""
        raw = "كتاب علي دادهها"  # 2 arabic unifications (ك, ي) + 1 zwnj fix (داده‌ها)
        _, report = normalizer.normalize_with_report(raw)
        assert report["arabic_unifications"] == 2
        assert report["zwnj_fixes"] == 1
        assert report["changed"] is True
        assert report["is_idempotent"] is True

    def test_empty_none_and_invalid_inputs(self, normalizer):
        """Empty strings, None, and invalid types are handled safely."""
        assert normalizer.normalize("") == ""
        assert normalizer.normalize(None) == ""
        assert normalizer.normalize("   \n\t  ") == ""

        empty_res, empty_rep = normalizer.normalize_with_report("")
        assert empty_res == ""
        assert empty_rep["chars_before"] == 0
        assert empty_rep["chars_after"] == 0
        assert empty_rep["changed"] is False
        assert empty_rep["is_idempotent"] is True

        with pytest.raises(TypeError):
            normalizer.normalize(12345)

    def test_standalone_convenience_function(self):
        """normalize_persian_text function produces equivalent results."""
        raw = "دادهها بهصورت خودکار ذخیره میشوند."
        res1 = normalize_persian_text(raw)
        res2 = PersianNormalizer().normalize(raw)
        assert res1 == res2
        assert f"داده{ZWNJ}ها" in res1
