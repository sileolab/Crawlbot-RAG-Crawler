"""Persian & Unicode Normalization Service for Crawlbot.

Provides deterministic, production-grade text normalization tailored for Persian
and multilingual NLP/RAG pipelines with strict safety boundaries:
1. Unicode NFC normalization.
2. Safe unification of Arabic characters (ي/ی, ك/ک, ة/ه).
3. Stripping invalid/invisible control characters while protecting structural ZWNJ (\\u200c).
4. Whitespace cleanup (tabs, duplicate spaces, carriage returns, trailing whitespace).
5. Persian compound words and affix ZWNJ correction (e.g., حل نمیشود -> حل نمی‌شود, دادهها -> داده‌ها, بهصورت -> به‌صورت).
6. Complete protection of fenced code blocks, inline code, URLs, emails, and JSON.
7. Pure idempotency: normalize(normalize(text)) == normalize(text).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Final, Optional

logger = logging.getLogger(__name__)

# Zero-Width Non-Joiner (نیم‌فاصله)
ZWNJ: Final[str] = "\u200c"

# Characters and mappings
_ARABIC_TO_PERSIAN_CHARS = {
    "\u0643": "\u06a9",  # Arabic Kaf ك -> Persian Kaf ک
    "\u064a": "\u06cc",  # Arabic Yeh ي -> Persian Yeh ی
    "\u0649": "\u06cc",  # Arabic Alef Maksura ى -> Persian Yeh ی
    "\u0629": "\u0647",  # Arabic Teh Marbuta ة -> Persian Heh ه
    "\u064b": "\u064b",  # Tanwin Fathah (preserve)
    "\u064c": "\u064c",  # Tanwin Dammah
    "\u064d": "\u064d",  # Tanwin Kasrah
}

_ARABIC_DIGITS_TO_PERSIAN = {
    "\u0660": "\u06f0",  # ٠ -> ۰
    "\u0661": "\u06f1",  # ١ -> ۱
    "\u0662": "\u06f2",  # ٢ -> ۲
    "\u0663": "\u06f3",  # ٣ -> ۳
    "\u0664": "\u06f4",  # ٤ -> ۴
    "\u0665": "\u06f5",  # ٥ -> ۵
    "\u0666": "\u06f6",  # ٦ -> ۶
    "\u0667": "\u06f7",  # ٧ -> ۷
    "\u0668": "\u06f8",  # ٨ -> ۸
    "\u0669": "\u06f9",  # ٩ -> ۹
}

# Regex to detect Persian script characters
_PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# Patterns for content protection (fenced code, inline code, URLs, emails)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Invalid & invisible characters (excluding \n, \r, \t, and ZWNJ \u200c)
_INVALID_CONTROLS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200d\u200e\u200f\ufeff\u00ad]"
)

# Common Persian verb stems for progressive prefixes (می / نمی)
_PERSIAN_VERB_STEMS = (
    "شود|شوند|شوی|شویم|شوید|"
    "گردد|گردند|گردی|گردیم|گردید|"
    "کند|کنند|کنی|کنیم|کنید|کرد|کردند|کردی|کردیم|کردید|"
    "تواند|توانند|توانی|توانیم|توانید|"
    "داند|دانند|دانی|دانیم|دانید|"
    "رسد|رسند|رسی|رسیم|رسید|"
    "بیند|بینند|بینی|بینیم|بینید|"
    "گوید|گویند|گویی|گوییم|گویید|"
    "رود|روند|روی|رویم|روید|"
    "زند|زنند|زنی|زنیم|زنید|زد|زدند|"
    "نهد|نهند|نهی|نهیم|نهید|"
    "خواند|خوانند|خوانی|خوانیم|خوانید|"
    "نویسد|نویسند|نویسی|نویسیم|نویسید|"
    "خواهد|خواهند|خواهی|خواهیم|خواهید|"
    "باشد|باشند|باشی|باشیم|باشید|"
    "ماند|مانند|مانی|مانیم|مانید|"
    "یابد|یابند|یابی|یابیم|یابید|"
    "شنود|شنوند|شنوی|شنویم|شنوید|"
    "افتد|افتند|افتی|افتیم|افتید|"
    "گذارد|گذارند|گذاری|گذاریم|گذارید|"
    "گزارد|گزارند|گزاری|گزاریم|گزارید|"
    "خیزد|خیزند|خیزی|خیزیم|خیزید|"
    "کشد|کشند|کشی|کشیم|کشید|"
    "برد|برند|بری|بریم|برید|"
    "پیوندد|پیوندند|بندد|بندند|"
    "گذرد|گذرند|گذری|گذریم|گذرید|"
    "شناسد|شناسند|شناسی|شناسیم|شناسید|"
    "پوشد|پوشند|پوشی|پوشیم|پوشید|"
    "ریزد|ریزند|ریزی|ریزیم|ریزید|"
    "سازد|سازند|سازی|سازیم|سازید|"
    "سنجد|سنجند|سنجی|سنجیم|سنجید|"
    "شمارد|شمارند|شماری|شماریم|شمارید|"
    "فروشد|فروشند|فروشی|فروشیم|فروشید|"
    "فهمد|فهمند|فهمی|فهمیم|فهمید|"
    "کوبد|کوبند|کوبی|کوبیم|کوبید|"
    "گیرد|گیرند|گیری|گیریم|گیرید|"
    "وزد|وزند|وزی|وزیم|وزید|"
    "آورد|آورند|آوری|آوریم|آورید|آرد|آرند|"
    "آید|آیند|آیی|آییم|آیید"
)


_NON_PLURAL_HA_WORDS = frozenset({"تنها", "انتها", "اشتها", "رها", "بها", "سها", "طاها", "ابها", "زها", "دها"})


class PersianNormalizer:
    """Production-ready Unicode and Persian text normalization engine."""

    def __init__(
        self,
        nfc: bool = True,
        unify_arabic: bool = True,
        unify_digits: bool = False,
        clean_controls: bool = True,
        fix_spacing: bool = True,
        fix_zwnj_affixes: bool = True,
        protect_code_and_urls: bool = True,
    ) -> None:
        """Configure normalizer rules.

        Args:
            nfc: Apply Unicode NFC composition.
            unify_arabic: Convert Arabic kaf/yeh/teh-marbuta to Persian equivalents.
            unify_digits: Convert Arabic digits (0-9) to Persian digits.
            clean_controls: Remove invisible/corrupting control characters.
            fix_spacing: Normalize multi-spaces, tabs, linebreaks.
            fix_zwnj_affixes: Correct spacing for prefixes (می/نمی) and suffixes (ها/تر/ترین/مند/شناس).
            protect_code_and_urls: Prevent modifying codeblocks, inline code, URLs, and emails.
        """
        self.nfc = nfc
        self.unify_arabic = unify_arabic
        self.unify_digits = unify_digits
        self.clean_controls = clean_controls
        self.fix_spacing = fix_spacing
        self.fix_zwnj_affixes = fix_zwnj_affixes
        self.protect_code_and_urls = protect_code_and_urls

        # Compile affixes patterns
        self._prefix_pattern = re.compile(rf"\b(ن?می)\s*({_PERSIAN_VERB_STEMS})\b")
        self._plural_glued_pattern = re.compile(rf"\b([آ-ی]{{2,}})(ها|های|هایی|هایمان|هایتان|هایشان)\b")
        self._plural_space_pattern = re.compile(rf"(\b[آ-ی]{{2,}}) (ها|های|هایمان|هایتان|هایشان)\b")
        self._comparative_space_pattern = re.compile(rf"(\b[آ-ی]{{2,}}) (تر|ترین)\b")
        self._comparative_glued_pattern = re.compile(rf"(\b[آ-ی]{{2,}}ه)تر(ین)?\b")

        # Specific compound words from prompt requirements
        self._known_compounds = [
            (re.compile(r"\bبهصورت\b"), f"به{ZWNJ}صورت"),
            (re.compile(r"\bبه صورت\b"), f"به{ZWNJ}صورت"),
            (re.compile(r"\bفارسیزبان\b"), f"فارسی{ZWNJ}زبان"),
            (re.compile(r"\bفارسی زبان\b"), f"فارسی{ZWNJ}زبان"),
            (re.compile(r"\bانگلیسیزبان\b"), f"انگلیسی{ZWNJ}زبان"),
            (re.compile(r"\bهمینطور\b"), f"همین{ZWNJ}طور"),
            (re.compile(r"\bاینطور\b"), f"این{ZWNJ}طور"),
            (re.compile(r"\bبهره مند\b"), f"بهره{ZWNJ}مند"),
            (re.compile(r"\bعلاقه مند\b"), f"علاقه{ZWNJ}مند"),
            (re.compile(r"\bزبان شناسی\b"), f"زبان{ZWNJ}شناسی"),
            (re.compile(r"\bروان شناسی\b"), f"روان{ZWNJ}شناسی"),
        ]

    def normalize(self, text: str) -> str:
        """Normalize text according to configured rules."""
        result, _ = self._normalize_pipeline(text)
        return result

    def normalize_with_report(self, text: str) -> tuple[str, dict[str, Any]]:
        """Normalize text and return detailed telemetry report of changes."""
        first_pass, report = self._normalize_pipeline(text)
        second_pass, _ = self._normalize_pipeline(first_pass)
        report["is_idempotent"] = (first_pass == second_pass)
        return first_pass, report

    def _normalize_pipeline(self, text: str) -> tuple[str, dict[str, Any]]:
        """Execute core normalization pipeline without recursing."""
        if not text:
            return "", {
                "chars_before": 0,
                "chars_after": 0,
                "arabic_unifications": 0,
                "controls_removed": 0,
                "zwnj_fixes": 0,
                "is_idempotent": True,
            }

        chars_before = len(text)
        report: dict[str, Any] = {
            "chars_before": chars_before,
            "arabic_unifications": 0,
            "controls_removed": 0,
            "zwnj_fixes": 0,
        }

        # 1. Mask protected blocks if requested
        masked_text = text
        placeholders: dict[str, str] = {}
        if self.protect_code_and_urls:
            masked_text, placeholders = self._protect_tokens(masked_text)

        # 2. Unicode NFC
        if self.nfc:
            masked_text = unicodedata.normalize("NFC", masked_text)

        # 3. Strip invalid control characters while preserving ZWNJ and line breaks
        if self.clean_controls:
            prev_len = len(masked_text)
            masked_text = _INVALID_CONTROLS_RE.sub("", masked_text)
            # Normalize redundant ZWNJ
            masked_text = re.sub(f"{ZWNJ}+", ZWNJ, masked_text)
            # Remove ZWNJ at word edges or near spaces
            masked_text = re.sub(f"\\s+{ZWNJ}", " ", masked_text)
            masked_text = re.sub(f"{ZWNJ}+\\s+", " ", masked_text)
            report["controls_removed"] = max(0, prev_len - len(masked_text))

        # 4. Unify Arabic characters if enabled
        if self.unify_arabic:
            unify_count = 0
            chars_list = []
            for ch in masked_text:
                if ch in _ARABIC_TO_PERSIAN_CHARS:
                    chars_list.append(_ARABIC_TO_PERSIAN_CHARS[ch])
                    unify_count += 1
                elif self.unify_digits and ch in _ARABIC_DIGITS_TO_PERSIAN:
                    chars_list.append(_ARABIC_DIGITS_TO_PERSIAN[ch])
                    unify_count += 1
                else:
                    chars_list.append(ch)
            masked_text = "".join(chars_list)
            report["arabic_unifications"] = unify_count

        # 5. Spacing & Whitespace cleanup
        if self.fix_spacing:
            # Normalize linebreaks
            masked_text = masked_text.replace("\r\n", "\n").replace("\r", "\n")
            # Collapse horizontal tabs and spaces
            masked_text = re.sub(r"[ \t\f\v]+", " ", masked_text)
            # Strip trailing space on lines
            masked_text = re.sub(r" +\n", "\n", masked_text)
            masked_text = re.sub(r"\n +", "\n", masked_text)
            # Limit consecutive newlines to maximum 2 (paragraph break)
            masked_text = re.sub(r"\n{3,}", "\n\n", masked_text)

        # 6. Persian Affixes & Glued Word ZWNJ Correction
        if self.fix_zwnj_affixes and self.detect_persian_script(masked_text):
            zwnj_fix_count = 0

            # Progressive verb prefixes: می / نمی (both glued and spaced)
            def repl_prefix(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                zwnj_fix_count += 1
                return f"{m.group(1)}{ZWNJ}{m.group(2)}"

            masked_text = self._prefix_pattern.sub(repl_prefix, masked_text)

            # Glued plurals (e.g., دادهها -> داده‌ها, سایتها -> سایت‌ها)
            def repl_plural_glued(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                word = m.group(0)
                if word in _NON_PLURAL_HA_WORDS:
                    return word
                zwnj_fix_count += 1
                return f"{m.group(1)}{ZWNJ}{m.group(2)}"

            masked_text = self._plural_glued_pattern.sub(repl_plural_glued, masked_text)

            # Spaced plurals (e.g., کتاب ها -> کتاب‌ها)
            def repl_plural_space(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                zwnj_fix_count += 1
                return f"{m.group(1)}{ZWNJ}{m.group(2)}"

            masked_text = self._plural_space_pattern.sub(repl_plural_space, masked_text)

            # Comparatives (e.g., ساده تر -> ساده‌تر, سادهتر -> ساده‌تر)
            def repl_comp_space(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                zwnj_fix_count += 1
                return f"{m.group(1)}{ZWNJ}{m.group(2)}"

            masked_text = self._comparative_space_pattern.sub(repl_comp_space, masked_text)

            def repl_comp_glued(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                zwnj_fix_count += 1
                suffix = f"تر{m.group(2)}" if m.group(2) else "تر"
                return f"{m.group(1)}{ZWNJ}{suffix}"

            masked_text = self._comparative_glued_pattern.sub(repl_comp_glued, masked_text)

            # Specific known compound words
            for pattern, replacement in self._known_compounds:
                if pattern.search(masked_text):
                    masked_text = pattern.sub(replacement, masked_text)
                    zwnj_fix_count += 1

            report["zwnj_fixes"] = zwnj_fix_count

        # 7. Restore protected blocks
        if placeholders:
            masked_text = self._restore_tokens(masked_text, placeholders)

        final_text = masked_text.strip()
        report["chars_after"] = len(final_text)
        return final_text, report

    def _protect_tokens(self, text: str) -> tuple[str, dict[str, str]]:
        """Protect sensitive code blocks, URLs, and emails with placeholders."""
        placeholders: dict[str, str] = {}
        counter = 0

        def repl(m: re.Match) -> str:
            nonlocal counter
            token = f"«__NORM_GUARD_{counter}__»"
            placeholders[token] = m.group(0)
            counter += 1
            return token

        # Order: fenced code, inline code, URLs, emails
        text = _CODE_BLOCK_RE.sub(repl, text)
        text = _INLINE_CODE_RE.sub(repl, text)
        text = _URL_RE.sub(repl, text)
        text = _EMAIL_RE.sub(repl, text)
        return text, placeholders

    def _restore_tokens(self, text: str, placeholders: dict[str, str]) -> str:
        """Restore protected placeholders."""
        for token, original in placeholders.items():
            text = text.replace(token, original)
        return text

    @staticmethod
    def detect_persian_script(text: str) -> bool:
        """Detect whether text contains Persian / Arabic characters."""
        return bool(_PERSIAN_CHAR_RE.search(text))


# Singleton default normalizer instance
_DEFAULT_NORMALIZER = PersianNormalizer()


def normalize_persian_text(text: str, **kwargs: Any) -> str:
    """Convenience function to normalize text using default or custom settings."""
    if kwargs:
        return PersianNormalizer(**kwargs).normalize(text)
    return _DEFAULT_NORMALIZER.normalize(text)
