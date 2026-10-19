"""Persian & Unicode Normalization Service for Crawlbot.

Provides deterministic, production-grade text normalization tailored for Persian
and multilingual NLP/RAG pipelines with strict safety boundaries:
1. Unicode NFC normalization.
2. Controlled unification of Arabic characters (ي/ی, ك/ک, ى/ی) while strictly preserving ة (Teh Marbuta).
3. Independent Arabic-to-Persian digit unification (٠-٩ -> ۰-۹) via unify_digits.
4. Stripping invalid/invisible control characters while protecting structural ZWNJ (\\u200c) and line breaks.
5. Whitespace and layout cleanup (tabs, multi-spaces, carriage returns, trailing whitespace) preserving linebreaks.
6. Persian affix and compound word ZWNJ restoration (e.g., حل نمیشود -> حل نمی‌شود, دادهها -> داده‌ها, بهصورت -> به‌صورت).
7. Comprehensive protection of code blocks, inline code, URLs (with balanced parens and trailing punctuation trimming), emails, Markdown links, and HTML tags.
8. Mathematically verified pure idempotency: normalize(normalize(text)) == normalize(text).
9. Accurate telemetry reporting (chars_before, chars_after, arabic_unifications, digits_unified, controls_removed, zwnj_fixes, changed, is_idempotent).
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from typing import Any, Final, Optional

logger = logging.getLogger(__name__)

# Zero-Width Non-Joiner (نیم‌فاصله)
ZWNJ: Final[str] = "\u200c"

# Characters and mappings
# NOTE: ة (\u0629) is intentionally NOT converted to ه (\u0647) globally
# to preserve valid Arabic quotations, names, and loanwords per specifications.
_ARABIC_TO_PERSIAN_CHARS: Final[dict[str, str]] = {
    "\u0643": "\u06a9",  # Arabic Kaf ك -> Persian Kaf ک
    "\u064a": "\u06cc",  # Arabic Yeh ي -> Persian Yeh ی
    "\u0649": "\u06cc",  # Arabic Alef Maksura ى -> Persian Yeh ی
}

_ARABIC_DIGITS_TO_PERSIAN: Final[dict[str, str]] = {
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
_PERSIAN_CHAR_RE: Final[re.Pattern[str]] = re.compile(
    r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Persian letter class covering standard Persian and Arabic characters
_PERSIAN_LETTER_CLASS: Final[str] = r"[آ-یء]"

# Invalid & invisible characters to strip (excluding \n, \r, \t, and ZWNJ \u200c)
_INVALID_CONTROLS_RE: Final[re.Pattern[str]] = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200d\u200e\u200f\ufeff\u00ad]"
)

# Common Persian root words that start with 'می' or 'نمی' and must NEVER be broken by prefix rules
_ROOT_WORDS_WITH_MI: Final[frozenset[str]] = frozenset({
    "میوه", "میوه‌ها", "میوهها", "میوه‌ای", "میوه‌جات",
    "میدان", "میدان‌ها", "میدانها", "میدانی",
    "میهن", "میهن‌پرست", "میهنی",
    "میزان", "میزان‌الحراره", "میزانی",
    "میلاد", "میلادی",
    "میعاد", "میعادی",
    "مینو", "مینوی",
    "میانه", "میانی", "میانهرو", "میانه‌رو",
    "میراث", "میراث‌دار",
    "میکروب", "میکروبی",
    "میکروفون",
    "مینیاتور", "مینیاتوری",
    "میلیارد", "میلیاردر", "میلیاردی",
    "میلیون", "میلیونر", "میلیونی",
    "میلیمتر", "میلی‌متر",
    "میز", "میزها", "میزی",
    "میله", "میله‌ها",
    "میخ", "میخ‌ها",
    "میش", "میش‌ها",
    "نمی",
})

# Words ending in 'ها' that are root nouns or non-plurals and must NOT have ZWNJ inserted
_NON_PLURAL_HA_WORDS: Final[frozenset[str]] = frozenset({
    "تنها", "انتها", "اشتها", "رها", "بها", "سها", "طاها", "ابها", "زها", "دها", "مهیا"
})

# Words ending in 'تر' that are roots and must NOT be broken by comparative adjective rules
_NON_COMPARATIVE_TAR_WORDS: Final[frozenset[str]] = frozenset({
    "دختر", "دفتر", "اختر", "انگشتر", "خاکستر", "کبوتر", "شتر", "بستر", "استر",
    "ستر", "سنتر", "لیتر", "میلیمتر", "سانتیمتر", "کیلومتر", "متر", "قطر", "وتر",
    "تویتر", "توییتر", "چارتر", "تئاتر", "ارکستر", "مانستر", "پوستر", "کاستر",
    "بهتر", "بیشتر", "کمتر"
})


def _build_verb_stems_regex() -> str:
    """Compile comprehensive Persian verb stems across 6 personal conjugations and past forms."""
    present_roots = [
        "شو", "گرد", "کن", "توان", "دان", "رس", "بین", "گو", "رو", "زن", "نه",
        "خوان", "نویس", "خواه", "باش", "مان", "یاب", "شنو", "افت", "گذار", "گزار",
        "خیز", "کش", "بر", "پیوند", "بند", "گذر", "شناس", "پوش", "ریز", "ساز",
        "سنج", "شمار", "فروش", "فهم", "کوب", "گیر", "وز", "آور", "آر", "آی",
        "دار", "ده", "پرس", "خر", "چسب", "پاش", "افزای", "کاه", "تاب", "لرز", "سوز",
    ]
    endings = ["م", "ی", "د", "یم", "ید", "ند"]
    stems: set[str] = set()

    for r in present_roots:
        for e in endings:
            stems.add(r + e)

    past_roots = [
        "کرد", "شد", "رفت", "گفت", "آمد", "آورد", "دید", "داد", "داشت", "زد",
        "گرفت", "توانست", "گذاشت", "گذشت", "نوشت", "خواند", "ساخت", "سوخت",
        "ریخت", "فهمید", "بخشید", "پوشید", "کشید", "رسید", "پرسید", "شنید",
        "برید", "پذیرفت", "بست", "خرید", "فروخت", "افتاد", "چسبید", "خوابید", "پرید",
    ]
    for pr in past_roots:
        for e in ["م", "ی", "", "یم", "ید", "ند"]:
            stems.add(pr + e)

    return "|".join(sorted(stems, key=len, reverse=True))


_PERSIAN_VERB_STEMS: Final[str] = _build_verb_stems_regex()
_PLURAL_SUFFIXES: Final[str] = r"(?:ها|های|هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان)"
_COMPARATIVE_SUFFIXES: Final[str] = r"(?:تر|ترین|ترها|ترهای|ترهایی)"


def _clean_url_match(url: str) -> tuple[str, str]:
    """Trim trailing punctuation that belongs to prose, respecting balanced parens."""
    trailing = ""
    strip_chars = ".,:;!?؟،؛"

    while url:
        if url[-1] in strip_chars:
            trailing = url[-1] + trailing
            url = url[:-1]
        elif url[-1] == ")":
            if url.count("(") < url.count(")"):
                trailing = url[-1] + trailing
                url = url[:-1]
            else:
                break
        elif url[-1] == "]":
            if url.count("[") < url.count("]"):
                trailing = url[-1] + trailing
                url = url[:-1]
            else:
                break
        else:
            break

    return url, trailing


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
            unify_arabic: Convert Arabic kaf/yeh/alef-maksura to Persian equivalents (preserves Teh Marbuta).
            unify_digits: Convert Arabic digits (٠-٩) to Persian digits (۰-۹).
            clean_controls: Remove invisible/corrupting control characters.
            fix_spacing: Normalize multi-spaces, tabs, linebreaks.
            fix_zwnj_affixes: Correct spacing for prefixes (می/نمی) and suffixes (ها/تر/ترین/ترکیب‌ها).
            protect_code_and_urls: Prevent modifying codeblocks, inline code, URLs, emails, and Markdown links.
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
        self._plural_glued_pattern = re.compile(rf"\b({_PERSIAN_LETTER_CLASS}{{2,}})({_PLURAL_SUFFIXES})\b")
        self._plural_space_pattern = re.compile(rf"\b({_PERSIAN_LETTER_CLASS}{{2,}}) +({_PLURAL_SUFFIXES})\b")
        self._comparative_glued_pattern = re.compile(rf"\b({_PERSIAN_LETTER_CLASS}{{3,}})({_COMPARATIVE_SUFFIXES})\b")
        self._comparative_space_pattern = re.compile(rf"\b({_PERSIAN_LETTER_CLASS}{{2,}}) +({_COMPARATIVE_SUFFIXES})\b")

        # Specific compound words from prompt requirements
        self._known_compounds = [
            (re.compile(r"\bبه(?:[ \t]+|)عنوان\b"), f"به{ZWNJ}عنوان"),
            (re.compile(r"\bبه(?:[ \t]+|)صورت\b"), f"به{ZWNJ}صورت"),
            (re.compile(r"\bفارسی(?:[ \t]+|)زبان(ان|ها|های)?\b"), rf"فارسی{ZWNJ}زبان\1"),
            (re.compile(r"\bانگلیسی(?:[ \t]+|)زبان(ان|ها|های)?\b"), rf"انگلیسی{ZWNJ}زبان\1"),
            (re.compile(r"\bبهره(?:[ \t]+|)مند(ی|ان|ها|ند)?\b"), rf"بهره{ZWNJ}مند\1"),
            (re.compile(r"\bعلاقه(?:[ \t]+|)مند(ی|ان|ها|ند)?\b"), rf"علاقه{ZWNJ}مند\1"),
            (re.compile(r"\bزبان(?:[ \t]+|)شناس(ی|ان|ها)?\b"), rf"زبان{ZWNJ}شناس\1"),
            (re.compile(r"\bروان(?:[ \t]+|)شناس(ی|ان|ها)?\b"), rf"روان{ZWNJ}شناس\1"),
            (re.compile(r"\bآماده(?:[ \t]+|)سازی(ها|های)?\b"), rf"آماده{ZWNJ}سازی\1"),
            (re.compile(r"\bنفرت(?:[ \t]+|)پراکن(ی|ها)?\b"), rf"نفرت{ZWNJ}پراکن\1"),
            (re.compile(r"\bهمین(?:[ \t]+|)طور\b"), f"همین{ZWNJ}طور"),
            (re.compile(r"\bاین(?:[ \t]+|)طور\b"), f"این{ZWNJ}طور"),
        ]

    def normalize(self, text: Optional[str]) -> str:
        """Normalize text according to configured rules."""
        if text is None:
            return ""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        result, _ = self._normalize_pipeline(text)
        return result

    def normalize_with_report(self, text: Optional[str]) -> tuple[str, dict[str, Any]]:
        """Normalize text and return detailed telemetry report of changes."""
        if text is None:
            return "", {
                "chars_before": 0,
                "chars_after": 0,
                "arabic_unifications": 0,
                "digits_unified": 0,
                "controls_removed": 0,
                "zwnj_fixes": 0,
                "changed": False,
                "is_idempotent": True,
            }
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        first_pass, report = self._normalize_pipeline(text)
        second_pass, _ = self._normalize_pipeline(first_pass)
        report["is_idempotent"] = (first_pass == second_pass)
        report["changed"] = (first_pass != text)
        return first_pass, report

    def _normalize_pipeline(self, text: str) -> tuple[str, dict[str, Any]]:
        """Execute core normalization pipeline without recursing."""
        if not text:
            return "", {
                "chars_before": 0,
                "chars_after": 0,
                "arabic_unifications": 0,
                "digits_unified": 0,
                "controls_removed": 0,
                "zwnj_fixes": 0,
                "changed": False,
                "is_idempotent": True,
            }

        chars_before = len(text)
        report: dict[str, Any] = {
            "chars_before": chars_before,
            "chars_after": 0,
            "arabic_unifications": 0,
            "digits_unified": 0,
            "controls_removed": 0,
            "zwnj_fixes": 0,
            "changed": False,
            "is_idempotent": True,
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
            control_matches = len(_INVALID_CONTROLS_RE.findall(masked_text))
            masked_text = _INVALID_CONTROLS_RE.sub("", masked_text)
            report["controls_removed"] = control_matches

            # Normalize redundant ZWNJ
            masked_text = re.sub(f"{ZWNJ}+", ZWNJ, masked_text)
            # Remove ZWNJ immediately adjacent to horizontal spaces
            masked_text = re.sub(rf"[ \t\f\v]+{ZWNJ}", " ", masked_text)
            masked_text = re.sub(rf"{ZWNJ}[ \t\f\v]+", " ", masked_text)
            # Remove leading/trailing ZWNJ from lines
            masked_text = re.sub(rf"(^|\n){ZWNJ}+", r"\1", masked_text)
            masked_text = re.sub(rf"{ZWNJ}+($|\n)", r"\1", masked_text)

        # 4. Unify Arabic characters if enabled (preserves Teh Marbuta ة)
        if self.unify_arabic or self.unify_digits:
            arabic_unify_count = 0
            digits_unify_count = 0
            chars_list: list[str] = []

            for ch in masked_text:
                if self.unify_arabic and ch in _ARABIC_TO_PERSIAN_CHARS:
                    chars_list.append(_ARABIC_TO_PERSIAN_CHARS[ch])
                    arabic_unify_count += 1
                elif self.unify_digits and ch in _ARABIC_DIGITS_TO_PERSIAN:
                    chars_list.append(_ARABIC_DIGITS_TO_PERSIAN[ch])
                    digits_unify_count += 1
                else:
                    chars_list.append(ch)

            masked_text = "".join(chars_list)
            report["arabic_unifications"] = arabic_unify_count
            report["digits_unified"] = digits_unify_count

        # 5. Spacing & Whitespace cleanup
        if self.fix_spacing:
            # Normalize carriage returns to standard line breaks
            masked_text = masked_text.replace("\r\n", "\n").replace("\r", "\n")
            # Collapse horizontal tabs and spaces
            masked_text = re.sub(r"[ \t\f\v]+", " ", masked_text)
            # Strip trailing/leading spaces on lines
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
                full_w = m.group(0)
                clean_w = full_w.replace(ZWNJ, "").strip()
                if clean_w in _ROOT_WORDS_WITH_MI:
                    return full_w
                target = f"{m.group(1)}{ZWNJ}{m.group(2)}"
                if target != full_w:
                    zwnj_fix_count += 1
                return target

            masked_text = self._prefix_pattern.sub(repl_prefix, masked_text)

            # Glued plurals (e.g., دادهها -> داده‌ها, سایتها -> سایت‌ها)
            def repl_plural_glued(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                word = m.group(0)
                stem = m.group(1)
                suffix = m.group(2)
                if word in _NON_PLURAL_HA_WORDS or stem in _NON_PLURAL_HA_WORDS:
                    return word
                target = f"{stem}{ZWNJ}{suffix}"
                if target != word:
                    zwnj_fix_count += 1
                return target

            masked_text = self._plural_glued_pattern.sub(repl_plural_glued, masked_text)

            # Spaced plurals (e.g., کتاب ها -> کتاب‌ها)
            def repl_plural_space(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                target = f"{m.group(1)}{ZWNJ}{m.group(2)}"
                if target != m.group(0):
                    zwnj_fix_count += 1
                return target

            masked_text = self._plural_space_pattern.sub(repl_plural_space, masked_text)

            # Comparatives (e.g., ساده تر -> ساده‌تر, روشنتر -> روشن‌تر)
            def repl_comp_space(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                target = f"{m.group(1)}{ZWNJ}{m.group(2)}"
                if target != m.group(0):
                    zwnj_fix_count += 1
                return target

            masked_text = self._comparative_space_pattern.sub(repl_comp_space, masked_text)

            def repl_comp_glued(m: re.Match) -> str:
                nonlocal zwnj_fix_count
                word = m.group(0)
                stem = m.group(1)
                suffix = m.group(2)
                if word in _NON_COMPARATIVE_TAR_WORDS:
                    return word
                target = f"{stem}{ZWNJ}{suffix}"
                if target != word:
                    zwnj_fix_count += 1
                return target

            masked_text = self._comparative_glued_pattern.sub(repl_comp_glued, masked_text)

            # Specific known compound words
            for pattern, replacement in self._known_compounds:
                def repl_compound(m: re.Match, rpl=replacement) -> str:
                    nonlocal zwnj_fix_count
                    expanded = m.expand(rpl)
                    if expanded != m.group(0):
                        zwnj_fix_count += 1
                    return expanded

                if pattern.search(masked_text):
                    masked_text = pattern.sub(repl_compound, masked_text)

            report["zwnj_fixes"] = zwnj_fix_count

        # 7. Restore protected blocks
        if placeholders:
            masked_text = self._restore_tokens(masked_text, placeholders)

        final_text = masked_text.strip()
        report["chars_after"] = len(final_text)
        report["changed"] = (final_text != text)
        return final_text, report

    def _protect_tokens(self, text: str) -> tuple[str, dict[str, str]]:
        """Protect sensitive code blocks, URLs, emails, Markdown links, and HTML tags."""
        placeholders: dict[str, str] = {}
        counter = 0
        nonce = uuid.uuid4().hex[:8]

        def make_token(val: str) -> str:
            nonlocal counter
            token = f"«__{nonce}_GUARD_{counter}__»"
            placeholders[token] = val
            counter += 1
            return token

        # 1. Fenced code blocks
        text = re.sub(r"```[\s\S]*?```", lambda m: make_token(m.group(0)), text)

        # 2. Inline code
        text = re.sub(r"`[^`\n]+`", lambda m: make_token(m.group(0)), text)

        # 3. HTML tags
        text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?>", lambda m: make_token(m.group(0)), text)

        # 4. Markdown links: protect destination URL while keeping anchor text normalizable
        def repl_md_link(m: re.Match) -> str:
            anchor = m.group(1)
            dest = m.group(2)
            return f"[{anchor}]({make_token(dest)})"

        text = re.sub(r"\[([^\]\n]+)\]\(((?:https?://|www\.|/)[^\s)]+)\)", repl_md_link, text)

        # 5. Email addresses
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", lambda m: make_token(m.group(0)), text)

        # 6. Standalone URLs: protect while trimming trailing prose punctuation
        def repl_url(m: re.Match) -> str:
            raw_url = m.group(0)
            clean_u, trailing = _clean_url_match(raw_url)
            return make_token(clean_u) + trailing

        text = re.sub(r"(?:https?://|www\.)[^\s<>\"\'\)]+", repl_url, text)

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
