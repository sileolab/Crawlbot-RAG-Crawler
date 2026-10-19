"""HTML Content Cleaner for RAG and Document Extraction.

This module provides the ContentCleaner service to strip boilerplate, scripts,
hidden elements, and navigation chrome from raw HTML pages while preserving
the primary informative content.
"""

from __future__ import annotations

import re
from typing import Final

from bs4 import BeautifulSoup, Comment, Tag


class ContentCleaner:
    """Cleans HTML documents to extract high-quality core content for RAG pipelines."""

    REMOVE_TAGS: Final[list[str]] = [
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "video",
        "audio",
        "source",
        "picture",
        "object",
        "embed",
        "form",
        "input",
        "button",
        "select",
        "textarea",
    ]

    BOILERPLATE_CLASSES: Final[list[str]] = [
        "nav",
        "navbar",
        "navigation",
        "menu",
        "header",
        "footer",
        "sidebar",
        "cookie",
        "banner",
        "popup",
        "modal",
        "social",
        "share",
        "comment",
        "widget",
        "advertisement",
        "ad-",
        "advert",
    ]

    BOILERPLATE_IDS: Final[list[str]] = [
        "nav",
        "navbar",
        "navigation",
        "menu",
        "header",
        "footer",
        "sidebar",
        "cookie-banner",
        "cookie-notice",
    ]

    # Tags that are inherently structural boilerplate in standard web layouts
    SEMANTIC_BOILERPLATE_TAGS: Final[list[str]] = ["nav", "footer", "aside"]

    def __init__(self) -> None:
        """Initialize ContentCleaner with compiled regex patterns."""
        # Precompile class and ID regexes for performance
        self._class_patterns: list[re.Pattern[str]] = [
            re.compile(rf"(^|[-_ ]){re.escape(cls_pat)}([-_ ]|$)", re.IGNORECASE)
            if cls_pat != "ad-"
            else re.compile(r"(^|[-_ ])ad[-_ ]", re.IGNORECASE)
            for cls_pat in self.BOILERPLATE_CLASSES
        ]
        self._id_patterns: list[re.Pattern[str]] = [
            re.compile(rf"(^|[-_ ]){re.escape(id_pat)}([-_ ]|$)", re.IGNORECASE)
            for id_pat in self.BOILERPLATE_IDS
        ]

    def clean(self, html: str) -> str:
        """Clean HTML by removing boilerplate and non-content elements.

        Steps:
        1. Parse HTML with lxml and remove comments.
        2. Remove unwanted tags entirely (script, style, etc.).
        3. Remove boilerplate elements by semantic tag, class, and ID patterns.
        4. Remove hidden elements (CSS display:none, hidden attributes).
        5. Extract main content container if identifiable; otherwise fall back to body.
        6. Return cleaned HTML string.

        Args:
            html: Raw HTML input string.

        Returns:
            Cleaned HTML string preserving structural semantic content.
        """
        if not html or not html.strip():
            return ""

        soup = BeautifulSoup(html, "lxml")

        # 1. Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 2. Remove unwanted tags entirely
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 3. Remove boilerplate by class/id patterns and semantic elements
        self._remove_boilerplate(soup)

        # 4. Remove hidden elements
        self._remove_hidden_elements(soup)

        # 5. Extract main content area if identifiable
        main_content = self._find_main_content(soup)
        if main_content is not None:
            return str(main_content)

        # Fallback to body or document root
        if soup.body:
            return str(soup.body)

        return str(soup)

    def _remove_boilerplate(self, soup: BeautifulSoup) -> None:
        """Remove elements matching boilerplate patterns.

        Removes semantic tags (<nav>, <footer>, <aside>) and elements whose
        classes or IDs match known boilerplate keywords, ensuring core containers
        like <main> or <article> are not accidentally removed.

        Args:
            soup: BeautifulSoup instance to modify in place.
        """
        # 1. Remove semantic boilerplate tags
        for tag_name in self.SEMANTIC_BOILERPLATE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove <header> only if it is outside <main> or <article>
        for header_tag in soup.find_all("header"):
            if not header_tag.find_parent(["article", "main"]):
                header_tag.decompose()

        # 2. Remove elements matching class or id boilerplate
        candidates: list[Tag] = []
        for el in soup.find_all(True):
            if not isinstance(el, Tag):
                continue

            # Never remove core layout roots
            if el.name in ("html", "body", "main", "article"):
                continue

            # Never remove an element that encloses the primary article/main
            if el.find(["main", "article"]) is not None:
                continue

            # Check classes
            classes = el.get("class", [])
            if isinstance(classes, str):
                classes = [classes]

            is_boilerplate = False
            for cls in classes:
                if any(pat.search(cls) for pat in self._class_patterns):
                    is_boilerplate = True
                    break

            # Check ID
            if not is_boilerplate:
                el_id = el.get("id")
                if el_id and isinstance(el_id, str):
                    if any(pat.search(el_id) for pat in self._id_patterns):
                        is_boilerplate = True

            if is_boilerplate:
                candidates.append(el)

        for el in candidates:
            # Check element is still in tree before decomposing
            if el.parent is not None:
                el.decompose()

    def _find_main_content(self, soup: BeautifulSoup) -> Tag | None:
        """Try to find the main content container.

        Heuristically searches for <main>, <article>, [role='main'], and
        common content ID/class patterns, ensuring candidate contains substantial text.

        Args:
            soup: BeautifulSoup instance.

        Returns:
            The main Tag if identifiable with sufficient content, or None.
        """
        # Candidate search selectors in priority order
        candidates: list[Tag | None] = [
            soup.find("main"),
            soup.find(attrs={"role": "main"}),
            soup.find("article"),
            soup.find(id=re.compile(r"^(main[-_]?content|content[-_]?main|post[-_]?content|article[-_]?content)$", re.I)),
            soup.find(class_=re.compile(r"^(main[-_]?content|content[-_]?main|post[-_]?content|article[-_]?content|entry[-_]?content)$", re.I)),
            soup.find(id=re.compile(r"^(main|content)$", re.I)),
            soup.find(class_=re.compile(r"^(main|content)$", re.I)),
        ]

        for candidate in candidates:
            if candidate is not None and isinstance(candidate, Tag):
                text_len = len(candidate.get_text(strip=True))
                # Only accept container if it contains substantive readable text
                if text_len >= 50:
                    return candidate

        return None

    def _remove_hidden_elements(self, soup: BeautifulSoup) -> None:
        """Remove elements with display:none, visibility:hidden, or hidden attributes.

        Args:
            soup: BeautifulSoup instance to modify in place.
        """
        hidden_style_re = re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)

        elements_to_remove: list[Tag] = []

        for el in soup.find_all(True):
            if not isinstance(el, Tag):
                continue

            # Hidden HTML attribute (e.g. <div hidden>)
            if el.has_attr("hidden"):
                elements_to_remove.append(el)
                continue

            # ARIA hidden attribute
            if el.get("aria-hidden") == "true":
                elements_to_remove.append(el)
                continue

            # Inline style attribute
            style = el.get("style", "")
            if isinstance(style, str) and hidden_style_re.search(style):
                elements_to_remove.append(el)

        for el in elements_to_remove:
            if el.parent is not None:
                el.decompose()

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace by collapsing multiple spaces, tabs, and excess newlines.

        Args:
            text: Input string.

        Returns:
            Clean string with normalized whitespace.
        """
        if not text:
            return ""

        # Normalize line breaks
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse horizontal whitespace
        cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)

        # Remove spaces at line boundaries
        cleaned = re.sub(r" *\n *", "\n", cleaned)

        # Collapse 3 or more newlines to a double newline (paragraph break)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()
