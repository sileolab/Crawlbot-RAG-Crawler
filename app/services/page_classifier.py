"""Page Type Classification Service for Crawlbot.

Classifies web pages into standardized categories based on:
1. URL path patterns and file extensions
2. Metadata & schema.org JSON-LD tags
3. DOM structural indicators (codeblocks, price blocks, accordion FAQs, SPA shells)
4. Script-to-text density ratio
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Valid Page Types
PAGE_TYPE_ARTICLE = "article"
PAGE_TYPE_DOCS = "docs"
PAGE_TYPE_POLICY = "policy"
PAGE_TYPE_PRODUCT = "product"
PAGE_TYPE_FAQ = "faq"
PAGE_TYPE_ABOUT = "about"
PAGE_TYPE_CONTACT = "contact"
PAGE_TYPE_SPA = "spa"
PAGE_TYPE_DASHBOARD = "dashboard"
PAGE_TYPE_FILE = "file"
PAGE_TYPE_UNKNOWN = "unknown"

VALID_PAGE_TYPES = {
    PAGE_TYPE_ARTICLE,
    PAGE_TYPE_DOCS,
    PAGE_TYPE_POLICY,
    PAGE_TYPE_PRODUCT,
    PAGE_TYPE_FAQ,
    PAGE_TYPE_ABOUT,
    PAGE_TYPE_CONTACT,
    PAGE_TYPE_SPA,
    PAGE_TYPE_DASHBOARD,
    PAGE_TYPE_FILE,
    PAGE_TYPE_UNKNOWN,
}

# File extensions for direct file classification
FILE_EXTENSIONS = (
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".csv", ".xlsx", ".xls",
    ".docx", ".doc", ".pptx", ".ppt", ".mp3", ".mp4", ".epub",
)

# URL path regex patterns
URL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"/(?:docs|documentation|guide|guides|manual|api-reference|reference|tutorial|sdk|wiki)(?:/|$)", re.I), PAGE_TYPE_DOCS),
    (re.compile(r"/(?:blog|post|posts|article|articles|news|journal|press-release|story)(?:/|$|\.html?)", re.I), PAGE_TYPE_ARTICLE),
    (re.compile(r"/(?:faq|faqs|frequently-asked-questions|qna|help-center|knowledgebase)(?:/|$)", re.I), PAGE_TYPE_FAQ),
    (re.compile(r"/(?:terms|privacy|policy|legal|disclaimer|tos|user-agreement|cookie-policy)(?:/|$|\.html?)", re.I), PAGE_TYPE_POLICY),
    (re.compile(r"/(?:product|products|item|items|shop|store|goods|detail|p)/", re.I), PAGE_TYPE_PRODUCT),
    (re.compile(r"/(?:about|about-us|team|who-we-are|company|our-story)(?:/|$|\.html?)", re.I), PAGE_TYPE_ABOUT),
    (re.compile(r"/(?:contact|contact-us|support|get-in-touch|reach-us)(?:/|$|\.html?)", re.I), PAGE_TYPE_CONTACT),
    (re.compile(r"/(?:dashboard|console|portal|admin|signin|signup|login|logout|app)(?:/|$)", re.I), PAGE_TYPE_DASHBOARD),
]


class PageClassifier:
    """Classifies web pages to select optimal extraction strategies and chunking."""

    @classmethod
    def classify(
        cls,
        url: str,
        html: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Classify page into one of the standardized page types.

        Args:
            url: The URL of the page.
            html: Optional raw HTML string.
            metadata: Optional extracted metadata dictionary (e.g. from JSON-LD).

        Returns:
            One of the valid PAGE_TYPE_* strings.
        """
        parsed_url = urlsplit(url)
        path = parsed_url.path.lower()

        # 1. Check for downloadable file extensions
        for ext in FILE_EXTENSIONS:
            if path.endswith(ext):
                return PAGE_TYPE_FILE

        # 2. Check metadata / schema.org if available
        if metadata:
            meta_type = cls._classify_from_metadata(metadata)
            if meta_type:
                return meta_type

        # 3. If HTML is provided, inspect JSON-LD and DOM structures
        if html and html.strip():
            dom_type = cls._classify_from_html(html, path)
            if dom_type:
                return dom_type

        # 4. Fallback to URL path heuristics
        for pattern, page_type in URL_PATTERNS:
            if pattern.search(path):
                return page_type

        return PAGE_TYPE_UNKNOWN

    @classmethod
    def _classify_from_metadata(cls, metadata: dict[str, Any]) -> Optional[str]:
        """Inspect schema.org types from metadata dictionary."""
        schema_type = str(metadata.get("@type", "")).lower()
        og_type = str(metadata.get("og:type", "")).lower()

        if any(t in schema_type for t in ["product", "individualproduct"]):
            return PAGE_TYPE_PRODUCT
        if any(t in schema_type for t in ["faqpage", "qapage"]):
            return PAGE_TYPE_FAQ
        if any(t in schema_type for t in ["article", "blogposting", "newsarticle", "techarticle"]):
            return PAGE_TYPE_ARTICLE
        if "aboutpage" in schema_type:
            return PAGE_TYPE_ABOUT
        if "contactpage" in schema_type:
            return PAGE_TYPE_CONTACT
        if og_type == "article":
            return PAGE_TYPE_ARTICLE
        if og_type == "product":
            return PAGE_TYPE_PRODUCT

        return None

    @classmethod
    def _classify_from_html(cls, html: str, path: str) -> Optional[str]:
        """Inspect HTML DOM for signals like SPA root, codeblocks, JSON-LD, or prices."""
        try:
            soup = BeautifulSoup(html[:150000], "html.parser")
        except Exception:
            return None

        # A. Check JSON-LD embedded scripts
        for script in soup.find_all("script", type=lambda t: t and "ld+json" in t.lower()):
            try:
                data = json.loads(script.string or "")
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict):
                        stype = str(item.get("@type", "")).lower()
                        if "faqpage" in stype or "qapage" in stype:
                            return PAGE_TYPE_FAQ
                        if "product" in stype:
                            return PAGE_TYPE_PRODUCT
                        if any(t in stype for t in ["article", "blogposting", "newsarticle", "techarticle"]):
                            return PAGE_TYPE_ARTICLE
            except Exception:
                continue

        # B. Check for SPA characteristics (JS-dependent empty shell)
        scripts = soup.find_all("script")
        script_count = len(scripts)
        body = soup.body
        body_text = body.get_text(separator=" ", strip=True) if body else ""
        text_len = len(body_text)

        # Check for typical SPA mounting points with very sparse initial text
        spa_mounts = soup.find(id=lambda i: i in ("root", "app", "__next", "__nuxt", "svelte-app"))
        has_noscript = bool(soup.find("noscript", string=re.compile(r"enable javascript", re.I)))

        if (spa_mounts or has_noscript) and text_len < 300 and script_count >= 2:
            return PAGE_TYPE_SPA

        # C. Check for Documentation signals (high codeblock count, sidebar navigation)
        code_blocks = soup.find_all(["pre", "code"])
        if len(code_blocks) >= 3 and any(cls_pattern.search(path) for cls_pattern, pt in URL_PATTERNS if pt == PAGE_TYPE_DOCS):
            return PAGE_TYPE_DOCS

        # D. Check for FAQ signals (multiple question/answer accordion pairs)
        faqs = soup.find_all(["details", "accordion"])
        faq_classes = soup.find_all(class_=re.compile(r"faq|question|accordion", re.I))
        if len(faqs) >= 2 or len(faq_classes) >= 3:
            return PAGE_TYPE_FAQ

        # E. Check for Product signals (price, add-to-cart, currency)
        price_tags = soup.find_all(class_=re.compile(r"price|cart|add-to-cart", re.I))
        currency_found = re.search(r"(?:\$|€|£|تومان|ریال)\s*\d+", body_text)
        if (price_tags or currency_found) and any(p.search(path) for p, pt in URL_PATTERNS if pt == PAGE_TYPE_PRODUCT):
            return PAGE_TYPE_PRODUCT

        # F. URL pattern match
        for pattern, page_type in URL_PATTERNS:
            if pattern.search(path):
                return page_type

        # G. Article signals: substantial body text with multiple paragraphs and h1
        h1s = soup.find_all("h1")
        paragraphs = soup.find_all("p")
        if len(paragraphs) >= 4 and len(h1s) >= 1 and text_len > 600:
            return PAGE_TYPE_ARTICLE

        return None
