"""Engine B: Beautiful Soup Extraction Engine.

Direct DOM control, custom rule extraction (Product, FAQ, Tables),
JSON-LD / OpenGraph metadata extraction, and selective structural cleaning.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from app.schemas.documents import Heading, LinkInfo
from .base import EngineResult

logger = logging.getLogger(__name__)

# Elements that are always stripped
_DROP_TAGS: set[str] = {
    "script", "style", "noscript", "template", "svg", "canvas",
    "iframe", "object", "embed", "applet",
}

# Selectors of purely non-content banners / popups / tracking widgets
_NOISE_SELECTORS: tuple[str, ...] = (
    ".cookie-banner", ".cookie-consent", "#cookie-notice",
    ".ad-banner", ".advertisement", ".google-ad",
    ".modal-overlay", ".popup-overlay",
)


class BeautifulSoupEngine:
    """Engine B implementation using BeautifulSoup."""

    ENGINE_NAME: str = "beautifulsoup"

    def __init__(self, parser: str = "lxml") -> None:
        self.parser = parser

    def extract(self, html: str, url: str) -> EngineResult:
        """Extract content, tables, and structured metadata using BeautifulSoup.

        Args:
            html: Raw HTML content.
            url: Source URL of the webpage.

        Returns:
            Standardized EngineResult.
        """
        start_time = time.perf_counter()
        result = EngineResult(engine=self.ENGINE_NAME, success=False)

        if not html or not html.strip():
            result.error = "HTML content is empty"
            result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return result

        try:
            soup = self._parse_soup(html)

            # 1. Extract metadata before destructive cleanup
            title, description, canonical, og_meta, json_ld_data = self._extract_metadata(soup, url)
            result.title = title
            result.description = description
            result.canonical_url = canonical

            metadata: dict[str, Any] = {
                "open_graph": og_meta,
                "json_ld": json_ld_data,
            }

            # 2. Extract Headings hierarchy
            headings = self._extract_headings(soup)
            result.headings = headings

            # 3. Extract Links
            links = self._extract_links(soup, url)
            result.links = links

            # 4. Extract structured domain rules (Tables, Product, FAQ)
            faq_items = self._extract_faq(soup, json_ld_data)
            if faq_items:
                metadata["faq_items"] = faq_items

            product_info = self._extract_product(soup, json_ld_data)
            if product_info:
                metadata["product_info"] = product_info

            tables_md = self._extract_tables(soup)
            if tables_md:
                metadata["tables_count"] = len(tables_md)

            # 5. Selective DOM cleaning (does NOT blindly strip documentation nav/asides)
            cleaned_soup = self._clean_dom(soup)

            # 6. Extract Main Text & Markdown
            plain_text = cleaned_soup.get_text(separator="\n", strip=True)
            # Remove excessive whitespace
            clean_text = re.sub(r"[ \t]+", " ", plain_text)
            clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

            markdown_text = self._soup_to_markdown(cleaned_soup, url)
            clean_markdown = re.sub(r"[ \t]+", " ", markdown_text)
            clean_markdown = re.sub(r"\n{3,}", "\n\n", clean_markdown).strip()

            result.text = clean_text or None
            result.markdown = clean_markdown or None
            result.metadata = metadata

            result.compute_counts()
            result.success = result.has_content
            if not result.success:
                result.warnings.append("BeautifulSoup returned empty textual content after cleaning")

        except Exception as exc:
            logger.warning("BeautifulSoup extraction failed for %s: %s", url, exc)
            result.error = f"BeautifulSoup error: {str(exc)}"
            result.success = False

        result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return result

    def _parse_soup(self, html: str) -> BeautifulSoup:
        try:
            return BeautifulSoup(html, self.parser)
        except Exception:
            return BeautifulSoup(html, "html.parser")

    def _extract_metadata(
        self, soup: BeautifulSoup, base_url: str
    ) -> tuple[Optional[str], Optional[str], Optional[str], dict[str, str], list[dict[str, Any]]]:
        """Extract title, meta description, canonical, OpenGraph tags, and JSON-LD."""
        # Title
        title: Optional[str] = None
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()

        # Meta description
        description: Optional[str] = None
        desc_meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        if desc_meta and desc_meta.get("content"):
            description = str(desc_meta["content"]).strip()

        # Canonical
        canonical: Optional[str] = None
        canon_tag = soup.find("link", attrs={"rel": re.compile(r"^canonical$", re.I)})
        if canon_tag and canon_tag.get("href"):
            canonical = urljoin(base_url, str(canon_tag["href"]).strip())

        # OpenGraph
        og: dict[str, str] = {}
        for m in soup.find_all("meta", property=re.compile(r"^(og|twitter):")):
            prop = m.get("property") or m.get("name")
            content = m.get("content")
            if prop and content:
                og[str(prop).strip()] = str(content).strip()

        if not title and "og:title" in og:
            title = og["og:title"]
        if not description and "og:description" in og:
            description = og["og:description"]

        # JSON-LD
        json_ld_list: list[dict[str, Any]] = []
        for script in soup.find_all("script", type="application/ld+json"):
            raw_text = script.string or script.get_text()
            if not raw_text or not raw_text.strip():
                continue
            try:
                data = json.loads(raw_text.strip())
                if isinstance(data, dict):
                    json_ld_list.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            json_ld_list.append(item)
            except Exception:
                continue

        return title, description, canonical, og, json_ld_list

    def _extract_headings(self, soup: BeautifulSoup) -> list[Heading]:
        """Extract hierarchical heading structure (H1-H6)."""
        headings: list[Heading] = []
        for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            text = el.get_text(strip=True)
            if not text:
                continue
            try:
                level = int(el.name[1])
                headings.append(Heading(level=level, text=text))
            except (ValueError, IndexError):
                continue
        return headings

    def _extract_links(self, soup: BeautifulSoup, base_url: str, base_domain: Optional[str] = None) -> list[LinkInfo]:
        """Extract and normalize links with anchor texts."""
        links: list[LinkInfo] = []
        seen: set[str] = set()
        base_host = (base_domain or urlparse(base_url).netloc).lower()

        for a in soup.find_all("a", href=True):
            href = str(a["href"]).strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            if parsed.scheme not in ("http", "https"):
                continue
            if full_url in seen:
                continue
            seen.add(full_url)

            text = a.get_text(strip=True)
            is_internal = (parsed.netloc.lower() == base_host) or (parsed.netloc.lower().endswith("." + base_host))
            links.append(LinkInfo(url=full_url, text=text, is_internal=is_internal))
        return links

    def _extract_faq(self, soup: BeautifulSoup, json_ld: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Extract FAQ questions and answers from JSON-LD or microdata."""
        faqs: list[dict[str, str]] = []

        # From JSON-LD
        for block in json_ld:
            b_type = str(block.get("@type", ""))
            if "FAQPage" in b_type or "QAPage" in b_type:
                for entity in block.get("mainEntity", []):
                    if isinstance(entity, dict):
                        q = entity.get("name") or entity.get("text", "")
                        accepted = entity.get("acceptedAnswer", {})
                        a = accepted.get("text", "") if isinstance(accepted, dict) else str(accepted)
                        if q and a:
                            # Strip any HTML tags inside answer
                            a_clean = re.sub(r"<[^>]+>", "", a).strip()
                            faqs.append({"question": str(q).strip(), "answer": a_clean})

        # From DOM fallback if JSON-LD had none
        if not faqs:
            for item in soup.find_all(class_=re.compile(r"faq[-_]item|accordion[-_]item", re.I)):
                q_el = item.find(class_=re.compile(r"question|title|header", re.I))
                a_el = item.find(class_=re.compile(r"answer|body|content", re.I))
                if q_el and a_el:
                    q_text = q_el.get_text(strip=True)
                    a_text = a_el.get_text(strip=True)
                    if q_text and a_text:
                        faqs.append({"question": q_text, "answer": a_text})

        return faqs

    def _extract_product(self, soup: BeautifulSoup, json_ld: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Extract product structured information (price, currency, availability, sku)."""
        for block in json_ld:
            if str(block.get("@type", "")).lower() == "product":
                prod: dict[str, Any] = {
                    "name": block.get("name"),
                    "description": block.get("description"),
                    "sku": block.get("sku"),
                    "brand": block.get("brand", {}).get("name") if isinstance(block.get("brand"), dict) else block.get("brand"),
                }
                offers = block.get("offers")
                if isinstance(offers, dict):
                    prod["price"] = offers.get("price")
                    prod["price_currency"] = offers.get("priceCurrency")
                    prod["availability"] = offers.get("availability")
                elif isinstance(offers, list) and offers and isinstance(offers[0], dict):
                    prod["price"] = offers[0].get("price")
                    prod["price_currency"] = offers[0].get("priceCurrency")
                return prod
        return None

    def _extract_tables(self, soup: BeautifulSoup) -> list[str]:
        """Convert HTML tables to clean Markdown table strings."""
        tables_md: list[str] = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue

            md_lines: list[str] = []
            header_processed = False

            for r_idx, row in enumerate(rows):
                cols = row.find_all(["th", "td"])
                if not cols:
                    continue
                cell_texts = [re.sub(r"\s+", " ", c.get_text(strip=True)).replace("|", "\\|") for c in cols]
                row_str = "| " + " | ".join(cell_texts) + " |"
                md_lines.append(row_str)

                # Insert divider after the header row (either <th> elements or first row)
                if not header_processed and (any(c.name == "th" for c in cols) or r_idx == 0):
                    div_str = "| " + " | ".join(["---"] * len(cols)) + " |"
                    md_lines.append(div_str)
                    header_processed = True

            if md_lines:
                tables_md.append("\n".join(md_lines))
        return tables_md

    def _clean_dom(self, original_soup: BeautifulSoup) -> BeautifulSoup:
        """Selectively clean unwanted elements while preserving documentation nav/asides."""
        # Work on a copy
        soup = BeautifulSoup(str(original_soup), "html.parser")

        # 1. Drop scripts, styles, noscript, etc.
        for tag_name in _DROP_TAGS:
            for el in soup.find_all(tag_name):
                el.decompose()

        # 2. Drop noise banners
        for sel in _NOISE_SELECTORS:
            try:
                for el in soup.select(sel):
                    el.decompose()
            except Exception:
                continue

        # 3. Handle <nav> and <aside> thoughtfully:
        # Check if the nav/aside contains documentation, sidebar, or table of contents
        for el in soup.find_all(["nav", "aside"]):
            classes = " ".join(el.get("class", []))
            id_val = el.get("id", "")
            combo = (classes + " " + id_val).lower()
            # If it's a documentation sidebar or table of contents, PRESERVE IT!
            if any(kw in combo for kw in ("doc", "sidebar", "toc", "menu-content", "content-nav", "article")):
                continue
            # If it's generic header navigation or footer navigation, unwrap or strip
            if el.name == "nav" and any(kw in combo for kw in ("header", "main-nav", "top-nav", "navbar", "footer")):
                el.decompose()

        # 4. Remove comments
        for comment in soup.find_all(string=lambda t: isinstance(t, NavigableString) and t.startswith("<!--")):
            comment.extract()

        return soup

    def _soup_to_markdown(self, soup: BeautifulSoup, base_url: str) -> str:
        """Convert a cleaned BeautifulSoup DOM tree into clean Markdown."""
        lines: list[str] = []

        # Find main content container if exists
        main_root = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"content|main", re.I))
            or soup.find(class_=re.compile(r"content|main|article-body|post-content", re.I))
            or soup.body
            or soup
        )

    def _soup_to_markdown(self, soup: BeautifulSoup, base_url: str = "") -> str:
        """Convert BeautifulSoup DOM to markdown preserving semantic structure and inline elements."""
        def _render_table(table: Tag) -> str:
            rows = table.find_all("tr")
            if not rows:
                return ""
            table_data: list[list[str]] = []
            for tr in rows:
                cells = tr.find_all(["th", "td"])
                row_cells = [
                    re.sub(r"\s+", " ", cell.get_text(strip=True)).replace("|", "\\|")
                    for cell in cells
                ]
                if row_cells:
                    table_data.append(row_cells)
            if not table_data:
                return ""

            num_cols = max(len(r) for r in table_data)
            padded = [r + [""] * (num_cols - len(r)) for r in table_data]

            lines: list[str] = []
            header = padded[0]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * num_cols) + " |")
            for r in padded[1:]:
                lines.append("| " + " | ".join(r) + " |")
            return "\n\n" + "\n".join(lines) + "\n\n"

        def _render_node(node: Any) -> str:
            if isinstance(node, NavigableString):
                return str(node)
            if not isinstance(node, Tag):
                return ""

            name = node.name.lower() if node.name else ""

            if name in ("strong", "b"):
                inner = "".join(_render_node(c) for c in node.children).strip()
                return f"**{inner}**" if inner else ""
            if name in ("em", "i"):
                inner = "".join(_render_node(c) for c in node.children).strip()
                return f"*{inner}*" if inner else ""
            if name == "code":
                if node.parent and node.parent.name == "pre":
                    return node.get_text()
                inner = node.get_text().strip()
                return f"`{inner}`" if inner else ""
            if name == "a":
                href = node.get("href", "").strip()
                inner = "".join(_render_node(c) for c in node.children).strip()
                if href and inner:
                    return f"[{inner}]({href})"
                return inner or href
            if name == "br":
                return "\n"
            if name == "hr":
                return "\n\n---\n\n"

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                inner = "".join(_render_node(c) for c in node.children).strip()
                return f"\n\n{'#' * level} {inner}\n\n"
            if name == "p":
                inner = "".join(_render_node(c) for c in node.children).strip()
                return f"\n\n{inner}\n\n" if inner else ""
            if name == "pre":
                code_text = node.get_text().strip()
                return f"\n\n```\n{code_text}\n```\n\n"
            if name == "blockquote":
                inner = "".join(_render_node(c) for c in node.children).strip()
                quoted = "\n".join(
                    f"> {line}" if line else ">" for line in inner.split("\n")
                )
                return f"\n\n{quoted}\n\n"
            if name == "ul":
                items: list[str] = []
                for li in node.find_all("li", recursive=False):
                    li_text = "".join(_render_node(c) for c in li.children).strip()
                    if li_text:
                        items.append(f"- {li_text}")
                return "\n\n" + "\n".join(items) + "\n\n"
            if name == "ol":
                items = []
                for idx, li in enumerate(node.find_all("li", recursive=False), 1):
                    li_text = "".join(_render_node(c) for c in li.children).strip()
                    if li_text:
                        items.append(f"{idx}. {li_text}")
                return "\n\n" + "\n".join(items) + "\n\n"
            if name == "table":
                return _render_table(node)

            return "".join(_render_node(c) for c in node.children)

        root = soup.body if soup.body else soup
        raw_markdown = "".join(_render_node(c) for c in root.children)
        return raw_markdown.strip()

    _html_to_markdown = _soup_to_markdown
