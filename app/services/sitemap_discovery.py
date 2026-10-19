"""Sitemap parser service for Crawlbot.

This module provides the SitemapParser class to parse XML sitemaps and sitemap indexes,
handle gzipped sitemaps, recover from malformed XML, and respect URL limits.
"""

from __future__ import annotations

import gzip
import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from lxml import etree

logger = logging.getLogger(__name__)


class SitemapParser:
    """Service to discover and parse XML sitemaps, supporting nested sitemap indexes and gzip."""

    SITEMAP_NS: dict[str, str] = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    def __init__(self, client: httpx.AsyncClient, max_urls: int = 5000) -> None:
        """Initialize the SitemapParser.

        Args:
            client: The httpx.AsyncClient to use for downloading sitemaps.
            max_urls: Maximum number of page URLs to discover and store.
        """
        self.client: httpx.AsyncClient = client
        self.max_urls: int = max_urls
        self._discovered_urls: list[str] = []
        self._discovered_set: set[str] = set()
        self._visited_sitemaps: set[str] = set()

    async def discover_from_robots(self, robots_sitemaps: list[str]) -> list[str]:
        """Parse sitemaps discovered from robots.txt.

        Args:
            robots_sitemaps: List of sitemap URLs extracted from robots.txt.

        Returns:
            List of unique page URLs discovered from the sitemaps.
        """
        results: list[str] = []
        for sitemap_url in robots_sitemaps:
            if len(self._discovered_urls) >= self.max_urls:
                break
            try:
                urls = await self.parse_sitemap(sitemap_url)
                results.extend(urls)
            except Exception as exc:
                logger.warning("Error processing robots sitemap %s: %s", sitemap_url, exc)

        return results

    async def parse_sitemap(self, sitemap_url: str) -> list[str]:
        """Parse a sitemap URL. Handles both sitemap index and urlset recursively.

        Args:
            sitemap_url: The URL of the sitemap or sitemap index.

        Returns:
            List of newly discovered page URLs from this sitemap and its children.
        """
        clean_url = sitemap_url.strip() if isinstance(sitemap_url, str) else ""
        if not clean_url:
            return []

        if clean_url in self._visited_sitemaps:
            return []
        self._visited_sitemaps.add(clean_url)

        if len(self._discovered_urls) >= self.max_urls:
            return []

        try:
            response = await self.client.get(clean_url, follow_redirects=True, timeout=15.0)
            if response.status_code != 200:
                logger.debug("Sitemap %s returned status %d", clean_url, response.status_code)
                return []
            content = response.content
        except Exception as exc:
            logger.warning("Failed to fetch sitemap %s: %s", clean_url, exc)
            return []

        if not content:
            return []

        page_urls, sub_sitemaps = self._parse_sitemap_xml(content)
        collected_page_urls: list[str] = []

        # Collect page URLs
        for url in page_urls:
            if len(self._discovered_urls) >= self.max_urls:
                break
            if url not in self._discovered_set:
                self._discovered_set.add(url)
                self._discovered_urls.append(url)
                collected_page_urls.append(url)

        # Recurse into nested sitemaps (sitemap index)
        for sub_url in sub_sitemaps:
            if len(self._discovered_urls) >= self.max_urls:
                break
            try:
                nested_urls = await self.parse_sitemap(sub_url)
                collected_page_urls.extend(nested_urls)
            except Exception as exc:
                logger.warning("Failed to parse nested sitemap %s: %s", sub_url, exc)

        return collected_page_urls

    async def try_default_sitemaps(self, base_url: str) -> list[str]:
        """Try common sitemap locations: /sitemap.xml, /sitemap_index.xml, /wp-sitemap.xml.

        Args:
            base_url: The base website URL or domain.

        Returns:
            List of newly discovered page URLs from default sitemap endpoints.
        """
        if not base_url or not isinstance(base_url, str):
            return []

        clean_base = base_url.strip().rstrip("/")
        if "://" not in clean_base:
            clean_base = f"https://{clean_base}"

        candidates = [
            f"{clean_base}/sitemap.xml",
            f"{clean_base}/sitemap_index.xml",
            f"{clean_base}/wp-sitemap.xml",
        ]

        discovered: list[str] = []
        for candidate_url in candidates:
            if len(self._discovered_urls) >= self.max_urls:
                break
            if candidate_url in self._visited_sitemaps:
                continue

            try:
                urls = await self.parse_sitemap(candidate_url)
                if urls:
                    discovered.extend(urls)
            except Exception as exc:
                logger.debug("Candidate sitemap %s failed: %s", candidate_url, exc)

        return discovered

    def _parse_sitemap_xml(self, content: bytes) -> tuple[list[str], list[str]]:
        """Parse XML content. Returns (page_urls, sub_sitemap_urls).

        Handles gzipped content, strips namespaces if necessary, and gracefully
        recovers from malformed XML.

        Args:
            content: Raw byte contents of the sitemap.

        Returns:
            A tuple containing (list of page URLs, list of sub-sitemap URLs).
        """
        if not content:
            return ([], [])

        if isinstance(content, str):
            content = content.encode("utf-8")

        # Handle gzip compressed sitemaps
        if content[:2] == b"\x1f\x8b":
            try:
                content = gzip.decompress(content)
            except Exception as exc:
                logger.warning("Failed to decompress gzipped sitemap: %s", exc)

        # Strip UTF-8 BOM if present
        if content.startswith(b"\xef\xbb\xbf"):
            content = content[3:]

        # Use recovery parser to handle malformed XML gracefully
        parser = etree.XMLParser(
            recover=True,
            resolve_entities=False,
            no_network=True,
            remove_comments=True,
        )

        try:
            root = etree.fromstring(content, parser=parser)
        except Exception as exc:
            logger.warning("XML parsing failed: %s", exc)
            return ([], [])

        if root is None:
            return ([], [])

        # Try standard sitemap namespace first
        try:
            page_elements = root.xpath("//sm:url/sm:loc/text()", namespaces=self.SITEMAP_NS)
            sitemap_elements = root.xpath(
                "//sm:sitemap/sm:loc/text()", namespaces=self.SITEMAP_NS
            )
        except Exception:
            page_elements = []
            sitemap_elements = []

        # Fallback to local-name() for non-standard or missing namespaces
        if not page_elements and not sitemap_elements:
            try:
                page_elements = root.xpath(
                    "//*[local-name()='url']/*[local-name()='loc']/text()"
                )
                sitemap_elements = root.xpath(
                    "//*[local-name()='sitemap']/*[local-name()='loc']/text()"
                )
            except Exception:
                page_elements = []
                sitemap_elements = []

        page_urls: list[str] = [
            str(elem).strip()
            for elem in page_elements
            if elem and str(elem).strip()
        ]
        sub_sitemaps: list[str] = [
            str(elem).strip()
            for elem in sitemap_elements
            if elem and str(elem).strip()
        ]

        return (page_urls, sub_sitemaps)

    @property
    def discovered_urls(self) -> list[str]:
        """Return a copy of all discovered page URLs."""
        return list(self._discovered_urls)
