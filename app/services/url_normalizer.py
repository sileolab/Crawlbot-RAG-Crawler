"""URL Normalization and Deduplication Service for Crawlbot.

This module provides the URLNormalizer class to clean, standardize, and deduplicate
URLs encountered during crawling, ensuring canonical URL representation and domain scoping.
"""

from __future__ import annotations

import posixpath
import re
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse

INVALID_SCHEME_PREFIXES = (
    "mailto:",
    "tel:",
    "javascript:",
    "data:",
    "sms:",
    "callto:",
    "viber:",
    "whatsapp:",
    "skype:",
    "file:",
    "ftp:",
    "blob:",
    "android-app:",
    "ios-app:",
)


class URLNormalizer:
    """Service for normalizing URLs and detecting duplicates across crawl runs."""

    def __init__(
        self,
        keep_query_params: bool = False,
        allowed_domains: list[str] | None = None,
        allow_subdomains: bool = False,
    ) -> None:
        """Initialize the URLNormalizer.

        Args:
            keep_query_params: Whether to retain and sort query parameters or strip them.
            allowed_domains: List of domain names that are permissible for crawling.
            allow_subdomains: Whether subdomains of allowed/base domains are considered in-scope.
        """
        self.keep_query_params: bool = keep_query_params
        self.allowed_domains: list[str] = [
            self.extract_domain(d) for d in (allowed_domains or []) if self.extract_domain(d)
        ]
        self.allow_subdomains: bool = allow_subdomains
        self._seen_urls: set[str] = set()

    def normalize(
        self,
        url: str | None,
        base_url: str | None = None,
        preserve_trailing_slash: bool = False,
    ) -> str | None:
        """Normalize a URL. Returns None if the URL is invalid or non-crawlable.

        Normalization steps:
        1. Resolve relative URLs against base_url
        2. Parse URL and validate scheme/host
        3. Lowercase scheme and hostname
        4. Remove default ports (80 for http, 443 for https)
        5. Remove fragment (#...)
        6. Remove trailing slash for non-root paths consistently (unless preserve_trailing_slash is True)
        7. Remove or sort query parameters based on config
        8. Normalize path (collapse multiple slashes, resolve . and ..)
        9. Return cleaned URL

        Args:
            url: The raw URL or relative link to normalize.
            base_url: Optional base URL used to resolve relative links.
            preserve_trailing_slash: Whether to preserve a trailing slash present on the input URL.

        Returns:
            The normalized URL string, or None if the URL is invalid.
        """
        if url is None or not isinstance(url, str):
            return None

        clean_url = url.strip()
        if not clean_url:
            return None

        # Check for non-HTTP schemes
        lower_raw = clean_url.lower()
        if any(lower_raw.startswith(prefix) for prefix in INVALID_SCHEME_PREFIXES):
            return None

        # 1. Resolve relative URLs against base_url
        if base_url:
            base_clean = base_url.strip()
            # If base has no scheme, prepend https://
            if "://" not in base_clean:
                base_clean = f"https://{base_clean}"
            clean_url = urljoin(base_clean, clean_url)
        elif clean_url.startswith(("#", "?", "/")):
            # Relative without base_url cannot be resolved
            return None

        # 2. Parse URL
        try:
            parsed = urlparse(clean_url)
        except Exception:
            return None

        # 3. Lowercase scheme and hostname
        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            return None

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return None

        # 4. Remove default ports (80 for http, 443 for https)
        port = parsed.port
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{hostname}:{port}"
        else:
            netloc = hostname

        # 5. Remove fragment (#...) -> always ignored in canonical crawler URLs
        # (handeled by omitting parsed.fragment in urlunparse)

        # 8. Normalize path (remove //, resolve /../)
        raw_path = parsed.path or "/"
        # Collapse multiple slashes
        collapsed_path = re.sub(r"/+", "/", raw_path)
        # Resolve '.' and '..' segments
        normalized_path = posixpath.normpath(collapsed_path)

        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"

        # 6. Remove trailing slash for non-root paths unless explicitly preserved
        had_trailing_slash = raw_path.endswith("/") and raw_path != "/"
        if not (preserve_trailing_slash and had_trailing_slash):
            if normalized_path != "/" and normalized_path.endswith("/"):
                normalized_path = normalized_path.rstrip("/")
        else:
            if not normalized_path.endswith("/"):
                normalized_path = f"{normalized_path}/"

        # 7. Remove or sort query parameters based on config
        query = ""
        if self.keep_query_params and parsed.query:
            try:
                params = parse_qsl(parsed.query, keep_blank_values=True)
                sorted_params = sorted(params, key=lambda kv: (kv[0], kv[1]))
                query = urlencode(sorted_params, doseq=True)
            except Exception:
                query = ""

        # 9. Return cleaned URL
        return urlunparse((scheme, netloc, normalized_path, "", query, ""))

    def extract_domain(self, url: str | None) -> str:
        """Extract the lowercase domain/hostname from a URL or raw domain string.

        Args:
            url: The URL or domain string.

        Returns:
            The extracted hostname in lowercase, or empty string if invalid.
        """
        if not url or not isinstance(url, str):
            return ""
        clean_url = url.strip().lower()
        if not clean_url:
            return ""
        if any(clean_url.startswith(prefix) for prefix in INVALID_SCHEME_PREFIXES):
            return ""
        if "://" not in clean_url:
            clean_url = f"http://{clean_url}"
        try:
            parsed = urlparse(clean_url)
            return (parsed.hostname or "").lower()
        except Exception:
            return ""

    def is_same_domain(self, url: str, reference_domain: str | None = None) -> bool:
        """Check if URL belongs to the allowed domain scope.

        Checks against reference_domain if provided, and falls back to
        the configured allowed_domains list.

        Args:
            url: The URL to evaluate.
            reference_domain: Optional reference domain to compare against.

        Returns:
            True if the URL matches the target domain scope, False otherwise.
        """
        url_domain = self.extract_domain(url)
        if not url_domain:
            return False

        targets: list[str] = []
        if reference_domain:
            ref_clean = self.extract_domain(reference_domain)
            if ref_clean:
                targets.append(ref_clean)

        for d in self.allowed_domains:
            if d and d not in targets:
                targets.append(d)

        if not targets:
            return False

        for target in targets:
            if url_domain == target:
                return True
            if self.allow_subdomains and url_domain.endswith(f".{target}"):
                return True

        return False

    def is_internal(self, url: str, base_domain: str) -> bool:
        """Check if URL is internal to the base domain.

        Args:
            url: The URL to check.
            base_domain: The base domain to compare against.

        Returns:
            True if the URL belongs to base_domain (or subdomains if allowed), False otherwise.
        """
        url_domain = self.extract_domain(url)
        target_base = self.extract_domain(base_domain)
        if not url_domain or not target_base:
            return False

        if url_domain == target_base:
            return True

        if self.allow_subdomains and url_domain.endswith(f".{target_base}"):
            return True

        return False

    def is_duplicate(self, url: str) -> bool:
        """Check if this normalized URL has been seen before.

        Args:
            url: The URL to check.

        Returns:
            True if the URL was previously marked as seen, False otherwise.
        """
        if not url:
            return False
        normalized = self.normalize(url)
        if not normalized:
            return False
        return normalized in self._seen_urls

    def add_seen(self, url: str) -> None:
        """Mark a URL as seen.

        Args:
            url: The URL to mark as seen.
        """
        if not url:
            return
        normalized = self.normalize(url)
        if normalized:
            self._seen_urls.add(normalized)

    def reset(self) -> None:
        """Clear seen URLs."""
        self._seen_urls.clear()
