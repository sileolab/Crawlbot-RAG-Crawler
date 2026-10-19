"""Robots.txt parser service for Crawlbot.

This module provides the RobotsChecker class to fetch, parse, and verify URL permissions,
crawl delays, and sitemap locations defined in site robots.txt files.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)


class RobotsChecker:
    """Service to fetch, parse, and cache robots.txt compliance rules for web crawling."""

    def __init__(self) -> None:
        """Initialize the RobotsChecker."""
        self._parsers: dict[str, RobotFileParser] = {}
        self._sitemaps: dict[str, list[str]] = {}
        self._crawl_delays: dict[str, dict[str, float]] = {}

    def _get_domain_key(self, url: str) -> str:
        """Extract a canonical domain key (hostname:port) from a URL.

        Args:
            url: The URL to derive the key from.

        Returns:
            Lowercase hostname (and port if non-default).
        """
        if not url or not isinstance(url, str):
            return ""
        clean_url = url.strip()
        if not clean_url:
            return ""
        if "://" not in clean_url:
            clean_url = f"https://{clean_url}"
        try:
            parsed = urlparse(clean_url)
            hostname = (parsed.hostname or "").lower()
            if not hostname:
                return ""
            port = parsed.port
            if port and not (
                (parsed.scheme == "http" and port == 80)
                or (parsed.scheme == "https" and port == 443)
            ):
                return f"{hostname}:{port}"
            return hostname
        except Exception:
            return ""

    async def fetch_robots(self, base_url: str, client: httpx.AsyncClient) -> None:
        """Fetch and parse robots.txt for a domain.

        Handles missing robots.txt (404/network failure) and malformed content gracefully
        by defaulting to permissive access.

        Args:
            base_url: The root URL or domain to fetch robots.txt for.
            client: The httpx.AsyncClient used for network requests.
        """
        domain_key = self._get_domain_key(base_url)
        if not domain_key:
            return

        clean_base = base_url.strip()
        if "://" not in clean_base:
            clean_base = f"https://{clean_base}"

        try:
            parsed = urlparse(clean_base)
            scheme = parsed.scheme if parsed.scheme in ("http", "https") else "https"
            netloc = parsed.netloc or parsed.path.split("/")[0]
            robots_url = f"{scheme}://{netloc}/robots.txt"
        except Exception as exc:
            logger.warning("Failed to construct robots.txt URL from %s: %s", base_url, exc)
            self._set_permissive_parser(domain_key)
            return

        try:
            response = await client.get(robots_url, follow_redirects=True, timeout=10.0)
            if response.status_code == 200:
                self._parse_robots_content(domain_key, response.text)
            else:
                # 404 or non-200 status -> allow all
                self._set_permissive_parser(domain_key)
        except Exception as exc:
            logger.debug("Could not fetch robots.txt for %s: %s", domain_key, exc)
            self._set_permissive_parser(domain_key)

    def _set_permissive_parser(self, domain_key: str) -> None:
        """Configure a permissive parser that allows all crawling for the given domain."""
        parser = RobotFileParser()
        parser.allow_all = True
        parser.modified()
        self._parsers[domain_key] = parser
        self._sitemaps.setdefault(domain_key, [])
        self._crawl_delays.setdefault(domain_key, {})

    def _parse_robots_content(self, domain_key: str, content: str) -> None:
        """Parse raw robots.txt text and populate parser, sitemaps, and crawl delays.

        Args:
            domain_key: The canonical domain key.
            content: Raw robots.txt text content.
        """
        parser = RobotFileParser()
        lines = content.splitlines()
        try:
            parser.parse(lines)
            parser.modified()
            # Conform to RFC 9309 (longer path matches take precedence, allow wins over disallow)
            all_entries = list(parser.entries)
            if parser.default_entry:
                all_entries.append(parser.default_entry)
            for entry in all_entries:
                if hasattr(entry, "rulelines"):
                    entry.rulelines.sort(key=lambda r: (len(getattr(r, "path", "")), getattr(r, "allowance", False)), reverse=True)
        except Exception as exc:
            logger.warning("Error parsing robots.txt for %s: %s", domain_key, exc)
            parser.allow_all = True
            parser.modified()

        self._parsers[domain_key] = parser

        # Custom extraction for sitemaps and float crawl-delays
        sitemaps: list[str] = []
        crawl_delays: dict[str, float] = {}

        # 0: waiting for user-agent, 1: reading user-agent(s), 2: reading rules
        state = 0
        current_agents: list[str] = []
        for raw_line in lines:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                state = 0
                current_agents = []
                continue
            if ":" not in line:
                continue

            field, val = line.split(":", 1)
            field = field.strip().lower()
            val = val.strip()

            if field == "user-agent":
                if state == 2:
                    current_agents = []
                state = 1
                agent = val.lower()
                current_agents.append(agent)
            elif field == "sitemap":
                if val and val not in sitemaps:
                    sitemaps.append(val)
            elif field == "crawl-delay":
                state = 2
                try:
                    delay = float(val)
                    agents_to_set = current_agents if current_agents else ["*"]
                    for ag in agents_to_set:
                        crawl_delays[ag] = delay
                except ValueError:
                    pass
            elif field in ("allow", "disallow"):
                state = 2

        # Also incorporate any sitemaps parsed by RobotFileParser itself
        try:
            official_sitemaps = parser.site_maps()
            if official_sitemaps:
                for sm in official_sitemaps:
                    if sm and sm not in sitemaps:
                        sitemaps.append(sm)
        except Exception:
            pass

        self._sitemaps[domain_key] = sitemaps
        self._crawl_delays[domain_key] = crawl_delays

    def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL is allowed by robots.txt.

        Args:
            url: The URL to evaluate.
            user_agent: The crawler user agent identifier.

        Returns:
            True if crawling is permitted, False otherwise.
        """
        if not url:
            return True

        domain_key = self._get_domain_key(url)
        parser = self._parsers.get(domain_key)
        if not parser:
            # If no robots.txt was fetched or loaded, permit crawling by default
            return True

        try:
            return bool(parser.can_fetch(user_agent, url))
        except Exception:
            return True

    def get_sitemaps(self, base_url: str) -> list[str]:
        """Get sitemap URLs extracted from robots.txt.

        Args:
            base_url: The domain or base URL.

        Returns:
            List of sitemap URLs found in robots.txt.
        """
        domain_key = self._get_domain_key(base_url)
        sitemaps = self._sitemaps.get(domain_key)
        if sitemaps is not None:
            return list(sitemaps)

        parser = self._parsers.get(domain_key)
        if parser:
            try:
                found = parser.site_maps()
                return list(found) if found else []
            except Exception:
                return []

        return []

    def get_crawl_delay(self, base_url: str, user_agent: str = "*") -> float | None:
        """Get crawl delay from robots.txt.

        Args:
            base_url: The domain or base URL.
            user_agent: The crawler user agent.

        Returns:
            Crawl delay in seconds as float, or None if unspecified.
        """
        domain_key = self._get_domain_key(base_url)
        delays = self._crawl_delays.get(domain_key)
        if delays is not None:
            ua_lower = user_agent.lower()
            if ua_lower in delays:
                return delays[ua_lower]
            if "*" in delays:
                return delays["*"]

        parser = self._parsers.get(domain_key)
        if parser:
            try:
                delay = parser.crawl_delay(user_agent)
                if delay is not None:
                    return float(delay)
            except Exception:
                return None

        return None
