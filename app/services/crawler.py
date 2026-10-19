"""Crawling orchestration service for Crawlbot.

Coordinates website discovery via XML sitemaps, robots.txt compliance, and BFS
hyperlink traversal with strict SSRF validation, concurrency throttling,
and configurable domain/path/pattern filtering.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Coroutine, NamedTuple, Optional
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.security import validate_url_safety
from app.services.link_discovery import LinkDiscovery
from app.services.robots_parser import RobotsChecker
from app.services.sitemap_discovery import SitemapParser
from app.services.url_normalizer import URLNormalizer

logger = get_logger(__name__)


class FetchResult(NamedTuple):
    """Result of fetching a page with redirect resolution and safety checks."""

    html: Optional[str]
    status_code: int
    final_url: Optional[str] = None
    error: Optional[str] = None


class DiscoveryResult:
    """Holds discovery state, discovered URLs, and crawl outcome metrics."""

    def __init__(self) -> None:
        """Initialize discovery result tracking containers."""
        # url -> {title: str | None, status: int, depth: int, source: str, error: str | None}
        self.discovered: dict[str, dict[str, Any]] = {}
        self.crawled: set[str] = set()
        self.failed: set[str] = set()
        self.blocked: set[str] = set()

    @property
    def total_discovered(self) -> int:
        """Return total number of identified unique URLs."""
        return len(self.discovered)

    @property
    def total_crawled(self) -> int:
        """Return count of successfully crawled pages."""
        return len(self.crawled)

    @property
    def total_failed(self) -> int:
        """Return count of failed page requests."""
        return len(self.failed)

    @property
    def total_blocked(self) -> int:
        """Return count of URLs blocked by robots.txt or SSRF policy."""
        return len(self.blocked)

    def to_dict(self) -> dict[str, Any]:
        """Serialize discovery state to a dictionary representation."""
        return {
            "discovered": self.discovered,
            "crawled": list(self.crawled),
            "failed": list(self.failed),
            "blocked": list(self.blocked),
            "total_discovered": self.total_discovered,
            "total_crawled": self.total_crawled,
            "total_failed": self.total_failed,
            "total_blocked": self.total_blocked,
        }


class CrawlerService:
    """Main crawler orchestration service."""

    USER_AGENT: str = "Crawlbot/1.0 (+https://github.com/crawlbot)"

    def __init__(
        self,
        root_url: str,
        max_depth: int = 3,
        max_pages: int = 500,
        request_timeout: float = 30.0,
        max_concurrency: int = 5,
        delay: float = 1.0,
        allowed_domains: Optional[list[str]] = None,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        allow_subdomains: bool = False,
        keep_query_params: bool = False,
        use_sitemap: bool = True,
        path_prefix: Optional[str] = None,
        max_response_size: int = 10 * 1024 * 1024,
        progress_callback: Optional[Callable[[DiscoveryResult], Coroutine[Any, Any, None] | None]] = None,
        cancel_event: Optional[asyncio.Event] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the crawler service.

        Args:
            root_url: Starting root URL for crawl and discovery.
            max_depth: Maximum link hop depth from the root page.
            max_pages: Maximum number of pages to crawl/discover.
            request_timeout: HTTP request timeout in seconds.
            max_concurrency: Maximum simultaneous requests.
            delay: Delay in seconds between HTTP requests per worker.
            allowed_domains: Explicit list of allowed domain names.
            include_patterns: Regex patterns a URL must match to be crawled.
            exclude_patterns: Regex patterns that disqualify a URL from being crawled.
            allow_subdomains: Whether subdomains of the base domain are permitted.
            keep_query_params: Whether query parameters are retained in normalized URLs.
            use_sitemap: Whether to attempt sitemap discovery.
            path_prefix: Optional path prefix filter (e.g. '/blog/').
            max_response_size: Maximum allowable response body in bytes (default 10MB).
            progress_callback: Optional callback invoked as progress occurs.
            cancel_event: Optional asyncio.Event for cancelling the crawling process.
        """
        self.root_url: str = root_url.strip()
        self.max_depth: int = max(0, max_depth)
        self.max_pages: int = max(1, max_pages)
        self.request_timeout: float = max(1.0, request_timeout)
        self.max_concurrency: int = max(1, max_concurrency)
        self.delay: float = max(0.0, delay)
        self.allow_subdomains: bool = allow_subdomains
        self.keep_query_params: bool = keep_query_params
        self.use_sitemap: bool = use_sitemap
        self.path_prefix: Optional[str] = path_prefix.strip() if path_prefix else None
        self.max_response_size: int = max_response_size
        self.progress_callback = progress_callback
        self.cancel_event: asyncio.Event = cancel_event if cancel_event is not None else asyncio.Event()
        self.client: Optional[httpx.AsyncClient] = client

        # Initialize URLNormalizer with configuration
        self.normalizer: URLNormalizer = URLNormalizer(
            keep_query_params=self.keep_query_params,
            allowed_domains=allowed_domains or [],
            allow_subdomains=self.allow_subdomains,
        )

        # Set up base_domain from root_url
        self.base_domain: str = self.normalizer.extract_domain(self.root_url)
        self.allowed_domains: list[str] = list(self.normalizer.allowed_domains)
        if self.base_domain and self.base_domain not in self.allowed_domains:
            self.allowed_domains.append(self.base_domain)
            self.normalizer.allowed_domains.append(self.base_domain)

        # Initialize RobotsChecker and LinkDiscovery
        self.robots_checker: RobotsChecker = RobotsChecker()
        self.link_discovery: LinkDiscovery = LinkDiscovery(self.normalizer)

        # Set up patterns
        settings = get_settings()
        configured_excludes = list(settings.default_exclude_patterns)
        if exclude_patterns:
            configured_excludes.extend(exclude_patterns)
        self.exclude_patterns: list[str] = configured_excludes
        self.include_patterns: list[str] = include_patterns or []

        self._compiled_include: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self.include_patterns if p
        ]
        self._compiled_exclude: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self.exclude_patterns if p
        ]

        self.result: DiscoveryResult = DiscoveryResult()
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrency)

    async def discover(self) -> DiscoveryResult:
        """Main discovery method. Orchestrates sitemap + link crawling.

        Returns:
            DiscoveryResult populated with discovered, crawled, failed, and blocked URLs.
        """
        logger.info("Starting crawler discovery for %s (max_pages=%d, max_depth=%d)",
                    self.root_url, self.max_pages, self.max_depth)

        # 1. Validate root_url safety against SSRF
        is_safe, reason = validate_url_safety(self.root_url)
        if not is_safe:
            logger.error("Root URL failed security validation: %s (%s)", self.root_url, reason)
            self.result.blocked.add(self.root_url)
            self.result.discovered[self.root_url] = {
                "title": None,
                "status": 403,
                "depth": 0,
                "source": "root",
                "error": f"Security validation blocked: {reason}",
            }
            await self._notify_progress()
            return self.result

        # Normalize root URL
        normalized_root = self.normalizer.normalize(self.root_url)
        if not normalized_root:
            logger.error("Failed to normalize root URL: %s", self.root_url)
            self.result.failed.add(self.root_url)
            self.result.discovered[self.root_url] = {
                "title": None,
                "status": 400,
                "depth": 0,
                "source": "root",
                "error": "Invalid root URL format",
            }
            await self._notify_progress()
            return self.result

        # 2. Execute with injected client or new AsyncClient
        if self.client is not None:
            await self._run_crawl_pipeline(self.client, normalized_root)
        else:
            limits = httpx.Limits(
                max_keepalive_connections=self.max_concurrency * 2,
                max_connections=self.max_concurrency * 4,
            )
            async with httpx.AsyncClient(
                timeout=self.request_timeout,
                headers={"User-Agent": self.USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                follow_redirects=False,
                verify=True,
                limits=limits,
            ) as client:
                await self._run_crawl_pipeline(client, normalized_root)

        # 6. Final notification and return
        await self._notify_progress()
        logger.info(
            "Crawl completed. Discovered: %d, Crawled: %d, Failed: %d, Blocked: %d",
            len(self.result.discovered),
            len(self.result.crawled),
            len(self.result.failed),
            len(self.result.blocked),
        )
        return self.result

    async def _run_crawl_pipeline(self, client: httpx.AsyncClient, normalized_root: str) -> None:
        """Run the robots, sitemap, and BFS stages using the provided HTTP client."""
        # 3. Fetch and parse robots.txt
        try:
            logger.info("Fetching robots.txt for %s", self.root_url)
            await self.robots_checker.fetch_robots(self.root_url, client)
        except Exception as exc:
            logger.warning("Error fetching robots.txt: %s", exc)

        # Check if robots.txt specifies a crawl delay
        robots_delay = self.robots_checker.get_crawl_delay(self.root_url, self.USER_AGENT)
        if robots_delay is not None and robots_delay > self.delay:
            logger.info("Respecting robots.txt crawl delay: %.2fs (was %.2fs)", robots_delay, self.delay)
            self.delay = robots_delay

        # Check if root URL is allowed by robots.txt
        if not self.robots_checker.is_allowed(normalized_root, self.USER_AGENT):
            logger.warning("Root URL is disallowed by robots.txt: %s", normalized_root)
            self.result.blocked.add(normalized_root)
            self.result.discovered[normalized_root] = {
                "title": None,
                "status": 403,
                "depth": 0,
                "source": "root",
                "error": "Disallowed by robots.txt",
            }
            await self._notify_progress()
            return

        # 4. If use_sitemap: discover URLs from sitemaps
        if self.use_sitemap and not self.cancel_event.is_set():
            await self._discover_sitemaps(client)
            await self._notify_progress()

        # 5. BFS crawl starting from root_url to discover internal links
        if not self.cancel_event.is_set() and len(self.result.crawled) < self.max_pages:
            await self._crawl_bfs(client, normalized_root)

        # 6. Final notification and return
        await self._notify_progress()
        logger.info(
            "Crawl completed. Discovered: %d, Crawled: %d, Failed: %d, Blocked: %d",
            len(self.result.discovered),
            len(self.result.crawled),
            len(self.result.failed),
            len(self.result.blocked),
        )
        return self.result

    async def _discover_sitemaps(self, client: httpx.AsyncClient) -> None:
        """Discover and record URLs from robots.txt sitemaps or standard locations."""
        try:
            robots_sitemaps = self.robots_checker.get_sitemaps(self.root_url)
            sitemap_parser = SitemapParser(client, max_urls=self.max_pages)

            sitemap_urls: list[str] = []
            if robots_sitemaps:
                logger.info("Parsing %d sitemaps identified in robots.txt", len(robots_sitemaps))
                sitemap_urls = await sitemap_parser.discover_from_robots(robots_sitemaps)
            else:
                logger.info("No sitemaps in robots.txt; attempting standard locations")
                sitemap_urls = await sitemap_parser.try_default_sitemaps(self.root_url)

            logger.info("Found %d URLs from sitemaps", len(sitemap_urls))

            for raw_url in sitemap_urls:
                if self.cancel_event.is_set():
                    break
                if len(self.result.discovered) >= self.max_pages:
                    break

                norm_url = self.normalizer.normalize(raw_url)
                if not norm_url:
                    continue

                if not self._should_crawl(norm_url, depth=1):
                    continue

                if norm_url not in self.result.discovered:
                    self.result.discovered[norm_url] = {
                        "title": None,
                        "status": 0,
                        "depth": 1,
                        "source": "sitemap",
                        "error": None,
                    }
                    self.normalizer.add_seen(norm_url)

        except Exception as exc:
            logger.warning("Sitemap discovery encountered an error: %s", exc)

    async def _crawl_bfs(self, client: httpx.AsyncClient, normalized_root: str) -> None:
        """Breadth-first crawl to discover internal links.

        Args:
            client: Shared httpx.AsyncClient.
            normalized_root: The normalized starting URL.
        """
        frontier_queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        visited: set[str] = set()

        # Enqueue root URL
        if normalized_root not in self.result.discovered:
            self.result.discovered[normalized_root] = {
                "title": None,
                "status": 0,
                "depth": 0,
                "source": "root",
                "error": None,
            }
        visited.add(normalized_root)
        self.normalizer.add_seen(normalized_root)
        await frontier_queue.put((normalized_root, 0))

        # Also seed queue with sitemap URLs (depth 1) if we haven't hit limits
        for url, meta in list(self.result.discovered.items()):
            if meta.get("source") == "sitemap" and url not in visited:
                if len(visited) < self.max_pages:
                    visited.add(url)
                    await frontier_queue.put((url, 1))

        active_count = 0
        active_lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal active_count
            while not self.cancel_event.is_set():
                if len(self.result.crawled) >= self.max_pages:
                    return

                try:
                    url, depth = await asyncio.wait_for(frontier_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    async with active_lock:
                        if frontier_queue.empty() and active_count == 0:
                            return
                    continue

                if self.cancel_event.is_set():
                    frontier_queue.task_done()
                    return

                async with active_lock:
                    active_count += 1

                try:
                    await self._process_page(client, frontier_queue, visited, url, depth)
                except Exception as exc:
                    logger.warning("Unexpected error crawling %s: %s", url, exc)
                    self.result.failed.add(url)
                    if url in self.result.discovered:
                        self.result.discovered[url]["status"] = 0
                        self.result.discovered[url]["error"] = str(exc)
                finally:
                    frontier_queue.task_done()
                    async with active_lock:
                        active_count -= 1

        # Spawn worker pool bounded by max_concurrency
        num_workers = max(1, min(self.max_concurrency, 10))
        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
        await asyncio.gather(*workers, return_exceptions=True)

    async def _process_page(
        self,
        client: httpx.AsyncClient,
        queue: asyncio.Queue[tuple[str, int]],
        visited: set[str],
        url: str,
        depth: int,
    ) -> None:
        """Process a single URL in the BFS crawl: fetch, inspect, extract internal links.

        Args:
            client: HTTP client.
            queue: The BFS queue for enqueueing discovered links.
            visited: Set of visited URLs to prevent loops.
            url: The page URL to process.
            depth: The hop depth of this page.
        """
        if self.cancel_event.is_set() or len(self.result.crawled) >= self.max_pages:
            return

        # Polite crawling delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Fetch page with SSRF and redirect validation
        fetch_res = await self._fetch_page(client, url)
        html = fetch_res.html
        status_code = fetch_res.status_code
        final_url = fetch_res.final_url or url
        error_info = fetch_res.error

        # Update tracking
        self.result.crawled.add(url)
        if final_url != url:
            self.normalizer.add_seen(final_url)

        if html is None or status_code >= 400 or status_code == 0 or status_code in (310, 311):
            self.result.failed.add(url)
            if error_info == "redirect_loop" or status_code == 311:
                status_val = "redirect_loop"
                err_msg = "حلقه ریدایرکت (Redirect Loop)"
            elif error_info == "too_many_redirects" or status_code == 310:
                status_val = "too_many_redirects"
                err_msg = "سقف مجاز ریدایرکت به پایان رسید (Too Many Redirects)"
            elif error_info and error_info.startswith("blocked_"):
                status_val = "blocked"
                err_msg = error_info
                self.result.blocked.add(url)
            elif status_code > 0:
                status_val = status_code
                err_msg = f"HTTP {status_code}"
            else:
                status_val = 0
                err_msg = error_info or "Fetch failed"

            if url in self.result.discovered:
                self.result.discovered[url]["status"] = status_val
                self.result.discovered[url]["error"] = err_msg
                self.result.discovered[url]["final_url"] = final_url
            else:
                self.result.discovered[url] = {
                    "title": None,
                    "status": status_val,
                    "depth": depth,
                    "source": "crawl",
                    "error": err_msg,
                    "final_url": final_url,
                }
            await self._notify_progress()
            return

        # Extract title and update status
        title = self.link_discovery.extract_title(html)
        if url in self.result.discovered:
            self.result.discovered[url]["title"] = title
            self.result.discovered[url]["status"] = status_code
            self.result.discovered[url]["depth"] = depth
            self.result.discovered[url]["error"] = None
            self.result.discovered[url]["final_url"] = final_url
        else:
            self.result.discovered[url] = {
                "title": title,
                "status": status_code,
                "depth": depth,
                "source": "crawl",
                "error": None,
                "final_url": final_url,
            }

        await self._notify_progress()

        # If depth is within limits, discover outgoing internal links
        if depth < self.max_depth and len(self.result.discovered) < self.max_pages:
            links = self.link_discovery.extract_links(html, page_url=final_url, base_domain=self.base_domain)
            for link in links:
                if self.cancel_event.is_set():
                    break
                if len(self.result.discovered) >= self.max_pages:
                    break
                if link in visited:
                    continue

                if not self._should_crawl(link, depth + 1):
                    continue

                visited.add(link)
                self.normalizer.add_seen(link)

                if link not in self.result.discovered:
                    self.result.discovered[link] = {
                        "title": None,
                        "status": 0,
                        "depth": depth + 1,
                        "source": "link",
                        "error": None,
                    }

                await queue.put((link, depth + 1))

    async def _fetch_page(self, client: httpx.AsyncClient, url: str) -> FetchResult:
        """Fetch a page with comprehensive safety checks and controlled redirects.

        Handles redirects manually, re-validating each hop for SSRF, domain scope,
        and robots.txt. Detects redirect cycles and preserves necessary trailing slashes.
        Enforces max_response_size by reading stream chunks incrementally.

        Args:
            client: The httpx.AsyncClient.
            url: The target URL to fetch.

        Returns:
            FetchResult with html, status_code, final_url, and error details.
        """
        max_redirects: int = 5
        current_url: str = url
        redirect_count: int = 0
        visited_chain: list[str] = [current_url]
        visited_set: set[str] = {current_url}

        while redirect_count <= max_redirects:
            if self.cancel_event.is_set():
                return FetchResult(None, 0, current_url, "cancelled")

            # 1. Validate URL safety (SSRF check)
            is_safe, reason = validate_url_safety(current_url)
            if not is_safe:
                logger.warning("SSRF check blocked URL: %s (%s)", current_url, reason)
                self.result.blocked.add(current_url)
                return FetchResult(None, 403, current_url, f"blocked_ssrf: {reason}")

            # 2. Check robots.txt
            if not self.robots_checker.is_allowed(current_url, self.USER_AGENT):
                logger.info("Robots.txt blocked URL: %s", current_url)
                self.result.blocked.add(current_url)
                return FetchResult(None, 403, current_url, "blocked_robots: Disallowed by robots.txt")

            # 3. Check domain scope for redirected URL
            if not self.normalizer.is_same_domain(current_url, self.base_domain):
                logger.info("Redirected outside target domain scope: %s", current_url)
                return FetchResult(None, 301, current_url, "blocked_domain: Outside target domain scope")

            # 4. Stream request to enforce size limits under semaphore control
            async with self._semaphore:
                try:
                    async with client.stream("GET", current_url, timeout=self.request_timeout) as response:
                        status_code = response.status_code

                        # Handle redirects manually
                        if status_code in (301, 302, 303, 307, 308):
                            location = response.headers.get("Location")
                            if not location:
                                return FetchResult(None, status_code, current_url, "empty_location_header")

                            # Preserve trailing slash if present on location to prevent normalizer loop
                            has_slash = location.rstrip().endswith("/") or (
                                urlparse(location).path.endswith("/") and urlparse(location).path != "/"
                            )
                            next_url = self.normalizer.normalize(
                                location,
                                base_url=current_url,
                                preserve_trailing_slash=has_slash,
                            )
                            if not next_url:
                                return FetchResult(None, status_code, current_url, "invalid_redirect_url")

                            # Loop detection: immediate self-redirect
                            if next_url == current_url:
                                logger.warning("Redirect loop (self-redirect) detected for %s", current_url)
                                return FetchResult(None, 311, current_url, "redirect_loop")

                            # Cycle detection: visited in current chain
                            if next_url in visited_set:
                                logger.warning(
                                    "Redirect cycle detected: %s -> %s",
                                    " -> ".join(visited_chain),
                                    next_url,
                                )
                                return FetchResult(None, 311, current_url, "redirect_loop")

                            redirect_count += 1
                            if redirect_count > max_redirects:
                                logger.warning(
                                    "Too many redirects encountered for %s (hops=%d)",
                                    url,
                                    redirect_count,
                                )
                                return FetchResult(None, 310, current_url, "too_many_redirects")

                            visited_chain.append(next_url)
                            visited_set.add(next_url)
                            current_url = next_url
                            continue

                        if status_code != 200:
                            return FetchResult(None, status_code, current_url, f"HTTP {status_code}")

                        # Filter non-HTML content types
                        content_type = response.headers.get("content-type", "").lower()
                        if content_type and not any(
                            ct in content_type for ct in ("text/html", "application/xhtml+xml", "text/plain")
                        ):
                            logger.debug("Skipping non-HTML content type '%s' for %s", content_type, current_url)
                            return FetchResult(None, status_code, current_url, f"non_html_content_type: {content_type}")

                        # Read stream incrementally up to max_response_size
                        chunks: list[bytes] = []
                        bytes_read = 0
                        async for chunk in response.aiter_bytes():
                            bytes_read += len(chunk)
                            if bytes_read > self.max_response_size:
                                logger.warning(
                                    "Response for %s exceeded max size limit (%d bytes)",
                                    current_url,
                                    self.max_response_size,
                                )
                                break
                            chunks.append(chunk)

                        raw_bytes = b"".join(chunks)
                        encoding = response.encoding or "utf-8"
                        try:
                            html_text = raw_bytes.decode(encoding, errors="replace")
                        except Exception:
                            html_text = raw_bytes.decode("utf-8", errors="replace")

                        return FetchResult(html_text, status_code, current_url, None)

                except (httpx.TimeoutException, httpx.RequestError) as exc:
                    logger.warning("HTTP connection error fetching %s: %s", current_url, exc)
                    return FetchResult(None, 0, current_url, f"connection_error: {exc}")
                except Exception as exc:
                    logger.error("Unexpected error fetching %s: %s", current_url, exc)
                    return FetchResult(None, 0, current_url, f"error: {exc}")

        logger.warning("Too many redirects encountered for %s", url)
        return FetchResult(None, 310, current_url, "too_many_redirects")

    def _should_crawl(self, url: str, depth: int) -> bool:
        """Check if URL should be crawled based on all filters and constraints.

        Args:
            url: Normalized URL to test.
            depth: The depth at which this URL would be processed.

        Returns:
            True if URL should be crawled, False otherwise.
        """
        if not url or not isinstance(url, str):
            return False

        if depth > self.max_depth:
            return False

        if len(self.result.crawled) >= self.max_pages:
            return False

        # Domain scoping check
        if not self.normalizer.is_internal(url, self.base_domain):
            if not self.normalizer.is_same_domain(url, self.base_domain):
                return False

        # Path prefix filter
        if self.path_prefix:
            try:
                parsed_path = urlparse(url).path
                if not parsed_path.startswith(self.path_prefix):
                    return False
            except Exception:
                return False

        # Regex pattern filters
        if self.exclude_patterns and self._matches_patterns(url, self.exclude_patterns):
            return False

        if self.include_patterns and not self._matches_patterns(url, self.include_patterns):
            return False

        # Robots.txt check
        if not self.robots_checker.is_allowed(url, self.USER_AGENT):
            return False

        return True

    def _matches_patterns(self, url: str, patterns: list[str]) -> bool:
        """Check if URL matches any regex patterns.

        Args:
            url: The URL string to test.
            patterns: List of regex pattern strings.

        Returns:
            True if any pattern matches, False otherwise.
        """
        if not patterns or not url:
            return False

        for pattern in patterns:
            try:
                if re.search(pattern, url, re.IGNORECASE):
                    return True
            except re.error:
                continue

        return False

    async def _notify_progress(self) -> None:
        """Trigger progress callback if registered."""
        if not self.progress_callback:
            return

        try:
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(self.result)
            else:
                res = self.progress_callback(self.result)
                if asyncio.iscoroutine(res):
                    await res
        except Exception as exc:
            logger.debug("Progress callback exception: %s", exc)
