"""Comprehensive tests for controlled redirect handling, loop/cycle detection,
trailing slash preservation, SSRF redirect defense, and crawler robustness.
"""

from __future__ import annotations

import asyncio
from typing import Any
import httpx
import pytest

from app.services.crawler import CrawlerService, FetchResult
from app.services.extractor import ContentExtractor
from app.services.url_normalizer import URLNormalizer


@pytest.fixture
def normalizer() -> URLNormalizer:
    return URLNormalizer(allowed_domains=["mocksite.local"])


class MockRedirectTransport(httpx.AsyncBaseTransport):
    """Custom async transport to simulate redirect scenarios and count requests."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.request_counts: dict[str, int] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        self.request_counts[url_str] = self.request_counts.get(url_str, 0) + 1

        route = self.routes.get(url_str)
        if not route:
            # Check path-only fallback
            path = request.url.raw_path.decode("ascii")
            route = self.routes.get(path)

        if not route:
            return httpx.Response(404, request=request, text="Not Found")

        status_code = route.get("status", 200)
        headers = dict(route.get("headers", {}))
        content = route.get("content", "<html><body><h1>Default Mock</h1></body></html>")

        if "content-type" not in [k.lower() for k in headers]:
            headers["content-type"] = "text/html; charset=utf-8"

        return httpx.Response(
            status_code=status_code,
            headers=headers,
            text=content,
            request=request,
        )


@pytest.mark.asyncio
async def test_scenario_1_single_valid_redirect_with_trailing_slash():
    """Scenario 1: Single valid redirect (e.g. /lobbies -> /lobbies/) succeeds without looping."""
    routes = {
        "http://mocksite.local/robots.txt": {"status": 404, "content": ""},
        "http://mocksite.local/sitemap.xml": {"status": 404, "content": ""},
        "http://mocksite.local/sitemap_index.xml": {"status": 404, "content": ""},
        "http://mocksite.local/wp-sitemap.xml": {"status": 404, "content": ""},
        "http://mocksite.local/": {
            "status": 200,
            "content": '<html><body><a href="/lobbies">Lobbies</a></body></html>',
        },
        "http://mocksite.local/lobbies": {
            "status": 301,
            "headers": {"Location": "http://mocksite.local/lobbies/"},
        },
        "http://mocksite.local/lobbies/": {
            "status": 200,
            "content": "<html><head><title>Lobbies Page</title></head><body><h1>All Lobbies</h1></body></html>",
        },
    }
    transport = MockRedirectTransport(routes)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = CrawlerService(
            root_url="http://mocksite.local/",
            max_depth=2,
            max_pages=10,
            client=client,
            delay=0.0,
        )
        # Mock domain safety to allow mocksite.local
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.validate_url_safety", lambda u: (True, "OK"))
            result = await crawler.discover()

    assert "http://mocksite.local/lobbies" in result.discovered
    lobbies_meta = result.discovered["http://mocksite.local/lobbies"]
    assert lobbies_meta["status"] == 200
    assert lobbies_meta["title"] == "Lobbies Page"
    assert lobbies_meta["final_url"] == "http://mocksite.local/lobbies/"
    assert "http://mocksite.local/lobbies" in result.crawled
    # Request count for /lobbies should be exactly 1, and for /lobbies/ exactly 1 (no infinite loop)
    assert transport.request_counts["http://mocksite.local/lobbies"] == 1
    assert transport.request_counts["http://mocksite.local/lobbies/"] == 1


@pytest.mark.asyncio
async def test_scenario_2_multihop_valid_redirect():
    """Scenario 2: Multi-hop valid redirect chain (/start -> /hop1 -> /final) succeeds."""
    routes = {
        "http://mocksite.local/start": {
            "status": 302,
            "headers": {"Location": "http://mocksite.local/hop1"},
        },
        "http://mocksite.local/hop1": {
            "status": 301,
            "headers": {"Location": "http://mocksite.local/final/"},
        },
        "http://mocksite.local/final/": {
            "status": 200,
            "content": "<html><head><title>Final Destination</title></head><body><h1>Final</h1></body></html>",
        },
    }
    transport = MockRedirectTransport(routes)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = CrawlerService(
            root_url="http://mocksite.local/start",
            max_depth=1,
            max_pages=5,
            client=client,
            delay=0.0,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.validate_url_safety", lambda u: (True, "OK"))
            fetch_res = await crawler._fetch_page(client, "http://mocksite.local/start")

    assert fetch_res.status_code == 200
    assert fetch_res.final_url == "http://mocksite.local/final/"
    assert fetch_res.html is not None
    assert transport.request_counts["http://mocksite.local/start"] == 1
    assert transport.request_counts["http://mocksite.local/hop1"] == 1
    assert transport.request_counts["http://mocksite.local/final/"] == 1


@pytest.mark.asyncio
async def test_scenario_3_self_redirect():
    """Scenario 3: Immediate self-redirect (A -> A) detected immediately without looping 5 times."""
    routes = {
        "http://mocksite.local/self-loop": {
            "status": 301,
            "headers": {"Location": "http://mocksite.local/self-loop"},
        },
    }
    transport = MockRedirectTransport(routes)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = CrawlerService(
            root_url="http://mocksite.local/self-loop",
            client=client,
            delay=0.0,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.validate_url_safety", lambda u: (True, "OK"))
            fetch_res = await crawler._fetch_page(client, "http://mocksite.local/self-loop")

    assert fetch_res.status_code == 311
    assert fetch_res.error == "redirect_loop"
    # Should stop after exactly 1 request, NOT 5!
    assert transport.request_counts["http://mocksite.local/self-loop"] == 1


@pytest.mark.asyncio
async def test_scenario_4_two_url_cycle():
    """Scenario 4: Two-URL cycle (A -> B -> A) detected on second hop."""
    routes = {
        "http://mocksite.local/cycle-a": {
            "status": 301,
            "headers": {"Location": "http://mocksite.local/cycle-b"},
        },
        "http://mocksite.local/cycle-b": {
            "status": 301,
            "headers": {"Location": "http://mocksite.local/cycle-a"},
        },
    }
    transport = MockRedirectTransport(routes)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = CrawlerService(
            root_url="http://mocksite.local/cycle-a",
            client=client,
            delay=0.0,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.validate_url_safety", lambda u: (True, "OK"))
            fetch_res = await crawler._fetch_page(client, "http://mocksite.local/cycle-a")

    assert fetch_res.status_code == 311
    assert fetch_res.error == "redirect_loop"
    assert transport.request_counts["http://mocksite.local/cycle-a"] == 1
    assert transport.request_counts["http://mocksite.local/cycle-b"] == 1


@pytest.mark.asyncio
async def test_scenario_5_exceeding_max_redirects():
    """Scenario 5: Chain exceeding 5 hops is halted with too_many_redirects."""
    routes = {
        f"http://mocksite.local/hop-{i}": {
            "status": 302,
            "headers": {"Location": f"http://mocksite.local/hop-{i+1}"},
        }
        for i in range(10)
    }
    transport = MockRedirectTransport(routes)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = CrawlerService(
            root_url="http://mocksite.local/hop-0",
            client=client,
            delay=0.0,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.validate_url_safety", lambda u: (True, "OK"))
            fetch_res = await crawler._fetch_page(client, "http://mocksite.local/hop-0")

    assert fetch_res.status_code == 310
    assert fetch_res.error == "too_many_redirects"
    # Total requests should be bounded by max_redirects + 1 = 6
    assert sum(transport.request_counts.values()) <= 6


@pytest.mark.asyncio
async def test_scenario_6_redirect_outside_allowed_domain():
    """Scenario 6: Redirect to domain outside allowed scope is blocked."""
    routes = {
        "http://mocksite.local/external": {
            "status": 302,
            "headers": {"Location": "http://external-evil.com/malicious"},
        },
    }
    transport = MockRedirectTransport(routes)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = CrawlerService(
            root_url="http://mocksite.local/",
            allowed_domains=["mocksite.local"],
            client=client,
            delay=0.0,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.validate_url_safety", lambda u: (True, "OK"))
            fetch_res = await crawler._fetch_page(client, "http://mocksite.local/external")

    assert fetch_res.status_code == 301
    assert "blocked_domain" in (fetch_res.error or "")
    # external-evil.com must never have been requested
    assert "http://external-evil.com/malicious" not in transport.request_counts


@pytest.mark.asyncio
async def test_scenario_7_redirect_to_private_ip():
    """Scenario 7: Redirect to private/loopback IP is blocked by SSRF defense."""
    routes = {
        "http://mocksite.local/ssrf": {
            "status": 302,
            "headers": {"Location": "http://127.0.0.1:8000/internal-admin"},
        },
    }
    transport = MockRedirectTransport(routes)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = CrawlerService(
            root_url="http://mocksite.local/",
            client=client,
            delay=0.0,
        )
        # Real validate_url_safety will reject 127.0.0.1
        with pytest.MonkeyPatch.context() as mp:
            # mocksite.local is allowed, 127.0.0.1 is rejected
            def custom_safety(url_str):
                if "127.0.0.1" in url_str:
                    return False, "Loopback/private IP disallowed"
                return True, "OK"

            mp.setattr("app.services.crawler.validate_url_safety", custom_safety)
            fetch_res = await crawler._fetch_page(client, "http://mocksite.local/ssrf")

    assert fetch_res.status_code == 403
    assert "blocked_ssrf" in (fetch_res.error or "")
    assert "http://127.0.0.1:8000/internal-admin" not in transport.request_counts


@pytest.mark.asyncio
async def test_scenario_8_bad_page_alongside_healthy_pages():
    """Scenario 8: A looping page does not prevent healthy pages from being crawled successfully."""
    routes = {
        "http://mocksite.local/robots.txt": {"status": 404, "content": ""},
        "http://mocksite.local/sitemap.xml": {"status": 404, "content": ""},
        "http://mocksite.local/sitemap_index.xml": {"status": 404, "content": ""},
        "http://mocksite.local/wp-sitemap.xml": {"status": 404, "content": ""},
        "http://mocksite.local/": {
            "status": 200,
            "content": """
                <html><body>
                    <a href="/healthy1">Healthy 1</a>
                    <a href="/bad-loop">Bad Loop</a>
                    <a href="/healthy2">Healthy 2</a>
                </body></html>
            """,
        },
        "http://mocksite.local/healthy1": {
            "status": 200,
            "content": "<html><head><title>Page 1</title></head><body>Content 1</body></html>",
        },
        "http://mocksite.local/bad-loop": {
            "status": 301,
            "headers": {"Location": "http://mocksite.local/bad-loop"},
        },
        "http://mocksite.local/healthy2": {
            "status": 200,
            "content": "<html><head><title>Page 2</title></head><body>Content 2</body></html>",
        },
    }
    transport = MockRedirectTransport(routes)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = CrawlerService(
            root_url="http://mocksite.local/",
            max_depth=2,
            max_pages=10,
            client=client,
            delay=0.0,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.validate_url_safety", lambda u: (True, "OK"))
            result = await crawler.discover()

    # Healthy pages succeeded
    assert "http://mocksite.local/healthy1" in result.crawled
    assert "http://mocksite.local/healthy2" in result.crawled
    assert result.discovered["http://mocksite.local/healthy1"]["status"] == 200
    assert result.discovered["http://mocksite.local/healthy2"]["status"] == 200

    # Bad page is recorded as failed with redirect_loop without stopping the crawl
    assert "http://mocksite.local/bad-loop" in result.failed
    assert result.discovered["http://mocksite.local/bad-loop"]["status"] == "redirect_loop"
    assert "Redirect Loop" in result.discovered["http://mocksite.local/bad-loop"]["error"]
