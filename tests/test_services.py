"""Unit tests for Crawlbot crawling services: URLNormalizer, RobotsChecker, SitemapParser, and LinkDiscovery."""

import gzip
import pytest
import httpx

from app.services.url_normalizer import URLNormalizer
from app.services.robots_parser import RobotsChecker
from app.services.sitemap_discovery import SitemapParser
from app.services.link_discovery import LinkDiscovery


# ============================================================================
# URLNormalizer Tests
# ============================================================================

class TestURLNormalizer:
    """Test suite for URLNormalizer service."""

    def test_normalize_relative_urls(self):
        norm = URLNormalizer()
        assert norm.normalize("/about", base_url="https://example.com/blog/") == "https://example.com/about"
        assert norm.normalize("team", base_url="https://example.com/about/") == "https://example.com/about/team"
        assert norm.normalize("../contact", base_url="https://example.com/company/team/") == "https://example.com/company/contact"

    def test_normalize_casing_and_ports(self):
        norm = URLNormalizer()
        assert norm.normalize("HTTP://EXAMPLE.COM:80/Path/") == "http://example.com/Path"
        assert norm.normalize("HTTPS://Example.Com:443/Path") == "https://example.com/Path"
        assert norm.normalize("https://example.com:8080/path") == "https://example.com:8080/path"
        assert norm.normalize("http://example.com:8000/") == "http://example.com:8000/"

    def test_normalize_paths_and_trailing_slashes(self):
        norm = URLNormalizer()
        assert norm.normalize("https://example.com") == "https://example.com/"
        assert norm.normalize("https://example.com/") == "https://example.com/"
        assert norm.normalize("https://example.com/about/") == "https://example.com/about"
        assert norm.normalize("https://example.com/about") == "https://example.com/about"
        assert norm.normalize("https://example.com/a//b///c/../d") == "https://example.com/a/b/d"
        assert norm.normalize("https://example.com/a/./b") == "https://example.com/a/b"

    def test_normalize_query_params(self):
        # Stripping query params by default
        norm_strip = URLNormalizer(keep_query_params=False)
        assert norm_strip.normalize("https://example.com/item?id=123&sort=asc") == "https://example.com/item"

        # Sorting query params when keep_query_params=True
        norm_keep = URLNormalizer(keep_query_params=True)
        assert norm_keep.normalize("https://example.com/item?b=2&a=1&c=3") == "https://example.com/item?a=1&b=2&c=3"
        assert norm_keep.normalize("https://example.com/item?z=9#section") == "https://example.com/item?z=9"

    def test_normalize_invalid_and_non_http(self):
        norm = URLNormalizer()
        assert norm.normalize(None) is None
        assert norm.normalize("") is None
        assert norm.normalize("   ") is None
        assert norm.normalize("mailto:user@example.com") is None
        assert norm.normalize("tel:+123456789") is None
        assert norm.normalize("javascript:void(0)") is None
        assert norm.normalize("data:text/plain;base64,SGVsbG8=") is None
        assert norm.normalize("#section") is None

    def test_extract_domain(self):
        norm = URLNormalizer()
        assert norm.extract_domain("https://Sub.Example.com:8080/path") == "sub.example.com"
        assert norm.extract_domain("example.com") == "example.com"
        assert norm.extract_domain("mailto:test@example.com") == ""
        assert norm.extract_domain(None) == ""

    def test_is_same_domain(self):
        norm = URLNormalizer(allowed_domains=["example.com"], allow_subdomains=True)
        assert norm.is_same_domain("https://example.com/page", "example.com") is True
        assert norm.is_same_domain("https://api.example.com/page", "example.com") is True
        assert norm.is_same_domain("https://other.com/page", "example.com") is False
        assert norm.is_same_domain("https://notexample.com/page", "example.com") is False

        # Subdomains disabled
        norm_no_sub = URLNormalizer(allowed_domains=["example.com"], allow_subdomains=False)
        assert norm_no_sub.is_same_domain("https://api.example.com/page", "example.com") is False
        assert norm_no_sub.is_same_domain("https://example.com/page", "example.com") is True

    def test_is_internal(self):
        norm = URLNormalizer(allow_subdomains=True)
        assert norm.is_internal("https://sub.example.com/page", "example.com") is True
        assert norm.is_internal("https://example.com/page", "example.com") is True
        assert norm.is_internal("https://other.com/page", "example.com") is False

    def test_deduplication(self):
        norm = URLNormalizer()
        url = "https://example.com/test?a=1"
        assert norm.is_duplicate(url) is False
        norm.add_seen(url)
        assert norm.is_duplicate(url) is True
        assert norm.is_duplicate("https://example.com/test#frag") is True  # Query stripped, matches base
        norm.reset()
        assert norm.is_duplicate(url) is False


# ============================================================================
# RobotsChecker Tests
# ============================================================================

class TestRobotsChecker:
    """Test suite for RobotsChecker service."""

    @pytest.mark.asyncio
    async def test_fetch_and_parse_robots(self):
        checker = RobotsChecker()
        robots_content = """
User-agent: *
Disallow: /admin/
Disallow: /private/
Allow: /private/public-file
Crawl-delay: 2.5
Sitemap: https://example.com/sitemap.xml
Sitemap: https://example.com/sitemap-images.xml

User-agent: BadCrawler
Disallow: /
Crawl-delay: 10
"""
        def handler(request):
            return httpx.Response(200, text=robots_content)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await checker.fetch_robots("https://example.com", client)

        # Permissions check
        assert checker.is_allowed("https://example.com/about") is True
        assert checker.is_allowed("https://example.com/admin/dashboard") is False
        assert checker.is_allowed("https://example.com/private/secret") is False
        assert checker.is_allowed("https://example.com/private/public-file") is True

        # Custom user agent
        assert checker.is_allowed("https://example.com/about", user_agent="BadCrawler") is False

        # Sitemaps extraction
        sitemaps = checker.get_sitemaps("https://example.com")
        assert "https://example.com/sitemap.xml" in sitemaps
        assert "https://example.com/sitemap-images.xml" in sitemaps

        # Crawl delay
        assert checker.get_crawl_delay("https://example.com") == 2.5
        assert checker.get_crawl_delay("https://example.com", user_agent="BadCrawler") == 10.0

    @pytest.mark.asyncio
    async def test_missing_robots_txt_allows_all(self):
        checker = RobotsChecker()

        def handler(request):
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await checker.fetch_robots("https://example.com", client)

        assert checker.is_allowed("https://example.com/admin") is True
        assert checker.get_sitemaps("https://example.com") == []
        assert checker.get_crawl_delay("https://example.com") is None

    @pytest.mark.asyncio
    async def test_malformed_robots_txt(self):
        checker = RobotsChecker()
        html_error = "<html><body><h1>500 Internal Server Error</h1></body></html>"

        def handler(request):
            return httpx.Response(200, text=html_error)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await checker.fetch_robots("https://example.com", client)

        assert checker.is_allowed("https://example.com/anything") is True


# ============================================================================
# SitemapParser Tests
# ============================================================================

class TestSitemapParser:
    """Test suite for SitemapParser service."""

    @pytest.mark.asyncio
    async def test_parse_urlset_and_index(self):
        urlset_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page-1</loc></url>
            <url><loc>https://example.com/page-2</loc></url>
        </urlset>"""
        gz_urlset = gzip.compress(urlset_xml)

        index_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap><loc>https://example.com/sub-sitemap.xml.gz</loc></sitemap>
        </sitemapindex>"""

        def handler(request):
            url = str(request.url)
            if "sitemap_index.xml" in url:
                return httpx.Response(200, content=index_xml)
            elif "sub-sitemap.xml.gz" in url:
                return httpx.Response(200, content=gz_urlset)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            parser = SitemapParser(client=client, max_urls=50)
            urls = await parser.parse_sitemap("https://example.com/sitemap_index.xml")
            assert "https://example.com/page-1" in urls
            assert "https://example.com/page-2" in urls
            assert len(parser.discovered_urls) == 2

    @pytest.mark.asyncio
    async def test_try_default_sitemaps(self):
        urlset = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/default-page</loc></url>
        </urlset>"""

        def handler(request):
            if "sitemap.xml" in str(request.url):
                return httpx.Response(200, content=urlset)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            parser = SitemapParser(client=client)
            discovered = await parser.try_default_sitemaps("https://example.com")
            assert "https://example.com/default-page" in discovered

    @pytest.mark.asyncio
    async def test_max_urls_enforcement(self):
        urlset = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/p1</loc></url>
            <url><loc>https://example.com/p2</loc></url>
            <url><loc>https://example.com/p3</loc></url>
        </urlset>"""

        def handler(request):
            return httpx.Response(200, content=urlset)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            parser = SitemapParser(client=client, max_urls=2)
            discovered = await parser.parse_sitemap("https://example.com/sitemap.xml")
            assert len(discovered) == 2
            assert len(parser.discovered_urls) == 2

    def test_malformed_xml_recovery(self):
        parser = SitemapParser(client=None)
        malformed = b"<urlset><url><loc>https://example.com/recovered</loc><unclosed></urlset>"
        pages, subs = parser._parse_sitemap_xml(malformed)
        assert pages == ["https://example.com/recovered"]
        assert subs == []


# ============================================================================
# LinkDiscovery Tests
# ============================================================================

class TestLinkDiscovery:
    """Test suite for LinkDiscovery service."""

    def test_extract_links_filtering(self):
        normalizer = URLNormalizer(allowed_domains=["example.com"], allow_subdomains=False)
        discovery = LinkDiscovery(normalizer=normalizer)

        html = """
        <html>
        <body>
            <a href="/internal-page">Internal Link</a>
            <a href="https://example.com/about">Full Internal</a>
            <a href="https://external.com/page">External Link</a>
            <a href="/file.pdf">PDF File</a>
            <a href="/image.jpg">Image File</a>
            <a href="/archive.zip">ZIP File</a>
            <a href="mailto:info@example.com">Email</a>
            <a href="tel:+123456789">Telephone</a>
            <a href="javascript:void(0)">JavaScript</a>
            <a href="#content-anchor">Anchor</a>
            <a href="/internal-page">Duplicate Internal</a>
        </body>
        </html>
        """
        links = discovery.extract_links(html, page_url="https://example.com/", base_domain="example.com")
        assert links == ["https://example.com/internal-page", "https://example.com/about"]

    def test_extract_metadata(self):
        discovery = LinkDiscovery(normalizer=URLNormalizer())
        html = """
        <!DOCTYPE html>
        <html lang="en-US">
        <head>
            <title>   Crawlbot Studio Test Title   </title>
            <meta name="description" content="A comprehensive crawler test page description.">
            <link rel="canonical" href="https://example.com/canonical-url">
        </head>
        <body></body>
        </html>
        """
        assert discovery.extract_title(html) == "Crawlbot Studio Test Title"
        assert discovery.extract_meta_description(html) == "A comprehensive crawler test page description."
        assert discovery.extract_canonical_url(html) == "https://example.com/canonical-url"
        assert discovery.extract_language(html) == "en-US"

    def test_extract_metadata_fallbacks(self):
        discovery = LinkDiscovery(normalizer=URLNormalizer())
        html = """
        <html>
        <head>
            <meta property="og:title" content="Social Title">
            <meta property="og:description" content="Social Description">
            <meta http-equiv="content-language" content="de-DE">
        </head>
        <body></body>
        </html>
        """
        assert discovery.extract_title(html) == "Social Title"
        assert discovery.extract_meta_description(html) == "Social Description"
        assert discovery.extract_language(html) == "de-DE"
        assert discovery.extract_canonical_url(html) is None
