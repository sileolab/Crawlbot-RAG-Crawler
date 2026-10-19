"""Unit tests for Crawlbot core modules: config, logging_config, and security."""

import os
from unittest.mock import patch
import pytest

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger, setup_logging
from app.core.security import (
    has_embedded_credentials,
    is_excluded_path,
    is_private_ip,
    resolve_and_validate_host,
    sanitize_filename,
    validate_url_safety,
    validate_url_scheme,
)


# ============================================================================
# Settings & Config Tests
# ============================================================================

def test_settings_defaults():
    """Verify default Settings configuration values."""
    settings = Settings()
    assert settings.app_name == "Crawlbot"
    assert settings.app_env == "development"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.max_crawl_depth == 3
    assert settings.max_pages == 500
    assert settings.request_timeout == 30.0
    assert settings.max_concurrency == 5
    assert settings.delay_between_requests == 1.0
    assert settings.max_response_size == 10 * 1024 * 1024
    assert settings.default_extraction_strategy == "trafilatura"
    assert settings.output_dir == "output"
    assert settings.log_level == "INFO"
    assert len(settings.default_exclude_patterns) > 0


def test_get_settings_caching():
    """Verify get_settings returns cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# ============================================================================
# Logging Config Tests
# ============================================================================

def test_setup_logging_and_get_logger(tmp_path):
    """Verify logger setup with custom settings and file output."""
    custom_settings = Settings(log_level="DEBUG")
    factory = setup_logging(custom_settings)
    assert callable(factory)

    logger = get_logger("crawlbot.test")
    logger.debug("Debug message test")
    logger.info("Info message test")

    assert os.path.exists("logs/crawlbot.log")


# ============================================================================
# Security Tests: is_private_ip
# ============================================================================

@pytest.mark.parametrize(
    "ip_str,expected",
    [
        # Loopback
        ("127.0.0.1", True),
        ("127.0.0.254", True),
        ("::1", True),
        # RFC 1918 Private
        ("10.0.0.1", True),
        ("10.254.254.254", True),
        ("172.16.0.1", True),
        ("172.31.255.255", True),
        ("192.168.0.1", True),
        ("192.168.100.50", True),
        # Link-local & Cloud metadata
        ("169.254.169.254", True),
        ("fe80::1", True),
        # Multicast & Reserved
        ("0.0.0.0", True),
        ("224.0.0.1", True),
        ("240.0.0.1", True),
        ("255.255.255.255", True),
        # IPv6 Unique Local
        ("fc00::1", True),
        ("fd00::1", True),
        # Carrier Grade NAT
        ("100.64.0.1", True),
        # IPv4-mapped IPv6
        ("::ffff:127.0.0.1", True),
        ("::ffff:192.168.1.1", True),
        # Public IPs (must be safe)
        ("8.8.8.8", False),
        ("1.1.1.1", False),
        ("93.184.216.34", False),
        ("2606:2800:220:1:248:1893:25c8:1946", False),
        # Invalid inputs
        ("invalid_ip", False),
        ("", False),
    ],
)
def test_is_private_ip(ip_str: str, expected: bool):
    assert is_private_ip(ip_str) == expected


# ============================================================================
# Security Tests: validate_url_scheme
# ============================================================================

@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://example.com", True),
        ("https://example.com/page", True),
        ("HTTP://EXAMPLE.COM", True),
        ("HTTPS://EXAMPLE.COM", True),
        ("ftp://example.com", False),
        ("file:///etc/passwd", False),
        ("gopher://example.com", False),
        ("javascript:alert(1)", False),
        ("data:text/html,test", False),
        ("", False),
        ("   ", False),
    ],
)
def test_validate_url_scheme(url: str, expected: bool):
    assert validate_url_scheme(url) == expected


# ============================================================================
# Security Tests: has_embedded_credentials
# ============================================================================

@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://user:pass@example.com", True),
        ("http://user@example.com", True),
        ("https://admin:secret@127.0.0.1:8080/path", True),
        ("http://example.com", False),
        ("https://example.com/path?foo=bar@baz", False),
        ("", False),
    ],
)
def test_has_embedded_credentials(url: str, expected: bool):
    assert has_embedded_credentials(url) == expected


# ============================================================================
# Security Tests: resolve_and_validate_host
# ============================================================================

def test_resolve_and_validate_host_private():
    with pytest.raises(ValueError, match="private/restricted IP"):
        resolve_and_validate_host("127.0.0.1")

    with pytest.raises(ValueError, match="private/restricted IP"):
        resolve_and_validate_host("192.168.1.1")

    with pytest.raises(ValueError, match="private/restricted IP"):
        resolve_and_validate_host("localhost")


def test_resolve_and_validate_host_empty():
    with pytest.raises(ValueError, match="empty"):
        resolve_and_validate_host("")


def test_resolve_and_validate_host_public():
    # Use mock or resolve a reliable public domain
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (2, 1, 0, "", ("93.184.216.34", 0)),
        ]
        ips = resolve_and_validate_host("example.com")
        assert ips == ["93.184.216.34"]


def test_resolve_and_validate_host_mocked_private_dns():
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        # Host points to private IP via DNS rebinding
        mock_getaddrinfo.return_value = [
            (2, 1, 0, "", ("10.0.0.5", 0)),
        ]
        with pytest.raises(ValueError, match="private/restricted IP"):
            resolve_and_validate_host("rebound.attacker.com")


# ============================================================================
# Security Tests: validate_url_safety
# ============================================================================

def test_validate_url_safety_private_ip():
    is_safe, reason = validate_url_safety("http://127.0.0.1:8000/api")
    assert not is_safe
    assert "private/restricted IP" in reason


def test_validate_url_safety_scheme():
    is_safe, reason = validate_url_safety("ftp://example.com/file")
    assert not is_safe
    assert "Invalid scheme" in reason


def test_validate_url_safety_credentials():
    is_safe, reason = validate_url_safety("http://user:pass@example.com")
    assert not is_safe
    assert "embedded credentials" in reason


def test_validate_url_safety_safe():
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (2, 1, 0, "", ("93.184.216.34", 0)),
        ]
        is_safe, reason = validate_url_safety("https://example.com/blog/article-1")
        assert is_safe
        assert reason == "URL is safe"


# ============================================================================
# Security Tests: is_excluded_path
# ============================================================================

def test_is_excluded_path():
    patterns = Settings().default_exclude_patterns

    assert is_excluded_path("https://example.com/wp-admin/index.php", patterns)
    assert is_excluded_path("https://example.com/login", patterns)
    assert is_excluded_path("https://example.com/cart", patterns)
    assert is_excluded_path("https://example.com/downloads/report.pdf", patterns)
    assert is_excluded_path("https://example.com/images/banner.png", patterns)
    assert is_excluded_path("https://example.com/article?replytocom=5", patterns)
    assert is_excluded_path("https://example.com/post#respond", patterns)

    # Safe paths
    assert not is_excluded_path("https://example.com/about-us", patterns)
    assert not is_excluded_path("https://example.com/products/item-42", patterns)
    assert not is_excluded_path("https://example.com/blog/2026/how-to-crawl", patterns)


# ============================================================================
# Security Tests: sanitize_filename
# ============================================================================

def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd") == "etc_passwd"
    assert sanitize_filename("..\\..\\Windows\\System32\\cmd.exe") == "Windows_System32_cmd.exe"
    assert sanitize_filename('invalid:name*with?chars"<>|.json') == "invalid_name_with_chars_.json"
    assert sanitize_filename("CON.txt") == "_CON.txt"
    assert sanitize_filename("PRN.csv") == "_PRN.csv"
    assert sanitize_filename("   spaced_name.txt   ") == "spaced_name.txt"
    assert sanitize_filename("...dotted_name.txt...") == "dotted_name.txt"
    assert sanitize_filename("") == "export"
    assert sanitize_filename("normal_file.json") == "normal_file.json"
