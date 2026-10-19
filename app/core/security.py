"""Security utilities for Crawlbot including SSRF protection, URL validation, and sanitization."""

import ipaddress
import re
import socket
from typing import List, Tuple
from urllib.parse import urlparse

PRIVATE_RANGES: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("224.0.0.0/4"),  # multicast
    ipaddress.ip_network("240.0.0.0/4"),  # reserved
    ipaddress.ip_network("255.255.255.255/32"),
]

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address string belongs to private, loopback, or reserved ranges.

    Args:
        ip_str: IP address string (IPv4 or IPv6).

    Returns:
        True if the IP is in private or restricted ranges, False otherwise.
    """
    if not ip_str or not isinstance(ip_str, str):
        return False

    clean_ip = ip_str.strip().strip("[]")
    try:
        ip = ipaddress.ip_address(clean_ip)
    except ValueError:
        return False

    # Check against explicit private networks
    if any(ip in net for net in PRIVATE_RANGES):
        return True

    # Check IPv4-mapped IPv6 addresses (e.g., ::ffff:127.0.0.1)
    ipv4_mapped = getattr(ip, "ipv4_mapped", None)
    if ipv4_mapped and any(ipv4_mapped in net for net in PRIVATE_RANGES):
        return True

    # Built-in properties from ipaddress module
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_reserved
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_url_scheme(url: str) -> bool:
    """Validate that the URL scheme is strictly 'http' or 'https'.

    Args:
        url: The URL string to inspect.

    Returns:
        True if the URL scheme is 'http' or 'https', False otherwise.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
        return parsed.scheme.lower() in ("http", "https")
    except Exception:
        return False


def has_embedded_credentials(url: str) -> bool:
    """Check whether the URL contains embedded credentials (e.g., user:pass@host).

    Args:
        url: The URL string to inspect.

    Returns:
        True if embedded credentials (username or password) are present, False otherwise.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
        if parsed.username is not None or parsed.password is not None:
            return True
        # Prevent obfuscation where @ appears in the network location
        if "@" in parsed.netloc:
            return True
        return False
    except Exception:
        return False


def resolve_and_validate_host(hostname: str) -> list[str]:
    """Resolve a hostname via DNS and validate that none of its IP addresses are private.

    Uses socket.getaddrinfo to resolve both IPv4 and IPv6 addresses.

    Args:
        hostname: Hostname or IP string to resolve and validate.

    Returns:
        list[str]: List of unique resolved public IP addresses.

    Raises:
        ValueError: If the hostname is empty, cannot be resolved, or resolves to
                    any private, loopback, or reserved IP address.
    """
    if not hostname or not isinstance(hostname, str):
        raise ValueError("Hostname cannot be empty")

    clean_host = hostname.strip().strip("[]")
    if not clean_host:
        raise ValueError("Hostname cannot be empty")

    # Check if hostname directly is an IP address
    try:
        direct_ip = ipaddress.ip_address(clean_host)
        if is_private_ip(str(direct_ip)):
            raise ValueError(
                f"Host '{hostname}' is a private/restricted IP address: {direct_ip}"
            )
        return [str(direct_ip)]
    except ValueError as err:
        # If it raised due to being a private IP, propagate the exception
        if "private/restricted IP" in str(err):
            raise
        # Otherwise, it's a domain name that needs DNS resolution

    try:
        addr_info = socket.getaddrinfo(
            clean_host,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as e:
        raise ValueError(f"Failed to resolve host '{hostname}': {e.strerror}") from e
    except Exception as e:
        raise ValueError(f"Failed to resolve host '{hostname}': {str(e)}") from e

    if not addr_info:
        raise ValueError(f"No DNS records found for host '{hostname}'")

    resolved_ips: list[str] = []
    for item in addr_info:
        ip = item[4][0]
        if ip not in resolved_ips:
            resolved_ips.append(ip)

    for ip in resolved_ips:
        if is_private_ip(ip):
            raise ValueError(
                f"Host '{hostname}' resolved to private/restricted IP address: {ip}"
            )

    return resolved_ips


def validate_url_safety(url: str) -> tuple[bool, str]:
    """Perform comprehensive SSRF and security checks on a URL.

    Verifies scheme, absence of embedded credentials, hostname presence and validity,
    and ensures that all resolved IP addresses are safe public IPs.

    Args:
        url: The URL to validate.

    Returns:
        tuple[bool, str]: (is_safe, reason) where is_safe is True if URL passed all
                          validations, or False with an explanation message.
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or invalid"

    stripped_url = url.strip()

    # 1. Scheme check
    if not validate_url_scheme(stripped_url):
        return False, "Invalid scheme: only http and https are allowed"

    # 2. Embedded credentials check
    if has_embedded_credentials(stripped_url):
        return False, "URL contains embedded credentials"

    # 3. Parse URL structure
    try:
        parsed = urlparse(stripped_url)
    except Exception as e:
        return False, f"Malformed URL: {str(e)}"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname"

    # Check port validity if present
    if parsed.port is not None:
        if not (1 <= parsed.port <= 65535):
            return False, f"Invalid port: {parsed.port}"

    # 4. Resolve DNS and validate host IPs
    try:
        resolve_and_validate_host(hostname)
    except ValueError as e:
        return False, str(e)

    return True, "URL is safe"


def is_excluded_path(url: str, patterns: list[str]) -> bool:
    """Check if a URL matches any exclusion regex patterns.

    Matches against both the full URL and URL path (including query and fragment).

    Args:
        url: The URL string to test.
        patterns: List of regex patterns to match against.

    Returns:
        True if any exclusion pattern matches, False otherwise.
    """
    if not url or not patterns:
        return False

    try:
        parsed = urlparse(url)
        path_with_query = parsed.path
        if parsed.query:
            path_with_query = f"{path_with_query}?{parsed.query}"
        if parsed.fragment:
            path_with_query = f"{path_with_query}#{parsed.fragment}"
    except Exception:
        path_with_query = url

    for pattern in patterns:
        if not pattern:
            continue
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            if (
                compiled.search(url)
                or compiled.search(path_with_query)
                or compiled.search(parsed.path)
            ):
                return True
        except re.error:
            # Fallback to case-insensitive substring match if regex compilation fails
            if pattern.lower() in url.lower():
                return True

    return False


def sanitize_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal and ensure filesystem compatibility.

    Removes directory traversal sequences ('..'), path separators ('/', '\\'),
    null bytes, Windows reserved characters (<>:"/\\|?*), and control characters.
    Handles Windows reserved filenames (CON, PRN, etc.) and enforces length limits.

    Args:
        name: Raw filename to sanitize.

    Returns:
        Safe filename string.
    """
    if not name or not isinstance(name, str):
        return "export"

    # Strip null bytes
    cleaned = name.replace("\x00", "")

    # Replace path separators with underscores to prevent path traversal
    cleaned = re.sub(r"[/\\]+", "_", cleaned)

    # Collapse multiple consecutive dots / directory traversal
    cleaned = re.sub(r"\.{2,}", "_", cleaned)

    # Strip illegal filesystem characters (<>:"/\|?*) and control chars (0x00-0x1F, 0x7F)
    cleaned = re.sub(r'[\x00-\x1f\x7f<>:"/\\|?*]', "_", cleaned)

    # Collapse multiple consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned)

    # Strip leading/trailing spaces, dots, and underscores
    cleaned = cleaned.strip(". _")

    if not cleaned:
        return "export"

    # Guard against Windows reserved device names
    base_stem = cleaned.split(".")[0].upper()
    if base_stem in WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"

    # Enforce maximum filename length (255 characters), preserving extension
    max_len = 255
    if len(cleaned) > max_len:
        if "." in cleaned:
            stem, ext = cleaned.rsplit(".", 1)
            ext = f".{ext}"
            cleaned = f"{stem[: max_len - len(ext)]}{ext}"
        else:
            cleaned = cleaned[:max_len]

    return cleaned
