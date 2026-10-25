"""Loading Screen and Splash Detector for Crawlbot.

Detects initial loading screens, preloaders, skeleton loaders, and SPAs
waiting for client-side hydration, both in Persian and English.
Provides automated helpers to detect, bypass, and remove loading overlays.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Regular expressions for loading indicator phrases in Persian and English
LOADING_PATTERNS_PERSIAN = [
    r"در\s*حال\s*(?:بارگذاری|بارگیری|دریافت|آماده‌?سازی|فراخوانی|لود|پردازش|اتصال)",
    r"لطفاً?\s*(?:کمی\s*)?صبر\s*کنید",
    r"کمی\s*صبر\s*کنید",
    r"چند\s*لحظه\s*صبر\s*کنید",
    r"صبور\s*باشید",
    r"بارگذاری\s*صفحه",
    r"در\s*حال\s*برقراری\s*ارتباط",
    r"اطلاعات\s*در\s*حال\s*بارگذاری",
    r"صفحه\s*در\s*حال\s*بارگذاری",
]

LOADING_PATTERNS_ENGLISH = [
    r"\bloading\b(?:\s+(?:content|data|page|application|app|please\s+wait))?",
    r"\bplease\s+wait\b",
    r"\bjust\s+a\s+moment\b",
    r"\bone\s+moment\s+please\b",
    r"\bfetching\s+(?:data|content|details|results)\b",
    r"\binitializing\b(?:\s+application|\s+app)?",
    r"\bgetting\s+(?:things\s+)?ready\b",
    r"\bhang\s+tight\b",
    r"\balmost\s+there\b",
    r"\bloading\.{2,}",
]

ALL_LOADING_REGEX = [
    re.compile(p, re.IGNORECASE) for p in (LOADING_PATTERNS_PERSIAN + LOADING_PATTERNS_ENGLISH)
]

# DOM indicators: class names or IDs indicative of loading screens / splash
LOADER_DOM_SELECTORS = [
    "preloader",
    "loader",
    "loading",
    "spinner",
    "splash",
    "splash-screen",
    "loading-screen",
    "loading-overlay",
    "page-loader",
    "site-loader",
    "loading-spinner",
    "skeleton",
    "skeleton-loader",
    "app-loading",
    "is-loading",
]

LOADER_DOM_CLASS_REGEX = re.compile(
    r"\b(?:preloader|loader|loading|spinner|splash-screen|loading-screen|loading-overlay|page-loader|site-loader|loading-spinner|skeleton(?:-loader)?|app-loading)\b",
    re.IGNORECASE,
)


def is_loading_text(text: Optional[str], title: Optional[str] = None) -> bool:
    """Determine if given text (or page title) represents an initial loading screen.

    Checks if:
    1. Title explicitly matches a loading phrase and text is short.
    2. Text is short (< 350 chars or < 45 words) and contains one or more loading patterns.
    3. Over 40% of the non-whitespace text matches loading indicator phrases.
    """
    if not text and not title:
        return False

    clean_text = (text or "").strip()
    clean_title = (title or "").strip()

    # 1. Check title
    if clean_title:
        for pattern in ALL_LOADING_REGEX:
            if pattern.search(clean_title) and len(clean_text) < 400:
                return True

    if not clean_text:
        return False

    # Word and character counts
    words = clean_text.split()
    word_count = len(words)
    char_count = len(clean_text)

    # 2. Check if text is short and contains loading phrases
    if char_count < 400 or word_count < 50:
        for pattern in ALL_LOADING_REGEX:
            if pattern.search(clean_text):
                return True

    # 3. Check for repetitive loading indicator text
    matches = 0
    for pattern in ALL_LOADING_REGEX:
        if pattern.search(clean_text):
            matches += 1
            if matches >= 2 and (char_count < 800 or word_count < 100):
                return True

    return False


def has_loading_dom(html: str) -> bool:
    """Inspect HTML string to determine if preloader, spinner, or skeleton elements exist."""
    if not html or not html.strip():
        return False

    # Fast substring check before BeautifulSoup parsing
    html_lower = html[:50000].lower()
    suspicious_terms = ("loader", "preloader", "spinner", "loading", "skeleton", "splash")
    if not any(term in html_lower for term in suspicious_terms):
        return False

    try:
        soup = BeautifulSoup(html[:100000], "html.parser")
    except Exception:
        return False

    # Check for IDs commonly used for preloaders
    for loader_id in ("preloader", "loader", "loading", "splash", "app-loading", "page-loader"):
        el = soup.find(id=loader_id)
        if el:
            return True

    # Check for elements with loader classes
    loader_elements = soup.find_all(class_=LOADER_DOM_CLASS_REGEX)
    if loader_elements:
        # Check if the page is mostly just the loader
        body = soup.body
        if body:
            body_text = body.get_text(separator=" ", strip=True)
            # If body text is short or mostly inside the loader, definitely true
            if len(body_text) < 400:
                return True
            for el in loader_elements:
                el_text = el.get_text(strip=True)
                if len(el_text) > 0 and len(el_text) >= len(body_text) * 0.4:
                    return True
        else:
            return True

    return False


def is_loading_screen(
    html: Optional[str] = None,
    text: Optional[str] = None,
    title: Optional[str] = None,
) -> bool:
    """Comprehensive check whether a page or extraction result is a loading screen."""
    if text and is_loading_text(text, title=title):
        return True

    if html and has_loading_dom(html):
        # If HTML has loader DOM and text is relatively short
        clean_text = (text or "").strip()
        if not clean_text or len(clean_text) < 450:
            return True

    return False


def get_loader_remover_js() -> str:
    """Return robust JavaScript to dismiss, remove, and unhide content blocked by loading screens.

    Executes safely in headless browser contexts (Playwright/Crawl4AI).
    """
    return """(() => {
    try {
        const loaderSelectors = [
            '#preloader', '#loader', '#loading', '#splash', '#app-loading', '#page-loader',
            '.preloader', '.loader', '.loading', '.spinner', '.splash',
            '.splash-screen', '.loading-screen', '.loading-overlay',
            '.page-loader', '.site-loader', '.loading-spinner', '.skeleton',
            '[class*="preloader"]', '[class*="loading-overlay"]', '[class*="splash-screen"]',
            '[id*="preloader"]', '[id*="loading-screen"]', '[aria-busy="true"]'
        ];

        loaderSelectors.forEach(sel => {
            try {
                document.querySelectorAll(sel).forEach(el => {
                    try {
                        el.style.setProperty('display', 'none', 'important');
                        el.style.setProperty('visibility', 'hidden', 'important');
                        el.style.setProperty('opacity', '0', 'important');
                        el.setAttribute('aria-hidden', 'true');
                        // If it is positioned fixed/absolute overlay, remove it from DOM
                        const pos = window.getComputedStyle(el).position;
                        if (pos === 'fixed' || pos === 'absolute') {
                            el.remove();
                        }
                    } catch (_) {}
                });
            } catch (_) {}
        });

        // Unhide primary app/content containers if hidden behind preloader
        const contentSelectors = [
            '#root', '#app', '#__next', '#__nuxt', 'main',
            '.main-content', '#content', '.content', '#main', '.main'
        ];
        contentSelectors.forEach(sel => {
            try {
                document.querySelectorAll(sel).forEach(el => {
                    try {
                        if (el.style.display === 'none') el.style.display = '';
                        if (el.style.visibility === 'hidden') el.style.visibility = 'visible';
                        if (el.style.opacity === '0') el.style.opacity = '1';
                    } catch (_) {}
                });
            } catch (_) {}
        });

        // Unlock page scrolling if loading screen locked body or html
        if (document.body) {
            document.body.style.overflow = 'auto';
            document.body.classList.remove('loading', 'is-loading', 'overflow-hidden', 'modal-open');
        }
        if (document.documentElement) {
            document.documentElement.style.overflow = 'auto';
            document.documentElement.classList.remove('loading', 'is-loading', 'overflow-hidden');
        }
    } catch (_) {}
})();"""
