"""Extraction engines package for Crawlbot.

Provides Engine A (Trafilatura), Engine B (BeautifulSoup), and Engine C (Crawl4AI),
along with the unified EngineResult model.
"""

from .base import EngineResult
from .bs4_engine import BeautifulSoupEngine
from .crawl4ai_engine import Crawl4AIEngine
from .playwright_engine import PlaywrightEngine
from .trafilatura_engine import TrafilaturaEngine

__all__ = [
    "EngineResult",
    "TrafilaturaEngine",
    "BeautifulSoupEngine",
    "Crawl4AIEngine",
    "PlaywrightEngine",
]
