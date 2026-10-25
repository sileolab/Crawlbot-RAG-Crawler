"""Crawlbot Application Package."""

import asyncio
import sys

# Ensure Windows ProactorEventLoop is active for subprocess spawning (Playwright, Chromium, Crawl4AI)
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

