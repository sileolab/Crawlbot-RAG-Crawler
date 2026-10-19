"""OpenAI Embedding Service for Crawlbot.

Generates dense vector embeddings for document chunks using the
OpenAI Embeddings API (``text-embedding-3-small`` by default).

Features:
- Automatic batching to respect API limits (max 2048 inputs per request).
- Configurable retry logic with exponential back-off.
- Graceful degradation: chunks that fail embedding are returned with
  ``embedding=None`` rather than raising an exception.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from ..schemas.chunks import DocumentChunk, EmbeddingStats

logger = logging.getLogger(__name__)

# OpenAI API batch limit for embeddings
_BATCH_SIZE = 500
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


def _get_openai_client(api_key: str, base_url: Optional[str] = None):
    """Lazily import and instantiate the OpenAI async client."""
    try:
        from openai import AsyncOpenAI  # type: ignore
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url and base_url.strip():
            kwargs["base_url"] = base_url.strip()
        return AsyncOpenAI(**kwargs)
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for embedding. "
            "Install it with: pip install openai>=1.0.0"
        ) from exc


class EmbeddingService:
    """Generates embeddings for DocumentChunk objects via OpenAI API."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        """Initialize the embedding service.

        Args:
            api_key: OpenAI API key.
            base_url: Optional base URL for OpenAI-compatible providers.
            model: Embedding model identifier.
            batch_size: Number of texts to send in a single API request.

        Raises:
            ValueError: If api_key is empty.
        """
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API key is required for the embedding service.")
        self.api_key = api_key.strip()
        self.base_url = base_url.strip() if base_url and base_url.strip() else None
        self.model = model
        self.batch_size = min(max(1, batch_size), 2048)
        self._client = None

    def _client_instance(self):
        if self._client is None:
            self._client = _get_openai_client(self.api_key, self.base_url)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> EmbeddingStats:
        """Embed a list of chunks in-place (mutates ``chunk.embedding``).

        Args:
            chunks: List of DocumentChunk objects to embed.

        Returns:
            EmbeddingStats with counts and token usage.
        """
        if not chunks:
            return EmbeddingStats(
                total_chunks=0, embedded_chunks=0, failed_chunks=0,
                model=self.model, total_tokens=0,
            )

        total_tokens = 0
        embedded = 0
        failed = 0

        # Process in batches
        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start: batch_start + self.batch_size]
            texts = [c.text for c in batch]

            vectors, tokens, error = await self._embed_with_retry(texts)

            if error:
                logger.warning(
                    "Embedding batch %d-%d failed: %s",
                    batch_start,
                    batch_start + len(batch) - 1,
                    error,
                )
                failed += len(batch)
            else:
                for chunk, vector in zip(batch, vectors):
                    chunk.embedding = vector
                    chunk.embedding_model = self.model
                embedded += len(batch)
                total_tokens += tokens

            # Small delay between batches to be polite to the API
            if batch_start + self.batch_size < len(chunks):
                await asyncio.sleep(0.1)

        logger.info(
            "Embedding complete: %d embedded, %d failed, %d tokens used.",
            embedded, failed, total_tokens,
        )

        return EmbeddingStats(
            total_chunks=len(chunks),
            embedded_chunks=embedded,
            failed_chunks=failed,
            model=self.model,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _embed_with_retry(
        self, texts: list[str]
    ) -> tuple[list[list[float]], int, Optional[str]]:
        """Call the OpenAI embedding API with retry on transient failures.

        Returns:
            Tuple of (vectors, tokens_used, error_message_or_None).
        """
        client = self._client_instance()
        delay = _RETRY_BASE_DELAY

        for attempt in range(_MAX_RETRIES):
            try:
                response = await client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                vectors = [item.embedding for item in response.data]
                tokens = response.usage.total_tokens if response.usage else 0
                return vectors, tokens, None

            except Exception as exc:
                err_str = str(exc)
                is_transient = any(
                    keyword in err_str.lower()
                    for keyword in ("rate", "timeout", "503", "502", "500", "connection")
                )
                if is_transient and attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Embedding attempt %d/%d failed (transient): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, err_str, delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    return [], 0, err_str

        return [], 0, "Max retries exceeded"
