"""Chunk schemas for Crawlbot AI pipeline.

Defines Pydantic models representing document chunks produced by the chunker service,
ready for embedding and RAG ingestion.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies for document processing."""

    AUTO = "auto"
    SEMANTIC_HEADING = "semantic_heading"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    FIXED_SIZE = "fixed_size"


class DocumentChunk(BaseModel):
    """A single chunk of text extracted from a document, ready for embedding."""

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(description="Unique identifier for this chunk (source_url + index + content_hash).")
    source_url: str = Field(description="Original URL of the document this chunk came from.")
    chunk_index: int = Field(ge=0, description="Zero-based sequential index of this chunk within its source document.")
    total_chunks: int = Field(ge=1, description="Total number of chunks produced from the source document.")
    text: str = Field(description="Plain text content of this chunk.")
    char_count: int = Field(ge=0, description="Number of characters in this chunk.")
    word_count: int = Field(ge=0, description="Approximate word count of this chunk.")
    heading_context: Optional[str] = Field(
        default=None,
        description="The hierarchical heading context (H1 > H2 > ... > H6) that precedes this chunk, if any.",
    )
    title: Optional[str] = Field(default=None, description="Title of the source document.")
    source_title: Optional[str] = Field(default=None, description="Same as title, for compatibility.")
    language: Optional[str] = Field(default=None, description="Language of the source document.")
    category: Optional[str] = Field(default=None, description="User-specified category/tag for the document.")
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding for this chunk produced by the embedding service.",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Name of the embedding model used to generate the vector.",
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 fingerprint hash of the chunk text.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached to this chunk.",
    )


class ChunkingResult(BaseModel):
    """Result of chunking a single document."""

    model_config = ConfigDict(populate_by_name=True)

    source_url: str = Field(description="URL of the source document.")
    total_chunks: int = Field(ge=0, description="Number of chunks produced.")
    strategy: str = Field(description="Chunking strategy that was applied or requested.")
    applied_strategy: Optional[str] = Field(
        default=None,
        description="The concrete strategy applied to the document (useful when strategy='auto').",
    )
    chunks: list[DocumentChunk] = Field(default_factory=list, description="List of produced chunks.")
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata about chunking execution, e.g. auto strategy selection reason.",
    )


class EmbeddingStats(BaseModel):
    """Statistics from an embedding run."""

    model_config = ConfigDict(populate_by_name=True)

    total_chunks: int = Field(ge=0)
    embedded_chunks: int = Field(ge=0)
    failed_chunks: int = Field(ge=0)
    model: str
    total_tokens: int = Field(ge=0, default=0)
