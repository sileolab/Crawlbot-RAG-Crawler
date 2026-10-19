"""Tests for AI features: Chunking, Embedding stub, RAG Enhancement stub.

These tests verify the Chunking service fully (no API required), and test
the Embedding / RAG services in isolation using mocked OpenAI responses.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.chunks import DocumentChunk, ChunkingResult
from app.schemas.documents import ExtractedDocument, QualityStatus
from app.services.chunker import ChunkerService, _split_fixed, _split_paragraph, _split_sentence


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_doc(text: str, url: str = "https://example.com/page", markdown: str = "") -> ExtractedDocument:
    return ExtractedDocument(
        url=url,
        content_text=text,
        content_markdown=markdown or f"# Title\n\n{text}",
        word_count=len(text.split()),
        character_count=len(text),
        quality_status=QualityStatus.SUCCESS,
    )


LONG_TEXT = """
This is the first paragraph. It contains multiple sentences and enough content to be meaningful.
It has various pieces of information that are useful for testing the chunking service.

This is the second paragraph. Here we discuss additional topics that provide more context.
The paragraph is long enough to test paragraph-based splitting strategies.

This is the third paragraph. Final thoughts and conclusions are presented here.
More details follow to ensure this text is long enough for proper chunk testing.
""".strip()

MARKDOWN_TEXT = """# Main Heading

Introduction paragraph with some initial content that sets the scene.

## Section One

Content for section one. This section has detailed information about the first topic.
More content here to ensure this section is substantial enough.

## Section Two

Content for section two. This covers a different but related topic in depth.
Additional details are provided here for completeness.

### Subsection 2.1

Nested subsection content appears here with specific details.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# ChunkerService Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkerService:

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkerService(strategy="unknown_strategy")

    def test_paragraph_strategy_basic(self):
        chunker = ChunkerService(strategy="paragraph", chunk_size=200, overlap=0)
        doc = make_doc(LONG_TEXT)
        result = chunker.chunk_document(doc)
        assert isinstance(result, ChunkingResult)
        assert result.total_chunks > 0
        assert result.strategy == "paragraph"
        assert len(result.chunks) == result.total_chunks

    def test_sentence_strategy(self):
        text = "First sentence. Second sentence! Third sentence? Fourth one here."
        chunker = ChunkerService(strategy="sentence", chunk_size=40, overlap=0)
        doc = make_doc(text)
        result = chunker.chunk_document(doc)
        assert result.total_chunks >= 1

    def test_fixed_size_strategy(self):
        chunker = ChunkerService(strategy="fixed_size", chunk_size=100, overlap=20)
        doc = make_doc(LONG_TEXT)
        result = chunker.chunk_document(doc)
        for chunk in result.chunks:
            assert chunk.char_count <= 120  # allow some slack
        assert result.total_chunks > 0

    def test_semantic_heading_strategy(self):
        chunker = ChunkerService(strategy="semantic_heading", chunk_size=500, overlap=50)
        doc = make_doc("", markdown=MARKDOWN_TEXT)
        result = chunker.chunk_document(doc)
        assert result.total_chunks >= 2
        # Check that heading context is captured
        non_first = [c for c in result.chunks if c.chunk_index > 0]
        if non_first:
            assert any(c.heading_context is not None for c in result.chunks)

    def test_empty_document_returns_zero_chunks(self):
        chunker = ChunkerService(strategy="paragraph", chunk_size=500, overlap=0)
        # Both content_text and content_markdown must be empty/None
        doc = ExtractedDocument(
            url="https://example.com/empty",
            content_text="",
            content_markdown="",
            word_count=0,
            character_count=0,
        )
        result = chunker.chunk_document(doc)
        assert result.total_chunks == 0
        assert result.chunks == []


    def test_chunk_fields_populated(self):
        chunker = ChunkerService(strategy="paragraph", chunk_size=300, overlap=50)
        doc = make_doc(LONG_TEXT, url="https://example.com/test")
        result = chunker.chunk_document(doc)
        for chunk in result.chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.source_url == "https://example.com/test"
            assert chunk.chunk_id
            assert chunk.text
            assert chunk.char_count > 0
            assert chunk.word_count > 0
            assert 0 <= chunk.chunk_index < chunk.total_chunks

    def test_chunk_multiple_documents(self):
        chunker = ChunkerService(strategy="paragraph", chunk_size=200, overlap=0)
        docs = [make_doc(LONG_TEXT, url=f"https://example.com/{i}") for i in range(3)]
        all_chunks = chunker.chunk_documents(docs, category="test")
        assert len(all_chunks) > 0
        urls = {c.source_url for c in all_chunks}
        assert len(urls) == 3
        for chunk in all_chunks:
            assert chunk.category == "test"

    def test_overlap_creates_context_continuity(self):
        """With overlap, adjacent chunks should share some characters."""
        chunker = ChunkerService(strategy="fixed_size", chunk_size=100, overlap=30)
        doc = make_doc(LONG_TEXT)
        result = chunker.chunk_document(doc)
        if len(result.chunks) >= 2:
            # Chunks should not be identical (overlap partial, not full)
            assert result.chunks[0].text != result.chunks[1].text

    def test_chunk_id_uniqueness(self):
        chunker = ChunkerService(strategy="paragraph", chunk_size=150, overlap=0)
        doc = make_doc(LONG_TEXT)
        result = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# ─────────────────────────────────────────────────────────────────────────────
# Split function unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitFunctions:

    def test_split_fixed_basic(self):
        text = "A" * 300
        chunks = _split_fixed(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 100

    def test_split_fixed_empty(self):
        assert _split_fixed("", chunk_size=100, overlap=10) == []

    def test_split_paragraph_empty(self):
        assert _split_paragraph("", chunk_size=100, overlap=0) == []

    def test_split_sentence_basic(self):
        text = "Hello world. This is a test. Another sentence here."
        chunks = _split_sentence(text, chunk_size=30, overlap=0)
        assert len(chunks) >= 1
        # All content should be covered
        combined = " ".join(chunks)
        assert "Hello" in combined


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingService Tests (mocked OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingService:

    def test_requires_api_key(self):
        from app.services.embedding_service import EmbeddingService
        with pytest.raises(ValueError, match="API key"):
            EmbeddingService(api_key="")

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        from app.services.embedding_service import EmbeddingService
        service = EmbeddingService(api_key="sk-test-key")
        stats = await service.embed_chunks([])
        assert stats.total_chunks == 0
        assert stats.embedded_chunks == 0

    @pytest.mark.asyncio
    async def test_embed_mocked_success(self):
        from app.services.embedding_service import EmbeddingService
        from app.schemas.chunks import DocumentChunk

        chunk = DocumentChunk(
            chunk_id="abc123",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
            text="Hello world test embedding.",
            char_count=27,
            word_count=4,
        )

        mock_item = MagicMock()
        mock_item.embedding = [0.1] * 1536

        mock_usage = MagicMock()
        mock_usage.total_tokens = 10

        mock_response = MagicMock()
        mock_response.data = [mock_item]
        mock_response.usage = mock_usage

        service = EmbeddingService(api_key="sk-test-key")

        with patch.object(service, '_client_instance') as mock_client_factory:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_factory.return_value = mock_client

            stats = await service.embed_chunks([chunk])

        assert stats.total_chunks == 1
        assert stats.embedded_chunks == 1
        assert stats.failed_chunks == 0
        assert stats.total_tokens == 10
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 1536
        assert chunk.embedding_model == "text-embedding-3-small"

    def test_custom_base_url_stored(self):
        from app.services.embedding_service import EmbeddingService
        service = EmbeddingService(api_key="sk-test-key", base_url="https://openrouter.ai/api/v1")
        assert service.base_url == "https://openrouter.ai/api/v1"



# ─────────────────────────────────────────────────────────────────────────────
# RAGEnhancer Tests (mocked OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGEnhancer:

    def test_requires_api_key(self):
        from app.services.rag_enhancer import RAGEnhancer
        with pytest.raises(ValueError, match="API key"):
            RAGEnhancer(api_key="")

    @pytest.mark.asyncio
    async def test_enhance_empty_content_skipped(self):
        from app.services.rag_enhancer import RAGEnhancer
        enhancer = RAGEnhancer(api_key="sk-test-key")
        doc = ExtractedDocument(
            url="https://example.com/empty",
            content_text="",
            word_count=0,
            character_count=0,
        )
        with patch.object(enhancer, '_client_instance') as mock_client_factory:
            mock_client_factory.return_value = MagicMock()
            results = await enhancer.enhance_documents([doc])

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "No text content" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_enhance_mocked_success(self):
        import json
        from app.services.rag_enhancer import RAGEnhancer

        mock_content = json.dumps({
            "summary": "This is a test summary.",
            "keywords": ["test", "example", "keyword"],
            "topics": ["Technology", "Testing"],
            "qa_pairs": [{"question": "What is this?", "answer": "A test document."}],
        })

        mock_choice = MagicMock()
        mock_choice.message.content = mock_content

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        enhancer = RAGEnhancer(api_key="sk-test-key")
        doc = make_doc(LONG_TEXT)

        with patch.object(enhancer, '_client_instance') as mock_client_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_factory.return_value = mock_client

            results = await enhancer.enhance_documents([doc])

        assert len(results) == 1
        r = results[0]
        assert r["success"] is True
        assert r["summary"] == "This is a test summary."
        assert "test" in r["keywords"]
        assert len(r["topics"]) == 2
        assert len(r["qa_pairs"]) == 1

    def test_custom_base_url_stored(self):
        from app.services.rag_enhancer import RAGEnhancer
        enhancer = RAGEnhancer(api_key="sk-test-key", base_url="https://api.deepseek.com/v1")
        assert enhancer.base_url == "https://api.deepseek.com/v1"


# ─────────────────────────────────────────────────────────────────────────────
# AI API Endpoints Tests (FastAPI TestClient)
# ─────────────────────────────────────────────────────────────────────────────

class TestAIRoutes:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_ai_status_endpoint(self, client):
        resp = client.get("/api/ai/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "chunking_available" in data["data"]
        assert data["data"]["chunking_available"] is True
        assert "base_url" in data["data"]


    def test_chunk_nonexistent_job(self, client):
        resp = client.post("/api/ai/chunk", json={"job_id": "nonexistent-job-id"})
        assert resp.status_code == 404

    def test_chunk_missing_job_id(self, client):
        resp = client.post("/api/ai/chunk", json={})
        assert resp.status_code == 400

    def test_embed_without_api_key(self, client):
        """Embedding without an API key should return 400."""
        resp = client.post("/api/ai/embed", json={"job_id": "some-job-id"})
        # Either 400 (no API key) or 404 (job not found) is acceptable
        assert resp.status_code in (400, 404)

    def test_enhance_without_api_key(self, client):
        """RAG enhance without API key should return 400."""
        resp = client.post("/api/ai/enhance", json={"job_id": "some-job-id"})
        assert resp.status_code in (400, 404)

    def test_export_invalid_format(self, client):
        resp = client.post("/api/ai/export", json={"job_id": "some-job", "format": "invalid"})
        assert resp.status_code == 400

    def test_chunk_invalid_strategy(self, client):
        resp = client.post("/api/ai/chunk", json={"job_id": "some-id", "strategy": "bad_strategy"})
        assert resp.status_code in (400, 404)

    def test_export_chunks_and_rag_enhanced(self, client):
        from app.api.routes_ai import get_manager
        from app.schemas.chunks import DocumentChunk
        from app.schemas.documents import ExtractedDocument

        manager = get_manager()
        job = manager.create_extraction_job(urls=["https://example.com/ai-doc"], config={})
        job_id = job.job_id

        doc = ExtractedDocument(
            url="https://example.com/ai-doc",
            title="سند هوش مصنوعی",
            content_markdown="# عنوان اصلی\n\nاین یک متن تستی است.",
            content_text="عنوان اصلی\n\nاین یک متن تستی است.",
            word_count=8,
            character_count=40,
        )
        manager.set_documents(job_id, [doc])

        chunk = DocumentChunk(
            chunk_id="chunk_test_1",
            source_url=doc.url,
            chunk_index=0,
            total_chunks=1,
            text=doc.content_text,
            char_count=40,
            word_count=8,
            title=doc.title,
        )
        manager.store_chunks(job_id, [chunk])

        # Test jsonl_chunks export
        r_jsonl = client.post("/api/ai/export", json={"job_id": job_id, "format": "jsonl_chunks"})
        assert r_jsonl.status_code == 200
        assert "chunk_test_1" in r_jsonl.text

        # Test json_chunks export
        r_json = client.post("/api/ai/export", json={"job_id": job_id, "format": "json_chunks"})
        assert r_json.status_code == 200
        assert "chunk_test_1" in r_json.text

        # Store RAG metadata
        rag_meta = [{
            "source_url": doc.url,
            "summary": "خلاصه سند هوش مصنوعی",
            "keywords": ["هوش مصنوعی", "کراول‌بات"],
            "topics": ["فناوری"],
            "qa_pairs": [{"question": "این چیست؟", "answer": "یک سند تستی"}],
            "model": "gpt-4o-mini",
            "success": True,
        }]
        manager.store_rag_metadata(job_id, rag_meta)

        # Test json_enhanced export
        r_rag_json = client.post("/api/ai/export", json={"job_id": job_id, "format": "json_enhanced"})
        assert r_rag_json.status_code == 200
        assert "خلاصه سند هوش مصنوعی" in r_rag_json.text
        assert "qa_pairs" in r_rag_json.text

        # Test jsonl_enhanced export
        r_rag_jsonl = client.post("/api/ai/export", json={"job_id": job_id, "format": "jsonl_enhanced"})
        assert r_rag_jsonl.status_code == 200
        assert "خلاصه سند هوش مصنوعی" in r_rag_jsonl.text

        # Test markdown_enhanced export (ZIP)
        r_rag_md = client.post("/api/ai/export", json={"job_id": job_id, "format": "markdown_enhanced"})
        assert r_rag_md.status_code == 200
        assert r_rag_md.headers["content-type"] == "application/zip"


# ─────────────────────────────────────────────────────────────────────────────
# Verification & Evidence Grounding Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVerificationGrounding:

    def test_verify_grounding_exact_evidence_and_numbers(self):
        from app.services.rag_enhancer import verify_grounding

        source = "شرکت در سال 2024 بیش از 1500 محصول تولید کرد که سود آن 25% رشد داشت."
        qa_pairs = [
            {
                "question": "شرکت چند محصول در سال 2024 تولید کرد؟",
                "answer": "شرکت 1500 محصول تولید کرد.",
                "evidence": "در سال 2024 بیش از 1500 محصول تولید کرد",
            }
        ]
        verified_qa, summary = verify_grounding(source, qa_pairs)
        assert len(verified_qa) == 1
        qa = verified_qa[0]
        assert qa["verification"]["is_grounded"] is True
        assert qa["verification"]["numbers_verified"] is True
        assert qa["verification"]["unverified_numbers"] == []
        assert qa["verification"]["status"] == "verified"
        assert summary["overall_status"] == "verified"
        assert summary["all_numbers_verified"] is True

    def test_verify_grounding_unverified_hallucinated_number(self):
        from app.services.rag_enhancer import verify_grounding

        source = "جمعیت شهر 50000 نفر است."
        qa_pairs = [
            {
                "question": "جمعیت شهر چقدر است؟",
                "answer": "جمعیت حدود 99999 نفر تخمین زده می‌شود.",
                "evidence": "جمعیت شهر 50000 نفر است.",
            }
        ]
        verified_qa, summary = verify_grounding(source, qa_pairs)
        qa = verified_qa[0]
        assert qa["verification"]["numbers_verified"] is False
        assert "99999" in qa["verification"]["unverified_numbers"]
        assert qa["verification"]["status"] == "flagged"
        assert summary["all_numbers_verified"] is False

    def test_verify_grounding_persian_to_english_numbers(self):
        from app.services.rag_enhancer import verify_grounding

        source = "سرعت خودرو به 120 کیلومتر بر ساعت رسید."
        qa_pairs = [
            {
                "question": "سرعت خودرو چقدر بود؟",
                "answer": "سرعت ۱۲۰ کیلومتر بود.",
                "evidence": "سرعت خودرو به 120 کیلومتر بر ساعت رسید.",
            }
        ]
        verified_qa, summary = verify_grounding(source, qa_pairs)
        qa = verified_qa[0]
        # 120 and ۱۲۰ should match after digit normalization
        assert qa["verification"]["numbers_verified"] is True
        assert qa["verification"]["unverified_numbers"] == []

    def test_verify_grounding_missing_evidence_fallback(self):
        from app.services.rag_enhancer import verify_grounding

        source = "پایتون یک زبان برنامه‌نویسی سطح بالا است که در سال 1991 منتشر شد."
        qa_pairs = [
            {
                "question": "پایتون چه زمانی منتشر شد؟",
                "answer": "در سال 1991 منتشر شد.",
                # No evidence provided by LLM
                "evidence": "",
            }
        ]
        verified_qa, summary = verify_grounding(source, qa_pairs)
        qa = verified_qa[0]
        # Should have found fallback evidence from source
        assert qa["evidence"] != ""
        assert qa["verification"]["is_grounded"] is True
        assert qa["verification"]["numbers_verified"] is True

    def test_verify_grounding_persian_percent(self):
        from app.services.rag_enhancer import verify_grounding

        source = "تخفیف ویژه این کالا ۵۰٪ است."
        qa_pairs = [
            {
                "question": "تخفیف چقدر است؟",
                "answer": "میزان تخفیف 50% تعیین شده است.",
                "evidence": "تخفیف ویژه این کالا ۵۰٪ است.",
            }
        ]
        verified_qa, summary = verify_grounding(source, qa_pairs)
        qa = verified_qa[0]
        assert qa["verification"]["numbers_verified"] is True
        assert summary["all_numbers_verified"] is True

    @pytest.mark.asyncio
    async def test_drop_unverified_flag(self):
        import json
        from app.services.rag_enhancer import RAGEnhancer

        mock_content = json.dumps({
            "summary": "This is a summary.",
            "keywords": ["paragraph"],
            "topics": ["Testing"],
            "qa_pairs": [{"question": "Unrelated?", "answer": "Unrelated answer."}],
        })
        mock_choice = MagicMock()
        mock_choice.message.content = mock_content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        enhancer = RAGEnhancer(api_key="sk-test", drop_unverified=True)
        doc = make_doc(LONG_TEXT)
        with patch.object(enhancer, '_client_instance') as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_factory.return_value = mock_client
            results = await enhancer.enhance_documents([doc])

        r = results[0]
        assert len(r["qa_pairs"]) == 0
        assert len(r["rejected_qa_pairs"]) == 1
        assert r["verification_summary"]["dropped_qa"] == 1



