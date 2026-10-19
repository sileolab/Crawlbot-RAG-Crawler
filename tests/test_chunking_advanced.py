"""Comprehensive Advanced Chunking Test Suite for Crawlbot.

Validates all 16 acceptance scenarios required by Section 11 of the
Chunking Engineering & Optimization Specification:
1. Multi-level Markdown headings (H1-H6) & hierarchical context tracking
2. Heading without body text (orphan heading prevention)
3. Content before the first heading (preamble preservation)
4. Oversized paragraphs (> chunk_size) sub-split cleanly
5. Content without headings (graceful fallback)
6. Multiple short paragraphs merged up to chunk_size
7. Persian text with Zero-Width Non-Joiner (\\u200c / نیم‌فاصله)
8. English text with URLs, decimals (3.14), versions, and abbreviations
9. Empty document handling (0 chunks)
10. Duplicate content handling (unique IDs, matching hashes)
11. Invalid settings validation (chunk_size, overlap, unknown strategy)
12. Auto strategy selection across various page types and structures
13. Parent heading hierarchy preservation and sibling separation
14. Independent SHA-256 content_hash and stable chunk_id generation
15. Pydantic schema compliance and JSON export serialization
16. UI/API parameter compatibility with /api/ai/chunk endpoint
"""

import hashlib
import pytest
from fastapi.testclient import TestClient

from app.api.routes_ai import get_manager, router as ai_router
from app.main import app
from app.schemas.chunks import ChunkingResult, ChunkingStrategy, DocumentChunk
from app.schemas.documents import ExtractedDocument
from app.services.chunker import (
    ChunkerService,
    _extract_sentences,
    _make_chunk_id,
    _split_fixed,
    _split_paragraph,
    _split_semantic_heading,
    _split_semantic_heading_with_context,
    _split_sentence,
    select_auto_strategy,
)


def _make_doc(text: str, markdown: str = "", url: str = "https://example.com/test", page_type: str = "article") -> ExtractedDocument:
    """Helper to create ExtractedDocument for chunking tests."""
    content_md = markdown if markdown else text
    content_txt = text if text else markdown
    return ExtractedDocument(
        url=url,
        source_url=url,
        title="Test Document",
        content_markdown=content_md,
        content_text=content_txt,
        word_count=len(content_txt.split()),
        character_count=len(content_txt),
        page_type=page_type,
        content_hash=hashlib.sha256(content_txt.encode("utf-8")).hexdigest(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-level Markdown headings (H1-H6)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_01_multilevel_markdown_headings():
    """Multi-level headings (H1-H4) must preserve full parent-child hierarchy in heading_context."""
    md = """# Guide
Welcome to the guide.

## Installation
Prerequisites must be met first.

### Linux
Run apt install crawlbot on Ubuntu.

#### Debian
Debian package details here.

## Usage
Run crawlbot start in CLI.
"""
    chunker = ChunkerService(strategy="semantic_heading", chunk_size=500, overlap=50)
    doc = _make_doc("", markdown=md)
    result = chunker.chunk_document(doc)

    assert result.total_chunks >= 4
    contexts = [c.heading_context for c in result.chunks]

    # Verify hierarchical contexts
    assert any(ctx == "Guide" for ctx in contexts)
    assert any(ctx == "Guide > Installation" for ctx in contexts)
    assert any(ctx == "Guide > Installation > Linux" for ctx in contexts)
    assert any(ctx == "Guide > Installation > Linux > Debian" for ctx in contexts)
    assert any(ctx == "Guide > Usage" for ctx in contexts)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Heading without body text (orphan heading prevention)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_02_heading_without_body_text_no_orphan():
    """Headings without substantive body text before subheadings must not become orphan chunks."""
    md = """# Documentation
## Architecture
### Storage
Storage layer uses PostgreSQL for relational data and Redis for queueing.

### Crawling Engine
The crawling engine uses Playwright and Trafilatura.
"""
    chunker = ChunkerService(strategy="semantic_heading", chunk_size=500, overlap=50)
    doc = _make_doc("", markdown=md)
    result = chunker.chunk_document(doc)

    # There should NOT be any chunk whose text is solely '# Documentation' or '## Architecture'
    for c in result.chunks:
        assert c.text.strip() != "# Documentation"
        assert c.text.strip() != "## Architecture"

    # Contexts must still retain the parent headings
    contexts = [c.heading_context for c in result.chunks]
    assert any("Documentation > Architecture > Storage" in (ctx or "") for ctx in contexts)
    assert any("Documentation > Architecture > Crawling Engine" in (ctx or "") for ctx in contexts)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Content before the first heading (preamble preservation)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_03_preamble_before_first_heading():
    """Text before the first heading must be preserved as its own chunk."""
    md = """This is the introduction and overview preamble before any Markdown headings exist.
It contains important context about the project.

# Main Title
Here is the main body under the first title.
"""
    chunker = ChunkerService(strategy="semantic_heading", chunk_size=500, overlap=50)
    doc = _make_doc("", markdown=md)
    result = chunker.chunk_document(doc)

    assert result.total_chunks >= 2
    first_chunk = result.chunks[0]
    assert "introduction and overview preamble" in first_chunk.text
    # Preamble should not inherit the subsequent heading context
    assert first_chunk.heading_context is None

    second_chunk = result.chunks[1]
    assert "main body under the first title" in second_chunk.text
    assert second_chunk.heading_context == "Main Title"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Oversized paragraphs sub-split cleanly
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_04_oversized_paragraph_sub_split():
    """A paragraph longer than chunk_size must not be emitted as an oversized chunk."""
    # Create a single paragraph with 1200 characters across multiple sentences
    long_para = " ".join([f"Sentence {i} explains a detailed feature of the crawler system in depth." for i in range(25)])
    assert len(long_para) > 1000

    chunker = ChunkerService(strategy="paragraph", chunk_size=400, overlap=50)
    doc = _make_doc(long_para)
    result = chunker.chunk_document(doc)

    assert result.total_chunks > 1
    # Check that chunks are properly bounded (allowing reasonable margin for overlap)
    for c in result.chunks:
        assert c.char_count <= 460
        assert c.char_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Content without headings (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_05_content_without_headings():
    """Documents without headings fallback to paragraph flow without error."""
    text = """First paragraph of plain text without any markdown hashes.

Second paragraph providing additional information.

Third paragraph concluding the note."""

    chunker = ChunkerService(strategy="semantic_heading", chunk_size=300, overlap=0)
    doc = _make_doc(text)
    result = chunker.chunk_document(doc)

    assert result.total_chunks >= 1
    full_text = " ".join(c.text for c in result.chunks)
    assert "First paragraph" in full_text
    assert "Third paragraph" in full_text


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multiple short paragraphs merged up to chunk_size
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_06_short_paragraphs_merged():
    """Multiple short paragraphs should be merged together up to chunk_size."""
    short_paras = "\n\n".join([f"Item number {i}: short note." for i in range(10)])
    chunker = ChunkerService(strategy="paragraph", chunk_size=300, overlap=0)
    doc = _make_doc(short_paras)
    result = chunker.chunk_document(doc)

    # 10 short items (~25 chars each) should merge into fewer chunks, not 10 chunks
    assert 1 <= result.total_chunks <= 3
    for c in result.chunks:
        assert c.char_count > 50


# ─────────────────────────────────────────────────────────────────────────────
# 7. Persian text with Zero-Width Non-Joiner (\u200c / نیم‌فاصله)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_07_persian_zwnj_preserved():
    """Persian words with half-space (ZWNJ / \\u200c) must be preserved without corruption."""
    persian_text = (
        "سیستم کراول‌بات به‌صورت کاملاً خودکار برنامه‌ریزی می‌شود. "
        "این سامانه از جدیدترین تکنولوژی‌های هوش مصنوعی بهره‌مند است. "
        "قابلیت‌های پیشرفته این ابزار باعث پردازش بهینه‌تر داده‌ها می‌گردد. "
        "آیا این ابزار برای توسعه‌دهندگان مناسب است؟ بله، کاملاً مفید می‌باشد."
    )
    chunker = ChunkerService(strategy="sentence", chunk_size=200, overlap=30)
    doc = _make_doc(persian_text)
    result = chunker.chunk_document(doc)

    assert result.total_chunks >= 1
    # Check that key Persian ZWNJ compounds remain intact in chunks
    all_chunks_text = "\n".join(c.text for c in result.chunks)
    assert "کراول‌بات" in all_chunks_text
    assert "به‌صورت" in all_chunks_text
    assert "می‌شود" in all_chunks_text
    assert "تکنولوژی‌های" in all_chunks_text
    assert "بهره‌مند" in all_chunks_text
    assert "توسعه‌دهندگان" in all_chunks_text


# ─────────────────────────────────────────────────────────────────────────────
# 8. English text with URLs, decimals (3.14), and abbreviations
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_08_decimals_urls_abbreviations_not_broken():
    """Sentence splitting must not break floating point numbers, versions, URLs, or abbreviations."""
    text = (
        "Visit https://crawlbot.ai/docs/v1.0 for setup. "
        "The model version is v2.5 and costs $3.14 per 1M tokens. "
        "You can use it, e.g. for scraping, vs. manual copying! "
        "Dr. Smith verified that approx. 99.9% of URLs succeeded."
    )
    sentences = _extract_sentences(text)
    assert len(sentences) == 4

    assert sentences[0] == "Visit https://crawlbot.ai/docs/v1.0 for setup."
    assert sentences[1] == "The model version is v2.5 and costs $3.14 per 1M tokens."
    assert sentences[2] == "You can use it, e.g. for scraping, vs. manual copying!"
    assert sentences[3] == "Dr. Smith verified that approx. 99.9% of URLs succeeded."


# ─────────────────────────────────────────────────────────────────────────────
# 9. Empty document handling (0 chunks)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_09_empty_document_handling():
    """ExtractedDocument with empty content returns total_chunks=0 and empty chunks list."""
    chunker = ChunkerService(strategy="auto", chunk_size=1000, overlap=150)
    empty_doc = ExtractedDocument(
        url="https://example.com/blank",
        content_markdown="",
        content_text="   \n   ",
        word_count=0,
        character_count=0,
    )
    result = chunker.chunk_document(empty_doc)
    assert result.total_chunks == 0
    assert result.chunks == []
    assert result.metadata.get("reason") is not None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Duplicate content handling (unique IDs, matching hashes)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_10_duplicate_content_chunk_ids_and_hashes():
    """Identical sections in a document produce identical content_hash but unique chunk_ids."""
    repeated_para = "This is an identical section with the exact same wording repeated for testing."
    text = f"{repeated_para}\n\n{repeated_para}\n\n{repeated_para}"

    chunker = ChunkerService(strategy="paragraph", chunk_size=100, overlap=0)
    doc = _make_doc(text)
    result = chunker.chunk_document(doc)

    assert result.total_chunks == 3
    # All chunks must have unique IDs
    ids = [c.chunk_id for c in result.chunks]
    assert len(ids) == len(set(ids))

    # Identical text must share the exact same content_hash
    hashes = [c.content_hash for c in result.chunks]
    assert len(set(hashes)) == 1
    assert hashes[0] == hashlib.sha256(repeated_para.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Invalid settings validation
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_11_invalid_settings_validation():
    """Invalid chunk_size, overlap, and strategy names raise appropriate exceptions."""
    # Unknown strategy
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        ChunkerService(strategy="unsupported_custom_strategy")

    # API validation via FastAPI test client
    client = TestClient(app)

    # 1. Invalid strategy
    res_strat = client.post("/api/ai/chunk", json={"job_id": "dummy_id", "strategy": "invalid_strat"})
    assert res_strat.status_code in {400, 404}

    # 2. Invalid chunk_size (< 100 or > 8000)
    res_size = client.post("/api/ai/chunk", json={"job_id": "dummy_id", "chunk_size": 20})
    assert res_size.status_code in {400, 404}

    # 3. Invalid overlap (>= chunk_size)
    res_ovlp = client.post("/api/ai/chunk", json={"job_id": "dummy_id", "chunk_size": 500, "overlap": 600})
    assert res_ovlp.status_code in {400, 404}


# ─────────────────────────────────────────────────────────────────────────────
# 12. Auto strategy selection across various page types
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_12_auto_strategy_selection_scenarios():
    """Auto strategy evaluates page characteristics to pick the optimal splitter."""
    # Case A: Document with >= 2 Markdown headings -> semantic_heading
    doc_heading = _make_doc("", markdown="# Section A\nText A\n\n## Section B\nText B")
    strat_a, reason_a = select_auto_strategy(doc_heading, doc_heading.content_markdown)
    assert strat_a == "semantic_heading"
    assert "structured Markdown headings" in reason_a

    # Case B: Standard article with prose paragraphs and no headings -> paragraph
    doc_article = _make_doc(
        "First long prose paragraph discussing crawler architectures in modern data science.\n\n"
        "Second long prose paragraph analyzing latency, network timeouts, and proxy rotation."
    )
    strat_b, reason_b = select_auto_strategy(doc_article, doc_article.content_text)
    assert strat_b == "paragraph"
    assert "well-formed paragraphs" in reason_b

    # Case C: Dense single-block content without paragraphs -> sentence
    doc_dense = _make_doc(
        "Clause 1 defines the service terms. "
        "Clause 2 outlines user obligations and confidentiality requirements. "
        "Clause 3 describes termination policies and liability disclaimers."
    )
    strat_c, reason_c = select_auto_strategy(doc_dense, doc_dense.content_text)
    assert strat_c == "sentence"

    # Case D: FAQ page without headings -> paragraph
    doc_faq = _make_doc(
        "Q: What is Crawlbot?\nA: A powerful web crawler and RAG engine.\n\n"
        "Q: Does it support Persian?\nA: Yes, complete multilingual support.",
        page_type="faq"
    )
    strat_d, reason_d = select_auto_strategy(doc_faq, doc_faq.content_text)
    assert strat_d == "paragraph"

    # Case E: Policy page with headings -> semantic_heading
    doc_policy = _make_doc(
        "",
        markdown="# Terms\nIntro\n## Privacy\nData details\n## Security\nEncryption details",
        page_type="policy"
    )
    strat_e, reason_e = select_auto_strategy(doc_policy, doc_policy.content_markdown)
    assert strat_e == "semantic_heading"


# ─────────────────────────────────────────────────────────────────────────────
# 13. Parent heading hierarchy preservation and sibling separation
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_13_parent_hierarchy_and_sibling_separation():
    """Parent headings must be preserved across children while sibling headings reset cleanly."""
    md = """## Skills
### Programming Languages
Python, TypeScript, Go.

### Database Systems
PostgreSQL, Redis.

## Education
B.Sc. in Computer Science.
"""
    chunker = ChunkerService(strategy="semantic_heading", chunk_size=500, overlap=0)
    doc = _make_doc("", markdown=md)
    result = chunker.chunk_document(doc)

    contexts = {c.text.strip(): c.heading_context for c in result.chunks}

    # Find the chunks
    lang_chunk = next(c for c in result.chunks if "TypeScript" in c.text)
    db_chunk = next(c for c in result.chunks if "PostgreSQL" in c.text)
    edu_chunk = next(c for c in result.chunks if "Computer Science" in c.text)

    # Programming Languages should inherit Skills
    assert lang_chunk.heading_context == "Skills > Programming Languages"

    # Database Systems should inherit Skills, but NOT Programming Languages
    assert db_chunk.heading_context == "Skills > Database Systems"

    # Education should reset and NOT inherit Skills
    assert edu_chunk.heading_context == "Education"


# ─────────────────────────────────────────────────────────────────────────────
# 14. Independent SHA-256 content_hash and stable chunk_id generation
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_14_hash_and_id_computation():
    """Each chunk must have its own SHA-256 content_hash derived from its own text."""
    doc = _make_doc(
        "First distinct paragraph for hash test with sufficient content.\n\n"
        "Second completely different paragraph with alternate content here."
    )
    chunker = ChunkerService(strategy="paragraph", chunk_size=70, overlap=0)
    result = chunker.chunk_document(doc)

    assert result.total_chunks == 2
    c0 = result.chunks[0]
    c1 = result.chunks[1]

    # Content hashes must match the sha256 of the respective chunk text
    expected_hash_0 = hashlib.sha256(c0.text.encode("utf-8")).hexdigest()
    expected_hash_1 = hashlib.sha256(c1.text.encode("utf-8")).hexdigest()

    assert c0.content_hash == expected_hash_0
    assert c1.content_hash == expected_hash_1
    assert c0.content_hash != c1.content_hash

    # Chunk IDs incorporate content hash and are distinct
    assert c0.chunk_id != c1.chunk_id
    assert len(c0.chunk_id) == 16


# ─────────────────────────────────────────────────────────────────────────────
# 15. Pydantic schema compliance and JSON export serialization
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_15_schema_compliance_and_json_export():
    """ChunkingResult and DocumentChunk must serialize cleanly to dict and JSON."""
    doc = _make_doc("# Title\nValid content paragraph.", markdown="# Title\nValid content paragraph.")
    chunker = ChunkerService(strategy="auto", chunk_size=500, overlap=50)
    result = chunker.chunk_document(doc, category="knowledge_base")

    # Validate Pydantic dump
    result_dict = result.model_dump()
    assert result_dict["source_url"] == doc.url
    assert result_dict["strategy"] == "auto"
    assert result_dict["applied_strategy"] in {"semantic_heading", "paragraph"}
    assert isinstance(result_dict["chunks"], list)
    assert len(result_dict["chunks"]) == result.total_chunks

    chunk_dict = result_dict["chunks"][0]
    assert chunk_dict["category"] == "knowledge_base"
    assert chunk_dict["content_hash"] is not None
    assert chunk_dict["char_count"] > 0
    assert chunk_dict["word_count"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 16. UI/API parameter compatibility with /api/ai/chunk endpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_16_ui_api_parameter_compatibility():
    """Verify end-to-end /api/ai/chunk invocation with auto strategy and parameter controls."""
    manager = get_manager()
    job_id = "test-chunking-job-123"

    # Seed manager with a test job and document
    doc = _make_doc(
        "Crawlbot is an advanced RAG crawling framework.\n\n"
        "It supports Trafilatura, BeautifulSoup, and Crawl4AI engines.",
        url="https://crawlbot.ai/features"
    )
    manager._jobs[job_id] = {"id": job_id, "status": "completed"}
    manager._documents[job_id] = [doc]

    client = TestClient(app)
    response = client.post(
        "/api/ai/chunk",
        json={
            "job_id": job_id,
            "strategy": "auto",
            "chunk_size": 800,
            "overlap": 100,
            "category": "ai_docs",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]

    assert data["job_id"] == job_id
    assert data["total_chunks"] >= 1
    assert data["strategy"] == "auto"
    assert data["applied_strategy"] in {"paragraph", "sentence", "semantic_heading"}
    assert data["chunk_size"] == 800
    assert data["overlap"] == 100

    # Verify stored chunks in manager
    stored_chunks = manager.get_chunks(job_id)
    assert len(stored_chunks) == data["total_chunks"]
    assert stored_chunks[0].category == "ai_docs"
    assert stored_chunks[0].content_hash is not None
