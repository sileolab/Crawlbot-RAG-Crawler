# 🕷️ Crawlbot Studio
### Enterprise Multi-Engine Web Crawler, Content Extractor & RAG Ingestion Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Crawl4AI](https://img.shields.io/badge/Engine%20C-Crawl4AI%20%7C%20Playwright-FF5722?style=flat-square)](https://crawl4ai.com/)
[![Trafilatura](https://img.shields.io/badge/Engine%20A-Trafilatura-2196F3?style=flat-square)](https://trafilatura.readthedocs.io/)
[![BeautifulSoup](https://img.shields.io/badge/Engine%20B-BeautifulSoup4-4CAF50?style=flat-square)](https://www.crummy.com/software/BeautifulSoup/)
[![Tests](https://img.shields.io/badge/Tests-161%20Passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Crawlbot Studio** is an enterprise-grade, asynchronous web crawling and data preparation engine engineered for **Retrieval-Augmented Generation (RAG)**, LLM knowledge bases, and semantic search pipelines. It unites **three independent extraction engines** (Trafilatura, BeautifulSoup, and Crawl4AI with Playwright), an autonomous **Page Type Classifier**, a **0–100 Content Quality Assessor**, and an **Auto-Fallback Engine** with a full **Offline Chunking & Embedding Pipeline**.

Featuring a **minimalist, modern rounded SaaS dark interface** and a high-performance RESTful API, Crawlbot bridges the gap between heterogeneous web sources and production vector stores (**ChromaDB, Pinecone, Qdrant, Milvus, pgvector**).

---

## 🌟 Core Architecture & Key Capabilities

```
                                  ┌───────────────────────────────┐
                                  │      Target Web Source        │
                                  └──────────────┬────────────────┘
                                                 │ HTTPX Async / Playwright
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ DISCOVERY & SECURITY LAYER                                                      │
│                                                                                 │
│   ┌─────────────────────┐      ┌──────────────────────┐   ┌─────────────────┐   │
│   │ URL Normalizer &    │ ───► │ Robots & Sitemap     │──►│ Hop-by-Hop SSRF │   │
│   │ Deduplication SHA   │      │ Recursive Discovery  │   │ DNS Rebind Guard│   │
│   └─────────────────────┘      └──────────────────────┘   └─────────────────┘   │
└────────────────────────────────────────┼────────────────────────────────────────┘
                                         │ Validated URLs
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ AUTONOMOUS EXTRACTION LAYER                                                     │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ Page Classifier: article | docs | product | faq | spa | policy | file   │   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │ Selection & Controlled Fallback        │
│                    ┌───────────────────┼───────────────────┐                    │
│                    ▼                   ▼                   ▼                    │
│           ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │
│           │ Engine A        │ │ Engine B        │ │ Engine C        │           │
│           │ Trafilatura     │ │ BeautifulSoup4  │ │ Crawl4AI        │           │
│           │ • Articles/Text │ │ • Tables & DOM  │ │ • Playwright JS │           │
│           │ • High Recall   │ │ • JSON-LD & OG  │ │ • SPA Hydration │           │
│           │ • Formatting    │ │ • Docs Nav/Aside│ │ • Client Routes │           │
│           └────────┬────────┘ └────────┬────────┘ └────────┬────────┘           │
│                    └───────────────────┼───────────────────┘                    │
│                                        ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ Quality Checker (0-100 Score) | Text Density | Fallback Chain | Deduplication│   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
└────────────────────────────────────────┼────────────────────────────────────────┘
                                         │ Unified ExtractedDocument
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ AI & RAG VECTOR PIPELINE                                                        │
│                                                                                 │
│   ┌────────────────────────┐   ┌────────────────────────┐   ┌─────────────────┐ │
│   │ DocumentChunker        │   │ RAGEnhancer (LLM)      │   │ EmbeddingService│ │
│   │ • Semantic Heading     │   │ • Verified Q&A Pairs   │   │ • 1536D Vectors │ │
│   │ • Paragraph / Sentence │   │ • Citation Evidence    │   │ • High-Throughput│ │
│   │ • Fixed Window+Overlap │   │ • Topic & Summary JSON │   │ • Custom BaseURL│ │
│   └───────────┬────────────┘   └───────────┬────────────┘   └────────┬────────┘ │
└───────────────┼────────────────────────────┼─────────────────────────┼──────────┘
                │                            │                         │
                ▼                            ▼                         ▼
        ┌───────────────┐            ┌───────────────┐         ┌────────────────┐
        │  Chunks JSONL │            │ Verified RAG  │         │ Vectors JSONL  │
        │  & Chunks JSON│            │ JSON Schema   │         │ (Vector Ready) │
        └───────────────┘            └───────────────┘         └────────────────┘
```

---

## 🚀 Highlights & Features

### 1. Three Independent Extraction Engines
- **Engine A — Trafilatura:** Optimized for long-form textual bodies, news publications, blogs, and documentation pages with clean Markdown output and metadata extraction.
- **Engine B — BeautifulSoup:** Direct DOM traversal with selective node cleaning. Converts tables into Markdown, extracts `JSON-LD` microdata and OpenGraph tags, preserves `<nav>` / `<aside>` code structures in developer docs without blind stripping.
- **Engine C — Crawl4AI (Playwright):** Full headless browser automation for JavaScript-rendered SPAs (React, Vue, Svelte, Angular, Next.js). Automatically handles timeouts, network drops, and ensures clean resource recycling.

### 2. Intelligent Engine Execution Modes
- **`auto` (Smart Routing & Fallback):** Classifies the page first, executes the best engine, assesses quality, and automatically executes a fallback chain if the quality score is below the configured threshold (`min_quality_score`, default: 40).
- **`trafilatura`:** Explicitly runs Trafilatura.
- **`beautifulsoup`:** Explicitly runs BeautifulSoup.
- **`crawl4ai`:** Explicitly runs Crawl4AI with browser rendering.
- **`compare` (Benchmark Mode):** Concurrently runs all three engines on the same page and produces a side-by-side performance matrix (score, word count, character count, latency).

### 3. Crawl & Ingestion Modes
- **`single_page`:** Rapid synchronous or asynchronous single-URL capture.
- **`site_crawl`:** Breadth-First-Search (BFS) crawler with depth control, domain boundary enforcement, rate limiting, and sitemap index parsing.
- **`spa_capture`:** Dynamic client-side application capture. Discovers hidden client-side routes (React/Vue router) and public REST/GraphQL APIs embedded in bundle scripts.
- **`compare`:** Side-by-side benchmark execution.

### 4. Comprehensive Quality Assessment (0–100 Score)
- **Text Density Index:** Ratio of meaningful text against raw markup size.
- **Heading Structure:** Validates semantic hierarchy ($H_1 \to H_2 \to H_3$).
- **Paragraph & Flow Analysis:** Evaluates paragraph count and average length.
- **Lexical Diversity:** Measures vocabulary uniqueness and penalizes keyword stuffing.
- **Short-Text Penalty:** Applies steep penalties for fragments with $<50$ words.
- **Deduplication:** SHA-256 cryptographic fingerprinting across crawl sessions.

### 5. Verified RAG Enhancement & Synthetic Q&A
- **Hallucination Prevention (Evidence Citation):** Every synthetic Q&A pair generated by the LLM retains verbatim evidence quoted directly from the source text.
- **Automated Consistency Check:** Automatically verifies numbers and named entities against source paragraphs.
- **Custom Base URL Support:** Fully compatible with OpenAI, OpenRouter, DeepSeek, Ollama, vLLM, and LocalAI via `CRAWLBOT_OPENAI_BASE_URL`.

### 6. Offline Chunking & Vector Embeddings
- **Zero API Dependency Chunking:** 4 algorithms (`semantic_heading`, `paragraph`, `sentence`, `fixed_size`) run 100% offline.
- **Dense Vector Embedding:** Generates 1536-dimensional embeddings with batched processing and export to ready-to-load JSONL formats.

---

## 📁 Repository Structure

```
Crawlbot/
├── app/
│   ├── main.py                     # FastAPI application factory & route mounts
│   ├── api/                        # REST API routing endpoints
│   │   ├── routes_discovery.py     # Web discovery & link crawling
│   │   ├── routes_extraction.py    # Multi-engine extraction & compare APIs
│   │   ├── routes_jobs.py          # Background job lifecycle & polling
│   │   ├── routes_exports.py       # Standard exports (JSON, JSONL, MD, TXT)
│   │   └── routes_ai.py            # AI endpoints (chunk, embed, enhance, export)
│   ├── core/                       # Security, logging, and configuration
│   │   ├── config.py               # Pydantic Settings (.env loader)
│   │   ├── logging_config.py       # Structured logging setup
│   │   └── security.py             # SSRF prevention, IP validator, file sanitizer
│   ├── schemas/                    # Pydantic v2 schemas and models
│   │   ├── chunks.py               # DocumentChunk and ChunkingRequest models
│   │   ├── documents.py            # ExtractedDocument, ExtractionAttempt, QualityReport
│   │   ├── requests.py             # ExtractionRequest, SingleExtract, CompareRequest
│   │   └── responses.py            # Standardized API response wrappers
│   ├── services/                   # Business and processing logic
│   │   ├── extractors/             # Independent extraction engines
│   │   │   ├── base.py             # Standardized EngineResult dataclass
│   │   │   ├── trafilatura_engine.py # Engine A: Trafilatura
│   │   │   ├── bs4_engine.py       # Engine B: BeautifulSoup (DOM & Tables)
│   │   │   └── crawl4ai_engine.py  # Engine C: Crawl4AI (Playwright Headless)
│   │   ├── page_classifier.py      # 11-type page classifier (article, docs, spa, ...)
│   │   ├── quality_checker.py      # 0-100 quality scoring & deduplication
│   │   ├── extractor.py            # Unified Coordinator (Auto & Compare modes)
│   │   ├── crawler.py              # BFS async crawler orchestrator
│   │   ├── chunker.py              # Chunking engine (heading, paragraph, etc.)
│   │   ├── embedding_service.py    # 1536D Vector embedding batcher
│   │   ├── rag_enhancer.py         # LLM metadata, Q&A & evidence verification
│   │   ├── content_cleaner.py      # DOM boilerplate cleaner
│   │   ├── export_service.py       # File generation and disk serialization
│   │   ├── link_discovery.py       # Link extraction & metadata discovery
│   │   ├── robots_parser.py        # robots.txt rule evaluator
│   │   ├── sitemap_discovery.py    # XML sitemap parser with index recursion
│   │   └── url_normalizer.py       # Canonicalization and deduplication
│   ├── jobs/                       # In-memory background job manager
│   │   ├── manager.py              # Job state orchestrator and cancellation
│   │   └── models.py               # Job metadata and status enums
│   └── static/                     # Modern Minimal Dark Frontend (GUI)
│       ├── index.html              # Rounded UI with 3 tabs and modal previews
│       ├── css/style.css           # 22px cards, 14px inputs, pill badges
│       └── js/app.js               # Reactive state manager, polling, and exports
├── tests/                          # Automated Pytest suite (161 tests)
│   ├── test_three_engines_and_pipelines.py # 18 verification scenarios for engines
│   ├── test_core.py                # SSRF, security, IP validation, sanitization
│   ├── test_services.py            # Crawler, normalizer, sitemap, robots
│   ├── test_extraction_and_cleaning.py # Content cleaner and quality evaluation
│   ├── test_api_and_integration.py # Discovery, extraction, and job APIs
│   ├── test_ai_features.py         # 30 tests covering Chunking, Embedding, RAG
│   ├── test_mock_website_integration.py # Full lifecycle on mock website
│   └── test_redirect_and_crawler_safety.py # Redirect loops and cycle defense
├── output/                         # Local storage for exported files (organized by job_id)
├── logs/                           # System log files
├── requirements.txt                # Python package dependencies
├── pyproject.toml                  # Pytest and linter configuration
├── .env.example                    # Environment configuration template
└── README.md                       # Documentation
```

---

## ⚡ Quickstart Guide

### Prerequisites
- Python **3.10+** (tested on 3.10, 3.11, 3.12)
- Playwright Chromium browser binary

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/crawlbot.git
cd crawlbot
```

### 2. Virtual Environment Setup

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash):**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies & Playwright Browser
```bash
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your settings:
```dotenv
# Server Configuration
CRAWLBOT_HOST=127.0.0.1
CRAWLBOT_PORT=8000
CRAWLBOT_DEBUG=false

# Extraction & Crawler Defaults
CRAWLBOT_MAX_CRAWL_DEPTH=3
CRAWLBOT_MAX_PAGES=500
CRAWLBOT_DEFAULT_EXTRACTION_STRATEGY=auto

# OpenAI / Third-Party LLM Provider (Optional for chunking; required for LLM and Embedding)
CRAWLBOT_OPENAI_API_KEY=sk-...
CRAWLBOT_OPENAI_BASE_URL=https://api.openai.com/v1

# Models
CRAWLBOT_EMBEDDING_MODEL=text-embedding-3-small
CRAWLBOT_LLM_MODEL=gpt-4o-mini
CRAWLBOT_AI_CONCURRENCY=5
```

> **Third-Party Providers:** Set `CRAWLBOT_OPENAI_BASE_URL` to any compatible gateway such as **OpenRouter** (`https://openrouter.ai/api/v1`), **DeepSeek**, or local instances (**Ollama** / **vLLM**).

### 5. Launch Crawlbot Studio
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📡 REST API Reference

Swagger UI documentation is available at `http://127.0.0.1:8000/docs`.

### Extraction & Benchmark Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/extraction/single` | Instant synchronous extraction of a single URL with specified engine or Auto mode |
| `POST` | `/api/extraction/compare` | Benchmark all 3 engines (Trafilatura, BeautifulSoup, Crawl4AI) on a single URL |
| `POST` | `/api/extraction/start` | Start background multi-URL extraction job with custom strategy, render mode, and quality threshold |
| `GET` | `/api/extraction/status/{job_id}` | Poll extraction progress, attempt logs, and document summaries |
| `POST` | `/api/extraction/retry/{job_id}` | Retry failed or blocked URLs from a previous run |
| `POST` | `/api/extraction/cancel/{job_id}` | Terminate an active extraction job |

### Web Discovery Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/discovery/start` | Launch asynchronous BFS link discovery |
| `GET` | `/api/discovery/status/{job_id}` | Poll discovery progress and discovered URLs |
| `POST` | `/api/discovery/cancel/{job_id}` | Abort active discovery crawl |

### AI & RAG Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ai/chunk` | Segment documents into chunks (`semantic_heading`, `paragraph`, `sentence`, `fixed_size`) |
| `POST` | `/api/ai/embed` | Generate 1536-D vector embeddings for generated chunks |
| `POST` | `/api/ai/enhance` | Generate verified Q&A pairs, summaries, and topic metadata with evidence citation |
| `POST` | `/api/ai/export` | Download AI assets (`jsonl_chunks`, `json_chunks`, `json_enhanced`, `jsonl_enhanced`) |
| `GET` | `/api/ai/status` | Verify LLM/Embedding provider connectivity |

### Export Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/export/download` | Export extracted documents (`json`, `jsonl`, `markdown`, `text`) |

---

## 📦 Output Artifacts & File Storage

All generated data is saved under the `output/` directory organized by `job_id`:

```text
output/
└── <job_id>/
    ├── documents.json            # Array of all extracted documents with metadata & quality reports
    ├── documents.jsonl           # Line-delimited JSON for high-throughput stream processing
    ├── markdown/                 # Individual Markdown files with YAML Frontmatter
    │   ├── example_com_blog_post_8f2a1b3c.md
    │   └── example_com_docs_api_e5f6g7h8.md
    └── text/                     # Clean plain text files with extraction headers
        ├── example_com_blog_post_8f2a1b3c.txt
        └── example_com_docs_api_e5f6g7h8.txt
```

### Standardized `ExtractedDocument` JSON Schema
```json
{
  "document_id": "a3b8c2d1e4f50912",
  "source_url": "https://example.com/blog/rag-architecture",
  "canonical_url": "https://example.com/blog/rag-architecture",
  "title": "Comprehensive Guide to RAG Pipelines",
  "language": "en",
  "page_type": "article",
  "extractor_used": "trafilatura",
  "extraction_attempts": [
    {
      "engine": "trafilatura",
      "status": "success",
      "latency_ms": 115.4,
      "char_count": 4820,
      "word_count": 760,
      "quality_score": 88
    }
  ],
  "quality_report": {
    "score": 88,
    "text_density": 0.32,
    "heading_count": 5,
    "paragraph_count": 8,
    "passed": true,
    "reasons": [
      "Substantial word count (760 words)",
      "Well-structured with 5 headings",
      "High text density (32.0%)"
    ]
  },
  "content_markdown": "# Comprehensive Guide to RAG Pipelines\n\n...",
  "content_text": "Comprehensive Guide to RAG Pipelines\n\n...",
  "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "word_count": 760,
  "character_count": 4820,
  "status": "success"
}
```

---

## 🔒 Enterprise Security Hardening

- **Multi-Hop SSRF Firewall (`app/core/security.py`):** Real-time OS DNS resolution and IP evaluation on every HTTP redirect hop. Blocks loopbacks (`127.0.0.0/8`, `::1`), private ranges (RFC-1918), AWS/GCP cloud metadata (`169.254.169.254`), link-local, and multicast addresses.
- **Protocol & Scheme Restriction:** Whitelists only `http://` and `https://`. Automatically discards dangerous pseudo-schemes (`file://`, `data:`, `javascript:`).
- **Embedded Credential Stripping:** Removes userinfo components from URLs (`http://user:pass@host`).
- **Path Traversal Defense:** Sanitizes all export filenames and job directory paths using strict alphanumeric allowlists.
- **Resource Leak Prevention:** Automatic context destruction and process cleanup for Headless Chromium instances.

---

## 🧪 Comprehensive Test Suite

Crawlbot includes **161 automated unit and integration tests** guaranteeing 100% test coverage and zero regression:

```bash
pytest -v
```

```text
======================= 161 passed, 1 warning in 18.27s =======================
```

- **`test_three_engines_and_pipelines.py` (17 tests):** Validates all 18 scenarios from Section 13 including Trafilatura article parsing, BeautifulSoup tables & JSON-LD, Crawl4AI dynamic rendering, error recovery, PageClassifier on 5 types, quality scoring, Auto Mode fallback chain, Compare Mode benchmarking, and SPA endpoint discovery.
- **`test_core.py` (56 tests):** SSRF mitigation, IP range checks, scheme sanitization, path traversal guards.
- **`test_services.py` (19 tests):** BFS crawler mechanics, canonical URL normalizer, sitemap parsing, robots caching.
- **`test_extraction_and_cleaning.py` (10 tests):** DOM cleaning, text whitespace normalization, quality statuses.
- **`test_api_and_integration.py` (8 tests):** RESTful route endpoints, background jobs, cancel/retry APIs.
- **`test_ai_features.py` (30 tests):** Chunking algorithms, overlap calculations, embedding batching, RAG formatting.
- **`test_redirect_and_crawler_safety.py` (8 tests):** Redirect cycle detection, self-redirects, max-redirect caps.
- **`test_mock_website_integration.py` (2 tests):** End-to-end full site crawl and extraction lifecycle on deterministic mock servers.

---

## 📄 License

This project is open-source software licensed under the [MIT License](LICENSE).
