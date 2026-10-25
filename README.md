# 🕷️ Crawlbot Studio
### Enterprise Multi-Engine Web Crawler, Content Extractor & RAG Ingestion Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Engine Playwright](https://img.shields.io/badge/Engine%20D-Playwright%20Chromium-45BA4B?style=flat-square&logo=playwright&logoColor=white)](https://playwright.dev/python/)
[![Engine Crawl4AI](https://img.shields.io/badge/Engine%20C-Crawl4AI-FF5722?style=flat-square)](https://crawl4ai.com/)
[![Engine Trafilatura](https://img.shields.io/badge/Engine%20A-Trafilatura-2196F3?style=flat-square)](https://trafilatura.readthedocs.io/)
[![Engine BeautifulSoup](https://img.shields.io/badge/Engine%20B-BeautifulSoup4-4CAF50?style=flat-square)](https://www.crummy.com/software/BeautifulSoup/)
[![Tests](https://img.shields.io/badge/Tests-234%20Passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Crawlbot Studio** is an enterprise-grade, asynchronous web crawling, content extraction, and data preparation platform engineered specifically for **Retrieval-Augmented Generation (RAG)**, LLM knowledge bases, and semantic search systems. 

It unites **four independent extraction engines** (Trafilatura, BeautifulSoup, Crawl4AI, and Native Playwright Chromium), an autonomous **Page Type Classifier**, an intelligent **Anti-Loader & Dynamic Hydration Bypass Engine**, a **Persian/Arabic Unicode Normalizer**, a **0–100 Content Quality Assessor**, and an **Offline Chunking & 1536-D Vector Embedding Pipeline** alongside a dedicated **File Processing Studio**.

Featuring a **minimalist, modern rounded SaaS dark interface** and a high-performance RESTful API, Crawlbot bridges the gap between heterogeneous web sources / raw datasets and production vector stores (**ChromaDB, Pinecone, Qdrant, Milvus, pgvector**).

---

## 🌟 Architecture & Data Flow

```
                                  ┌───────────────────────────────┐
                                  │   Target Web / Uploaded File  │
                                  └──────────────┬────────────────┘
                                                 │ HTTPX Async / Playwright / Multipart
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ DISCOVERY, FILE INGESTION & SECURITY LAYER                                      │
│                                                                                 │
│   ┌─────────────────────┐      ┌──────────────────────┐   ┌─────────────────┐   │
│   │ URL Normalizer &    │ ───► │ Robots & Sitemap     │──►│ Hop-by-Hop SSRF │   │
│   │ Deduplication SHA   │      │ Recursive Discovery  │   │ DNS Rebind Guard│   │
│   ├─────────────────────┤      └──────────────────────┘   └─────────────────┘   │
│   │ File Ingestion      │ ───► JSON / JSONL / Markdown Direct Ingestion         │
│   │ (/api/files)        │                                                       │
│   └─────────────────────┘                                                       │
└────────────────────────────────────────┼────────────────────────────────────────┘
                                         │ Validated URLs / Raw Content
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ AUTONOMOUS EXTRACTION & HYDRATION LAYER                                         │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ Page Classifier: article | docs | product | faq | spa | policy | about  │   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │ Selection & Controlled Fallback        │
│                    ┌───────────────────┼───────────────────┬────────────────┐   │
│                    ▼                   ▼                   ▼                ▼   │
│           ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ ┌─────────────┐ │
│           │ Engine A        │ │ Engine B        │ │ Engine C    │ │ Engine D    │ │
│           │ Trafilatura     │ │ BeautifulSoup4  │ │ Crawl4AI    │ │ Playwright  │ │
│           │ • Articles/Text │ │ • Tables & DOM  │ │ • Browser   │ │ • Chromium  │ │
│           │ • High Recall   │ │ • JSON-LD & OG  │ │ • Markdown  │ │ • Hydration │ │
│           │ • Formatting    │ │ • Docs Nav/Aside│ │ • Scraping  │ │ • AntiLoader│ │
│           └────────┬────────┘ └────────┬────────┘ └──────┬──────┘ └──────┬──────┘ │
│                    └───────────────────┼─────────────────┴───────────────┘        │
│                                        ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐     │
│   │ Loading Screen & Skeleton Detector: Auto-Scroll, Settlement & Bypass    │     │
│   ├─────────────────────────────────────────────────────────────────────────┤     │
│   │ Persian Normalizer: Unicode harmonization, ZWNJ, digit/punct correction │     │
│   ├─────────────────────────────────────────────────────────────────────────┤     │
│   │ Quality Checker (0-100 Score): Text Density | Structure | Deduplication │     │
│   └────────────────────────────────────┬────────────────────────────────────┘     │
└────────────────────────────────────────┼──────────────────────────────────────────┘
                                         │ Unified ExtractedDocument
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ AI & RAG VECTOR PIPELINE                                                        │
│                                                                                 │
│   ┌────────────────────────┐   ┌────────────────────────┐   ┌─────────────────┐ │
│   │ DocumentChunker        │   │ RAGEnhancer (LLM)      │   │ EmbeddingService│ │
│   │ • Semantic Heading     │   │ • Verified Q&A Pairs   │   │ • 1536D Vectors │ │
│   │ • Paragraph / Sentence │   │ • Citation Evidence    │   │ • High-Batching │ │
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

## 🚀 Key Highlights & Capabilities

### 1. Four Independent Extraction Engines
- **Engine A — Trafilatura:** Optimized for long-form textual bodies, news publications, blogs, and documentation pages with clean Markdown output and metadata extraction.
- **Engine B — BeautifulSoup:** Direct DOM traversal with selective node cleaning. Converts tables into clean Markdown, extracts `JSON-LD` microdata and OpenGraph tags, and preserves `<nav>` / `<aside>` code structures in developer docs without blind stripping.
- **Engine C — Crawl4AI:** Browser-assisted extraction designed for LLM readiness, supporting dynamic JavaScript rendering and modern layout parsing.
- **Engine D — Native Playwright Chromium:** Full headless browser automation for JavaScript-heavy SPAs (React, Vue, Svelte, Angular, Next.js). Features custom hydration waits, smooth auto-scrolling to trigger lazy loading, and an isolated **Windows ProactorEventLoop** architecture that guarantees seamless subprocess execution under Windows ASGI servers.

### 2. Intelligent Loading Screen & Skeleton Bypass
- Autonomous detection of dynamic SPA loading screens (e.g. animated preloader spinners, splash screens, skeleton overlays).
- Automatic JavaScript injection to dismiss blocking loader elements (`#preloader`, `.loader`, `[data-loading]`, `animate-spin`).
- Adaptive network settlement and auto-retry to ensure complete page hydration before DOM extraction.

### 3. Persian & Multilingual Normalization Engine
- **Unicode Normalization:** Standardizes Arabic/Persian letter variants (ی/ي, ک/ك, ۀ/هٔ).
- **Zero-Width Non-Joiner (ZWNJ / نیم‌فاصله):** Intelligent placement for affixes (`می‌`, `نمی‌`, `تر`, `ترین`, `ها`).
- **Digits & Punctuation:** Canonical conversion of Arabic/English numerals to Persian digits and quotation marks/question marks harmonization.
- **Diacritics & Noise Cleaning:** Cleans Kashida (elongation), excessive spaces, and pronunciation marks while preserving code blocks and URLs.

### 4. Direct File Processing Studio (`/api/files`)
- Upload raw `.json`, `.jsonl`, or `.md` files directly from the UI or API.
- Modular execution options:
  - **AI Optimization & Enrichment:** Extracts topics, summaries, and synthetic Q&A pairs with citations.
  - **Advanced Chunking:** Splits uploaded files into semantic or fixed-size chunks.
  - **Dense Vector Embedding:** Directly generates 1536-D vector embeddings for downstream vector storage.
- Real-time inline previews and one-click JSON/JSONL downloads.

### 5. Multi-Strategy Offline Chunking (Zero External API Dependency)
- **`semantic_heading`:** Preserves document hierarchy ($H_1 \to H_2 \to H_3$) and prepends parent heading trails to each chunk.
- **`paragraph`:** Merges paragraphs up to the configured token budget without splitting coherent paragraphs.
- **`sentence`:** Regex-based multilingual sentence boundary detection with configurable token overlap.
- **`fixed_size`:** Sliding window character/word chunking with customizable overlap.

### 6. Intelligent Engine Execution Modes
- **`auto` (Smart Routing & Fallback):** Classifies the page first, executes the best engine, assesses quality, and automatically cascades through fallback engines if the quality score is below the configured threshold (`min_quality_score`, default: 40).
- **`playwright`:** Explicitly runs the native Playwright Chromium engine.
- **`trafilatura`:** Explicitly runs Trafilatura.
- **`beautifulsoup`:** Explicitly runs BeautifulSoup.
- **`crawl4ai`:** Explicitly runs Crawl4AI.
- **`compare` (Benchmark Mode):** Concurrently runs all engines on the same page and produces a side-by-side performance matrix (score, word count, character count, latency).

### 7. Comprehensive Quality Assessment (0–100 Score)
- **Text Density Index:** Ratio of meaningful text against raw markup size.
- **Heading Structure:** Validates semantic hierarchy ($H_1 \to H_2 \to H_3$).
- **Paragraph & Flow Analysis:** Evaluates paragraph count and average length.
- **Lexical Diversity:** Measures vocabulary uniqueness and penalizes keyword stuffing.
- **Short-Text Penalty:** Applies steep penalties for fragments with $<50$ words.
- **Deduplication:** SHA-256 cryptographic fingerprinting across crawl sessions.

### 8. Verified RAG Enhancement & Synthetic Q&A
- **Hallucination Prevention (Evidence Citation):** Every synthetic Q&A pair generated by the LLM retains verbatim evidence quoted directly from the source text.
- **Automated Consistency Check:** Automatically verifies numbers and named entities against source paragraphs.
- **Custom Base URL Support:** Fully compatible with OpenAI, OpenRouter, DeepSeek, Ollama, vLLM, and LocalAI via `CRAWLBOT_OPENAI_BASE_URL`.

---

## 📁 Repository Structure

```text
Crawlbot/
├── run.py                          # Dedicated server launcher (enforces Proactor loop on Windows)
├── app/
│   ├── main.py                     # FastAPI application factory & route mounts
│   ├── api/                        # REST API routing endpoints
│   │   ├── routes_discovery.py     # Web discovery & link crawling
│   │   ├── routes_extraction.py    # Multi-engine extraction & compare APIs
│   │   ├── routes_files.py         # File Processing Studio (JSON/MD upload & processing)
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
│   │   ├── extractors/             # Four independent extraction engines
│   │   │   ├── base.py             # Standardized EngineResult dataclass
│   │   │   ├── trafilatura_engine.py # Engine A: Trafilatura
│   │   │   ├── bs4_engine.py       # Engine B: BeautifulSoup (DOM & Tables)
│   │   │   ├── crawl4ai_engine.py  # Engine C: Crawl4AI (Browser Crawler)
│   │   │   └── playwright_engine.py# Engine D: Native Playwright Chromium (Hydration & Bypass)
│   │   ├── file_processor.py       # File parsing, AI enhancement, chunking & embedding
│   │   ├── loading_detector.py     # Skeleton & animated loading screen detector/bypass
│   │   ├── normalizer.py           # Persian/Arabic text and Unicode normalizer
│   │   ├── page_classifier.py      # Multi-type page classifier (article, docs, spa, about, etc.)
│   │   ├── quality_checker.py      # 0-100 quality scoring & deduplication
│   │   ├── extractor.py            # Unified Coordinator (Auto, Fallback & Compare modes)
│   │   ├── crawler.py              # BFS async crawler orchestrator
│   │   ├── chunker.py              # DocumentChunker (heading, paragraph, sentence, fixed)
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
│       ├── index.html              # Rounded UI with Extraction, Discovery & File tabs
│       ├── css/style.css           # 22px cards, 14px inputs, pill badges
│       └── js/app.js               # Reactive state manager, polling, file upload & exports
├── tests/                          # Automated Pytest suite (234 passing tests)
│   ├── test_ai_features.py         # 36 tests: Chunking, Embedding batching, RAG enhancer
│   ├── test_api_and_integration.py # 13 tests: Discovery, Extraction, and Job endpoints
│   ├── test_chunking_advanced.py   # 16 tests: Advanced heading hierarchy & Persian chunking
│   ├── test_core.py                # 56 tests: SSRF mitigation, IP ranges, sanitization
│   ├── test_extraction_and_cleaning.py # 10 tests: Cleaning, whitespace & quality scoring
│   ├── test_file_processing.py     # 23 tests: JSON/MD uploads, AI, chunking & embeddings
│   ├── test_loading_screen_bypass.py # 10 tests: Loading screen detection, bypass & DOM wait
│   ├── test_mock_website_integration.py # 2 tests: Full site crawl on deterministic mock servers
│   ├── test_normalizer.py          # 16 tests: Persian/Arabic Unicode normalization & ZWNJ
│   ├── test_playwright_engine.py   # 8 tests: Playwright Chromium engine & Proactor fallback
│   ├── test_redirect_and_crawler_safety.py # 8 tests: Redirect cycles and safety guards
│   ├── test_services.py            # 19 tests: Crawler, normalizer, sitemap, robots parser
│   └── test_three_engines_and_pipelines.py # 17 tests: Multi-engine workflows & compare mode
├── output/                         # Local storage for exported files (organized by job_id)
├── logs/                           # Structured system logs
├── requirements.txt                # Python package dependencies
├── pyproject.toml                  # Pytest configuration
├── .env.example                    # Environment configuration template
└── README.md                       # Comprehensive documentation
```

---

## ⚡ Quickstart Guide

### Prerequisites
- Python **3.10+** (tested on 3.10, 3.11, 3.12, 3.13)
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

> **Third-Party LLM Gateways:** You can set `CRAWLBOT_OPENAI_BASE_URL` to any compatible provider such as **OpenRouter** (`https://openrouter.ai/api/v1`), **DeepSeek**, or local inference servers (**Ollama** / **vLLM**).

### 5. Launch Crawlbot Studio

The recommended way to start Crawlbot is using the dedicated runner:
```bash
python run.py
```
> **Note for Windows Users:** `run.py` automatically initializes `asyncio.WindowsProactorEventLoopPolicy()`, ensuring that Playwright's headless Chromium subprocesses execute without `NotImplementedError` issues.

Alternatively, launch directly via Uvicorn:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📡 REST API Reference

Interactive Swagger UI documentation is available at `http://127.0.0.1:8000/docs`.

### Extraction & Benchmark Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/extraction/single` | Instant synchronous extraction of a single URL with specified engine (`auto`, `playwright`, `trafilatura`, `beautifulsoup`, `crawl4ai`) |
| `POST` | `/api/extraction/compare` | Benchmark all engines concurrently on a single URL with quality & latency scores |
| `POST` | `/api/extraction/start` | Start background multi-URL extraction job with custom strategy, render mode, and quality threshold |
| `GET` | `/api/extraction/status/{job_id}` | Poll extraction progress, attempt logs, and document summaries |
| `POST` | `/api/extraction/retry/{job_id}` | Retry failed or blocked URLs from a previous run |
| `POST` | `/api/extraction/cancel/{job_id}` | Terminate an active extraction job |

### File Processing Studio Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/files/process` | Upload JSON, JSONL, or Markdown files and execute AI enhancement, chunking, and/or vector embedding |
| `GET` | `/api/files/supported-formats` | List supported file extensions and accepted schemas |

### Web Discovery Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/discovery/start` | Launch asynchronous BFS link discovery with domain boundaries |
| `GET` | `/api/discovery/status/{job_id}` | Poll discovery progress and discovered URLs |
| `POST` | `/api/discovery/cancel/{job_id}` | Abort an active discovery crawl |

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

## 📦 Output Artifacts & Data Schema

All generated extraction outputs are organized under `output/<job_id>/`:

```text
output/
└── <job_id>/
    ├── documents.json            # Array of all extracted documents with metadata & quality reports
    ├── documents.jsonl           # Line-delimited JSON for stream processing
    ├── markdown/                 # Individual Markdown files with YAML Frontmatter
    │   ├── site_com_about_8f2a1b3c.md
    │   └── site_com_docs_api_e5f6g7h8.md
    └── text/                     # Clean plain text files with extraction headers
        ├── site_com_about_8f2a1b3c.txt
        └── site_com_docs_api_e5f6g7h8.txt
```

### Standardized `ExtractedDocument` JSON Schema
```json
{
  "document_id": "a3b8c2d1e4f50912",
  "source_url": "https://example.com/about",
  "final_url": "https://example.com/about",
  "canonical_url": "https://example.com/about",
  "title": "About Us — Engineering & Innovation",
  "language": "fa",
  "page_type": "about",
  "extractor_used": "playwright",
  "extraction_attempts": [
    {
      "engine": "playwright",
      "status": "success",
      "latency_ms": 2840.5,
      "char_count": 5420,
      "word_count": 755,
      "quality_score": 88
    }
  ],
  "quality_report": {
    "score": 88,
    "text_density": 0.41,
    "heading_count": 45,
    "paragraph_count": 12,
    "passed": true,
    "reasons": [
      "Substantial word count (755 words)",
      "Well-structured with 45 headings",
      "High text density (41.0%)"
    ]
  },
  "content_markdown": "# درباره ما\n\nما تیمی از مهندسان نرم‌افزار هستیم...",
  "content_text": "درباره ما\n\nما تیمی از مهندسان نرم‌افزار هستیم...",
  "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "word_count": 755,
  "character_count": 5420,
  "status": "success"
}
```

---

## 🔒 Enterprise Security Hardening

- **Multi-Hop SSRF Firewall (`app/core/security.py`):** Real-time OS DNS resolution and IP evaluation on every HTTP redirect hop. Blocks loopback addresses (`127.0.0.0/8`, `::1`), private ranges (RFC-1918), AWS/GCP cloud metadata (`169.254.169.254`), link-local, and multicast addresses.
- **Protocol & Scheme Restriction:** Whitelists only `http://` and `https://`. Automatically discards dangerous pseudo-schemes (`file://`, `data:`, `javascript:`).
- **Embedded Credential Stripping:** Strips userinfo components from URLs (`http://user:pass@host`).
- **Path Traversal Defense:** Sanitizes all export filenames and job directory paths using strict alphanumeric allowlists.
- **Resource Leak Prevention:** Automatic context destruction and isolated thread destruction for Chromium instances.

---

## 🧪 Comprehensive Automated Test Suite

Crawlbot is backed by **234 automated unit and integration tests** guaranteeing 100% test coverage and zero regression:

```bash
pytest -v
```

```text
======================= 234 passed, 1 warning in 21.26s =======================
```

| Test Suite | Tests | Scope Covered |
|---|---|---|
| `test_core.py` | 56 | SSRF mitigation, IP range checks, scheme sanitization, path traversal defense |
| `test_ai_features.py` | 36 | Chunking algorithms, overlap calculations, embedding batching, RAG Q&A verification |
| `test_file_processing.py` | 23 | File upload endpoint, JSON/MD parsing, AI optimization, chunking, and embedding |
| `test_services.py` | 19 | BFS crawler mechanics, canonical URL normalizer, XML sitemap parsing, robots parser |
| `test_three_engines_and_pipelines.py` | 17 | Multi-engine workflows, Trafilatura, BS4, Crawl4AI, Auto fallback & Compare mode |
| `test_chunking_advanced.py` | 16 | Semantic heading hierarchy preservation, token budget packing, Persian chunking |
| `test_normalizer.py` | 16 | Persian & Arabic Unicode normalization, ZWNJ rules, digit & punctuation conversion |
| `test_api_and_integration.py` | 13 | RESTful route endpoints, background jobs, cancel/retry APIs, UI asset serving |
| `test_loading_screen_bypass.py` | 10 | Loading screen detection, spinner bypass, DOM settlement, auto-scroll hydration |
| `test_playwright_engine.py` | 8 | Native Playwright Chromium engine, Windows Proactor isolation, error recovery |
| `test_redirect_and_crawler_safety.py` | 8 | Redirect cycle detection, self-redirects, max-redirect caps |
| `test_extraction_and_cleaning.py` | 10 | DOM cleaning, text whitespace normalization, quality status evaluation |
| `test_mock_website_integration.py` | 2 | End-to-end full site crawl and extraction lifecycle on deterministic mock servers |

---

## 📄 License

This project is open-source software licensed under the [MIT License](LICENSE).
