# AI Policy & Product Helper

A local-first RAG (Retrieval-Augmented Generation) assistant that answers questions about company policy documents with source citations, streaming responses, and MMR-reranked retrieval.

**Stack:** Next.js 14 → FastAPI → Qdrant · `all-MiniLM-L6-v2` embeddings · OpenRouter / Ollama / Stub LLM

> **Demo note:** The demo video uses **Ollama (`llama3.2`)** as the LLM provider. The OpenRouter API key supplied in the starter pack returns `401 User not found` and is no longer valid, so Ollama was used as the working alternative.

---

## Quick Start (Docker)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com) (optional — for free local LLM fallback)

### 1. Set up Ollama (optional but recommended)

Ollama is used as the fallback LLM when OpenRouter is unavailable. Run these **once** before starting Docker:

```bash
ollama pull llama3.2   # download the model (~2 GB, one-time)
ollama serve           # start the Ollama server (keep this terminal open)
```

> **Windows note:** `make` is not built into Windows.
> - **PowerShell (VS Code default):** install via `winget install GnuWin32.Make`, then add `C:\Program Files (x86)\GnuWin32\bin` to your PATH.
> - **Git Bash:** same install, then `export PATH=$PATH:"/c/Program Files (x86)/GnuWin32/bin"` (add to `~/.bashrc` to persist).
> - **No `make`?** Run the Docker command directly: `docker compose up --build`

### 2. Start all services

```bash
cp .env.example .env          # edit LLM_PROVIDER and API key as needed
docker compose up --build

# Frontend:      http://localhost:3000
# Backend docs:  http://localhost:8000/docs
# Qdrant UI:     http://localhost:6333/dashboard
```

### 3. Use the app

1. **Admin tab** → **Ingest sample docs**
2. **Chat tab** → ask a question, e.g. *"Can a customer return a damaged blender after 20 days?"*
3. Watch the answer stream token-by-token; click a **citation badge** to expand the source chunk
4. Give 👍 / 👎 feedback on each answer

### LLM fallback chain

The backend supports three providers, each falling back to Stub on failure:

```
OpenRouter (gpt-4o-mini)  →  Stub (offline deterministic)
Ollama (llama3.2)         →  Stub (offline deterministic)
Stub                      →  (no fallback needed)
```

The default provider is **Ollama**. Set `LLM_PROVIDER` in `.env` to switch.

---

## Quick Start (No Docker)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
DATA_DIR=../data VECTOR_STORE=memory uvicorn app.main:app --app-dir . --port 8000

# Frontend (separate terminal)
cd frontend
npm install && npm run dev      # http://localhost:3000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                        │
│  ┌──────────────────┐   ┌─────────────────────────────────────┐ │
│  │  AdminPanel      │   │  Chat                               │ │
│  │  stat cards      │   │  streaming · citations · feedback   │ │
│  │  ingest / metrics│   │  auto-scroll · PII warning          │ │
│  └────────┬─────────┘   └──────────────┬──────────────────────┘ │
│           │  Next.js 14 (port 3000)    │                        │
└───────────┼────────────────────────────┼────────────────────────┘
            │ POST /api/ingest           │ POST /api/ask/stream (SSE)
            │ GET  /api/metrics          │ POST /api/ask
            │ GET  /api/health           │ POST /api/feedback
            ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                                   │
│                                                                 │
│  guardrails.py ──▶ mask_pii(query)                              │
│                                                                 │
│  ingest.py                                                      │
│    load_documents() — parse H1>H2>H3 breadcrumb sections        │
│    chunk_text()     — word-count chunks, heading-prefixed        │
│                                                                 │
│  rag.py — RAGEngine                                             │
│    SentenceTransformerEmbedder (all-MiniLM-L6-v2, 384-dim)      │
│    retrieve()    — cosine search + MMR reranking                 │
│    generate()    — LLM call with fallback + query cache          │
│    stream_generate() — async token streaming via thread+queue   │
│                                                                 │
│  LLMs: OpenRouterLLM | OllamaLLM | StubLLM                     │
│        all support generate() + stream()                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ gRPC / HTTP
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Qdrant  (port 6333) — vector DB, cosine similarity, persistent │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Ingest:**
```
data/*.md  →  parse H1/H2/H3 sections  →  word-count chunks (700w, 80 overlap)
           →  all-MiniLM-L6-v2 embed  →  UUID point IDs  →  upsert Qdrant
```

**Query (`/api/ask/stream`):**
```
user query  →  PII masking  →  embed  →  cosine search (k×3)
            →  MMR reranking (λ=0.5)  →  build prompt
            →  LLM stream tokens  →  SSE to browser
            citations sent first so UI shows sources while answer streams
```

---

## Environment Variables

| Variable               | Default                              | Description                                               |
|------------------------|--------------------------------------|-----------------------------------------------------------|
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2`                   | Sentence-transformer model; `local-384` for offline tests |
| `LLM_PROVIDER`         | `ollama`                             | `openrouter` \| `ollama` \| `stub`                        |
| `OPENROUTER_API_KEY`   | *(empty)*                            | Required when `LLM_PROVIDER=openrouter`                   |
| `LLM_MODEL`            | `openai/gpt-4o-mini`                 | OpenRouter model ID (only used with `openrouter`)         |
| `OLLAMA_HOST`          | `http://host.docker.internal:11434`  | Ollama URL (Linux: `http://172.17.0.1:11434`)             |
| `OLLAMA_MODEL`         | `llama3.2`                           | Any model installed via `ollama pull`                     |
| `VECTOR_STORE`         | `qdrant`                             | `qdrant` \| `memory`                                      |
| `CHUNK_SIZE`           | `700`                                | Words per chunk                                           |
| `CHUNK_OVERLAP`        | `80`                                 | Overlapping words between chunks                          |
| `MMR_ENABLED`          | `true`                               | Enable Maximal Marginal Relevance reranking                |
| `MMR_LAMBDA`           | `0.5`                                | MMR λ — 1.0 = pure relevance, 0.0 = pure diversity        |
| `CACHE_ENABLED`        | `true`                               | Cache responses for identical queries                     |
| `FEEDBACK_FILE`        | `/app/data/feedback.jsonl`           | Append-only JSONL feedback log                            |

### Switching LLM providers

```bash
# Ollama (free, local) — default, install from https://ollama.com
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2

# OpenRouter (cloud, requires API key — provided in .env.example)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=openai/gpt-4o-mini   # or any OpenRouter-supported model

# Stub (offline, deterministic — tests / no internet)
LLM_PROVIDER=stub
```

---

## Running Tests

Tests use in-memory vector store and stub LLM — **no Qdrant, no API key, no internet**.

```bash
# Inside Docker (recommended)
docker compose run --rm backend pytest -q

# Outside Docker
cd backend
VECTOR_STORE=memory LLM_PROVIDER=stub pytest -q

# Single test
docker compose run --rm backend pytest -q -k test_citation_accuracy_damaged_blender
```

### Test suite (13 tests)

| Test                                             | Coverage                                 |
|--------------------------------------------------|------------------------------------------|
| `test_health`                                    | Returns `status` + `qdrant` fields       |
| `test_ingest_returns_doc_and_chunk_counts`       | `indexed_docs > 0`, `indexed_chunks > 0` |
| `test_ingest_is_idempotent`                      | Re-ingest does not error                 |
| `test_ask_returns_expected_fields`               | answer, citations, chunks, metrics       |
| `test_empty_query_rejected`                      | 422 on empty / whitespace                |
| `test_citations_have_title_and_section`          | Every citation has a title               |
| `test_chunk_text_is_non_empty`                   | Chunk text non-empty                     |
| `test_citation_accuracy_damaged_blender`         | **Acceptance check 1**                   |
| `test_citation_accuracy_shipping_sla`            | **Acceptance check 2**                   |
| `test_metrics_contains_expected_fields`          | All 7 fields present                     |
| `test_metrics_total_chunks_populated_after_ingest` | `total_chunks > 0`                     |
| `test_metrics_query_count_increments`            | +1 per ask                               |
| `test_metrics_latency_populated_after_query`     | Retrieval + generation > 0               |

---

## Eval Script (acceptance regression)

```bash
# Requires backend to be running
pip install httpx
python scripts/eval.py

# Skip re-ingest if already done
python scripts/eval.py --no-ingest

# Against a different host
python scripts/eval.py --base-url http://localhost:8000
```

---

## API Reference

| Endpoint          | Method | Description                                                                         |
|-------------------|--------|-------------------------------------------------------------------------------------|
| `/api/health`     | GET    | `{status, qdrant}` — real Qdrant connectivity                                       |
| `/api/ingest`     | POST   | Load `data/*.md`, embed, upsert to Qdrant                                           |
| `/api/ask`        | POST   | `{query, k}` → `{answer, citations, chunks, metrics, cached, pii_redacted}`         |
| `/api/ask/stream` | POST   | Same as `/api/ask` but SSE stream (citations → tokens → done)                       |
| `/api/metrics`    | GET    | `{total_docs, total_chunks, query_count, avg_retrieval_ms, avg_generation_ms, …}`   |
| `/api/feedback`   | POST   | `{query, answer, rating:"up"\|"down", comment?}` → logged to JSONL                 |

---

## What Changed from the Starter Pack

### P0 — Critical fixes
- `docker-compose.yml`: removed stray `OPENAI_API_KEY` env var
- `.env` added to `.gitignore`
- Qdrant healthcheck: `curl` → bash TCP check (newer images lack `curl`)
- Qdrant point IDs: SHA-256 hex → UUID (Qdrant rejects plain hex strings)
- Removed obsolete `version:` from `docker-compose.yml`

### P1 — Functionality & correctness
- Replaced `LocalEmbedder` (hash-based) with `SentenceTransformerEmbedder` (`all-MiniLM-L6-v2`, pre-downloaded at build time)
- `_md_sections`: H1 > H2 > H3 breadcrumb section names; non-first chunks prefixed with heading
- `GET /api/metrics`: added `query_count`; all latency fields populated
- `GET /api/health`: real Qdrant ping, `{"status":"ok"|"degraded","qdrant":true|false}`
- Added `OllamaLLM` (OpenAI-compatible endpoint at `/v1`) and `OpenRouterLLM` (cloud)
- LLM fallback chain: primary provider → Stub on any error (instead of returning 500)
- Three providers supported: `openrouter` | `ollama` | `stub`
- `Makefile`: added `ollama-setup` (pull model) and `ollama-serve` targets; `make dev` runs Ollama setup before Docker

### P2 — Testing
- `conftest.py`: sets `VECTOR_STORE=memory` + `LLM_PROVIDER=stub` before app import
- 13 tests — health, ingest, ask shape, both acceptance citation checks, empty query 422, metrics increment, latency

### P3 — Code quality
- Return type annotations on all backend functions
- Pydantic `@field_validator` rejects empty/whitespace queries (422)
- Shared TypeScript types in `frontend/lib/types.ts`; fully typed `api.ts`
- ESLint config (`frontend/.eslintrc.json`)
- `docker-compose.yml` `EMBEDDING_MODEL` default updated to `all-MiniLM-L6-v2`

### P4 — UX polish
- `AdminPanel`: stat card grid, inline error display, `aria-label` on all buttons
- `Chat`: auto-scroll, typing indicator, input disabled while loading, specific error text, `role="log"` + `aria-live="polite"`, `aria-label` on all controls

### P5 — Docs
- This README: setup, architecture diagram, env ref, trade-offs, what's next
- Windows-specific setup notes for PowerShell and Git Bash

### P6 — Performance & observability
- **Structured logging**: `logging.getLogger(__name__)` throughout; logs ingest start/end, retrieval latency, generation latency, LLM errors, cache hits
- **Query-level cache**: `RAGEngine._cache` keyed on normalised query; cleared on ingest; `cached: bool` flag in response
- **Latency**: `retrieval_ms` and `generation_ms` returned per-response in `/api/ask` and at the end of the SSE stream

### P7 — Bonus
- **MMR reranking**: fetch `k×3` candidates, apply Maximal Marginal Relevance (λ=0.5) to pick the `k` most relevant-yet-diverse chunks
- **Streaming**: `POST /api/ask/stream` SSE endpoint; citations sent first, then tokens stream via thread+asyncio.Queue bridge; all three LLMs support `.stream()`
- **PDPA/PII masking**: `guardrails.py` redacts Malaysian IC numbers, phone numbers, and emails before sending to LLM; `pii_redacted: bool` flag shown in UI
- **Feedback logging**: `POST /api/feedback` appends `{ts, query, answer, rating, comment}` to `feedback.jsonl`; 👍 / 👎 buttons on every assistant message
- **Eval script**: `scripts/eval.py` runs 4 acceptance queries, asserts citations, prints metrics summary; usable as a CI regression check

---

## Trade-offs

| Decision                    | Trade-off                                                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `all-MiniLM-L6-v2`          | Good quality at 384-dim; pre-downloaded at build time. Could be replaced with an Ollama-served embedding model (e.g. `nomic-embed-text` via `ollama pull nomic-embed-text`) for a fully Ollama-based stack with no separate Python model download. |
| Word-count chunking         | Simple and reproducible. Sentence-boundary chunking would produce more coherent chunks but requires an NLP tokeniser.       |
| MMR λ=0.5                   | Balanced relevance/diversity. Tunable via `MMR_LAMBDA` env var.                                                             |
| In-process singleton engine | No serialisation overhead; no shared state across replicas. Production: split state to Qdrant + Redis cache.                |
| SSE streaming               | Simpler than WebSockets; works with plain `fetch`. No reconnect logic — acceptable for a demo.                              |
| JSONL feedback              | Zero dependencies; easy to `grep`. Production: write to SQLite or a proper DB with a review UI.                            |
| Hash dedup in InMemoryStore | Prevents re-ingesting the same chunk but doesn't handle content edits. Production: content-addressable versioning.          |

## What I'd Ship Next

1. **File upload** — `POST /api/ingest/upload` for multipart PDF/DOCX ingestion beyond static `data/`
2. **Reranker model** — cross-encoder reranker (e.g. `ms-marco-MiniLM`) after MMR for higher precision
3. **Auth** — API key middleware on `/api/ingest` and `/api/feedback` before any production deploy
4. **Feedback review UI** — simple page to browse `feedback.jsonl` and flag bad answers
5. **CI eval gate** — run `scripts/eval.py` in GitHub Actions on every PR; fail if citation accuracy drops
6. **Observability** — export latency metrics to Prometheus / Grafana; alert on p95 > 2 s
