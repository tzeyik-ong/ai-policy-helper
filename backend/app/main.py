import json
import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .guardrails import mask_pii
from .ingest import load_documents
from .models import (
    AskRequest,
    AskResponse,
    Citation,
    Chunk,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    MetricsResponse,
)
from .rag import RAGEngine, build_chunks_from_docs
from .settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Policy & Product Helper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    qdrant_ok = engine.qdrant_healthy()
    return {"status": "ok" if qdrant_ok else "degraded", "qdrant": qdrant_ok}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**engine.stats())


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@app.post("/api/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    logger.info("Ingest triggered")
    docs = load_documents(settings.data_dir)
    chunks = build_chunks_from_docs(docs, settings.chunk_size, settings.chunk_overlap)
    new_docs, new_chunks = engine.ingest_chunks(chunks)
    return IngestResponse(indexed_docs=new_docs, indexed_chunks=new_chunks)


# ---------------------------------------------------------------------------
# Ask (standard — with cache)
# ---------------------------------------------------------------------------


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    clean_query = mask_pii(req.query)
    pii_redacted = clean_query != req.query

    # Cache hit
    cached_data = engine.get_cache(clean_query)
    if cached_data:
        logger.info("Cache hit for query: %.60s", clean_query)
        return AskResponse(**{**cached_data, "cached": True, "pii_redacted": pii_redacted})

    ctx = engine.retrieve(clean_query, k=req.k or 4)
    answer = engine.generate(clean_query, ctx)
    citations = [Citation(title=c.get("title"), section=c.get("section")) for c in ctx]
    chunks = [Chunk(title=c.get("title"), section=c.get("section"), text=c.get("text", "")) for c in ctx]
    stats = engine.stats()

    response = AskResponse(
        query=req.query,
        answer=answer,
        citations=citations,
        chunks=chunks,
        metrics={
            "retrieval_ms": stats["avg_retrieval_latency_ms"],
            "generation_ms": stats["avg_generation_latency_ms"],
        },
        cached=False,
        pii_redacted=pii_redacted,
        llm_fallback=engine.last_llm_fallback,
        llm_fallback_provider=engine.last_llm_fallback_provider,
    )
    engine.set_cache(clean_query, response.model_dump())
    return response


# ---------------------------------------------------------------------------
# Ask (streaming — SSE, no cache)
# ---------------------------------------------------------------------------


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    clean_query = mask_pii(req.query)
    pii_redacted = clean_query != req.query

    async def event_stream():
        # 1. Retrieve chunks
        import asyncio

        ctx = await asyncio.to_thread(engine.retrieve, clean_query, req.k or 4)
        citations = [
            {"title": c.get("title"), "section": c.get("section")} for c in ctx
        ]
        chunks_data = [
            {"title": c.get("title"), "section": c.get("section"), "text": c.get("text", "")}
            for c in ctx
        ]

        # 2. Send citations immediately so UI can show sources while answer streams
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "citations",
                    "citations": citations,
                    "chunks": chunks_data,
                    "pii_redacted": pii_redacted,
                }
            )
            + "\n\n"
        )

        # 3. Stream LLM tokens
        t0 = time.time()
        try:
            async for token in engine.stream_generate(clean_query, ctx):
                yield "data: " + json.dumps({"type": "token", "content": token}) + "\n\n"
        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"

        # 4. Done event with final metrics
        stats = engine.stats()
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "done",
                    "metrics": {
                        "retrieval_ms": stats["avg_retrieval_latency_ms"],
                        "generation_ms": round((time.time() - t0) * 1000, 2),
                    },
                    "llm_fallback": engine.last_llm_fallback,
                    "llm_fallback_provider": engine.last_llm_fallback_provider,
                }
            )
            + "\n\n"
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": req.query,
        "answer": req.answer[:500],
        "rating": req.rating,
        "comment": req.comment,
    }
    try:
        with open(settings.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Feedback logged: %s for query: %.50s", req.rating, req.query)
    except Exception as e:
        logger.warning("Could not write feedback: %s", e)
    return FeedbackResponse(status="ok")
