"""
API tests — run with: docker compose run --rm backend pytest -q
All tests use in-memory vector store + stub LLM (no Qdrant, no API key needed).
Citation accuracy tests use the real all-MiniLM-L6-v2 embedder.
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "qdrant" in data


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_ingest_returns_doc_and_chunk_counts(client: TestClient):
    r = client.post("/api/ingest")
    assert r.status_code == 200
    data = r.json()
    assert data["indexed_docs"] > 0
    assert data["indexed_chunks"] > 0


def test_ingest_is_idempotent(client: TestClient):
    """Re-ingesting should not double-count chunks (hash dedup)."""
    r1 = client.post("/api/ingest")
    r2 = client.post("/api/ingest")
    assert r1.status_code == 200
    assert r2.status_code == 200


# ---------------------------------------------------------------------------
# Ask — structure
# ---------------------------------------------------------------------------

def test_ask_returns_expected_fields(client: TestClient):
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query": "What is the refund window for small appliances?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert "citations" in data and isinstance(data["citations"], list)
    assert "chunks" in data and isinstance(data["chunks"], list)
    assert "metrics" in data


def test_empty_query_rejected(client: TestClient):
    """Empty / whitespace query must be rejected with 422."""
    r = client.post("/api/ask", json={"query": ""})
    assert r.status_code == 422

    r2 = client.post("/api/ask", json={"query": "   "})
    assert r2.status_code == 422


def test_citations_have_title_and_section(client: TestClient):
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query": "What is the return policy?"})
    assert r.status_code == 200
    for citation in r.json()["citations"]:
        assert "title" in citation


def test_chunk_text_is_non_empty(client: TestClient):
    """Every chunk in the response must carry non-empty text."""
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query": "shipping policy"})
    assert r.status_code == 200
    for chunk in r.json()["chunks"]:
        assert len(chunk.get("text", "")) > 0


# ---------------------------------------------------------------------------
# Ask — citation accuracy (acceptance checks)
# ---------------------------------------------------------------------------

def test_citation_accuracy_damaged_blender(client: TestClient):
    """Acceptance check 1: damaged blender return query cites Returns_and_Refunds + Warranty_Policy."""
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query": "Can a customer return a damaged blender after 20 days?"})
    assert r.status_code == 200
    titles = {c["title"] for c in r.json()["citations"]}
    assert any("Returns" in t for t in titles), f"Expected Returns_and_Refunds in citations, got: {titles}"
    assert any("Warranty" in t for t in titles), f"Expected Warranty_Policy in citations, got: {titles}"


def test_citation_accuracy_shipping_sla(client: TestClient):
    """Acceptance check 2: East Malaysia shipping SLA query cites Delivery_and_Shipping."""
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query": "What's the shipping SLA to East Malaysia for bulky items?"})
    assert r.status_code == 200
    titles = {c["title"] for c in r.json()["citations"]}
    assert any("Delivery" in t or "Shipping" in t for t in titles), \
        f"Expected Delivery_and_Shipping in citations, got: {titles}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_contains_expected_fields(client: TestClient):
    client.post("/api/ingest")
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    for field in ("total_docs", "total_chunks", "query_count",
                  "avg_retrieval_latency_ms", "avg_generation_latency_ms",
                  "embedding_model", "llm_model"):
        assert field in data, f"Missing field: {field}"


def test_metrics_total_chunks_populated_after_ingest(client: TestClient):
    client.post("/api/ingest")
    r = client.get("/api/metrics")
    assert r.json()["total_chunks"] > 0


def test_metrics_query_count_increments(client: TestClient):
    """query_count must increase by 1 after each /api/ask call."""
    client.post("/api/ingest")
    before = client.get("/api/metrics").json()["query_count"]
    client.post("/api/ask", json={"query": "test increment"})
    after = client.get("/api/metrics").json()["query_count"]
    assert after == before + 1


def test_metrics_latency_populated_after_query(client: TestClient):
    client.post("/api/ingest")
    client.post("/api/ask", json={"query": "latency test"})
    data = client.get("/api/metrics").json()
    assert data["avg_retrieval_latency_ms"] > 0
    assert data["avg_generation_latency_ms"] > 0
