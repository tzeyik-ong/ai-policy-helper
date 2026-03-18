from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator


class IngestResponse(BaseModel):
    indexed_docs: int
    indexed_chunks: int


class AskRequest(BaseModel):
    query: str
    k: int | None = 4

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v


class Citation(BaseModel):
    title: str
    section: str | None = None


class Chunk(BaseModel):
    title: str
    section: str | None = None
    text: str


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    chunks: List[Chunk]
    metrics: Dict[str, Any]
    cached: bool = False
    pii_redacted: bool = False
    llm_fallback: bool = False
    llm_fallback_provider: str = ""


class MetricsResponse(BaseModel):
    total_docs: int
    total_chunks: int
    query_count: int
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    embedding_model: str
    llm_model: str


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
