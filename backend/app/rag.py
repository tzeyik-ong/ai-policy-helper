import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .ingest import chunk_text, doc_hash
from .settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedder ready — dim=%d", self.dim)

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode(text, normalize_embeddings=True)
        return v.astype("float32")


class LocalEmbedder:
    """Hash-based fallback embedder (no semantic meaning — for offline tests only)."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha1(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "big") % (2**32 - 1)
        rng = np.random.default_rng(rng_seed)
        v = rng.standard_normal(self.dim).astype("float32")
        v = v / (np.linalg.norm(v) + 1e-9)
        return v


# ---------------------------------------------------------------------------
# Vector stores
# ---------------------------------------------------------------------------


class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes: set = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]) -> None:
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def _cosine_scores(self, query: np.ndarray) -> np.ndarray:
        A = np.vstack(self.vecs)
        q = query.reshape(1, -1)
        return (A @ q.T).ravel() / (
            np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9
        )

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        sims = self._cosine_scores(query)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

    def search_with_vectors(
        self, query: np.ndarray, k: int = 4
    ) -> List[Tuple[float, Dict, np.ndarray]]:
        if not self.vecs:
            return []
        sims = self._cosine_scores(query)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i], self.vecs[i]) for i in idx]


class QdrantStore:
    def __init__(self, collection: str, dim: int = 384):
        from qdrant_client import QdrantClient, models as qm

        self._qm = qm
        self.client = QdrantClient(url="http://qdrant:6333", timeout=10.0)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models as qm

        try:
            info = self.client.get_collection(self.collection)
            if info.config.params.vectors.size != self.dim:
                raise ValueError("dimension mismatch")
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]) -> None:
        from qdrant_client import models as qm

        points = []
        for i, (v, m) in enumerate(zip(vectors, metadatas)):
            h = m.get("hash") or m.get("id") or str(i)
            point_id = str(uuid.UUID(h[:32]))
            points.append(
                qm.PointStruct(id=point_id, vector=v.tolist(), payload=m)
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True,
        )
        return [(float(r.score), dict(r.payload)) for r in res]

    def search_with_vectors(
        self, query: np.ndarray, k: int = 4
    ) -> List[Tuple[float, Dict, np.ndarray]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True,
            with_vectors=True,
        )
        return [
            (float(r.score), dict(r.payload), np.array(r.vector, dtype="float32"))
            for r in res
        ]


# ---------------------------------------------------------------------------
# MMR helper
# ---------------------------------------------------------------------------


def _mmr_rerank(
    candidates: List[Tuple[float, Dict, np.ndarray]],
    k: int,
    lambda_mult: float = 0.5,
) -> List[Dict]:
    """Maximal Marginal Relevance: balance relevance vs. diversity."""
    if len(candidates) <= k:
        return [m for _, m, _ in candidates]

    selected: List[int] = []
    remaining = list(range(len(candidates)))

    while len(selected) < k and remaining:
        best_idx, best_score = None, -float("inf")
        for i in remaining:
            rel, _, vec = candidates[i]
            if selected:
                max_sim = max(
                    float(
                        np.dot(vec, candidates[j][2])
                        / (np.linalg.norm(vec) * np.linalg.norm(candidates[j][2]) + 1e-9)
                    )
                    for j in selected
                )
            else:
                max_sim = 0.0
            score = lambda_mult * rel - (1 - lambda_mult) * max_sim
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i][1] for i in selected]


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

_POLICY_SYSTEM = (
    "You are a helpful company policy assistant. "
    "Cite sources by title and section when relevant."
)


def _build_prompt(query: str, contexts: List[Dict]) -> str:
    prompt = f"Question: {query}\nSources:\n"
    for c in contexts:
        prompt += f"- {c.get('title')} | {c.get('section')}\n{c.get('text', '')[:600]}\n---\n"
    prompt += "Write a concise, accurate answer grounded in the sources. If unsure, say so."
    return prompt


class StubLLM:
    name = "stub"

    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = ["Answer (stub): Based on the following sources:"]
        for c in contexts:
            lines.append(f"  - {c.get('title')} — {c.get('section') or 'Section'}")
        joined = " ".join(c.get("text", "") for c in contexts)
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)

    def stream(self, query: str, contexts: List[Dict]) -> Iterator[str]:
        for word in self.generate(query, contexts).split():
            yield word + " "


class OpenRouterLLM:
    name = "openrouter"

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        import openai

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    def generate(self, query: str, contexts: List[Dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _POLICY_SYSTEM},
                {"role": "user", "content": _build_prompt(query, contexts)},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content

    def stream(self, query: str, contexts: List[Dict]) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _POLICY_SYSTEM},
                {"role": "user", "content": _build_prompt(query, contexts)},
            ],
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaLLM:
    def __init__(self, host: str = "http://host.docker.internal:11434", model: str = "llama3.2"):
        from openai import OpenAI

        self.client = OpenAI(api_key="ollama", base_url=f"{host}/v1")
        self.model = model

    def generate(self, query: str, contexts: List[Dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _POLICY_SYSTEM},
                {"role": "user", "content": _build_prompt(query, contexts)},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content

    def stream(self, query: str, contexts: List[Dict]) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _POLICY_SYSTEM},
                {"role": "user", "content": _build_prompt(query, contexts)},
            ],
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class Metrics:
    def __init__(self) -> None:
        self.t_retrieval: List[float] = []
        self.t_generation: List[float] = []
        self.query_count: int = 0

    def add_retrieval(self, ms: float) -> None:
        self.t_retrieval.append(ms)
        self.query_count += 1

    def add_generation(self, ms: float) -> None:
        self.t_generation.append(ms)

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval) / len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation) / len(self.t_generation) if self.t_generation else 0.0
        return {
            "query_count": self.query_count,
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------


class RAGEngine:
    def __init__(self) -> None:
        # Embedder
        if settings.embedding_model == "local-384":
            self.embedder = LocalEmbedder(dim=384)
            emb_dim = 384
        else:
            self.embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
            emb_dim = self.embedder.dim

        # Vector store
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=emb_dim)
                logger.info("Using QdrantStore")
            except Exception as e:
                logger.warning("Qdrant unavailable (%s), falling back to InMemoryStore", e)
                self.store = InMemoryStore(dim=emb_dim)
        else:
            self.store = InMemoryStore(dim=emb_dim)
            logger.info("Using InMemoryStore")

        # LLM — three possible providers: openrouter | ollama | stub
        if settings.llm_provider == "openrouter" and settings.openrouter_api_key:
            self.llm = OpenRouterLLM(
                api_key=settings.openrouter_api_key, model=settings.llm_model
            )
            self.llm_name = f"openrouter:{settings.llm_model}"
        elif settings.llm_provider == "ollama":
            self.llm = OllamaLLM(host=settings.ollama_host, model=settings.ollama_model)
            self.llm_name = f"ollama:{settings.ollama_model}"
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"
        logger.info("LLM provider: %s", self.llm_name)

        # Fallback is always Stub (both OpenRouter and Ollama fall back to Stub)
        self._fallback_llm = StubLLM()
        self._fallback_name = "stub"

        self.metrics = Metrics()
        self._doc_titles: set = set()
        self._chunk_count: int = 0
        self._cache: Dict[str, Dict] = {}
        self.last_llm_fallback: bool = False
        self.last_llm_fallback_provider: str = ""

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        logger.info("Ingesting %d chunks", len(chunks))
        vectors, metas = [], []
        doc_titles_before = set(self._doc_titles)

        for ch in chunks:
            text = ch["text"]
            h = doc_hash(text)
            meta = {
                "id": h,
                "hash": h,
                "title": ch["title"],
                "section": ch.get("section"),
                "text": text,
            }
            vectors.append(self.embedder.embed(text))
            metas.append(meta)
            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

        self.store.upsert(vectors, metas)
        self._cache.clear()  # invalidate cache after new ingest
        new_docs = len(self._doc_titles) - len(doc_titles_before)
        logger.info("Ingest complete — %d new docs, %d chunks upserted", new_docs, len(metas))
        return new_docs, len(metas)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        t0 = time.time()
        qv = self.embedder.embed(query)

        if settings.mmr_enabled:
            candidates = self.store.search_with_vectors(qv, k=min(k * 3, 20))
            results = _mmr_rerank(candidates, k=k, lambda_mult=settings.mmr_lambda)
            logger.debug("MMR retrieved %d results from %d candidates", len(results), len(candidates))
        else:
            raw = self.store.search(qv, k=k)
            results = [m for _s, m in raw]

        elapsed = (time.time() - t0) * 1000.0
        self.metrics.add_retrieval(elapsed)
        logger.info("Retrieve '%.50s' → %d chunks in %.1f ms", query, len(results), elapsed)
        return results

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, query: str, contexts: List[Dict]) -> str:
        t0 = time.time()
        self.last_llm_fallback = False
        self.last_llm_fallback_provider = ""
        try:
            answer = self.llm.generate(query, contexts)
        except Exception as e:
            logger.warning("LLM error (%s), falling back to %s", e, self._fallback_name)
            self.last_llm_fallback = True
            self.last_llm_fallback_provider = self.llm_name
            try:
                answer = self._fallback_llm.generate(query, contexts)
            except Exception as e2:
                logger.warning("Fallback LLM error (%s), using stub", e2)
                answer = StubLLM().generate(query, contexts)
        elapsed = (time.time() - t0) * 1000.0
        self.metrics.add_generation(elapsed)
        logger.info("Generate completed in %.1f ms", elapsed)
        return answer

    async def stream_generate(
        self, query: str, contexts: List[Dict]
    ) -> AsyncIterator[str]:
        """Bridge synchronous LLM stream() to an async generator."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        self.last_llm_fallback = False
        self.last_llm_fallback_provider = ""

        def _run() -> None:
            try:
                for token in self.llm.stream(query, contexts):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
            except Exception as e:
                logger.warning("Streaming LLM error: %s, falling back to %s", e, self._fallback_name)
                provider = self.llm_name
                loop.call_soon_threadsafe(lambda: setattr(self, "last_llm_fallback", True))
                loop.call_soon_threadsafe(lambda: setattr(self, "last_llm_fallback_provider", provider))
                try:
                    for token in self._fallback_llm.stream(query, contexts):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                except Exception as e2:
                    logger.warning("Fallback streaming error: %s, using stub", e2)
                    for word in StubLLM().generate(query, contexts).split():
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", word + " "))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        threading.Thread(target=_run, daemon=True).start()

        t0 = time.time()
        while True:
            type_, data = await asyncio.wait_for(queue.get(), timeout=60.0)
            if type_ == "done":
                self.metrics.add_generation((time.time() - t0) * 1000.0)
                return
            yield data

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def get_cache(self, query: str) -> Optional[Dict]:
        return self._cache.get(query.strip().lower())

    def set_cache(self, query: str, response: Dict) -> None:
        if settings.cache_enabled:
            self._cache[query.strip().lower()] = response
            logger.debug("Cached response for query: %.50s", query)

    # ------------------------------------------------------------------
    # Stats / health
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": settings.embedding_model,
            "llm_model": self.llm_name,
            **m,
        }

    def qdrant_healthy(self) -> bool:
        if not hasattr(self.store, "client"):
            return False
        try:
            self.store.client.get_collections()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap, heading=d.get("section", "")):
            out.append({"title": d["title"], "section": d["section"], "text": ch})
    return out
