from pydantic import BaseModel
import os


class Settings(BaseModel):
    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "local-384")

    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # stub | ollama | openrouter
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Vector store
    vector_store: str = os.getenv("VECTOR_STORE", "qdrant")  # qdrant | memory
    collection_name: str = os.getenv("COLLECTION_NAME", "policy_helper")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # Data
    data_dir: str = os.getenv("DATA_DIR", "/app/data")
    feedback_file: str = os.getenv("FEEDBACK_FILE", "/app/data/feedback.jsonl")

    # MMR reranking
    mmr_enabled: bool = os.getenv("MMR_ENABLED", "true").lower() == "true"
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.5"))

    # Query cache
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"


settings = Settings()
