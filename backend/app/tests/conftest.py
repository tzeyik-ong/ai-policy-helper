import os
from pathlib import Path

# Override settings before any app modules are imported.
# This makes all tests run with in-memory store + stub LLM so no Qdrant is needed.
os.environ["VECTOR_STORE"] = "memory"
os.environ["LLM_PROVIDER"] = "stub"
# Use real semantic embedder so citation accuracy tests are meaningful.
# The model is pre-downloaded in the Docker image; outside Docker it will download on first run.
os.environ.setdefault("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Resolve data dir: works both outside Docker (relative path) and inside Docker (/app/data default).
_data_dir = Path(__file__).parents[3] / "data"
if _data_dir.exists():
    os.environ.setdefault("DATA_DIR", str(_data_dir))

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)
