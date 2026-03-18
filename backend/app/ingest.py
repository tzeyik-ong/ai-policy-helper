import logging
import os
import re
import hashlib
from typing import List, Dict, Tuple

from .settings import settings

logger = logging.getLogger(__name__)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _md_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown by headings, building a H1 > H2 > H3 breadcrumb as the section name."""
    parts = re.split(r"\n(?=#{1,3} )", text)
    out = []
    h1 = h2 = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        m = re.match(r"^(#{1,3})\s+(.*)", lines[0]) if lines else None
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            if level == 1:
                h1, h2 = heading, ""
                section = heading
            elif level == 2:
                h2 = heading
                section = f"{h1} > {h2}" if h1 else heading
            else:
                section = f"{h2} > {heading}" if h2 else (f"{h1} > {heading}" if h1 else heading)
        else:
            section = "Body"
        out.append((section, p))
    return out or [("Body", text)]


def chunk_text(text: str, chunk_size: int, overlap: int, heading: str = "") -> List[str]:
    """Split text into word-count chunks. Non-first chunks are prefixed with the heading
    so each chunk carries enough context for semantic retrieval."""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i : i + chunk_size])
        if i > 0 and heading:
            chunk = f"{heading}\n{chunk}"
        chunks.append(chunk)
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks


def load_documents(data_dir: str) -> List[Dict[str, str]]:
    logger.info("Loading documents from %s", data_dir)
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, fname)
        text = _read_text_file(path)
        for section, body in _md_sections(text):
            docs.append({"title": fname, "section": section, "text": body})
    logger.info("Loaded %d sections from %d files", len(docs), len({d["title"] for d in docs}))
    return docs


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
