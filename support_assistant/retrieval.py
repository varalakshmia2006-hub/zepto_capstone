"""Policy ingestion, local embedding, ChromaDB storage, and retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "zepto_policy"
MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def load_documents() -> list[Chunk]:
    chunks = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        chunks.append(Chunk(path.stem, path.name, text))
    if len(chunks) != 8:
        raise RuntimeError(f"Expected exactly 8 policy documents, found {len(chunks)}")
    return chunks


def _fallback_embedding(text: str, dimensions: int = 384) -> list[float]:
    values = np.zeros(dimensions, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        values[index] += 1.0 if digest[4] % 2 else -1.0
    norm = np.linalg.norm(values)
    return (values / norm if norm else values).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        return model.encode(texts, normalize_embeddings=True).tolist()
    except Exception as error:
        print(f"SentenceTransformer unavailable ({error}); using deterministic local fallback embeddings.")
        return [_fallback_embedding(text) for text in texts]


def collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def build_index() -> None:
    chunks = load_documents()
    target = collection()
    embeddings = embed_texts([chunk.text for chunk in chunks])
    target.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[{"source": chunk.source} for chunk in chunks],
        embeddings=embeddings,
    )
    print(f"Indexed {len(chunks)} policy documents in ChromaDB collection {COLLECTION_NAME!r}.")


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    result = collection().query(query_embeddings=embed_texts([question]), n_results=top_k)
    documents = result.get("documents", [[]])[0]
    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    return [{"id": chunk_id, "text": text, "source": metadata.get("source", "unknown")} for chunk_id, text, metadata in zip(ids, documents, metadatas)]


if __name__ == "__main__":
    build_index()
