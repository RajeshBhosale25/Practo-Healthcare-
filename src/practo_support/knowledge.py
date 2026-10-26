"""JSON policy ingestion, chunking, local embeddings, and ChromaDB indexing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from practo_support.schemas import PolicyDocument

POLICY_DIR = Path("data/policies")
FIXED_COLLECTION = "policy_fixed_overlap_v1"
SENTENCE_COLLECTION = "policy_sentence_v1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    version: str
    status: str
    topic: str
    text: str


def load_policy_documents(directory: Path = POLICY_DIR) -> list[PolicyDocument]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No policy JSON files found in {directory}")

    documents: list[PolicyDocument] = []
    errors: list[str] = []
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            documents.append(PolicyDocument.model_validate(raw))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            errors.append(f"{path}: {error}")
    if errors:
        raise ValueError("Policy ingestion validation failed:\n" + "\n".join(errors))
    if len({document.source_id for document in documents}) != len(documents):
        raise ValueError("Policy source_id values must be unique")
    return documents


def fixed_size_chunks(
    document: PolicyDocument, *, chunk_size: int = 180, overlap: int = 40
) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = document.text.strip()
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        content = text[start:end]
        if content.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_id}:fixed:{len(chunks)}",
                    source_id=document.source_id,
                    title=document.title,
                    version=document.version,
                    status=document.status,
                    topic=document.topic,
                    text=content,
                )
            )
        if end == len(text):
            break
        start = end - overlap
    return chunks


def sentence_chunks(document: PolicyDocument) -> list[Chunk]:
    sentences = re.split(r"(?<=[.!?])\s+", document.text.strip())
    return [
        Chunk(
            chunk_id=f"{document.source_id}:sentence:{index}",
            source_id=document.source_id,
            title=document.title,
            version=document.version,
            status=document.status,
            topic=document.topic,
            text=sentence.strip(),
        )
        for index, sentence in enumerate(sentences)
        if sentence.strip()
    ]


class LocalSentenceTransformer:
    """SentenceTransformer wrapper that refuses to download a model at runtime."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, local_files_only=True)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.tolist()


def _chunk_documents(
    documents: list[PolicyDocument], strategy: str
) -> list[Chunk]:
    chunker = fixed_size_chunks if strategy == "fixed" else sentence_chunks
    return [chunk for document in documents for chunk in chunker(document)]


def index_policy_collections(
    *,
    directory: Path = POLICY_DIR,
    persist_directory: str = ".chroma",
    embedder: Embedder | None = None,
) -> dict[str, int]:
    import chromadb

    documents = load_policy_documents(directory)
    model = embedder or LocalSentenceTransformer()
    client = chromadb.PersistentClient(path=persist_directory)
    result: dict[str, int] = {}
    for strategy, collection_name in (
        ("fixed", FIXED_COLLECTION),
        ("sentence", SENTENCE_COLLECTION),
    ):
        chunks = _chunk_documents(documents, strategy)
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        embeddings = model.encode(
            [f"{chunk.title}. {chunk.topic}. {chunk.text}" for chunk in chunks]
        )
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "source_id": chunk.source_id,
                    "title": chunk.title,
                    "version": chunk.version,
                    "status": chunk.status,
                    "topic": chunk.topic,
                }
                for chunk in chunks
            ],
        )
        result[collection_name] = len(chunks)
    return result


def main() -> None:
    counts = index_policy_collections()
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
