"""Persistent local ChromaDB adapter for the support service."""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb

from practo_support.knowledge import (
    FIXED_COLLECTION,
    LocalSentenceTransformer,
    SENTENCE_COLLECTION,
)
from practo_support.rag import RetrievedEvidence, retrieve


class LocalPolicyRetriever:
    def __init__(
        self,
        *,
        strategy: str = "sentence",
        persist_directory: str = ".chroma",
        threshold: float,
    ) -> None:
        collections = {
            "fixed": FIXED_COLLECTION,
            "sentence": SENTENCE_COLLECTION,
        }
        if strategy not in collections:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_collection(collections[strategy])
        self.embedder = LocalSentenceTransformer()
        self.threshold = threshold
        self.corpus_version = hashlib.sha256(
            f"{strategy}:{self.collection.count()}".encode("utf-8")
        ).hexdigest()

    def retrieve(self, query: str) -> list[RetrievedEvidence]:
        ranked = retrieve(query, self.collection, self.embedder, top_k=3)
        evidence: list[RetrievedEvidence] = []
        seen: set[str] = set()
        for item in ranked:
            if item.cosine_similarity < self.threshold or item.source_id in seen:
                continue
            seen.add(item.source_id)
            chunks = self.collection.get(
                where={"source_id": item.source_id},
                include=["documents", "metadatas"],
            )
            ordered = sorted(
                zip(chunks["ids"], chunks["documents"]),
                key=lambda chunk: int(chunk[0].rsplit(":", 1)[-1]),
            )
            full_text = ""
            for _, text in ordered:
                if not full_text:
                    full_text = text
                    continue
                overlap = min(len(full_text), len(text))
                while overlap and full_text[-overlap:] != text[:overlap]:
                    overlap -= 1
                full_text += text[overlap:] if overlap else f" {text}"
            evidence.append(
                RetrievedEvidence(
                    source_id=item.source_id,
                    title=item.title,
                    version=item.version,
                    text=full_text,
                    cosine_similarity=item.cosine_similarity,
                )
            )
        return evidence


def calibrated_local_threshold(
    *,
    strategy: str,
    measurements: dict[str, list[str]],
    persist_directory: str = ".chroma",
) -> tuple[float, dict[str, list[float]]]:
    from practo_support.rag import calibrated_threshold

    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = FIXED_COLLECTION if strategy == "fixed" else SENTENCE_COLLECTION
    collection = client.get_collection(collection_name)
    embedder = LocalSentenceTransformer()
    similarities: dict[str, list[float]] = {}
    for group, queries in measurements.items():
        scores = []
        for query in queries:
            found = retrieve(query, collection, embedder, top_k=1)
            scores.append(found[0].cosine_similarity if found else -1.0)
        similarities[group] = scores
    threshold = calibrated_threshold(
        similarities["in_scope"], similarities["out_of_scope"]
    )
    return threshold, similarities
