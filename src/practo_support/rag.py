"""Dual-collection retrieval, grounded MOCK_LLM generation, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from practo_support.knowledge import (
    FIXED_COLLECTION,
    SENTENCE_COLLECTION,
    Embedder,
)

UNKNOWN_ANSWER = "I don't know"


class QueryableCollection(Protocol):
    def query(
        self, *, query_embeddings: list[list[float]], n_results: int
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RetrievedEvidence:
    source_id: str
    title: str
    version: str
    text: str
    cosine_similarity: float


def retrieve(
    query: str,
    collection: QueryableCollection,
    embedder: Embedder,
    *,
    top_k: int = 3,
) -> list[RetrievedEvidence]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    vector = embedder.encode([query])
    result = collection.query(query_embeddings=vector, n_results=top_k)
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metadata = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    evidence = []
    for index, text in enumerate(docs):
        meta = metadata[index] or {}
        distance = float(distances[index]) if distances else 1.0
        evidence.append(
            RetrievedEvidence(
                source_id=str(meta.get("source_id", ids[index])),
                title=str(meta.get("title", "")),
                version=str(meta.get("version", "")),
                text=str(text),
                cosine_similarity=max(-1.0, min(1.0, 1.0 - distance)),
            )
        )
    return evidence


def calibrated_threshold(
    in_scope_similarities: list[float], out_of_scope_similarities: list[float]
) -> float:
    if len(in_scope_similarities) < 3 or len(out_of_scope_similarities) < 2:
        raise ValueError("Need >=3 in-scope and >=2 out-of-scope measurements")
    lower_in_scope = min(in_scope_similarities)
    upper_out_of_scope = max(out_of_scope_similarities)
    if upper_out_of_scope >= lower_in_scope:
        raise ValueError(
            "Measured similarity clusters overlap; threshold calibration is "
            "not supported by these samples"
        )
    return (upper_out_of_scope + lower_in_scope) / 2


def generate_grounded_answer(
    evidence: list[RetrievedEvidence], *, threshold: float
) -> tuple[str, list[dict[str, str]]]:
    if not evidence or evidence[0].cosine_similarity < threshold:
        return UNKNOWN_ANSWER, []
    best = evidence[0]
    return best.text, [
        {"source_id": best.source_id, "title": best.title, "version": best.version}
    ]


def document_metrics(
    relevant_document_ids: set[str],
    retrieved: list[RetrievedEvidence],
) -> dict[str, float | int]:
    retrieved_ids = {item.source_id for item in retrieved}
    relevant_retrieved = relevant_document_ids & retrieved_ids
    precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = (
        len(relevant_retrieved) / len(relevant_document_ids)
        if relevant_document_ids
        else 0.0
    )
    return {
        "relevant_retrieved": len(relevant_retrieved),
        "retrieved_documents": len(retrieved_ids),
        "relevant_documents": len(relevant_document_ids),
        "precision": precision,
        "recall": recall,
    }


def collection_for_name(client: Any, collection_name: str) -> Any:
    if collection_name not in {FIXED_COLLECTION, SENTENCE_COLLECTION}:
        raise ValueError(f"Unknown policy collection: {collection_name}")
    return client.get_collection(collection_name)
