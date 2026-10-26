"""Compare document-level retrieval precision and recall on five shared queries."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from practo_support.knowledge import (
    FIXED_COLLECTION,
    SENTENCE_COLLECTION,
    LocalSentenceTransformer,
)
from practo_support.rag import document_metrics, retrieve

CASES = [
    ("How can I book an appointment?", "POL-BOOKING-001"),
    ("What is the cancellation or rescheduling window?", "POL-CANCEL-002"),
    ("How much does a specialty consultation cost?", "POL-FEES-003"),
    ("How does an insurance claim work?", "POL-INSURANCE-004"),
    ("Can you renew my prescription?", "POL-REFILL-005"),
]


def main() -> None:
    client = chromadb.PersistentClient(path=".chroma")
    embedder = LocalSentenceTransformer()
    results: dict[str, dict[str, object]] = {}
    for strategy, name in (
        ("fixed", FIXED_COLLECTION),
        ("sentence", SENTENCE_COLLECTION),
    ):
        collection = client.get_collection(name)
        queries = []
        for query, relevant_id in CASES:
            retrieved = retrieve(query, collection, embedder, top_k=3)
            metrics = document_metrics({relevant_id}, retrieved)
            queries.append(
                {
                    "query": query,
                    "relevant_document_ids": [relevant_id],
                    "retrieved_document_ids": sorted(
                        {item.source_id for item in retrieved}
                    ),
                    "arithmetic": (
                        f"precision={metrics['relevant_retrieved']}/"
                        f"{metrics['retrieved_documents']}="
                        f"{metrics['precision']:.3f}; recall="
                        f"{metrics['relevant_retrieved']}/"
                        f"{metrics['relevant_documents']}="
                        f"{metrics['recall']:.3f}"
                    ),
                    **metrics,
                }
            )
        results[strategy] = {
            "collection": name,
            "per_query": queries,
            "macro_precision": round(
                sum(row["precision"] for row in queries) / len(queries), 3
            ),
            "macro_recall": round(
                sum(row["recall"] for row in queries) / len(queries), 3
            ),
        }
    output = Path("data/eval/rag_comparison.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
