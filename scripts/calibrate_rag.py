"""Measure cosine separation and write reproducible threshold calibration data."""

from __future__ import annotations

import json
from pathlib import Path

from practo_support.local_rag import calibrated_local_threshold

MEASUREMENTS = {
    "in_scope": [
        "What is the cancellation window for an appointment?",
        "How do I request an insurance claim?",
        "Who is eligible for telemedicine?",
    ],
    "out_of_scope": [
        "Diagnose my rash and prescribe treatment.",
        "What is the meaning of quantum entanglement?",
    ],
}


def main() -> None:
    strategies = {}
    for strategy in ("fixed", "sentence"):
        threshold, similarities = calibrated_local_threshold(
            strategy=strategy, measurements=MEASUREMENTS
        )
        strategies[strategy] = {
            "threshold": threshold,
            "measurements": similarities,
        }
    comparison = json.loads(
        Path("data/eval/rag_comparison.json").read_text(encoding="utf-8")
    )
    recommended = max(
        strategies,
        key=lambda name: (
            comparison[name]["macro_recall"],
            comparison[name]["macro_precision"],
        ),
    )
    result = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "recommended_strategy": recommended,
        "recommendation_basis": (
            "Highest document-level macro recall on the shared five-query set; "
            "macro precision breaks ties."
        ),
        "comparison": {
            strategy: {
                "macro_precision": comparison[strategy]["macro_precision"],
                "macro_recall": comparison[strategy]["macro_recall"],
            }
            for strategy in strategies
        },
        "strategies": strategies,
    }
    output = Path("data/eval/rag_calibration.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
