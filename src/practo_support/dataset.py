"""Deterministic synthetic appointment generation and rubric summary."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

SEED = 271828
RECORD_COUNT = 50
CATEGORY_WEIGHTS = {
    "General Medicine": 0.40,
    "Cardiology": 0.20,
    "Dermatology": 0.16,
    "Pediatrics": 0.14,
    "Orthopedics": 0.10,
}
STATUS_WEIGHTS = {
    "Scheduled": 0.36,
    "Completed": 0.24,
    "Cancelled": 0.14,
    "No-Show": 0.10,
    "Rescheduled": 0.16,
}
FOLLOW_UP_PROBABILITY = 0.20
CATEGORY_FEES_INR = {
    "General Medicine": (300, 800),
    "Cardiology": (700, 2000),
    "Dermatology": (500, 1500),
    "Pediatrics": (400, 1200),
    "Orthopedics": (600, 1800),
}


def _weighted_values(weights: dict[str, float], count: int) -> list[str]:
    """Allocate weighted counts by largest remainder, then shuffle later."""
    raw = {key: value * count for key, value in weights.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(raw, key=lambda item: raw[item] - quotas[item], reverse=True)[
        :remaining
    ]:
        quotas[key] += 1
    return [key for key, quota in quotas.items() for _ in range(quota)]


def generate_appointments(seed: int = SEED) -> list[dict[str, Any]]:
    """Generate reproducible synthetic records and reject seeds outside the rubric."""
    rng = random.Random(seed)
    categories = _weighted_values(CATEGORY_WEIGHTS, RECORD_COUNT)
    statuses = _weighted_values(STATUS_WEIGHTS, RECORD_COUNT)
    rng.shuffle(categories)
    rng.shuffle(statuses)

    appointments: list[dict[str, Any]] = []
    for index in range(RECORD_COUNT):
        category = categories[index]
        lower_fee, upper_fee = CATEGORY_FEES_INR[category]
        appointments.append(
            {
                "record_id": f"APT-{index + 1:04d}",
                "category": category,
                "status": statuses[index],
                "consultation_fee_inr": rng.randrange(lower_fee, upper_fee + 1, 50),
                "days_since_created": rng.randint(0, 30),
                "follow_up_required": rng.random() < FOLLOW_UP_PROBABILITY,
            }
        )
    validate_appointments(appointments)
    return appointments


def validate_appointments(appointments: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate every structural threshold and return its reproducible summary."""
    if len(appointments) < 40:
        raise ValueError("The appointment dataset must contain at least 40 records")

    required_fields = {
        "record_id",
        "category",
        "status",
        "consultation_fee_inr",
        "days_since_created",
        "follow_up_required",
    }
    for index, record in enumerate(appointments, start=1):
        if set(record) != required_fields:
            raise ValueError(
                f"Record {index} must contain exactly the required appointment fields"
            )
        if record["category"] not in CATEGORY_WEIGHTS:
            raise ValueError(f"Record {index} has an unsupported category")
        if record["status"] not in STATUS_WEIGHTS:
            raise ValueError(f"Record {index} has an unsupported appointment status")
        lower_fee, upper_fee = CATEGORY_FEES_INR[record["category"]]
        fee = record["consultation_fee_inr"]
        if type(fee) is not int or not lower_fee <= fee <= upper_fee or fee % 50:
            raise ValueError(f"Record {index} has an invalid consultation fee")
        days = record["days_since_created"]
        if type(days) is not int or not 0 <= days <= 30:
            raise ValueError(f"Record {index} has an invalid days_since_created value")
        if type(record["follow_up_required"]) is not bool:
            raise ValueError(f"Record {index} follow_up_required must be a boolean")

    record_ids = [record["record_id"] for record in appointments]
    if not all(isinstance(record_id, str) and record_id for record_id in record_ids):
        raise ValueError("Appointment record IDs must be non-empty strings")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Appointment record IDs must be unique")

    summary = summarize_appointments(appointments)
    underrepresented_categories = [
        category
        for category in CATEGORY_WEIGHTS
        if summary["by_category"].get(category, 0) < 3
    ]
    if underrepresented_categories:
        raise ValueError(
            "Every required category must have at least three records; "
            f"underrepresented: {underrepresented_categories}"
        )
    missing_statuses = [
        status for status in STATUS_WEIGHTS if summary["by_status"].get(status, 0) < 1
    ]
    if missing_statuses:
        raise ValueError(f"Required statuses are missing: {missing_statuses}")
    if not 10 <= summary["follow_up_true_percent"] <= 30:
        raise ValueError(
            "Seeded follow-up draw must produce 10%-30% follow-up records; "
            f"observed {summary['follow_up_true_percent']}%"
        )
    return summary


def summarize_appointments(
    appointments: list[dict[str, Any]],
) -> dict[str, Any]:
    follow_up_count = sum(bool(item["follow_up_required"]) for item in appointments)
    total = len(appointments)
    return {
        "total": total,
        "by_category": dict(sorted(Counter(item["category"] for item in appointments).items())),
        "by_status": dict(sorted(Counter(item["status"] for item in appointments).items())),
        "follow_up_true_count": follow_up_count,
        "follow_up_true_percent": round(100 * follow_up_count / total, 2) if total else 0.0,
    }


APPOINTMENTS = generate_appointments()


def main() -> None:
    summary = summarize_appointments(APPOINTMENTS)
    print(json.dumps(summary, indent=2, sort_keys=True))
    output = Path("data") / "fixtures" / "appointments.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(APPOINTMENTS, indent=2), encoding="utf-8")
    print(f"Wrote {len(APPOINTMENTS)} synthetic records to {output}")


if __name__ == "__main__":
    main()
