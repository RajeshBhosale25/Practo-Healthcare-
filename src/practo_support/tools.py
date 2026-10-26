"""Appointment lookup and dataset-derived escalation scoring."""

from __future__ import annotations

import math
from typing import Any

from practo_support.dataset import APPOINTMENTS, summarize_appointments


def escalation_score(record: dict[str, Any]) -> float:
    follow_up = int(bool(record["follow_up_required"]))
    normalized_age = int(record["days_since_created"]) / 30
    return round(0.6 * follow_up + 0.4 * normalized_age, 4)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values or not 0 <= percentile_value <= 100:
        raise ValueError("Values must be non-empty and percentile in [0, 100]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


ESCALATION_RECENCY_DAY_THRESHOLD = percentile(
    [float(record["days_since_created"]) for record in APPOINTMENTS], 80
)
ESCALATION_THRESHOLD = round(
    0.4 * ESCALATION_RECENCY_DAY_THRESHOLD / 30, 4
)


def check_appointment_status(record_id: str) -> dict[str, Any]:
    """Look up one synthetic appointment and return its support-safe fields."""
    record = next(
        (appointment for appointment in APPOINTMENTS if appointment["record_id"] == record_id),
        None,
    )
    if record is None:
        return {"found": False, "record_id": record_id, "error": "not_found"}
    score = escalation_score(record)
    return {
        "found": True,
        "record_id": record["record_id"],
        "status": record["status"],
        "consultation_fee_inr": record["consultation_fee_inr"],
        "escalation_score": score,
        "escalation_recommended": score >= ESCALATION_THRESHOLD,
    }


def dataset_escalation_summary() -> dict[str, Any]:
    scores = [escalation_score(record) for record in APPOINTMENTS]
    summary = summarize_appointments(APPOINTMENTS)
    summary.update(
        {
            "formula": "0.6 * int(follow_up_required) + 0.4 * (days_since_created / 30)",
            "threshold_percentile": 80,
            "threshold_basis": "80th percentile of days_since_created",
            "threshold_days_since_created": ESCALATION_RECENCY_DAY_THRESHOLD,
            "threshold": ESCALATION_THRESHOLD,
            "escalation_recommended_count": sum(
                score >= ESCALATION_THRESHOLD for score in scores
            ),
            "score_min": min(scores),
            "score_max": max(scores),
            "score_mean": round(sum(scores) / len(scores), 4),
        }
    )
    return summary
