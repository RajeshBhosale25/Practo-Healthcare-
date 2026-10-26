"""Deterministic input/output checks and least-privilege helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTACT_PATTERN = re.compile(r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)")
INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|system)\s+instructions", re.I),
    re.compile(r"reveal\s+(the\s+)?(system prompt|hidden prompt|secrets?)", re.I),
    re.compile(r"act\s+as\s+system", re.I),
)
OUT_OF_SCOPE_PATTERNS = (
    re.compile(r"\b(diagnos(e|is)|prescrib(e|ing)|which medicine|what medication)\b", re.I),
    re.compile(
        r"\b(another|someone else's|other)\s+patient\b.*\b(appointment|record|visit)\b",
        re.I,
    ),
)


@dataclass(frozen=True)
class GuardrailResult:
    text: str
    masked_contact: bool
    injection_detected: bool
    scope_violation: bool
    blocked: bool


def mask_contact_numbers(text: str) -> tuple[str, bool]:
    def mask(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return "*" * max(0, len(match.group(0)) - 4) + digits[-4:]

    masked, count = CONTACT_PATTERN.subn(mask, text)
    return masked, count > 0


def input_guardrail(text: str) -> GuardrailResult:
    masked, did_mask = mask_contact_numbers(text)
    injection = any(pattern.search(masked) for pattern in INJECTION_PATTERNS)
    scope_violation = any(
        pattern.search(masked) for pattern in OUT_OF_SCOPE_PATTERNS
    )
    return GuardrailResult(
        text=masked,
        masked_contact=did_mask,
        injection_detected=injection,
        scope_violation=scope_violation,
        blocked=injection or scope_violation,
    )


def groundedness_guardrail(answer: str, evidence: list[str]) -> bool:
    """Accept a generated statement only when its factual text is in evidence."""
    normalized_answer = " ".join(answer.casefold().split())
    normalized_evidence = " ".join(" ".join(e.casefold().split()) for e in evidence)
    if answer.strip() == "I don't know":
        return True
    return bool(normalized_answer) and normalized_answer in normalized_evidence


def appointment_tool_allowed(agent_name: str) -> bool:
    return agent_name.casefold() == "lookup"


def estimate_request_cost(query: str, *, rate_per_1k_tokens: float = 0.002) -> float:
    estimated_tokens = max(1, len(query) // 4)
    return estimated_tokens / 1000 * rate_per_1k_tokens


def enforce_cost_budget(query: str, budget_usd: float) -> float:
    estimated = estimate_request_cost(query)
    if estimated > budget_usd:
        raise ValueError(
            f"Estimated request cost ${estimated:.6f} exceeds "
            f"the ${budget_usd:.6f} budget"
        )
    return estimated
