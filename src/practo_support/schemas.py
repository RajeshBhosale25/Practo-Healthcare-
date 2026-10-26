"""Validated API, policy, tool, and orchestration contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AppointmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^APT-\d{4}$")
    category: Literal[
        "General Medicine",
        "Cardiology",
        "Dermatology",
        "Pediatrics",
        "Orthopedics",
    ]
    status: Literal["Scheduled", "Completed", "Cancelled", "No-Show", "Rescheduled"]
    consultation_fee_inr: int = Field(ge=0)
    days_since_created: int = Field(ge=0, le=30)
    follow_up_required: bool


class PolicyDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    owner: str
    effective_date: str
    review_date: str
    version: str
    access_classification: str
    status: Literal["current", "superseded", "withdrawn"]
    source_location: str
    synthetic: bool
    topic: str
    text: str = Field(min_length=1)


class SupportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    conversation_id: str = Field(min_length=1, max_length=120)
    record_id: str | None = Field(default=None, pattern=r"^APT-\d{4}$")


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    version: str


class AppointmentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^APT-\d{4}$")
    status: Literal["Scheduled", "Completed", "Cancelled", "No-Show", "Rescheduled"]
    consultation_fee_inr: int = Field(ge=0)


class SupportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    status: Literal["answered", "clarification_needed", "refused", "error"]
    answer: str
    intent: Literal["policy", "appointment", "mixed", "out_of_scope"]
    sources: list[SourceReference] = Field(default_factory=list)
    appointment: AppointmentSummary | None = None
    escalation_score: float | None = Field(default=None, ge=0, le=1)
    cache_status: Literal["hit", "miss", "bypass"] | None = None

    @model_validator(mode="after")
    def validate_appointment_fields(self) -> "SupportResponse":
        if self.intent != "appointment" and self.appointment is not None:
            raise ValueError("Appointment details require appointment intent")
        return self


class CrewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    intent: Literal["policy", "appointment", "mixed", "out_of_scope"]
    source_ids: list[str] = Field(default_factory=list)
    appointment_status: str | None = None
    escalation_score: float | None = Field(default=None, ge=0, le=1)


class VerdictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    final_answer: str
    reason: str


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[str]
    source_ids: list[str]


class LookupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    record_id: str
    status: str | None = None
    consultation_fee_inr: int | None = None
    escalation_score: float | None = Field(default=None, ge=0, le=1)


class EvaluationScore(BaseModel):
    query_id: str
    query: str
    accuracy: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)


class ErrorResponse(BaseModel):
    detail: str
