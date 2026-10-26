"""Application service composing guardrails, retrieval, lookup, and memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from practo_support.cache import GroundedResponseCache
from practo_support.governance import (
    enforce_cost_budget,
    groundedness_guardrail,
    input_guardrail,
)
from practo_support.memory import SessionMemory
from practo_support.rag import RetrievedEvidence, generate_grounded_answer
from practo_support.schemas import CrewResponse, SupportRequest, SupportResponse
from practo_support.tools import check_appointment_status
from practo_support.autogen_review import review_draft_sync
from practo_support.crew import run_crew


class Retriever(Protocol):
    corpus_version: str

    def retrieve(self, query: str) -> list[RetrievedEvidence]: ...

    @property
    def threshold(self) -> float: ...


@dataclass
class GroundedDraft:
    answer: str
    sources: list[dict[str, str]]
    evidence: list[str]


class SupportService:
    def __init__(
        self,
        retriever: Retriever,
        *,
        memory: SessionMemory | None = None,
        cache: GroundedResponseCache | None = None,
        cost_budget_usd: float = 0.0005,
    ) -> None:
        self.retriever = retriever
        self.memory = memory or SessionMemory()
        self.cache = cache or GroundedResponseCache()
        self.cost_budget_usd = cost_budget_usd
        self.metrics: dict[str, int] = {}
        self.rag_calls = 0
        self.generation_calls = 0
        self.lookup_calls = 0
        self.crew_calls = 0
        self.review_calls = 0

    def _retrieve_policy(self, query: str) -> GroundedDraft:
        self.rag_calls += 1
        evidence = self.retriever.retrieve(query)
        self.generation_calls += 1
        answer, sources = generate_grounded_answer(
            evidence, threshold=self.retriever.threshold
        )
        draft = GroundedDraft(
            answer=answer,
            sources=sources,
            evidence=[item.text for item in evidence],
        )
        return draft

    def ask(self, request: SupportRequest, *, trace_id: str | None = None) -> SupportResponse:
        trace_id = trace_id or str(uuid4())
        safe_input = input_guardrail(request.query)
        if safe_input.blocked:
            answer = (
                "I can't help with requests to override safeguards or reveal private instructions."
                if safe_input.injection_detected
                else "I don't know"
            )
            response = SupportResponse(
                trace_id=trace_id,
                status="refused",
                answer=answer,
                intent="out_of_scope",
            )
            self.memory.add_turn(request.conversation_id, safe_input.text, response.answer)
            return response

        try:
            enforce_cost_budget(safe_input.text, self.cost_budget_usd)
        except ValueError:
            response = SupportResponse(
                trace_id=trace_id,
                status="refused",
                answer="This request exceeds the service's configured processing limit.",
                intent="out_of_scope",
            )
            self.memory.add_turn(request.conversation_id, safe_input.text, response.answer)
            return response

        record_id = request.record_id or self._extract_record_id(safe_input.text)
        if record_id:
            lookup_result: dict[str, Any] = {}

            def lookup_callback(value: str) -> dict[str, Any]:
                self.lookup_calls += 1
                lookup_result.update(check_appointment_status(value))
                return lookup_result

            self.crew_calls += 1
            crew_result = run_crew(
                f"{safe_input.text} {record_id}",
                rag_callback=lambda value: {},
                use_lookup=True,
                lookup_callback=lookup_callback,
            )
            result = lookup_result
            if not result["found"]:
                response = SupportResponse(
                    trace_id=trace_id,
                    status="answered",
                    answer="I couldn't find that appointment reference.",
                    intent="appointment",
                )
            else:
                self.review_calls += 1
                verdict = review_draft_sync(
                    query=safe_input.text,
                    draft=crew_result.answer,
                    evidence=[crew_result.answer],
                )
                response = SupportResponse(
                    trace_id=trace_id,
                    status="answered",
                    answer=verdict.final_answer if verdict.approved else "I don't know",
                    intent="appointment",
                    appointment={
                        "record_id": record_id,
                        "status": result["status"],
                        "consultation_fee_inr": result["consultation_fee_inr"],
                    },
                    escalation_score=result["escalation_score"],
                )
            self.memory.add_turn(request.conversation_id, safe_input.text, response.answer)
            return response

        previous_messages = self.memory.messages(request.conversation_id)
        if previous_messages and any(
            phrase in safe_input.text.casefold()
            for phrase in ("repeat that", "what did you just say", "say that again")
        ):
            previous_answer = next(
                (
                    message.content
                    for message in reversed(previous_messages)
                    if message.type == "ai"
                ),
                None,
            )
            if previous_answer:
                response = SupportResponse(
                    trace_id=trace_id,
                    status="answered",
                    answer=str(previous_answer),
                    intent="policy",
                    cache_status="bypass",
                )
                self.memory.add_turn(
                    request.conversation_id, safe_input.text, response.answer
                )
                return response

        draft = self.cache.get(safe_input.text, self.retriever.corpus_version)
        cache_status = "hit" if draft is not None else "miss"
        if draft is None:
            self.crew_calls += 1
            retrieved_draft: GroundedDraft | None = None

            def retrieval_callback(query: str) -> dict[str, Any]:
                nonlocal retrieved_draft
                retrieved_draft = self._retrieve_policy(query)
                return {
                    "answer": retrieved_draft.answer,
                    "intent": "policy",
                    "sources": retrieved_draft.sources,
                    "evidence": retrieved_draft.evidence,
                }

            crew_result = run_crew(
                safe_input.text,
                rag_callback=retrieval_callback,
            )
            if retrieved_draft is None:
                raise RuntimeError("CrewAI completed without invoking the RAG tool")
            draft = GroundedDraft(
                answer=crew_result.answer,
                sources=retrieved_draft.sources,
                evidence=retrieved_draft.evidence,
            )
            self.review_calls += 1
            verdict = review_draft_sync(
                query=safe_input.text,
                draft=draft.answer,
                evidence=draft.evidence,
            )
            draft = GroundedDraft(
                answer=verdict.final_answer,
                sources=draft.sources,
                evidence=draft.evidence,
            )
        is_grounded = groundedness_guardrail(draft.answer, draft.evidence)
        if is_grounded and cache_status == "miss":
            self.cache.put(
                safe_input.text,
                self.retriever.corpus_version,
                draft,
                is_appointment_or_personal=False,
            )
        if not is_grounded:
            response = SupportResponse(
                trace_id=trace_id,
                status="refused",
                answer="I don't know",
                intent="policy",
                cache_status=cache_status,
            )
        elif draft.answer == "I don't know":
            response = SupportResponse(
                trace_id=trace_id,
                status="answered",
                answer=draft.answer,
                intent="policy",
                cache_status=cache_status,
            )
        else:
            response = SupportResponse(
                trace_id=trace_id,
                status="answered",
                answer=draft.answer,
                intent="policy",
                sources=draft.sources,
                cache_status=cache_status,
            )
        self.memory.add_turn(request.conversation_id, safe_input.text, response.answer)
        return response

    @staticmethod
    def _extract_record_id(query: str) -> str | None:
        import re

        match = re.search(r"\bAPT-\d{4}\b", query, flags=re.IGNORECASE)
        return match.group(0).upper() if match else None

    def validate_crew_output(self, value: dict[str, Any]) -> CrewResponse:
        return CrewResponse.model_validate(value)
