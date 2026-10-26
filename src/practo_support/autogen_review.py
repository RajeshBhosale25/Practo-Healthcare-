"""Two-agent, local-only AutoGen review stage with structured verdicts."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    RequestUsage,
)
from pydantic import BaseModel

from practo_support.schemas import VerdictModel


class MockAutoGenClient(ChatCompletionClient):
    """Deterministic AutoGen model client that operates without network access."""

    def __init__(self) -> None:
        super().__init__()
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        }

    @property
    def capabilities(self) -> dict[str, Any]:
        return self.model_info

    @staticmethod
    def _text(messages: Sequence[LLMMessage]) -> str:
        fragments: list[str] = []
        for message in messages:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                fragments.append(content)
            elif isinstance(content, list):
                fragments.extend(
                    str(getattr(part, "text", part))
                    for part in content
                    if getattr(part, "text", None) is not None
                )
        return "\n".join(fragments)

    @staticmethod
    def _verdict(prompt: str) -> VerdictModel:
        draft = ""
        evidence = ""
        for line in prompt.splitlines():
            if line.startswith("Draft:"):
                draft = line.removeprefix("Draft:").strip()
            elif line.startswith("Evidence:"):
                evidence = line.removeprefix("Evidence:").strip()
        if draft and evidence and draft not in evidence:
            supported = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", draft)
                if sentence.strip() and sentence.strip() in evidence
            ]
            cleaned = " ".join(supported)
            return VerdictModel(
                approved=False,
                final_answer=cleaned,
                reason="Removed claims not present in the supplied evidence.",
            )
        if draft and evidence and draft in evidence:
            return VerdictModel(
                approved=True,
                final_answer=draft,
                reason="Draft is supported by the supplied evidence.",
            )
        if draft == "I don't know":
            return VerdictModel(
                approved=True,
                final_answer=draft,
                reason="Safe uncertainty response; no unsupported claim.",
            )
        return VerdictModel(
            approved=False,
            final_answer="I don't know",
            reason="The draft could not be verified against supplied evidence.",
        )

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        del tools, tool_choice, extra_create_args, cancellation_token
        prompt = self._text(messages)
        if isinstance(json_output, type) and issubclass(json_output, BaseModel):
            content = self._verdict(prompt).model_dump_json()
        elif json_output:
            content = self._verdict(prompt).model_dump_json()
        elif "Policy-Compliance-Reviewer" in prompt or "review" in prompt.casefold():
            content = "The draft should be checked against the supplied evidence."
        else:
            content = "Review completed."
        self._usage = RequestUsage(
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(content) // 4),
        )
        return CreateResult(
            finish_reason="stop", content=content, usage=self._usage, cached=False
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

    def count_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()
    ) -> int:
        del tools
        return max(1, len(self._text(messages)) // 4)

    def remaining_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()
    ) -> int:
        return max(0, 8192 - self.count_tokens(messages, tools=tools))

    def total_usage(self) -> RequestUsage:
        return self._usage

    def actual_usage(self) -> RequestUsage:
        return self._usage

    async def close(self) -> None:
        return None


def build_review_team() -> RoundRobinGroupChat:
    client = MockAutoGenClient()
    reviewer = AssistantAgent(
        name="PolicyComplianceReviewer",
        model_client=client,
        description="Checks claims against the supplied source evidence.",
        system_message="Review only against the supplied evidence.",
    )
    editor = AssistantAgent(
        name="FinalEditor",
        model_client=client,
        description="Approves or safely revises the response draft.",
        system_message=(
            "Return a structured verdict with approved, final_answer, and reason. "
            "Remove unsupported claims."
        ),
        output_content_type=VerdictModel,
    )
    return RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[VerdictModel]],
        name="SupportResponseReview",
    )


async def review_draft(
    *, query: str, draft: str, evidence: list[str]
) -> VerdictModel:
    team = build_review_team()
    evidence_text = " ".join(evidence)
    result = await team.run(
        task=(
            f"Query: {query}\nDraft: {draft}\nEvidence: {evidence_text}\n"
            "Return a verdict. Approve supported text unchanged; revise "
            "unsupported claims."
        )
    )
    for message in reversed(result.messages):
        content = getattr(message, "content", None)
        if isinstance(content, VerdictModel):
            return content
        if isinstance(content, dict):
            try:
                return VerdictModel.model_validate(content)
            except ValueError:
                continue
    raise RuntimeError("AutoGen review completed without a structured verdict")


def review_draft_sync(*, query: str, draft: str, evidence: list[str]) -> VerdictModel:
    import asyncio

    return asyncio.run(review_draft(query=query, draft=draft, evidence=evidence))
