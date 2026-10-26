"""In-process, session-keyed LangChain message memory."""

from __future__ import annotations

from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class SessionMemory:
    def __init__(self) -> None:
        self._histories: dict[str, InMemoryChatMessageHistory] = {}

    def history_for(self, session_id: str) -> InMemoryChatMessageHistory:
        if not session_id:
            raise ValueError("session_id must not be empty")
        return self._histories.setdefault(session_id, InMemoryChatMessageHistory())

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        history = self.history_for(session_id)
        history.add_user_message(user_message)
        history.add_ai_message(assistant_message)

    def messages(self, session_id: str) -> list[Any]:
        return list(self.history_for(session_id).messages)

    def reset(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    def wrap(self, runnable: Any) -> RunnableWithMessageHistory:
        return RunnableWithMessageHistory(
            runnable,
            self.history_for,
            input_messages_key="input",
            history_messages_key="history",
        )
