"""Small in-memory cache for policy grounding only."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class GroundedResponseCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = ttl_seconds
        self._entries: dict[tuple[str, str], CacheEntry] = {}
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def normalize(query: str) -> str:
        return " ".join(query.casefold().split())

    def get(self, query: str, corpus_version: str) -> Any | None:
        key = (self.normalize(query), corpus_version)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.expires_at <= time.monotonic():
                self._entries.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return entry.value

    def put(
        self,
        query: str,
        corpus_version: str,
        value: Any,
        *,
        is_appointment_or_personal: bool = False,
    ) -> None:
        if is_appointment_or_personal:
            raise ValueError("Appointment or identity-dependent results cannot be cached")
        key = (self.normalize(query), corpus_version)
        with self._lock:
            self._entries[key] = CacheEntry(
                value=value, expires_at=time.monotonic() + self.ttl_seconds
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0
