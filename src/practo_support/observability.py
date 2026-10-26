"""Privacy-conscious one-record-per-request JSON Lines logging."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from fastapi import Request

LOGGER = logging.getLogger("practo_support")


def configure_logging() -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def emit_request_log(
    *,
    trace_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    outcome: str,
) -> None:
    payload = {
        "trace_id": trace_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
        "outcome": outcome,
        "event": "request_complete",
    }
    LOGGER.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def emit_websocket_log(
    *, trace_id: str, path: str, duration_ms: float, outcome: str
) -> None:
    payload = {
        "trace_id": trace_id,
        "method": "WEBSOCKET_MESSAGE",
        "path": path,
        "status_code": 200 if outcome == "answered" else 400,
        "duration_ms": round(duration_ms, 3),
        "outcome": outcome,
        "event": "request_complete",
    }
    LOGGER.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))


class RequestLogMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        import uuid

        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        status_code = 500
        outcome = "error"

        async def traced_send(message: dict[str, Any]) -> None:
            nonlocal status_code, outcome
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                outcome = "answered" if status_code < 400 else "error"
                headers = list(message.get("headers", []))
                headers.append((b"x-trace-id", trace_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        state = scope.setdefault("state", {})
        state["trace_id"] = trace_id
        try:
            await self.app(scope, receive, traced_send)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            emit_request_log(
                trace_id=trace_id,
                method=scope.get("method", "UNKNOWN"),
                path=scope.get("path", ""),
                status_code=status_code,
                duration_ms=duration_ms,
                outcome=outcome,
            )


def record_request_metric(metrics: dict[str, int], key: str) -> None:
    metrics[key] = metrics.get(key, 0) + 1


def safe_request_context(request: Request) -> dict[str, str]:
    return {
        "trace_id": request.state.trace_id,
        "method": request.method,
        "path": request.url.path,
    }
