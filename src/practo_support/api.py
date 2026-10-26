"""FastAPI HTTP and WebSocket endpoints for local support conversations."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict

from practo_support.knowledge import POLICY_DIR, index_policy_collections
from practo_support.local_rag import LocalPolicyRetriever
from practo_support.memory import SessionMemory
from practo_support.observability import (
    RequestLogMiddleware,
    configure_logging,
    emit_websocket_log,
)
from practo_support.schemas import PolicyDocument, SupportRequest, SupportResponse
from practo_support.service import SupportService


class AddDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: PolicyDocument


class HealthResponse(BaseModel):
    status: str
    mode: str


class AddDocumentResponse(BaseModel):
    trace_id: str
    source_id: str
    indexed_chunks: dict[str, int]


def _configured_rag() -> tuple[str, float]:
    path = Path("data/eval/rag_calibration.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    strategy = data["recommended_strategy"]
    return strategy, float(data["strategies"][strategy]["threshold"])


def _default_service() -> SupportService:
    strategy, threshold = _configured_rag()
    return SupportService(
        LocalPolicyRetriever(
            strategy=strategy,
            threshold=threshold,
        ),
        memory=SessionMemory(),
    )


def create_app(service: SupportService | None = None) -> FastAPI:
    configure_logging()
    application = FastAPI(
        title="Practo Domain Support Agent",
        version="0.1.0",
        description="Synthetic, offline-capable healthcare support capstone.",
    )
    application.add_middleware(RequestLogMiddleware)
    application.state.support_service = service

    def get_service() -> SupportService:
        if application.state.support_service is None:
            application.state.support_service = _default_service()
        return application.state.support_service

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", mode="MOCK_LLM")

    @application.post("/ask", response_model=SupportResponse)
    def ask(payload: SupportRequest, request: Request) -> SupportResponse:
        return get_service().ask(payload, trace_id=request.state.trace_id)

    @application.post("/add-document", response_model=AddDocumentResponse)
    def add_document(
        payload: AddDocumentRequest, request: Request
    ) -> AddDocumentResponse:
        document = payload.document
        if not document.synthetic:
            raise HTTPException(
                status_code=422,
                detail="The local capstone endpoint accepts synthetic documents only",
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", document.source_id):
            raise HTTPException(
                status_code=422,
                detail="source_id may contain only letters, digits, '_' and '-'",
            )
        POLICY_DIR.mkdir(parents=True, exist_ok=True)
        destination = POLICY_DIR / f"{document.source_id}.json"
        destination.write_text(
            json.dumps(document.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        counts = index_policy_collections(directory=POLICY_DIR)
        current_service = get_service()
        current_service.cache.clear()
        return AddDocumentResponse(
            trace_id=request.state.trace_id,
            source_id=document.source_id,
            indexed_chunks=counts,
        )

    @application.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket) -> None:
        await websocket.accept()
        service_instance = get_service()
        try:
            while True:
                started = time.perf_counter()
                trace_id = str(uuid4())
                outcome = "error"
                try:
                    message = await websocket.receive_json()
                    payload = SupportRequest.model_validate(message)
                    response = await run_in_threadpool(
                        service_instance.ask, payload, trace_id=trace_id
                    )
                    await websocket.send_json(response.model_dump())
                    outcome = "answered"
                except WebSocketDisconnect:
                    raise
                except (ValueError, TypeError) as error:
                    await websocket.send_json(
                        {
                            "trace_id": trace_id,
                            "status": "error",
                            "answer": "The chat message was invalid.",
                            "detail": str(error),
                        }
                    )
                finally:
                    emit_websocket_log(
                        trace_id=trace_id,
                        path="/ws/chat",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        outcome=outcome,
                    )
        except WebSocketDisconnect:
            return

    return application


app = create_app()
