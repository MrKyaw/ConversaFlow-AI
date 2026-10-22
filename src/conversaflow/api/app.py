"""FastAPI application factory.

The HTTP layer is intentionally thin: authenticate, validate, delegate to the
runtime, serialise. Business rules live in the engine so the same behaviour is
available from the CLI, the WebSocket gateway and tests without an HTTP server.

Every error leaves through one envelope, ``{"error": {...}}``, including
framework-level validation errors. Clients therefore need one parser, not one
per failure mode.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import Depends, FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError

from .. import __version__
from ..config import Principal, Settings
from ..errors import ConversaFlowError, NotFoundError
from ..errors import ValidationError as DomainValidationError
from ..models import (
    BargeInRequest,
    BargeInResponse,
    HealthResponse,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    PlanRequest,
    PlanResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionView,
    TurnRequest,
    TurnResponse,
)
from ..observability.logging import configure_logging, log_event
from ..runtime.bootstrap import Runtime, build_runtime
from .auth import authenticate, require_role
from .ratelimit import RateLimiter
from .schemas import (
    AgentsResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    FlowSummary,
    KnowledgeDeleteResponse,
    KnowledgeListResponse,
    RouteRequest,
    RouteResponse,
)

LOGGER = logging.getLogger("conversaflow.api")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# --------------------------------------------------------------------- helpers


def _runtime(request: Request) -> Runtime:
    return cast(Runtime, request.app.state.runtime)


def _ws_runtime(websocket: WebSocket) -> Runtime:
    return cast(Runtime, cast(Any, websocket.scope["app"]).state.runtime)


def _request_id(request: Request) -> str:
    return cast(str, getattr(request.state, "request_id", secrets.token_hex(8)))


def _error(
    code: str,
    message: str,
    status_code: int,
    request_id: str,
    details: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": {"code": code, "message": message, "request_id": request_id}
    }
    if details:
        payload["error"]["details"] = details
    return JSONResponse(payload, status_code=status_code, headers={"X-Request-ID": request_id})


def _validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for error in exc.errors():
        details.append(
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": str(error.get("msg", "invalid value")),
                "type": str(error.get("type", "value_error")),
            }
        )
    return details


def _auth_dependency(
    *, write: bool = False, admin: bool = False
) -> Callable[[Request], Awaitable[Principal]]:
    async def dependency(request: Request) -> Principal:
        runtime = _runtime(request)
        principal = authenticate(
            runtime.settings.api_keys,
            api_key=request.headers.get("x-api-key"),
            authorization=request.headers.get("authorization"),
        )
        limiter = cast(RateLimiter, request.app.state.limiter)
        limiter.check(principal.key)
        require_role(principal, write=write, admin=admin)
        request.state.principal = principal
        return principal

    return dependency


get_principal = _auth_dependency()
get_writer = _auth_dependency(write=True)
get_admin = _auth_dependency(admin=True)


def _summarise(flow: Any) -> FlowSummary:
    return FlowSummary(
        id=flow.id,
        name=flow.name,
        description=flow.description,
        version=flow.version,
        intents=list(flow.triggers.intents),
        slots=[slot.name for slot in flow.slots],
        nodes=len(flow.nodes),
        start=flow.start,
    )


# ------------------------------------------------------------------ app factory


def create_app(
    settings: Settings | None = None,
    *,
    runtime: Runtime | None = None,
) -> FastAPI:
    resolved = settings or (runtime.settings if runtime is not None else Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved.log_level, resolved.log_format)
        if app.state.runtime is None:
            app.state.runtime = build_runtime(resolved)
        log_event(
            LOGGER,
            logging.INFO,
            "service_started",
            version=__version__,
            store=resolved.store,
            auth_required=True,
        )
        try:
            yield
        finally:
            store = cast(Runtime, app.state.runtime).store
            close = getattr(store, "close", None)
            if callable(close):
                close()
            log_event(LOGGER, logging.INFO, "service_stopped")

    app = FastAPI(
        title="ConversaFlow AI",
        summary="Multi-turn conversational AI runtime with hybrid NLU and dialogue flows.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.runtime = runtime
    app.state.limiter = RateLimiter(resolved.rate_limit_per_minute)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = (
            incoming if _REQUEST_ID_RE.fullmatch(incoming) else secrets.token_hex(8)
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = _request_id(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    # ------------------------------------------------------------- exceptions

    @app.exception_handler(ConversaFlowError)
    async def domain_error_handler(request: Request, exc: ConversaFlowError) -> JSONResponse:
        return _error(exc.code, exc.message, exc.status_code, _request_id(request))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error(
            "validation_error",
            "request validation failed",
            422,
            _request_id(request),
            _validation_details(exc),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log_event(
            LOGGER,
            logging.ERROR,
            "unhandled_error",
            request_id=_request_id(request),
            error_type=type(exc).__name__,
        )
        return _error(
            "internal_error",
            "an unexpected error occurred",
            500,
            _request_id(request),
        )

    # ----------------------------------------------------------------- health

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health(request: Request) -> HealthResponse:
        return _runtime(request).engine.health()

    @app.get("/v1/health", response_model=HealthResponse, tags=["operations"])
    async def authenticated_health(
        request: Request, _: Principal = Depends(get_principal)
    ) -> HealthResponse:
        return _runtime(request).engine.health()

    @app.get("/metrics", response_class=PlainTextResponse, tags=["operations"])
    async def metrics(request: Request, _: Principal = Depends(get_principal)) -> PlainTextResponse:
        return PlainTextResponse(
            _runtime(request).metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4",
        )

    # --------------------------------------------------------------- sessions

    @app.post(
        "/v1/sessions",
        response_model=SessionCreateResponse,
        status_code=201,
        tags=["sessions"],
    )
    async def create_session(
        payload: SessionCreateRequest,
        request: Request,
        principal: Principal = Depends(get_writer),
    ) -> SessionCreateResponse:
        session = await _runtime(request).engine.create_session(
            principal.tenant_id,
            channel=payload.channel.value,
            locale=payload.locale,
            metadata=payload.metadata,
        )
        return SessionCreateResponse(session_id=session.id, expires_at=session.expires_at)

    @app.get("/v1/sessions/{session_id}", response_model=SessionView, tags=["sessions"])
    async def get_session(
        session_id: str,
        request: Request,
        principal: Principal = Depends(get_principal),
    ) -> SessionView:
        runtime = _runtime(request)
        session = await runtime.engine.get_session(session_id, principal.tenant_id)
        return runtime.engine.session_view(session)

    @app.delete("/v1/sessions/{session_id}", status_code=204, tags=["sessions"])
    async def delete_session(
        session_id: str,
        request: Request,
        principal: Principal = Depends(get_writer),
    ) -> Response:
        await _runtime(request).engine.delete_session(session_id, principal.tenant_id)
        return Response(status_code=204)

    # ------------------------------------------------------------------ turns

    @app.post("/v1/turns", response_model=TurnResponse, tags=["turns"])
    async def turn(
        payload: TurnRequest,
        request: Request,
        principal: Principal = Depends(get_writer),
    ) -> TurnResponse:
        return await _runtime(request).engine.handle_turn(
            principal.tenant_id, payload, trace_id=_request_id(request)
        )

    @app.post(
        "/v1/sessions/{session_id}/turns",
        response_model=TurnResponse,
        tags=["turns"],
    )
    async def session_turn(
        session_id: str,
        payload: TurnRequest,
        request: Request,
        principal: Principal = Depends(get_writer),
    ) -> TurnResponse:
        if payload.session_id not in (None, session_id):
            raise DomainValidationError("session_id in the body must match the path")
        scoped = payload.model_copy(update={"session_id": session_id})
        return await _runtime(request).engine.handle_turn(
            principal.tenant_id, scoped, trace_id=_request_id(request)
        )

    @app.post(
        "/v1/sessions/{session_id}/barge-in",
        response_model=BargeInResponse,
        tags=["turns"],
    )
    async def barge_in(
        session_id: str,
        request: Request,
        payload: BargeInRequest | None = None,
        principal: Principal = Depends(get_writer),
    ) -> BargeInResponse:
        reason = payload.reason if payload is not None else "user_speech"
        return await _runtime(request).engine.barge_in(
            session_id, principal.tenant_id, reason=reason
        )

    # -------------------------------------------------------------- knowledge

    @app.post(
        "/v1/knowledge",
        response_model=KnowledgeIngestResponse,
        status_code=201,
        tags=["knowledge"],
    )
    async def ingest_knowledge(
        payload: KnowledgeIngestRequest,
        request: Request,
        principal: Principal = Depends(get_admin),
    ) -> KnowledgeIngestResponse:
        knowledge = _runtime(request).knowledge
        ingested = knowledge.ingest(principal.tenant_id, payload.documents)
        return KnowledgeIngestResponse(
            ingested=ingested,
            chunks=knowledge.count_chunks(principal.tenant_id),
            document_ids=knowledge.document_ids(principal.tenant_id),
        )

    @app.get("/v1/knowledge", response_model=KnowledgeListResponse, tags=["knowledge"])
    async def list_knowledge(
        request: Request,
        principal: Principal = Depends(get_principal),
    ) -> KnowledgeListResponse:
        knowledge = _runtime(request).knowledge
        return KnowledgeListResponse(
            tenant_id=principal.tenant_id,
            documents=knowledge.count_documents(principal.tenant_id),
            chunks=knowledge.count_chunks(principal.tenant_id),
            document_ids=knowledge.document_ids(principal.tenant_id),
        )

    @app.delete(
        "/v1/knowledge/{document_id}",
        response_model=KnowledgeDeleteResponse,
        tags=["knowledge"],
    )
    async def delete_knowledge(
        document_id: str,
        request: Request,
        principal: Principal = Depends(get_admin),
    ) -> KnowledgeDeleteResponse:
        knowledge = _runtime(request).knowledge
        if not knowledge.delete(principal.tenant_id, document_id):
            raise NotFoundError(f"knowledge document {document_id!r} was not found")
        return KnowledgeDeleteResponse(deleted=True, document_id=document_id)

    # ------------------------------------------------------------------ flows

    @app.get("/v1/flows", response_model=list[FlowSummary], tags=["flows"])
    async def list_flows(
        request: Request, _: Principal = Depends(get_principal)
    ) -> list[FlowSummary]:
        return [_summarise(flow) for flow in _runtime(request).flows.values()]

    @app.get("/v1/flows/{flow_id}", tags=["flows"])
    async def get_flow(
        flow_id: str,
        request: Request,
        _: Principal = Depends(get_principal),
    ) -> dict[str, Any]:
        flow = _runtime(request).flows.get(flow_id)
        if flow is None:
            raise NotFoundError(f"flow {flow_id!r} was not found")
        return flow.model_dump(mode="json")

    # ----------------------------------------------------------------- agents

    @app.get("/v1/agents", response_model=AgentsResponse, tags=["agents"])
    async def list_agents(
        request: Request, _: Principal = Depends(get_principal)
    ) -> AgentsResponse:
        return AgentsResponse(agents=_runtime(request).router.catalogue())

    @app.post("/v1/agents/route", response_model=RouteResponse, tags=["agents"])
    async def route_agent(
        payload: RouteRequest,
        request: Request,
        _: Principal = Depends(get_principal),
    ) -> RouteResponse:
        decision = _runtime(request).router.route(
            payload.intent,
            payload.confidence,
            sentiment_score=payload.sentiment_score,
        )
        if decision is None:
            raise NotFoundError(f"no specialist handles intent {payload.intent!r}")
        return RouteResponse(
            agent=decision.agent.name,
            department=decision.agent.department,
            reason=decision.reason,
            confidence=decision.confidence,
            tools=list(decision.agent.tools),
        )

    # ------------------------------------------------------------------- NLU

    @app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["nlu"])
    async def analyze(
        payload: AnalyzeRequest,
        request: Request,
        _: Principal = Depends(get_principal),
    ) -> AnalyzeResponse:
        result = _runtime(request).nlu.analyse(payload.text, locale=payload.locale)
        return AnalyzeResponse(
            text=result.text,
            normalized=result.normalized,
            prediction=result.prediction(),
            entities=result.entities,
            sentiment=result.sentiment,
            language=result.language,
            out_of_vocabulary=round(result.out_of_vocabulary, 4),
            low_confidence=result.low_confidence,
            ambiguous=result.ambiguous,
        )

    @app.post("/v1/plan", response_model=PlanResponse, tags=["agents"])
    async def plan(
        payload: PlanRequest,
        request: Request,
        _: Principal = Depends(get_principal),
    ) -> PlanResponse:
        runtime = _runtime(request)
        result = runtime.nlu.analyse(payload.text)
        return runtime.planner.plan(
            result.intent,
            result.confidence,
            payload.available_slots,
            sentiment_score=result.sentiment.score,
        )

    # ------------------------------------------------------------- websocket

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        runtime = _ws_runtime(websocket)
        key = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")
        authorization = websocket.headers.get("authorization")
        try:
            principal = authenticate(
                runtime.settings.api_keys, api_key=key, authorization=authorization
            )
        except ConversaFlowError:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "ready",
                "tenant_id": principal.tenant_id,
                "role": principal.role,
                "version": __version__,
            }
        )
        try:
            while True:
                raw = await websocket.receive_text()
                response = await _handle_stream_frame(runtime, principal, raw)
                await websocket.send_json(response)
        except WebSocketDisconnect:
            return

    return app


async def _handle_stream_frame(runtime: Runtime, principal: Principal, raw: str) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        return _ws_error("invalid_json", "frame must be a JSON object")
    if not isinstance(message, dict):
        return _ws_error("invalid_json", "frame must be a JSON object")
    kind = message.get("type")
    try:
        if kind == "ping":
            return {"type": "pong"}
        if kind in {"session.create", "turn", "barge_in", "session.delete"}:
            require_role(principal, write=True)
        if kind == "session.create":
            request = SessionCreateRequest.model_validate(
                {key: value for key, value in message.items() if key != "type"}
            )
            session = await runtime.engine.create_session(
                principal.tenant_id,
                channel=request.channel.value,
                locale=request.locale,
                metadata=request.metadata,
            )
            return {
                "type": "session.created",
                "session_id": session.id,
                "expires_at": session.expires_at.isoformat(),
            }
        if kind == "turn":
            payload = TurnRequest.model_validate(
                {key: value for key, value in message.items() if key != "type"}
            )
            response = await runtime.engine.handle_turn(principal.tenant_id, payload)
            return {"type": "turn", "response": response.model_dump(mode="json")}
        if kind == "barge_in":
            session_id = str(message.get("session_id", ""))
            reason = str(message.get("reason", "user_speech"))
            ack = await runtime.engine.barge_in(session_id, principal.tenant_id, reason=reason)
            return {"type": "barge_in", "response": ack.model_dump(mode="json")}
        if kind == "session.get":
            session = await runtime.engine.get_session(
                str(message.get("session_id", "")), principal.tenant_id
            )
            return {
                "type": "session",
                "session": runtime.engine.session_view(session).model_dump(mode="json"),
            }
        if kind == "session.delete":
            await runtime.engine.delete_session(
                str(message.get("session_id", "")), principal.tenant_id
            )
            return {"type": "session.deleted"}
    except ConversaFlowError as exc:
        return _ws_error(exc.code, exc.message)
    except ValidationError as exc:
        return _ws_error(
            "validation_error",
            "frame validation failed",
            details=[
                {
                    "loc": [str(part) for part in error.get("loc", ())],
                    "msg": str(error.get("msg", "invalid value")),
                    "type": str(error.get("type", "value_error")),
                }
                for error in exc.errors()
            ],
        )
    return _ws_error("unsupported_type", f"unsupported frame type {kind!r}")


def _ws_error(
    code: str, message: str, details: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"type": "error", "error": error}


app = create_app()
