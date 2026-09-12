"""FastAPI application factory for SatQuery AI.

Phase 0 wires every route in the frozen contract to a schema-valid fixture, so
the frontend can generate a typed client from ``openapi.json`` and integrate
against real shapes before a single model is loaded.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from satquery.agent.concurrency import DeviceGates
from satquery.api.jobs import get_job_store
from satquery.api.routers import analyze, artifacts, health, jobs, registry, traces, validate
from satquery.core.config import Settings, get_settings, load_env_file
from satquery.core.logging import configure_logging, get_logger
from satquery.ingest.errors import IngestError
from satquery.schemas.api import ApiErrorResponse

TRACE_ID_HEADER = "X-Trace-Id"

DESCRIPTION = """
Agentic vision-language analysis for satellite imagery.

**Schema version 1.0 — frozen.** Fields may be added within `1.0`; renaming,
removing or retyping any field requires a `2.0` bump. See `DOCS/API_CONTRACT.md`.

Phase 0: every endpoint returns schema-valid fixtures.
""".strip()


def _new_trace_id() -> str:
    """Return a 32-char lowercase hex trace id."""
    return secrets.token_hex(16)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts injected settings so tests can vary them."""
    # Before settings are read, and before the tool catalog probes for weights:
    # the checkpoint paths are read straight from os.environ by the tools that
    # need them, so a .env that has not been copied across leaves the segmenter
    # and the detector looking unavailable.
    from_file = load_env_file()
    settings = settings or get_settings()
    configure_logging(settings)
    log = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The one set of device gates every request's executor shares. Created
        # here, on the serving loop, rather than at import: see
        # satquery.agent.concurrency for why a module global is wrong.
        app.state.gates = DeviceGates()
        log.info(
            "app.startup",
            app_name=settings.app_name,
            environment=settings.environment,
            version=settings.version,
            schema_version=settings.schema_version,
            env_file_applied=from_file,
        )
        yield
        # A job still running at shutdown would otherwise keep its GPU thread
        # and its upload directory: cancel, wait, and let _execute's finally
        # clean up. Subscribers get a terminal error event, not a dropped socket.
        await get_job_store().shutdown()
        log.info("app.shutdown")

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.version,
        lifespan=lifespan,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        # DELETE is /v1/jobs/{id}: the Vite proxy hides its absence in dev, a
        # cross-origin deploy (the nginx image) does not.
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[TRACE_ID_HEADER],
    )

    @app.middleware("http")
    async def attach_trace_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Echo a trace id on every response, per API_CONTRACT §1."""
        trace_id = request.headers.get(TRACE_ID_HEADER) or _new_trace_id()
        response = await call_next(request)
        response.headers[TRACE_ID_HEADER] = trace_id
        return response

    @app.exception_handler(IngestError)
    async def handle_ingest_error(request: Request, exc: IngestError) -> JSONResponse:
        """Render an ingestion failure as the frozen error envelope (API_CONTRACT §6)."""
        trace_id = request.headers.get(TRACE_ID_HEADER)
        log.warning("ingest.rejected", code=exc.code, status=exc.http_status, ref=exc.ref)
        body = ApiErrorResponse(error=exc.to_api_error(trace_id=trace_id))
        return JSONResponse(status_code=exc.http_status, content=body.model_dump(mode="json"))

    for module in (health, registry, validate, analyze, jobs, artifacts, traces):
        app.include_router(module.router, prefix=settings.api_prefix)

    return app


app = create_app()
