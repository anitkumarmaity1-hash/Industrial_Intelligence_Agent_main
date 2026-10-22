"""FastAPI application entrypoint.

The Engine is created once here, in the lifespan, and stored on
`app.state.db_engine` — not at import time (importing app.main shouldn't
require DATABASE_URL, e.g. for tooling that only needs the OpenAPI schema)
and not per-request (that would defeat connection pooling). It is disposed
on shutdown so the process doesn't leak connections if reloaded in-process
by a test runner.

The retriever is deliberately NOT built here. `get_retriever()` is already
lazy and process-wide-cached (app/rag/retriever.py); building it eagerly on
every app startup would pay its cost (BM25 indexing, or a Pinecone
describe_index round trip) even for requests that never touch it, which in
Phase 4 is every request — /investigate and /chat are stubs.

Run with: uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from fastapi import Depends

from app.api.auth import login_router, users_router
from app.api.connectors import router as connectors_router
from app.api.dashboard import router as dashboard_router
from app.api.dependencies import enforce_route_permission
from app.api.routes import router
from app.api.uploads import router as uploads_router
from app.core.config import database_is_remote, get_settings
from app.core.logging_config import setup_logging
from app.core.request_context import set_request_id
from app.database.tenant_context import bypasses_rls, install

# Called at import time (not inside lifespan) so logging is active for
# anything that imports app.main, including `uvicorn app.main:app` and the
# TestClient(app) used throughout tests/test_api.py.
setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.engine_url:
        # Phase 2: RLS tenant context is applied per transaction (see
        # app/database/tenant_context.py); the engine uses APP_DATABASE_URL
        # (least-privilege role) when set, else DATABASE_URL.
        app.state.db_engine = install(create_engine(settings.engine_url))
        logger.info("Database engine created.")
        _check_rls_role(app.state.db_engine, settings)
        if database_is_remote(settings.engine_url) and not settings.auth_required:
            logger.warning(
                "AUTH_REQUIRED is off while DATABASE_URL points at a non-local host: "
                "every tenant-scoped route is open to unauthenticated callers (as the demo tenant). "
                "Unset AUTH_REQUIRED (the default is on for a remote database) unless this is intended.")
    else:
        app.state.db_engine = None
        logger.warning(
            "DATABASE_URL not set — database-backed endpoints will return 503.")

    yield

    if app.state.db_engine is not None:
        app.state.db_engine.dispose()
        logger.info("Database engine disposed.")


def _check_rls_role(engine, settings) -> None:
    """An owner/superuser/BYPASSRLS connection silently ignores every RLS
    policy. Warn locally; refuse to start when RLS_REQUIRED (default on for a
    remote database). An unreachable database is not this check's concern."""
    try:
        bypass = bypasses_rls(engine)
    except SQLAlchemyError as exc:
        logger.debug("RLS role check skipped (database unreachable): %s", exc)
        return
    if not bypass:
        return
    msg = ("The API's database role bypasses row-level security (superuser or "
           "BYPASSRLS): tenant isolation rests on WHERE clauses only. Set "
           "APP_DATABASE_URL to the iia_app role (scripts/provision_app_role.py).")
    if settings.rls_required:
        raise RuntimeError(msg)
    logger.warning(msg)


app = FastAPI(
    title="Industrial Intelligence Agent API",
    description="AI operations decision-support API for factory machine investigation.",
    version="0.6.0",
    lifespan=lifespan,
)

# CORS (production-readiness fix 5): an explicit allow-list from
# CORS_ALLOW_ORIGINS, or no middleware at all — the browser default of
# "cross-origin calls are refused" is the safe posture, so it is what you
# get unless you opt origins in. Credentials (cookies) are never allowed:
# this API authenticates with an X-API-Key or Authorization: Bearer header, not cookies. The bundled
# Streamlit frontend calls the API server-to-server and needs no entry.
_cors_origins = get_settings().cors_allow_origins
if _cors_origins:
    if "*" in _cors_origins:
        logger.warning(
            "CORS_ALLOW_ORIGINS contains '*': any website can call this API from a "
            "browser. Prefer an explicit list of origins.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Authorization", "Content-Type"],
        max_age=600,
    )

# Phase 4: every data/upload/user route goes through enforce_route_permission, which looks
# the matched route up in app.core.permissions.ROUTE_PERMISSIONS (undeclared -> 403).
# /auth/login is public (it is where a credential is obtained); /, /healthz, /readyz
# are defined below on the app itself.
_enforce = [Depends(enforce_route_permission)]
app.include_router(router, dependencies=_enforce)
app.include_router(dashboard_router, dependencies=_enforce)
app.include_router(uploads_router, dependencies=_enforce)
app.include_router(connectors_router, dependencies=_enforce)
app.include_router(users_router, dependencies=_enforce)
app.include_router(login_router)


# Production-readiness fix 14: every request gets a request ID — reused
# from an inbound X-Request-ID so a caller's own trace ID survives the
# hop, generated otherwise — set in app.core.request_context (a
# contextvar, so it stays correctly isolated per request even with many
# requests interleaved on one process) before anything else runs, and
# echoed back in the response header so a caller can quote it when
# reporting an issue. Every log line for this request now carries it
# (see logging_config's %(request_id)s), and app/api/routes.py's
# investigation-audit insert stores it too.
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    set_request_id(request_id)
    try:
        response = await call_next(request)
    finally:
        set_request_id(None)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", tags=["meta"])
def root() -> dict:
    """Liveness/landing endpoint — not part of the master-prompt API design."""
    return {"service": "industrial-intelligence-agent", "version": app.version, "docs": "/docs"}


@app.get("/healthz", tags=["meta"])
def healthz() -> dict:
    """Liveness: the process is up and serving. Touches nothing else, so an
    orchestrator restarting on failure never restarts because the DB is down.
    Never requires authentication."""
    return {"status": "ok"}


@app.get("/readyz", tags=["meta"])
def readyz(request: Request):
    """Readiness: the database answers and the schema is present (`machines`
    is queryable; the app role has no access to `tenants` by design). 503 otherwise, with a generic body - the
    real error is logged server-side, not leaked to the caller. Never
    requires authentication (a load balancer must be able to call it)."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        return JSONResponse({"status": "not_ready", "detail": "database not configured"}, status_code=503)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM machines LIMIT 0"))
    except Exception as exc:  # noqa: BLE001 - any failure means not ready
        logger.error("Readiness check failed: %s", exc)
        return JSONResponse({"status": "not_ready", "detail": "database unavailable"}, status_code=503)
    return {"status": "ready"}
