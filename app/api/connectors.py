"""Connector endpoints (Phase 5), admin only (permissions in app/core/permissions.py).

    GET  /connectors, /connectors/{id}, /connectors/{id}/runs
    POST /connectors                    create
    POST /connectors/{id}               update (partial)
    POST /connectors/{id}/enable | disable | delete
    POST /connectors/{id}/test          validate access to the bucket; ingests nothing
    POST /connectors/{id}/run           run now: makes the connector due immediately

Everything is POST/GET like the rest of the API (CORS stays GET/POST only). The tenant is
always the caller's; credentials are never accepted, returned or stored: only a `secret_ref`
name. The config is exactly bucket/prefix/pattern (extra fields are refused), so no endpoint
URL or credential-shaped key can be saved.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import get_conn, get_tenant
from app.api.rate_limit import limiter, retry_after_header
from app.connectors.base import ConnectorError, StoreFactory
from app.connectors.gcs import emulator_configured, make_gcs_store
from app.connectors.settings import connector_settings
from app.connectors.validation import build_gcs_config, next_run, validate_schedule
from app.core.secrets_provider import (
    SecretError,
    SecretsProvider,
    get_secrets_provider,
    validate_secret_ref,
)
from app.core.tenancy import TenantContext
from app.database import queries
from app.onboarding.sensors import SensorMapping, SensorMappingError

logger = logging.getLogger(__name__)
router = APIRouter()

TEST_LIMIT_PER_MINUTE = 10


class ConnectorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    type: Literal["gcs"] = "gcs"
    bucket: str = Field(max_length=222)
    prefix: str = Field(default="", max_length=512)
    pattern: str = Field(default="*.csv", max_length=100)
    schedule: str = Field(max_length=100, description="5-field cron expression, UTC")
    secret_ref: str | None = Field(default=None, max_length=100)
    mapping: dict[str, Any] | None = None
    mapping_from_upload_id: str | None = None
    enabled: bool = True


class ConnectorPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=100)
    bucket: str | None = Field(default=None, max_length=222)
    prefix: str | None = Field(default=None, max_length=512)
    pattern: str | None = Field(default=None, max_length=100)
    schedule: str | None = Field(default=None, max_length=100)
    secret_ref: str | None = Field(default=None, max_length=100)
    mapping: dict[str, Any] | None = None
    mapping_from_upload_id: str | None = None


class ConnectorOut(BaseModel):
    connector_id: str
    name: str
    type: str
    bucket: str
    prefix: str
    pattern: str
    schedule: str
    secret_ref: str | None
    enabled: bool
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_status: str
    last_message: str | None
    created_at: datetime
    updated_at: datetime


class RunOut(BaseModel):
    run_id: str
    trigger: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    objects_seen: int
    objects_new: int
    objects_queued: int
    objects_failed: int
    message: str | None


class TestOut(BaseModel):
    ok: bool
    message: str


def get_store_factory() -> StoreFactory:
    return make_gcs_store


def get_secrets() -> SecretsProvider:
    return get_secrets_provider()


def _out(row: dict[str, Any]) -> ConnectorOut:
    cfg = row["config"]
    return ConnectorOut(
        connector_id=str(row["id"]), bucket=cfg.get("bucket", ""), prefix=cfg.get("prefix", ""),
        pattern=cfg.get("pattern", ""),
        **{k: row[k] for k in ("name", "type", "schedule", "secret_ref", "enabled", "next_run_at",
                               "last_run_at", "last_status", "last_message", "created_at",
                               "updated_at")})


def _422(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=message)


def _get_or_404(conn: Connection, connector_id: str, tenant: TenantContext) -> dict[str, Any]:
    try:
        uuid.UUID(connector_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Connector not found.") from None
    row = queries.get_connector(conn, connector_id, tenant_id=tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Connector not found.")
    return row


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _check_schedule(expr: str) -> None:
    try:
        validate_schedule(expr, connector_settings().min_interval_seconds)
    except ValueError as exc:
        raise _422(str(exc)) from None


def _check_secret_ref(ref: str | None) -> None:
    if ref is None:
        if not emulator_configured():
            raise _422("secret_ref is required (the name of the credential the worker resolves; "
                       "credentials are never sent to this API)")
        return
    try:
        validate_secret_ref(ref)
    except SecretError as exc:
        raise _422(str(exc)) from None


def _resolve_mapping(conn: Connection, tenant: TenantContext, mapping: dict | None,
                     from_upload: str | None) -> dict[str, Any]:
    if (mapping is None) == (from_upload is None):
        raise _422("give exactly one of mapping or mapping_from_upload_id")
    if from_upload is not None:
        try:
            uuid.UUID(from_upload)
        except ValueError:
            raise _422("mapping_from_upload_id is not a valid upload id") from None
        upload = queries.get_upload(conn, from_upload, tenant_id=tenant.tenant_id)
        if upload is None or not upload["mapping"]:
            raise _422("that upload does not exist or has no confirmed mapping")
        mapping = upload["mapping"]
    try:
        SensorMapping.from_dict(mapping)
    except SensorMappingError as exc:
        problems = [ln.strip("- ").strip() for ln in str(exc).splitlines()[1:] if ln.strip()]
        raise HTTPException(status_code=422, detail={"message": "Invalid mapping.",
                                                     "problems": problems}) from exc
    return mapping


def _config(bucket: str, prefix: str, pattern: str) -> dict[str, str]:
    try:
        return build_gcs_config(bucket, prefix, pattern)
    except ValueError as exc:
        raise _422(str(exc)) from None


@router.get("/connectors", response_model=list[ConnectorOut])
def list_connectors(conn: Connection = Depends(get_conn),
                    tenant: TenantContext = Depends(get_tenant)) -> list[ConnectorOut]:
    return [_out(r) for r in queries.list_connectors(conn, tenant_id=tenant.tenant_id)]


@router.post("/connectors", response_model=ConnectorOut, status_code=201)
def create_connector(body: ConnectorIn, conn: Connection = Depends(get_conn),
                     tenant: TenantContext = Depends(get_tenant)) -> ConnectorOut:
    config = _config(body.bucket, body.prefix, body.pattern)
    _check_schedule(body.schedule)
    _check_secret_ref(body.secret_ref)
    mapping = _resolve_mapping(conn, tenant, body.mapping, body.mapping_from_upload_id)
    if queries.count_connectors(conn, tenant_id=tenant.tenant_id) >= connector_settings().max_per_tenant:
        raise HTTPException(status_code=409, detail="Connector limit reached for this tenant.")
    try:
        row = queries.create_connector(
            conn, str(uuid.uuid4()), name=body.name, type=body.type, config=config,
            schedule=body.schedule, mapping=mapping, secret_ref=body.secret_ref,
            enabled=body.enabled,
            next_run_at=next_run(body.schedule, _now()) if body.enabled else None,
            tenant_id=tenant.tenant_id)
        conn.commit()
    except IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=409, detail="A connector with that name already exists.") from None
    return _out(row)


@router.get("/connectors/{connector_id}", response_model=ConnectorOut)
def get_connector(connector_id: str, conn: Connection = Depends(get_conn),
                  tenant: TenantContext = Depends(get_tenant)) -> ConnectorOut:
    return _out(_get_or_404(conn, connector_id, tenant))


@router.post("/connectors/{connector_id}", response_model=ConnectorOut)
def update_connector(connector_id: str, body: ConnectorPatch, conn: Connection = Depends(get_conn),
                     tenant: TenantContext = Depends(get_tenant)) -> ConnectorOut:
    row = _get_or_404(conn, connector_id, tenant)
    fields: dict[str, Any] = {}
    if body.name is not None:
        fields["name"] = body.name
    if body.bucket is not None or body.prefix is not None or body.pattern is not None:
        cur = row["config"]
        fields["config"] = _config(body.bucket if body.bucket is not None else cur.get("bucket", ""),
                                   body.prefix if body.prefix is not None else cur.get("prefix", ""),
                                   body.pattern if body.pattern is not None else cur.get("pattern", "*.csv"))
        if fields["config"] != cur:
            # A different bucket/prefix/pattern is a different source: re-list from the start
            # (the (key, version) ledger still stops anything already queued being re-ingested).
            fields["list_watermark"] = None
    if body.schedule is not None:
        _check_schedule(body.schedule)
        fields["schedule"] = body.schedule
        if row["enabled"]:
            fields["next_run_at"] = next_run(body.schedule, _now())
    if body.secret_ref is not None:
        _check_secret_ref(body.secret_ref)
        fields["secret_ref"] = body.secret_ref
    if body.mapping is not None or body.mapping_from_upload_id is not None:
        fields["mapping"] = _resolve_mapping(conn, tenant, body.mapping, body.mapping_from_upload_id)
    if not fields:
        raise _422("nothing to update")
    try:
        updated = queries.update_connector(conn, connector_id, fields, tenant_id=tenant.tenant_id)
        conn.commit()
    except IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=409, detail="A connector with that name already exists.") from None
    return _out(updated)


def _set_enabled(conn: Connection, connector_id: str, tenant: TenantContext, enabled: bool) -> ConnectorOut:
    row = _get_or_404(conn, connector_id, tenant)
    updated = queries.update_connector(
        conn, connector_id,
        {"enabled": enabled, "next_run_at": next_run(row["schedule"], _now()) if enabled else None},
        tenant_id=tenant.tenant_id)
    conn.commit()
    return _out(updated)


@router.post("/connectors/{connector_id}/enable", response_model=ConnectorOut)
def enable_connector(connector_id: str, conn: Connection = Depends(get_conn),
                     tenant: TenantContext = Depends(get_tenant)) -> ConnectorOut:
    return _set_enabled(conn, connector_id, tenant, True)


@router.post("/connectors/{connector_id}/disable", response_model=ConnectorOut)
def disable_connector(connector_id: str, conn: Connection = Depends(get_conn),
                      tenant: TenantContext = Depends(get_tenant)) -> ConnectorOut:
    return _set_enabled(conn, connector_id, tenant, False)


@router.post("/connectors/{connector_id}/delete", status_code=204, response_class=Response)
def delete_connector(connector_id: str, conn: Connection = Depends(get_conn),
                     tenant: TenantContext = Depends(get_tenant)) -> Response:
    _get_or_404(conn, connector_id, tenant)
    queries.delete_connector(conn, connector_id, tenant_id=tenant.tenant_id)
    conn.commit()
    return Response(status_code=204)


@router.post("/connectors/{connector_id}/run", response_model=ConnectorOut, status_code=202)
def run_connector(connector_id: str, conn: Connection = Depends(get_conn),
                  tenant: TenantContext = Depends(get_tenant)) -> ConnectorOut:
    _get_or_404(conn, connector_id, tenant)
    row = queries.request_connector_run(conn, connector_id, tenant_id=tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=409, detail="The connector is disabled; enable it first.")
    conn.commit()
    return _out(row)


@router.post("/connectors/{connector_id}/test", response_model=TestOut)
def test_connector(connector_id: str, conn: Connection = Depends(get_conn),
                   tenant: TenantContext = Depends(get_tenant),
                   store_factory: StoreFactory = Depends(get_store_factory),
                   secrets: SecretsProvider = Depends(get_secrets)) -> TestOut:
    """Check that the credentials work and the bucket is listable. Reads no object, ingests nothing."""
    row = _get_or_404(conn, connector_id, tenant)
    wait = limiter.check(f"connector-test|{tenant.tenant_id}", TEST_LIMIT_PER_MINUTE, 60.0)
    if wait is not None:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.",
                            headers={"Retry-After": retry_after_header(wait)})
    try:
        secret = secrets.get(row["secret_ref"]) if row["secret_ref"] else None
    except SecretError:
        return TestOut(ok=False, message="the configured secret is not available")
    try:
        store_factory(row, secret).check_access()
    except ConnectorError as exc:
        return TestOut(ok=False, message=exc.message)
    except ImportError:
        raise HTTPException(status_code=503, detail="Connector support is not installed.") from None
    except Exception as exc:  # noqa: BLE001
        logger.warning("connector test failed (%s)", type(exc).__name__)
        return TestOut(ok=False, message="unexpected error while contacting the storage service")
    return TestOut(ok=True, message="access OK")


@router.get("/connectors/{connector_id}/runs", response_model=list[RunOut])
def list_runs(connector_id: str, limit: int = Query(default=20, ge=1, le=200),
              offset: int = Query(default=0, ge=0), conn: Connection = Depends(get_conn),
              tenant: TenantContext = Depends(get_tenant)) -> list[RunOut]:
    _get_or_404(conn, connector_id, tenant)
    return [RunOut(run_id=str(r["id"]), **{k: r[k] for k in RunOut.model_fields if k != "run_id"})
            for r in queries.list_connector_runs(conn, connector_id, limit, offset,
                                                 tenant_id=tenant.tenant_id)]
