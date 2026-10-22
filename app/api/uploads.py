"""Upload, mapping and job endpoints (Phase 3).

    POST /uploads                      create the upload record (filename + content type)
    POST /uploads/{id}/content         the file itself: a raw request body (Content-Type
                                       text/csv), streamed to storage; size-capped
    POST /uploads/{id}/preview         first N rows + detected columns
    POST /uploads/{id}/mapping         validate a Phase 1 sensor mapping against a sample;
                                       with "confirm": true, store it and queue the job
    GET  /uploads, /uploads/{id}
    GET  /jobs, /jobs/{id}, /jobs/{id}/rejected-rows
    POST /jobs/{id}/retry              failed jobs only

Why a streamed POST and not a presigned PUT: the Streamlit frontend uploads
server-to-server, and a direct-to-storage upload could not enforce the size cap.
The body is read in chunks into a bounded spool, hashed, and handed
to the StorageProvider; no multipart parser (and no extra dependency) is involved.

Every route is tenant-scoped through get_tenant, requires a real API key (an
anonymous demo request is refused: uploads write data), and is rate limited per
tenant. The object key is generated here from the tenant and a server-made UUID;
no request field names a path or key, and no response exposes one.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.engine import Connection

from app.api.dependencies import get_conn, get_tenant
from app.api.rate_limit import limiter, retry_after_header
from app.core.ingest_settings import CSV_CONTENT_TYPES, ingest_settings
from app.core.tenancy import TenantContext
from app.database import queries
from app.onboarding import sensors
from app.onboarding.columns import missing_source_columns
from app.onboarding.sensors import SensorMapping, SensorMappingError, normalize_sensor_readings
from app.storage import StorageError, StorageProvider, get_storage, new_upload_id, upload_key

logger = logging.getLogger(__name__)
router = APIRouter()

SPOOL_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class UploadCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(default="text/csv", max_length=100)


class UploadOut(BaseModel):
    upload_id: str
    filename: str
    content_type: str
    status: str
    size_bytes: int | None
    sha256: str | None
    has_mapping: bool
    created_at: datetime
    updated_at: datetime


class PreviewRequest(BaseModel):
    rows: int = Field(default=10, ge=1, le=1000)


class ColumnOut(BaseModel):
    name: str
    detected_type: str          # "number" | "timestamp" | "text" | "empty" (from the sampled rows)


class PreviewOut(BaseModel):
    columns: list[ColumnOut]
    rows: list[dict[str, str]]
    rows_returned: int


class MappingRequest(BaseModel):
    mapping: dict[str, Any]
    confirm: bool = False


class JobOut(BaseModel):
    job_id: str
    upload_id: str
    type: str
    status: str
    stage: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    rows_in: int
    rows_loaded: int
    rows_rejected: int
    error: str | None
    retry_count: int
    max_retries: int


class MappingOut(BaseModel):
    valid: bool
    sampled_rows: int
    report: dict[str, Any]       # Phase 1 NormalizationReport.to_dict(), for the sample
    job: JobOut | None = None


class RejectedRowOut(BaseModel):
    row_number: int | None
    code: str
    reason: str
    raw: dict[str, Any] | None


# ---------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------

def get_upload_storage() -> StorageProvider:
    try:
        return get_storage()
    except Exception as exc:  # noqa: BLE001 - e.g. boto3 missing; keep detail out of the response
        logger.error("Storage unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="File storage is unavailable.") from exc


def upload_tenant(tenant: TenantContext = Depends(get_tenant)) -> TenantContext:
    """Authenticated + rate limited. Keyed per tenant, apart from the LLM limit."""
    if not tenant.authenticated:
        raise HTTPException(status_code=401, detail="Uploads require an API key.")
    wait = limiter.check(f"upload|{tenant.tenant_id}",
                         ingest_settings().upload_rate_limit_per_minute, 60.0)
    if wait is not None:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.",
                            headers={"Retry-After": retry_after_header(wait)})
    return tenant


def _upload_out(row: dict[str, Any]) -> UploadOut:
    fields = {k: row[k] for k in ("filename", "content_type", "status", "size_bytes", "sha256",
                                  "created_at", "updated_at")}
    return UploadOut(upload_id=str(row["upload_id"]), has_mapping=row["mapping"] is not None, **fields)


def _job_out(row: dict[str, Any]) -> JobOut:
    fields = {k: row[k] for k in JobOut.model_fields if k not in ("job_id", "upload_id")}
    return JobOut(job_id=str(row["job_id"]), upload_id=str(row["upload_id"]), **fields)


def _get_upload_or_404(conn: Connection, upload_id: str, tenant: TenantContext) -> dict[str, Any]:
    try:
        uuid.UUID(upload_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Upload not found.") from None
    row = queries.get_upload(conn, upload_id, tenant_id=tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Upload not found.")
    return row


def _get_job_or_404(conn: Connection, job_id: str, tenant: TenantContext) -> dict[str, Any]:
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found.") from None
    row = queries.get_job(conn, job_id, tenant_id=tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return row


def _base_type(value: str | None) -> str:
    return (value or "").split(";")[0].strip().lower()


def _read_sample(storage: StorageProvider, upload: dict[str, Any], tenant: TenantContext,
                 nrows: int) -> pd.DataFrame:
    if upload["status"] == "created":
        raise HTTPException(status_code=409, detail="Upload the file content first.")
    try:
        with storage.open(upload["storage_key"], tenant_id=tenant.tenant_id) as f:
            return pd.read_csv(f, dtype=str, keep_default_na=False, nrows=nrows)
    except StorageError as exc:
        logger.error("storage read failed: %s", exc)
        raise HTTPException(status_code=503, detail="File storage is unavailable.") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"The file is not a readable CSV: {exc}") from exc


def _detect_type(values: pd.Series) -> str:
    cells = values[values.str.strip() != ""]
    if cells.empty:
        return "empty"
    if pd.to_numeric(cells, errors="coerce").notna().all():
        return "number"
    try:
        parsed = pd.to_datetime(cells, errors="coerce", format="ISO8601")
        if parsed.notna().all():
            return "timestamp"
    except (ValueError, TypeError):
        pass
    return "text"


# ---------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------

@router.post("/uploads", response_model=UploadOut, status_code=201)
def create_upload(body: UploadCreate, conn: Connection = Depends(get_conn),
                  tenant: TenantContext = Depends(upload_tenant)) -> UploadOut:
    if _base_type(body.content_type) not in CSV_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"content_type must be a CSV type "
                            f"({', '.join(sorted(CSV_CONTENT_TYPES))}).")
    upload_id = new_upload_id()
    filename = os.path.basename(body.filename.replace("\\", "/")) or "upload.csv"   # display only
    row = queries.create_upload(conn, upload_id, filename[:255], _base_type(body.content_type),
                                upload_key(tenant.tenant_id, upload_id), tenant_id=tenant.tenant_id)
    conn.commit()
    return _upload_out(row)


def _store_content(conn: Connection, storage: StorageProvider, upload: dict[str, Any],
                   spool, size: int, digest: str, tenant: TenantContext) -> dict[str, Any]:
    if queries.get_active_job_for_upload(conn, upload["upload_id"], tenant_id=tenant.tenant_id):
        raise HTTPException(status_code=409, detail="A job for this upload is queued or running.")
    try:
        storage.put(upload["storage_key"], spool, tenant_id=tenant.tenant_id)
    except StorageError as exc:
        logger.error("storage write failed: %s", exc)
        raise HTTPException(status_code=503, detail="File storage is unavailable.") from exc
    row = queries.set_upload_content(conn, upload["upload_id"], size, digest, tenant_id=tenant.tenant_id)
    conn.commit()
    return row


@router.post("/uploads/{upload_id}/content", response_model=UploadOut)
async def upload_content(upload_id: str, request: Request, conn: Connection = Depends(get_conn),
                         tenant: TenantContext = Depends(upload_tenant),
                         storage: StorageProvider = Depends(get_upload_storage)) -> UploadOut:
    s = ingest_settings()
    upload = await run_in_threadpool(_get_upload_or_404, conn, upload_id, tenant)
    if _base_type(request.headers.get("content-type")) not in CSV_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Send the file as the request body with a CSV Content-Type.")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > s.upload_max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the {s.upload_max_bytes}-byte limit.")

    digest, size, first = hashlib.sha256(), 0, True
    spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_BYTES)
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            if first and b"\x00" in chunk[:8192]:
                raise HTTPException(status_code=415, detail="That looks like a binary file, not CSV.")
            first = False
            size += len(chunk)
            if size > s.upload_max_bytes:
                raise HTTPException(status_code=413, detail=f"File exceeds the {s.upload_max_bytes}-byte limit.")
            digest.update(chunk)
            spool.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The request body is empty.")
        spool.seek(0)
        row = await run_in_threadpool(_store_content, conn, storage, upload, spool, size,
                                      digest.hexdigest(), tenant)
    finally:
        spool.close()
    return _upload_out(row)


@router.get("/uploads", response_model=list[UploadOut])
def list_uploads(limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0),
                 conn: Connection = Depends(get_conn),
                 tenant: TenantContext = Depends(upload_tenant)) -> list[UploadOut]:
    return [_upload_out(r) for r in queries.list_uploads(
        conn, limit, offset, tenant_id=tenant.tenant_id)]


@router.get("/uploads/{upload_id}", response_model=UploadOut)
def get_upload(upload_id: str, conn: Connection = Depends(get_conn),
               tenant: TenantContext = Depends(upload_tenant)) -> UploadOut:
    return _upload_out(_get_upload_or_404(conn, upload_id, tenant))


@router.post("/uploads/{upload_id}/preview", response_model=PreviewOut)
def preview_upload(upload_id: str, body: PreviewRequest | None = None,
                   conn: Connection = Depends(get_conn),
                   tenant: TenantContext = Depends(upload_tenant),
                   storage: StorageProvider = Depends(get_upload_storage)) -> PreviewOut:
    upload = _get_upload_or_404(conn, upload_id, tenant)
    n = min((body.rows if body else 10), ingest_settings().preview_max_rows)
    df = _read_sample(storage, upload, tenant, n)
    return PreviewOut(
        columns=[ColumnOut(name=str(c), detected_type=_detect_type(df[c])) for c in df.columns],
        rows=df.to_dict("records"), rows_returned=len(df))


@router.post("/uploads/{upload_id}/mapping", response_model=MappingOut)
def map_upload(upload_id: str, body: MappingRequest, conn: Connection = Depends(get_conn),
               tenant: TenantContext = Depends(upload_tenant),
               storage: StorageProvider = Depends(get_upload_storage)) -> MappingOut:
    """Validate (and with confirm=true, store + queue) a Phase 1 sensor mapping.

    The mapping format and the per-row error report are Phase 1's
    (app.onboarding.sensors). The report here covers the first
    MAPPING_DRY_RUN_ROWS rows; the job validates the whole file."""
    upload = _get_upload_or_404(conn, upload_id, tenant)
    try:
        mapping = SensorMapping.from_dict(body.mapping)
    except SensorMappingError as exc:
        problems = [ln.strip("- ").strip() for ln in str(exc).splitlines()[1:] if ln.strip()]
        raise HTTPException(status_code=422, detail={"message": "Invalid mapping.", "problems": problems}) from exc

    n = ingest_settings().mapping_dry_run_rows
    sample = _read_sample(storage, upload, tenant, n)
    missing = missing_source_columns(sample.columns, mapping)
    if missing:
        raise HTTPException(status_code=422, detail={
            "message": "Columns named in the mapping were not found in the file.",
            "missing_columns": missing, "file_columns": [str(c) for c in sample.columns]})
    result = normalize_sensor_readings(sample, mapping)
    report = result.report
    fatal = report.counts.get(sensors.MISSING_COLUMN, 0) > 0 or (
        report.readings_accepted == 0)
    out = MappingOut(valid=not fatal, sampled_rows=len(sample), report=report.to_dict())
    if not body.confirm:
        return out
    if fatal:
        raise HTTPException(status_code=422, detail={
            "message": "The mapping cannot be applied to this file.", "report": report.to_dict()})
    if queries.get_active_job_for_upload(conn, upload_id, tenant_id=tenant.tenant_id):
        raise HTTPException(status_code=409, detail="A job for this upload is queued or running.")

    queries.set_upload_mapping(conn, upload_id, body.mapping, tenant_id=tenant.tenant_id)
    job = queries.create_job(conn, str(uuid.uuid4()), upload_id, ingest_settings().job_max_retries,
                             tenant_id=tenant.tenant_id)
    conn.commit()
    out.job = _job_out(job)
    return out


# ---------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------

@router.get("/jobs", response_model=list[JobOut])
def list_jobs(status: str | None = Query(default=None, pattern="^(queued|running|succeeded|failed)$"),
              limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0),
              conn: Connection = Depends(get_conn),
              tenant: TenantContext = Depends(upload_tenant)) -> list[JobOut]:
    return [_job_out(r) for r in queries.list_jobs(
        conn, status, limit, offset, tenant_id=tenant.tenant_id)]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, conn: Connection = Depends(get_conn),
            tenant: TenantContext = Depends(upload_tenant)) -> JobOut:
    return _job_out(_get_job_or_404(conn, job_id, tenant))


@router.get("/jobs/{job_id}/rejected-rows", response_model=list[RejectedRowOut])
def get_rejected_rows(job_id: str, limit: int = Query(default=100, ge=1, le=1000),
                      offset: int = Query(default=0, ge=0), conn: Connection = Depends(get_conn),
                      tenant: TenantContext = Depends(upload_tenant)) -> list[RejectedRowOut]:
    _get_job_or_404(conn, job_id, tenant)
    return [RejectedRowOut(**r) for r in queries.list_rejected_rows(
        conn, job_id, limit, offset, tenant_id=tenant.tenant_id)]


@router.post("/jobs/{job_id}/retry", response_model=JobOut, status_code=202)
def retry_job(job_id: str, conn: Connection = Depends(get_conn),
              tenant: TenantContext = Depends(upload_tenant)) -> JobOut:
    _get_job_or_404(conn, job_id, tenant)
    row = queries.retry_failed_job(conn, job_id, ingest_settings().job_max_retries,
                                   tenant_id=tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=409, detail="Only failed jobs can be retried.")
    conn.commit()
    return _job_out(row)
