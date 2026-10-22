"""Dashboard endpoints (Phase 6): fleet summary, per-sensor series, stored investigations,
unit catalog and the sensor-registry editor.

Every route is tenant-scoped through get_tenant (the tenant is never a request field),
paginated or capped, and declared in app.core.permissions.ROUTE_PERMISSIONS.
Reads are `data:read` / `investigate:run` (viewers allowed); the registry edit is
`registry:write` (admin only) and is audited by enforce_route_permission like every write.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection

from app.agents.risk import assess_risk
from app.api.dependencies import get_conn, get_tenant, get_tenant_calibration
from app.api.schemas import HealthStatus, RiskLevel
from app.core.tenancy import TenantContext
from app.core.tenant_settings import TenantCalibration
from app.database import queries
from app.onboarding.units import UNITS

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_SERIES_SPAN = timedelta(days=31)
DEFAULT_SERIES_SPAN = timedelta(days=7)
MIN_BUCKET_SECONDS = 60
TREND_WINDOWS = 24      # same window count the agent's fleet_scan uses


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class FleetItem(BaseModel):
    machine_id: str
    production_line: str
    name: str | None = None
    location: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    health_status: HealthStatus | None = None
    max_anomaly_score: int | None = None
    anomalous_reading_count: int | None = None
    reading_count: int | None = None
    risk_level: RiskLevel


class FleetPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[FleetItem]


class SeriesPoint(BaseModel):
    ts: datetime
    avg: float
    min: float
    max: float
    n: int


class SeriesOut(BaseModel):
    machine_id: str
    sensor: str
    unit: str
    normal_min: float | None = None
    normal_max: float | None = None
    since: datetime | None = None
    until: datetime | None = None
    bucket_seconds: int
    points: list[SeriesPoint]


class InvestigationSummary(BaseModel):
    id: str
    created_by_label: str
    endpoint: str
    question: str
    machine_id: str | None
    intent: str | None
    risk_level: str | None
    created_at: datetime


class InvestigationDetail(InvestigationSummary):
    result: dict[str, Any]


class UnitOut(BaseModel):
    unit: str
    dimension: str


class SensorPatch(BaseModel):
    """Editable sensor-registry fields. A field sent as null clears it (range/threshold)."""
    model_config = ConfigDict(extra="forbid")
    normal_min: float | None = None
    normal_max: float | None = None
    z_score_threshold: float | None = Field(default=None, gt=0, le=20)
    enabled: bool | None = None


class SensorOut(BaseModel):
    sensor_name: str
    unit: str
    normal_min: float | None = None
    normal_max: float | None = None
    z_score_threshold: float | None = None
    enabled: bool


# ---------------------------------------------------------------------
# Fleet summary
# ---------------------------------------------------------------------

@router.get("/fleet/summary", response_model=FleetPage)
def fleet_summary(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
    calibration: TenantCalibration = Depends(get_tenant_calibration),
) -> dict:
    """One page of machines with their latest window and the sustained-rate risk_level —
    the same assess_risk rule /investigate's fleet scan uses, over the same 24 windows."""
    tid = tenant.tenant_id
    rows = queries.get_fleet_summary_page(conn, limit, offset, tenant_id=tid)
    ids = [r["machine_id"] for r in rows]
    trends = queries.get_machines_trend(conn, ids, TREND_WINDOWS, tenant_id=tid)
    high = queries.get_machines_recent_high_anomalies(conn, ids, TREND_WINDOWS, tenant_id=tid)
    items = []
    for r in rows:
        if r["window_start"] is None:
            risk = "UNKNOWN"
        else:
            metrics = [{"severity": "HIGH", "detected_at": None}] if r["machine_id"] in high else []
            risk = assess_risk(r, metrics, trends.get(r["machine_id"], []),
                               thresholds=calibration.risk_thresholds)
        items.append({**r, "risk_level": risk})
    return {"total": queries.count_machines(conn, tenant_id=tid),
            "limit": limit, "offset": offset, "items": items}


# ---------------------------------------------------------------------
# Per-sensor series (long-format readings, bucketed in SQL)
# ---------------------------------------------------------------------

@router.get("/machines/{machine_id}/series", response_model=SeriesOut)
def machine_series(
    machine_id: str,
    sensor: str = Query(description="A sensor_name from GET /sensors."),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    days: int | None = Query(default=None, ge=1, le=31, description="Window length when 'since' is omitted (default 7)."),
    max_points: int = Query(default=300, ge=10, le=1000),
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> dict:
    """Bucketed avg/min/max of one sensor plus its registry unit and normal range.

    Defaults to the 7 days ending at the machine's newest reading; the window may not
    exceed 31 days (422) and never returns more than `max_points` buckets."""
    tid = tenant.tenant_id
    if queries.get_machine(conn, machine_id, tenant_id=tid) is None:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id!r} not found.")
    registry = {s["sensor_name"]: s for s in queries.list_sensor_registry(conn, tenant_id=tid)}
    spec = registry.get(sensor)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor!r} is not declared.")
    base = {"machine_id": machine_id, "sensor": sensor, "unit": spec["unit"],
            "normal_min": spec["normal_min"], "normal_max": spec["normal_max"]}

    if until is None:
        until = queries.get_latest_reading_ts(conn, machine_id, sensor, tenant_id=tid)
        if until is None:
            return {**base, "since": None, "until": None,
                    "bucket_seconds": MIN_BUCKET_SECONDS, "points": []}
    if since is None:
        since = until - (timedelta(days=days) if days else DEFAULT_SERIES_SPAN)
    if since >= until:
        raise HTTPException(status_code=422, detail="'since' must be earlier than 'until'.")
    if until - since > MAX_SERIES_SPAN:
        raise HTTPException(status_code=422, detail="The time range may not exceed 31 days.")
    bucket = max(MIN_BUCKET_SECONDS, math.ceil((until - since).total_seconds() / max_points))
    rows = queries.get_reading_series(conn, machine_id, sensor, since, until, bucket,
                                      max_points, tenant_id=tid)
    points = [{"ts": r["bucket"], "avg": float(r["avg_value"]), "min": float(r["min_value"]),
               "max": float(r["max_value"]), "n": int(r["n"])} for r in rows]
    return {**base, "since": since, "until": until, "bucket_seconds": bucket, "points": points}


# ---------------------------------------------------------------------
# Stored investigations
# ---------------------------------------------------------------------

def _inv(row: dict) -> dict:
    return {**row, "id": str(row["id"])}


@router.get("/investigations", response_model=list[InvestigationSummary])
def list_investigations(
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    machine_id: str | None = Query(default=None, max_length=64),
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> list[dict]:
    """This tenant's past investigations, newest first (summary rows, no result JSON)."""
    return [_inv(r) for r in queries.list_investigations(
        conn, limit, offset, machine_id, tenant_id=tenant.tenant_id)]


@router.get("/investigations/{investigation_id}", response_model=InvestigationDetail)
def get_investigation(
    investigation_id: str,
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> dict:
    try:
        uuid.UUID(investigation_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Investigation not found.") from None
    row = queries.get_investigation(conn, investigation_id, tenant_id=tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return _inv(row)


# ---------------------------------------------------------------------
# Unit catalog and sensor-registry editor
# ---------------------------------------------------------------------

@router.get("/units", response_model=list[UnitOut])
def list_units(tenant: TenantContext = Depends(get_tenant)) -> list[dict]:
    """The units a mapping may declare (app.onboarding.units), so clients don't hardcode them."""
    return [{"unit": u, "dimension": d[0]} for u, d in sorted(UNITS.items(), key=lambda kv: (kv[1][0], kv[0]))]


@router.post("/sensors/{sensor_name}", response_model=SensorOut)
def edit_sensor(
    sensor_name: str,
    body: SensorPatch,
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> dict:
    """Edit one existing sensor's normal range, z-score threshold or enabled flag (admin).

    Takes effect for detection runs after the edit; stored anomalies are not recomputed."""
    tid = tenant.tenant_id
    fields = {k: getattr(body, k) for k in body.model_fields_set}
    if not fields:
        raise HTTPException(status_code=422, detail="Nothing to change.")
    if fields.get("enabled", True) is None:
        raise HTTPException(status_code=422, detail="'enabled' cannot be null.")
    current = {s["sensor_name"]: s for s in queries.list_sensor_registry(conn, tenant_id=tid)}.get(sensor_name)
    if current is None:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_name!r} is not declared.")
    merged = {**current, **fields}
    lo, hi = merged["normal_min"], merged["normal_max"]
    if lo is not None and hi is not None and not lo < hi:
        raise HTTPException(status_code=422, detail="normal_min must be less than normal_max.")
    row = queries.update_sensor_registry_entry(conn, sensor_name, fields, tenant_id=tid)
    conn.commit()
    return row
