"""
Database query layer for the Industrial Intelligence Agent.

These functions are the actual data-access surface the Phase 4 FastAPI
endpoints and Phase 5 LangGraph tools will call. Written and tested against
a real PostgreSQL instance *before* FastAPI/LangGraph exist, per the phase
plan, so the query logic is validated in isolation.

Deliberately plain SQL via SQLAlchemy's text(), not an ORM: every function
here is either a straightforward filtered/aggregated read or a bounded
ranking query. An ORM would add a mapping layer with no benefit for
read-mostly reporting queries like these.

Every function takes a Connection as its first argument rather than
opening its own connection, so callers (tests, FastAPI's dependency
injection, agent tools) control connection lifecycle and pooling. Typed
as `Connection`, not the broader `Connectable` (which also matches
`Engine`): every real caller passes a request-scoped Connection already
checked out from the pool (app.api.dependencies.get_conn), and calling
`.execute()` on a bare Engine directly — which Connectable would also
accept — bypasses that pooling/lifecycle control this module's docstring
just promised.

Tenant isolation (production-readiness fix 1). Every function that reads
company data takes a REQUIRED keyword-only `tenant_id` and filters every
table it touches on it, so a caller can only ever see its own tenant's
rows. It is keyword-only and has no default on purpose: forgetting it is a
TypeError at the call site, not a silent cross-tenant read. Adding a query
here without a tenant filter fails tests/test_tenant_isolation.py's static
check. The one exception is `get_ai4i_failure_mode_rates`, which reads the
public UCI AI4I reference dataset — shared reference data, not company data.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

_UTC_NOW = "(NOW() AT TIME ZONE 'UTC')"


def list_machines(conn: Connection, *, tenant_id: str) -> list[dict[str, Any]]:
    """All machines in the tenant's fleet with static metadata. Backs GET /machines."""
    rows = conn.execute(text("""
        SELECT machine_id, production_line, type, name, install_date, location
        FROM machines
        WHERE tenant_id = :tenant_id
        ORDER BY machine_id
    """), {"tenant_id": tenant_id}).mappings().all()
    return [dict(r) for r in rows]


def get_machine(conn: Connection, machine_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Single machine's metadata. Backs GET /machines/{machine_id}."""
    row = conn.execute(
        text("""
            SELECT machine_id, production_line, type, name, install_date, location
            FROM machines
            WHERE tenant_id = :tenant_id AND machine_id = :machine_id
        """),
        {"tenant_id": tenant_id, "machine_id": machine_id},
    ).mappings().first()
    return dict(row) if row else None


def get_sensor_summary(
    conn: Connection,
    machine_id: str,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 168,  # default: last 7 days of hourly windows
    *,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Hourly sensor history for one machine, most recent first.
    Backs the sensor-history evidence a root-cause investigation needs."""
    query = """
        SELECT window_start, window_end, avg_air_temp_k, avg_process_temp_k,
               avg_rotational_speed_rpm, avg_torque_nm, tool_wear_min,
               avg_production_rate, avg_energy_consumption_kwh, avg_defect_rate,
               reading_count, anomalous_reading_count, max_anomaly_score, health_status
        FROM sensor_summary
        WHERE tenant_id = :tenant_id AND machine_id = :machine_id
    """
    params: dict[str, Any] = {
        "tenant_id": tenant_id, "machine_id": machine_id, "limit": limit}
    if since is not None:
        query += " AND window_start >= :since"
        params["since"] = since
    if until is not None:
        query += " AND window_start <= :until"
        params["until"] = until
    query += " ORDER BY window_start DESC LIMIT :limit"

    rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]


def get_machine_health(conn: Connection, machine_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Latest known health snapshot for one machine.
    Backs GET /machines/{machine_id}/health.

    Audit F11: avg_defect_rate and tool_wear_min added — the Streamlit
    Overview tab's "key metrics" row previously had nothing to show but
    the window's own start/end timestamps, because this query never
    selected any of sensor_summary's actual measured columns beyond the
    anomaly-count bookkeeping fields.
    """
    row = conn.execute(
        text("""
            SELECT machine_id, window_start, window_end, health_status,
                   max_anomaly_score, anomalous_reading_count, reading_count,
                   avg_defect_rate, tool_wear_min
            FROM sensor_summary
            WHERE tenant_id = :tenant_id AND machine_id = :machine_id
            ORDER BY window_start DESC
            LIMIT 1
        """),
        {"tenant_id": tenant_id, "machine_id": machine_id},
    ).mappings().first()
    return dict(row) if row else None


def get_machine_anomalies(
    conn: Connection,
    machine_id: str,
    since: datetime | None = None,
    severity: str | None = None,
    limit: int = 100,
    *,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Flagged anomaly events for one machine, most recent first.
    Backs GET /machines/{machine_id}/anomalies."""
    query = """
        SELECT detected_at, anomaly_score, severity, triggered_reasons
        FROM machine_anomalies
        WHERE tenant_id = :tenant_id AND machine_id = :machine_id
    """
    params: dict[str, Any] = {
        "tenant_id": tenant_id, "machine_id": machine_id, "limit": limit}
    if since is not None:
        query += " AND detected_at >= :since"
        params["since"] = since
    if severity is not None:
        query += " AND severity = :severity"
        params["severity"] = severity
    query += " ORDER BY detected_at DESC LIMIT :limit"

    rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]


def get_maintenance_records(conn: Connection, machine_id: str, *, tenant_id: str) -> list[dict[str, Any]]:
    """Maintenance history for one machine, most recent first."""
    rows = conn.execute(
        text("""
            SELECT event_date, event_type, technician_notes, resolved
            FROM maintenance_records
            WHERE tenant_id = :tenant_id AND machine_id = :machine_id
            ORDER BY event_date DESC
        """),
        {"tenant_id": tenant_id, "machine_id": machine_id},
    ).mappings().all()
    return [dict(r) for r in rows]


def get_fleet_status(conn: Connection, *, tenant_id: str) -> list[dict[str, Any]]:
    """One row per machine: its most recent health window. Backs
    "which machines currently show abnormal behavior" — a fleet-wide
    scan using each machine's latest window, not a single global timestamp,
    since machines are independent in principle even though this dataset
    happens to have them all aligned to the same hourly grid."""
    rows = conn.execute(text("""
        SELECT DISTINCT ON (s.machine_id)
               s.machine_id, m.production_line, s.window_start, s.health_status,
               s.max_anomaly_score, s.anomalous_reading_count
        FROM sensor_summary s
        JOIN machines m ON m.tenant_id = s.tenant_id AND m.machine_id = s.machine_id
        WHERE s.tenant_id = :tenant_id
        ORDER BY s.machine_id, s.window_start DESC
    """), {"tenant_id": tenant_id}).mappings().all()
    return [dict(r) for r in rows]


def get_fleet_sensor_trend(
    conn: Connection, windows: int = 24, *, tenant_id: str
) -> dict[str, list[dict[str, Any]]]:
    """The last `windows` hourly sensor_summary rows for every machine at
    once, keyed by machine_id. Same shape/ordering as get_sensor_summary,
    just for the whole fleet in one round trip instead of one query per
    machine.

    Phase 9 fix. Exists so fleet_scan (app/agents/nodes.py) can run each
    machine's rows through risk.assess_risk() — the exact same
    sustained-anomalous-rate rule a single-machine /investigate uses —
    instead of flagging on get_fleet_status's single latest hourly window,
    which is noisy enough that a healthy machine's one-off WATCH hour used
    to get it flagged fleet-wide while the same machine's own investigation
    correctly called it LOW risk.
    """
    rows = conn.execute(text("""
        SELECT machine_id, window_start, window_end, health_status,
               max_anomaly_score, anomalous_reading_count, reading_count
        FROM (
            SELECT s.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY s.machine_id ORDER BY s.window_start DESC
                   ) AS rn
            FROM sensor_summary s
            WHERE s.tenant_id = :tenant_id
        ) ranked
        WHERE rn <= :windows
        ORDER BY machine_id, window_start DESC
    """), {"tenant_id": tenant_id, "windows": windows}).mappings().all()

    trend_by_machine: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trend_by_machine.setdefault(row["machine_id"], []).append(dict(row))
    return trend_by_machine


def get_fleet_recent_high_anomalies(conn: Connection, windows: int = 24, *, tenant_id: str) -> set[str]:
    """machine_ids with a HIGH-severity anomaly inside their own last
    `windows` hourly windows — the fleet-wide equivalent of
    risk.assess_risk()'s `_has_recent_high_anomaly` check.

    Computed as one query (a per-machine trend floor timestamp, joined
    against machine_anomalies) rather than a global "detected_at >= NOW() -
    24h" cutoff: the trend floor is exact for every machine regardless of
    whether its data is clock-aligned to "now", which a synthetic dataset
    generated at a fixed reference time is not guaranteed to be.
    """
    rows = conn.execute(text("""
        WITH trend AS (
            SELECT machine_id, window_start,
                   ROW_NUMBER() OVER (
                       PARTITION BY machine_id ORDER BY window_start DESC
                   ) AS rn
            FROM sensor_summary
            WHERE tenant_id = :tenant_id
        ),
        trend_floor AS (
            SELECT machine_id, MIN(window_start) AS floor_ts
            FROM trend
            WHERE rn <= :windows
            GROUP BY machine_id
        )
        SELECT DISTINCT a.machine_id
        FROM machine_anomalies a
        JOIN trend_floor t ON t.machine_id = a.machine_id
        WHERE a.tenant_id = :tenant_id
          AND a.severity = 'HIGH' AND a.detected_at >= t.floor_ts
    """), {"tenant_id": tenant_id, "windows": windows}).mappings().all()
    return {row["machine_id"] for row in rows}


def get_ai4i_failure_mode_rates(conn: Connection) -> dict[str, dict[str, Any]]:
    """Real-world incidence of each AI4I failure mode (TWF/HDF/PWF/OSF/RNF)
    among ai4i_reference's 10,000 published rows.

    P2 cleanup: ai4i_reference was loaded and indexed (Phase 3/schema.sql)
    but no tool ever queried it — the "real data" story in the audit was
    decorative. synthesis.infer_failure_modes already maps every anomaly's
    triggered_reasons to one of these same five codes, so this is the one
    query needed to attach a real base rate to each root-cause candidate
    (see app.agents.synthesis.build_root_cause_candidates) instead of only
    ever reasoning over the synthetic fleet.

    One row per mode: how many of the 10,000 real machines tripped that
    flag, and what share of the real *machine_failure=1* rows (not all
    10,000 — a mode's rate is only meaningful relative to failures, since
    a machine can trip a flag without an overall failure) each mode
    accounts for. Modes can co-occur, so the per-mode shares don't sum to
    1.0 — that's expected, not a bug.

    Not tenant-scoped, on purpose: ai4i_reference is the public UCI dataset
    (shared read-only reference data), not any company's data.
    """
    total_row = conn.execute(
        text("SELECT COUNT(*) AS n FROM ai4i_reference")).mappings().one()
    total_rows = total_row["n"]
    failed_row = conn.execute(
        text("SELECT COUNT(*) AS n FROM ai4i_reference WHERE machine_failure = 1")
    ).mappings().one()
    total_failures = failed_row["n"]

    rows = conn.execute(text("""
        SELECT
            SUM(twf) AS twf, SUM(hdf) AS hdf, SUM(pwf) AS pwf,
            SUM(osf) AS osf, SUM(rnf) AS rnf
        FROM ai4i_reference
        WHERE machine_failure = 1
    """)).mappings().one()

    rates: dict[str, dict[str, Any]] = {}
    for mode in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        count = rows[mode.lower()] or 0
        rates[mode] = {
            "failure_count": count,
            "share_of_real_failures": (count / total_failures) if total_failures else 0.0,
            "total_real_rows": total_rows,
            "total_real_failures": total_failures,
        }
    return rates


def rank_machines_for_inspection(
    conn: Connection, since: datetime, top_n: int = 5, *, tenant_id: str
) -> list[dict[str, Any]]:
    """Rank machines by anomaly volume/severity since a given time.
    Backs "which machines should the maintenance team inspect first" —
    deterministic ranking by count and severity-weighted score, computed
    entirely in SQL so the LLM never invents the ordering."""
    rows = conn.execute(
        text("""
            SELECT machine_id,
                   COUNT(*) AS anomaly_count,
                   SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) AS high_severity_count,
                   MAX(detected_at) AS most_recent_anomaly
            FROM machine_anomalies
            WHERE tenant_id = :tenant_id AND detected_at >= :since
            GROUP BY machine_id
            ORDER BY high_severity_count DESC, anomaly_count DESC
            LIMIT :top_n
        """),
        {"tenant_id": tenant_id, "since": since, "top_n": top_n},
    ).mappings().all()
    return [dict(r) for r in rows]


def get_tenant_settings(conn: Connection, *, tenant_id: str) -> dict[str, Any] | None:
    """This tenant's calibration overrides (production-readiness fix 11/12:
    risk thresholds, RAG coverage floor, machine-ID scheme), or None when
    they haven't set any — see app.core.tenant_settings for how the
    documented defaults are filled in around whatever this returns."""
    row = conn.execute(
        text("SELECT config FROM tenant_settings WHERE tenant_id = :tenant_id"),
        {"tenant_id": tenant_id},
    ).mappings().first()
    return dict(row["config"]) if row else None


def upsert_tenant_settings(conn: Connection, config: dict[str, Any], *, tenant_id: str) -> None:
    """Replace this tenant's calibration config wholesale (see
    app.core.tenant_settings for its shape). Used by
    scripts/manage_tenants.py `calibrate`, never by request-serving code."""
    conn.execute(
        text("""
            INSERT INTO tenant_settings (tenant_id, config, updated_at)
            VALUES (:tenant_id, CAST(:config AS JSONB), NOW() AT TIME ZONE 'UTC')
            ON CONFLICT (tenant_id)
            DO UPDATE SET config = EXCLUDED.config, updated_at = EXCLUDED.updated_at
        """),
        {"tenant_id": tenant_id, "config": json.dumps(config)},
    )


def insert_investigation_audit(
    conn: Connection,
    *,
    tenant_id: str,
    request_id: str | None,
    endpoint: str,
    machine_id: str | None,
    intent: str | None,
    risk_level: str | None,
    authenticated: bool,
    question: str,
) -> None:
    """Record who asked what, of which machine, and what the agent
    concluded — production-readiness fix 19. Called once, after a
    successful /investigate or /chat call (app/api/routes.py); a call that
    raises before returning is never audited here, which is fine — the
    thing worth an audit trail is "tenant X's credential was used to
    investigate machine Y", not the request's outcome."""
    conn.execute(
        text("""
            INSERT INTO investigation_audit_log
                (tenant_id, request_id, endpoint, machine_id, intent,
                 risk_level, authenticated, question)
            VALUES (:tenant_id, :request_id, :endpoint, :machine_id, :intent,
                    :risk_level, :authenticated, :question)
        """),
        {
            "tenant_id": tenant_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "machine_id": machine_id,
            "intent": intent,
            "risk_level": risk_level,
            "authenticated": authenticated,
            "question": question,
        },
    )


def get_ingestion_checkpoint(conn: Connection, source: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Where an incremental ingestion run for this tenant/source left off
    (production-readiness fix 17) — see scripts/incremental_load.py."""
    row = conn.execute(
        text("""
            SELECT source, last_window_end, last_run_at, rows_processed
            FROM ingestion_checkpoints
            WHERE tenant_id = :tenant_id AND source = :source
        """),
        {"tenant_id": tenant_id, "source": source},
    ).mappings().first()
    return dict(row) if row else None


def upsert_ingestion_checkpoint(
    conn: Connection,
    source: str,
    last_window_end: datetime,
    rows_processed: int,
    *,
    tenant_id: str,
) -> None:
    """Advance (or create) this tenant/source's incremental-ingestion
    checkpoint. `rows_processed` is this run's count, not a running total —
    scripts/incremental_load.py logs the total separately."""
    conn.execute(
        text("""
            INSERT INTO ingestion_checkpoints
                (tenant_id, source, last_window_end, last_run_at, rows_processed)
            VALUES (:tenant_id, :source, :last_window_end, NOW() AT TIME ZONE 'UTC', :rows_processed)
            ON CONFLICT (tenant_id, source) DO UPDATE SET
                last_window_end = EXCLUDED.last_window_end,
                last_run_at = EXCLUDED.last_run_at,
                rows_processed = EXCLUDED.rows_processed
        """),
        {
            "tenant_id": tenant_id,
            "source": source,
            "last_window_end": last_window_end,
            "rows_processed": rows_processed,
        },
    )


# ---------------------------------------------------------------------
# Generic sensor model (migration 0005): sensor_registry + sensor_readings
# ---------------------------------------------------------------------

def list_sensor_registry(conn: Connection, *, tenant_id: str) -> list[dict[str, Any]]:
    """This tenant's declared sensors and their anomaly-rule settings."""
    rows = conn.execute(text("""
        SELECT sensor_name, unit, normal_min, normal_max, z_score_threshold, enabled
        FROM sensor_registry
        WHERE tenant_id = :tenant_id
        ORDER BY sensor_name
    """), {"tenant_id": tenant_id}).mappings().all()
    return [dict(r) for r in rows]


def upsert_sensor_registry(
    conn: Connection, sensors: list[dict[str, Any]], *, tenant_id: str
) -> int:
    """Create or update this tenant's sensors (idempotent on sensor_name)."""
    if not sensors:
        return 0
    rows = [{
        "tenant_id": tenant_id,
        "sensor_name": s["sensor_name"],
        "unit": s["unit"],
        "normal_min": s.get("normal_min"),
        "normal_max": s.get("normal_max"),
        "z_score_threshold": s.get("z_score_threshold"),
        "enabled": s.get("enabled", True),
    } for s in sensors]
    conn.execute(text("""
        INSERT INTO sensor_registry
            (tenant_id, sensor_name, unit, normal_min, normal_max, z_score_threshold, enabled)
        VALUES (:tenant_id, :sensor_name, :unit, :normal_min, :normal_max,
                :z_score_threshold, :enabled)
        ON CONFLICT (tenant_id, sensor_name) DO UPDATE SET
            unit = EXCLUDED.unit, normal_min = EXCLUDED.normal_min,
            normal_max = EXCLUDED.normal_max,
            z_score_threshold = EXCLUDED.z_score_threshold, enabled = EXCLUDED.enabled
    """), rows)
    return len(rows)


def upsert_machines(
    conn: Connection, machines: list[dict[str, Any]], *, tenant_id: str
) -> int:
    """Create or update machine rows (machine_id, production_line, type)."""
    if not machines:
        return 0
    def _clean(value):   # pandas hands back NaN for "no value"; the column wants NULL
        return None if value is None or (isinstance(value, float) and value != value) or value == "" else value

    rows = [{
        "tenant_id": tenant_id,
        "machine_id": m["machine_id"],
        "production_line": _clean(m.get("production_line")) or "UNASSIGNED",
        "type": _clean(m.get("type")),
    } for m in machines]
    conn.execute(text("""
        INSERT INTO machines (tenant_id, machine_id, production_line, type)
        VALUES (:tenant_id, :machine_id, :production_line, :type)
        ON CONFLICT (tenant_id, machine_id) DO UPDATE SET
            production_line = EXCLUDED.production_line, type = EXCLUDED.type
    """), rows)
    return len(rows)


def insert_sensor_readings(
    conn: Connection, readings: list[dict[str, Any]], *, tenant_id: str
) -> int:
    """Bulk-insert long-format readings. Idempotent: the primary key
    (tenant_id, machine_id, sensor_name, ts) is the conflict target and an
    already-present reading is left untouched (first write wins, same as
    scripts/incremental_load.py). Returns the number actually inserted."""
    if not readings:
        return 0
    rows = [{
        "tenant_id": tenant_id, "machine_id": r["machine_id"],
        "sensor_name": r["sensor_name"], "ts": r["ts"], "value": r["value"],
    } for r in readings]
    before = conn.execute(text(
        "SELECT COUNT(*) FROM sensor_readings WHERE tenant_id = :tenant_id"),
        {"tenant_id": tenant_id}).scalar()
    conn.execute(text("""
        INSERT INTO sensor_readings (tenant_id, machine_id, sensor_name, ts, value)
        VALUES (:tenant_id, :machine_id, :sensor_name, :ts, :value)
        ON CONFLICT (tenant_id, machine_id, sensor_name, ts) DO NOTHING
    """), rows)
    after = conn.execute(text(
        "SELECT COUNT(*) FROM sensor_readings WHERE tenant_id = :tenant_id"),
        {"tenant_id": tenant_id}).scalar()
    return after - before


def get_sensor_readings(
    conn: Connection,
    machine_id: str,
    sensor_name: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 1000,
    *,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Raw readings for one machine (optionally one sensor), newest first."""
    query = """
        SELECT sensor_name, ts, value
        FROM sensor_readings
        WHERE tenant_id = :tenant_id AND machine_id = :machine_id
    """
    params: dict[str, Any] = {
        "tenant_id": tenant_id, "machine_id": machine_id, "limit": limit}
    if sensor_name is not None:
        query += " AND sensor_name = :sensor_name"
        params["sensor_name"] = sensor_name
    if since is not None:
        query += " AND ts >= :since"
        params["since"] = since
    if until is not None:
        query += " AND ts <= :until"
        params["until"] = until
    query += " ORDER BY ts DESC, sensor_name LIMIT :limit"
    rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]


def merge_tenant_anomaly_settings(
    conn: Connection, anomaly: dict[str, Any], *, tenant_id: str
) -> None:
    """Set ONLY the `anomaly` key of this tenant's settings, leaving the
    risk / RAG / machine-id keys alone (upsert_tenant_settings, by contrast,
    replaces the whole config)."""
    conn.execute(
        text("""
            INSERT INTO tenant_settings (tenant_id, config, updated_at)
            VALUES (:tenant_id, jsonb_build_object('anomaly', CAST(:anomaly AS JSONB)),
                    NOW() AT TIME ZONE 'UTC')
            ON CONFLICT (tenant_id) DO UPDATE SET
                config = tenant_settings.config || EXCLUDED.config,
                updated_at = EXCLUDED.updated_at
        """),
        {"tenant_id": tenant_id, "anomaly": json.dumps(anomaly)},
    )


def get_tenant_by_key_hash(conn: Connection, api_key_hash: str) -> dict[str, Any] | None:
    """The active tenant whose credential hashes to `api_key_hash`, or None.

    This is the authentication lookup (app.api.dependencies.get_tenant), so
    it is the one query here that is keyed by credential rather than by
    tenant_id — the tenant id is its *output*.
    """
    row = conn.execute(
        text("""
            SELECT tenant_id, name
            FROM tenants
            WHERE api_key_hash = :api_key_hash AND active
        """),
        {"api_key_hash": api_key_hash},
    ).mappings().first()
    return dict(row) if row else None


# ---------------------------------------------------------------------
# Uploads, jobs, rejected rows (migration 0007, Phase 3)
# ---------------------------------------------------------------------
# The API reads/creates these through its tenant-bound connection; the worker
# uses the same functions after claiming a job (the claim itself is the one
# cross-tenant step and lives in app/jobs/worker.py, as a SECURITY DEFINER call).

_UPLOAD_COLS = ("upload_id, filename, content_type, storage_key, status, size_bytes, sha256, "
                "mapping, created_at, updated_at")
_JOB_COLS = ("job_id, upload_id, type, status, stage, created_at, started_at, finished_at, "
             "run_after, rows_in, rows_loaded, rows_rejected, error, retry_count, max_retries")


def create_upload(
    conn: Connection, upload_id: str, filename: str, content_type: str, storage_key: str,
    *, tenant_id: str,
) -> dict[str, Any]:
    row = conn.execute(text(f"""
        INSERT INTO uploads (upload_id, tenant_id, filename, content_type, storage_key)
        VALUES (:upload_id, :tenant_id, :filename, :content_type, :storage_key)
        RETURNING {_UPLOAD_COLS}
    """), {"upload_id": upload_id, "tenant_id": tenant_id, "filename": filename,
           "content_type": content_type, "storage_key": storage_key}).mappings().one()
    return dict(row)


def get_upload(conn: Connection, upload_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    row = conn.execute(text(f"""
        SELECT {_UPLOAD_COLS} FROM uploads
        WHERE tenant_id = :tenant_id AND upload_id = :upload_id
    """), {"tenant_id": tenant_id, "upload_id": upload_id}).mappings().first()
    return dict(row) if row else None


def list_uploads(
    conn: Connection, limit: int = 50, offset: int = 0, *, tenant_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(text(f"""
        SELECT {_UPLOAD_COLS} FROM uploads
        WHERE tenant_id = :tenant_id
        ORDER BY created_at DESC, upload_id LIMIT :limit OFFSET :offset
    """), {"tenant_id": tenant_id, "limit": limit, "offset": offset}).mappings().all()
    return [dict(r) for r in rows]


def set_upload_content(
    conn: Connection, upload_id: str, size_bytes: int, sha256: str, *, tenant_id: str,
) -> dict[str, Any] | None:
    """Record the stored object's size/hash; a new file invalidates any earlier mapping."""
    row = conn.execute(text(f"""
        UPDATE uploads SET status = 'uploaded', size_bytes = :size_bytes, sha256 = :sha256,
               mapping = NULL, updated_at = {_UTC_NOW}
        WHERE tenant_id = :tenant_id AND upload_id = :upload_id
        RETURNING {_UPLOAD_COLS}
    """), {"tenant_id": tenant_id, "upload_id": upload_id,
           "size_bytes": size_bytes, "sha256": sha256}).mappings().first()
    return dict(row) if row else None


def set_upload_mapping(
    conn: Connection, upload_id: str, mapping: dict[str, Any], *, tenant_id: str,
) -> dict[str, Any] | None:
    row = conn.execute(text(f"""
        UPDATE uploads SET status = 'mapped', mapping = CAST(:mapping AS JSONB),
               updated_at = {_UTC_NOW}
        WHERE tenant_id = :tenant_id AND upload_id = :upload_id
        RETURNING {_UPLOAD_COLS}
    """), {"tenant_id": tenant_id, "upload_id": upload_id,
           "mapping": json.dumps(mapping)}).mappings().first()
    return dict(row) if row else None


def create_job(
    conn: Connection, job_id: str, upload_id: str, max_retries: int, *, tenant_id: str,
) -> dict[str, Any]:
    row = conn.execute(text(f"""
        INSERT INTO jobs (job_id, tenant_id, upload_id, max_retries)
        VALUES (:job_id, :tenant_id, :upload_id, :max_retries)
        RETURNING {_JOB_COLS}
    """), {"job_id": job_id, "tenant_id": tenant_id, "upload_id": upload_id,
           "max_retries": max_retries}).mappings().one()
    return dict(row)


def get_job(conn: Connection, job_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    row = conn.execute(text(f"""
        SELECT {_JOB_COLS} FROM jobs
        WHERE tenant_id = :tenant_id AND job_id = :job_id
    """), {"tenant_id": tenant_id, "job_id": job_id}).mappings().first()
    return dict(row) if row else None


def list_jobs(
    conn: Connection, status: str | None = None, limit: int = 50, offset: int = 0,
    *, tenant_id: str,
) -> list[dict[str, Any]]:
    query = f"SELECT {_JOB_COLS} FROM jobs WHERE tenant_id = :tenant_id"
    params: dict[str, Any] = {"tenant_id": tenant_id, "limit": limit, "offset": offset}
    if status is not None:
        query += " AND status = :status"
        params["status"] = status
    query += " ORDER BY created_at DESC, job_id LIMIT :limit OFFSET :offset"
    return [dict(r) for r in conn.execute(text(query), params).mappings().all()]


def get_active_job_for_upload(
    conn: Connection, upload_id: str, *, tenant_id: str,
) -> dict[str, Any] | None:
    row = conn.execute(text(f"""
        SELECT {_JOB_COLS} FROM jobs
        WHERE tenant_id = :tenant_id AND upload_id = :upload_id
          AND status IN ('queued', 'running')
        ORDER BY created_at DESC LIMIT 1
    """), {"tenant_id": tenant_id, "upload_id": upload_id}).mappings().first()
    return dict(row) if row else None


def set_job_stage(conn: Connection, job_id: str, stage: str, *, tenant_id: str) -> None:
    """Record the current pipeline stage and refresh the heartbeat (locked_at)."""
    conn.execute(text(f"""
        UPDATE jobs SET stage = :stage, locked_at = {_UTC_NOW}
        WHERE tenant_id = :tenant_id AND job_id = :job_id AND status = 'running'
    """), {"tenant_id": tenant_id, "job_id": job_id, "stage": stage})


def set_job_counts(
    conn: Connection, job_id: str, *, rows_in: int, rows_loaded: int, rows_rejected: int,
    tenant_id: str,
) -> None:
    conn.execute(text("""
        UPDATE jobs SET rows_in = :rows_in, rows_loaded = :rows_loaded,
               rows_rejected = :rows_rejected
        WHERE tenant_id = :tenant_id AND job_id = :job_id
    """), {"tenant_id": tenant_id, "job_id": job_id, "rows_in": rows_in,
           "rows_loaded": rows_loaded, "rows_rejected": rows_rejected})


def finish_job(
    conn: Connection, job_id: str, *, status: str, error: str | None, tenant_id: str,
) -> None:
    if status not in ("succeeded", "failed"):
        raise ValueError(f"finish_job: status must be succeeded/failed, got {status!r}")
    conn.execute(text(f"""
        UPDATE jobs SET status = :status, error = :error, stage = NULL, locked_at = NULL,
               finished_at = {_UTC_NOW}
        WHERE tenant_id = :tenant_id AND job_id = :job_id
    """), {"tenant_id": tenant_id, "job_id": job_id, "status": status, "error": error})


def requeue_job(
    conn: Connection, job_id: str, *, error: str, delay_seconds: float, tenant_id: str,
) -> None:
    """Put a running job back on the queue after a transient failure."""
    conn.execute(text(f"""
        UPDATE jobs SET status = 'queued', retry_count = retry_count + 1, error = :error,
               stage = NULL, locked_at = NULL,
               run_after = {_UTC_NOW} + make_interval(secs => :delay)
        WHERE tenant_id = :tenant_id AND job_id = :job_id
    """), {"tenant_id": tenant_id, "job_id": job_id, "error": error, "delay": delay_seconds})


def retry_failed_job(
    conn: Connection, job_id: str, extra_retries: int, *, tenant_id: str,
) -> dict[str, Any] | None:
    """Manually re-queue a FAILED job (any other status -> None). Grants
    `extra_retries` further automatic retries on top of those already used."""
    row = conn.execute(text(f"""
        UPDATE jobs SET status = 'queued', retry_count = retry_count + 1,
               max_retries = retry_count + 1 + :extra, stage = NULL, locked_at = NULL,
               finished_at = NULL, run_after = {_UTC_NOW}
        WHERE tenant_id = :tenant_id AND job_id = :job_id AND status = 'failed'
        RETURNING {_JOB_COLS}
    """), {"tenant_id": tenant_id, "job_id": job_id, "extra": extra_retries}).mappings().first()
    return dict(row) if row else None


def replace_rejected_rows(
    conn: Connection, job_id: str, rows: list[dict[str, Any]], *, tenant_id: str,
) -> int:
    """Store this job's rejected rows, replacing any from an earlier attempt
    (a retry must not list the same rejection twice)."""
    conn.execute(text("DELETE FROM rejected_rows WHERE tenant_id = :tenant_id AND job_id = :job_id"),
                 {"tenant_id": tenant_id, "job_id": job_id})
    if not rows:
        return 0
    conn.execute(text("""
        INSERT INTO rejected_rows (tenant_id, job_id, row_number, code, reason, raw)
        VALUES (:tenant_id, :job_id, :row_number, :code, :reason, CAST(:raw AS JSONB))
    """), [{
        "tenant_id": tenant_id, "job_id": job_id, "row_number": r.get("row_number"),
        "code": r["code"], "reason": r["reason"],
        "raw": None if r.get("raw") is None else json.dumps(r["raw"]),
    } for r in rows])
    return len(rows)


def list_rejected_rows(
    conn: Connection, job_id: str, limit: int = 100, offset: int = 0, *, tenant_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(text("""
        SELECT row_number, code, reason, raw FROM rejected_rows
        WHERE tenant_id = :tenant_id AND job_id = :job_id
        ORDER BY id LIMIT :limit OFFSET :offset
    """), {"tenant_id": tenant_id, "job_id": job_id, "limit": limit, "offset": offset}).mappings().all()
    return [dict(r) for r in rows]


# --- worker writes -------------------------------------------------------

def bulk_insert_sensor_readings(
    conn: Connection, machine_ids: list[str], sensor_names: list[str],
    timestamps: list[datetime], values: list[float], *, tenant_id: str,
) -> int:
    """One-statement bulk insert (parallel arrays through unnest). Same
    idempotency as insert_sensor_readings (first write wins) but the returned
    count is the statement's own rowcount, with no per-chunk COUNT(*) scans."""
    if not machine_ids:
        return 0
    result = conn.execute(text("""
        INSERT INTO sensor_readings (tenant_id, machine_id, sensor_name, ts, value)
        SELECT :tenant_id, m, s, t, v
        FROM unnest(CAST(:m AS text[]), CAST(:s AS text[]), CAST(:t AS timestamp[]),
                    CAST(:v AS double precision[])) AS u(m, s, t, v)
        ON CONFLICT (tenant_id, machine_id, sensor_name, ts) DO NOTHING
    """), {"tenant_id": tenant_id, "m": machine_ids, "s": sensor_names,
           "t": timestamps, "v": values})
    return result.rowcount


def upsert_machines_keep_known(
    conn: Connection, machines: list[dict[str, Any]], *, tenant_id: str,
) -> int:
    """Like upsert_machines, but an upload that doesn't carry a machine's
    line/type (UNASSIGNED / NULL) never erases what an earlier one recorded."""
    if not machines:
        return 0
    rows = [{
        "tenant_id": tenant_id, "machine_id": m["machine_id"],
        "production_line": (m.get("production_line") or "UNASSIGNED"),
        "type": m.get("type") if isinstance(m.get("type"), str) and m.get("type") else None,
    } for m in machines]
    conn.execute(text("""
        INSERT INTO machines (tenant_id, machine_id, production_line, type)
        VALUES (:tenant_id, :machine_id, :production_line, :type)
        ON CONFLICT (tenant_id, machine_id) DO UPDATE SET
            production_line = CASE WHEN EXCLUDED.production_line = 'UNASSIGNED'
                                   THEN machines.production_line ELSE EXCLUDED.production_line END,
            type = COALESCE(EXCLUDED.type, machines.type)
    """), rows)
    return len(rows)


def fetch_readings_for_detection(
    conn: Connection, machine_ids: list[str], since: list[datetime], sensor_names: list[str],
    lookback: int, *, tenant_id: str,
) -> list[tuple[str, str, datetime, float]]:
    """For each machine: every stored reading at or after its `since` plus the
    readings at the `lookback` distinct timestamps before it (the rolling
    baseline's context). `machine_ids` and `since` are parallel lists."""
    rows = conn.execute(text("""
        WITH w AS (
            SELECT * FROM unnest(CAST(:m AS text[]), CAST(:s AS timestamp[])) AS w(machine_id, since)
        ), ctx AS (
            SELECT machine_id, MIN(ts) AS cutoff FROM (
                SELECT d.machine_id, d.ts,
                       DENSE_RANK() OVER (PARTITION BY d.machine_id ORDER BY d.ts DESC) AS rk
                FROM (SELECT DISTINCT r0.machine_id, r0.ts
                      FROM sensor_readings r0 JOIN w ON w.machine_id = r0.machine_id
                      WHERE r0.tenant_id = :tenant_id AND r0.ts < w.since
                        AND r0.sensor_name = ANY(CAST(:names AS text[]))) d
            ) q WHERE rk <= :lookback GROUP BY machine_id
        )
        SELECT r.machine_id, r.sensor_name, r.ts, r.value
        FROM sensor_readings r
        JOIN w ON w.machine_id = r.machine_id
        LEFT JOIN ctx ON ctx.machine_id = r.machine_id
        WHERE r.tenant_id = :tenant_id AND r.sensor_name = ANY(CAST(:names AS text[]))
          AND r.ts >= COALESCE(ctx.cutoff, w.since)
        ORDER BY r.machine_id, r.ts, r.sensor_name
    """), {"tenant_id": tenant_id, "m": machine_ids, "s": since, "names": sensor_names,
           "lookback": lookback}).all()
    return [(r[0], r[1], r[2], float(r[3])) for r in rows]


_SUMMARY_COLUMNS = (
    "window_end", "avg_air_temp_k", "avg_process_temp_k", "avg_rotational_speed_rpm",
    "avg_torque_nm", "tool_wear_min", "avg_production_rate", "avg_energy_consumption_kwh",
    "avg_defect_rate", "reading_count", "anomalous_reading_count", "max_anomaly_score",
    "health_status",
)


def upsert_sensor_summary(
    conn: Connection, rows: list[dict[str, Any]], *, tenant_id: str,
) -> int:
    """Insert hourly rollups, UPDATING a window that already exists: an upload
    that adds readings to an hour already summarised must refresh that hour."""
    if not rows:
        return 0
    params = [{"tenant_id": tenant_id, "machine_id": r["machine_id"],
               "window_start": r["window_start"],
               **{c: r.get(c) for c in _SUMMARY_COLUMNS}} for r in rows]
    cols = ", ".join(_SUMMARY_COLUMNS)
    conn.execute(text(f"""
        INSERT INTO sensor_summary (tenant_id, machine_id, window_start, {cols})
        VALUES (:tenant_id, :machine_id, :window_start, {", ".join(":" + c for c in _SUMMARY_COLUMNS)})
        ON CONFLICT (tenant_id, machine_id, window_start) DO UPDATE SET
            {", ".join(f"{c} = EXCLUDED.{c}" for c in _SUMMARY_COLUMNS)}
    """), params)
    return len(params)


def upsert_machine_anomalies(
    conn: Connection, rows: list[dict[str, Any]], *, tenant_id: str,
) -> int:
    """Insert anomalies keyed by (tenant, machine, detected_at); a re-detected
    one is refreshed in place, so re-running an upload adds nothing."""
    if not rows:
        return 0
    params = [{"tenant_id": tenant_id, "machine_id": r["machine_id"],
               "detected_at": r["detected_at"], "anomaly_score": int(r["anomaly_score"]),
               "severity": r["severity"], "triggered_reasons": r.get("triggered_reasons")}
              for r in rows]
    conn.execute(text("""
        INSERT INTO machine_anomalies
            (tenant_id, machine_id, detected_at, anomaly_score, severity, triggered_reasons)
        VALUES (:tenant_id, :machine_id, :detected_at, :anomaly_score, :severity, :triggered_reasons)
        ON CONFLICT (tenant_id, machine_id, detected_at) DO UPDATE SET
            anomaly_score = EXCLUDED.anomaly_score, severity = EXCLUDED.severity,
            triggered_reasons = EXCLUDED.triggered_reasons
    """), params)
    return len(params)


# ---------------------------------------------------------------------
# Users and the general audit log (migration 0008, Phase 4)
# ---------------------------------------------------------------------
# The API role cannot SELECT users.password_hash (column-level grants), so these
# functions never name it; the hash is read only by the SECURITY DEFINER login lookup
# (app.database.tenant_context.lookup_user_for_login).

_USER_COLS = "id, email, role, is_active, created_at, last_login_at"


def create_user(conn: Connection, user_id: str, email: str, password_hash: str, role: str,
                *, tenant_id: str) -> dict[str, Any]:
    """Insert a user. Raises sqlalchemy.exc.IntegrityError if (tenant, email) exists."""
    return dict(conn.execute(
        text(f"""
            INSERT INTO users (id, tenant_id, email, password_hash, role)
            VALUES (:id, :tenant_id, :email, :password_hash, :role)
            RETURNING {_USER_COLS}
        """),
        {"id": user_id, "tenant_id": tenant_id, "email": email,
         "password_hash": password_hash, "role": role},
    ).mappings().one())


def list_users(conn: Connection, *, tenant_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        text(f"SELECT {_USER_COLS} FROM users WHERE tenant_id = :tenant_id ORDER BY email"),
        {"tenant_id": tenant_id}).mappings().all()
    return [dict(r) for r in rows]


def get_user(conn: Connection, user_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        text(f"SELECT {_USER_COLS} FROM users WHERE tenant_id = :tenant_id AND id = :id"),
        {"tenant_id": tenant_id, "id": user_id}).mappings().first()
    return dict(row) if row else None


def lock_active_admins(conn: Connection, *, tenant_id: str) -> list[str]:
    """Ids of the tenant's active admins, row-locked until the transaction ends, so two
    concurrent demotions cannot both see 'another admin exists'."""
    rows = conn.execute(
        text("""SELECT id FROM users
                WHERE tenant_id = :tenant_id AND role = 'admin' AND is_active
                ORDER BY id FOR UPDATE"""),
        {"tenant_id": tenant_id}).all()
    return [str(r[0]) for r in rows]


def set_user_role(conn: Connection, user_id: str, role: str, *, tenant_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        text(f"""UPDATE users SET role = :role WHERE tenant_id = :tenant_id AND id = :id
                 RETURNING {_USER_COLS}"""),
        {"tenant_id": tenant_id, "id": user_id, "role": role}).mappings().first()
    return dict(row) if row else None


def set_user_active(conn: Connection, user_id: str, active: bool, *, tenant_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        text(f"""UPDATE users SET is_active = :active WHERE tenant_id = :tenant_id AND id = :id
                 RETURNING {_USER_COLS}"""),
        {"tenant_id": tenant_id, "id": user_id, "active": active}).mappings().first()
    return dict(row) if row else None


def touch_last_login(conn: Connection, user_id: str, *, tenant_id: str) -> None:
    conn.execute(
        text("""UPDATE users SET last_login_at = (NOW() AT TIME ZONE 'UTC')
                WHERE tenant_id = :tenant_id AND id = :id"""),
        {"tenant_id": tenant_id, "id": user_id})


def insert_audit_log(
    conn: Connection, *, tenant_id: str, request_id: str | None, principal_kind: str,
    principal_id: str | None, actor: str | None, action: str, resource: str | None,
    outcome: str, detail: dict[str, Any] | None,
) -> None:
    conn.execute(
        text("""
            INSERT INTO audit_log (tenant_id, request_id, principal_kind, principal_id,
                                   actor, action, resource, outcome, detail)
            VALUES (:tenant_id, :request_id, :principal_kind, :principal_id,
                    :actor, :action, :resource, :outcome, CAST(:detail AS JSONB))
        """),
        {"tenant_id": tenant_id, "request_id": request_id, "principal_kind": principal_kind,
         "principal_id": principal_id, "actor": actor, "action": action, "resource": resource,
         "outcome": outcome, "detail": json.dumps(detail) if detail is not None else None},
    )


# ---------------------------------------------------------------------
# Pull connectors (migration 0009, Phase 5)
# ---------------------------------------------------------------------

_CONNECTOR_COLS = ("id, name, type, config, schedule, mapping, secret_ref, enabled, next_run_at, "
                   "run_requested, claimed_at, list_watermark, last_run_at, last_status, last_message, "
                   "created_at, updated_at")
_CONNECTOR_UPDATABLE = {"name": False, "config": True, "schedule": False, "mapping": True,
                        "secret_ref": False, "enabled": False, "next_run_at": False,
                        "list_watermark": False}   # value -> is JSON


def create_connector(
    conn: Connection, connector_id: str, *, name: str, type: str, config: dict[str, Any],
    schedule: str, mapping: dict[str, Any], secret_ref: str | None, enabled: bool,
    next_run_at: datetime | None, tenant_id: str,
) -> dict[str, Any]:
    row = conn.execute(text(f"""
        INSERT INTO connectors (id, tenant_id, name, type, config, schedule, mapping,
                                secret_ref, enabled, next_run_at)
        VALUES (:id, :tenant_id, :name, :type, CAST(:config AS JSONB), :schedule,
                CAST(:mapping AS JSONB), :secret_ref, :enabled, :next_run_at)
        RETURNING {_CONNECTOR_COLS}
    """), {"id": connector_id, "tenant_id": tenant_id, "name": name, "type": type,
           "config": json.dumps(config), "schedule": schedule, "mapping": json.dumps(mapping),
           "secret_ref": secret_ref, "enabled": enabled, "next_run_at": next_run_at}).mappings().one()
    return dict(row)


def get_connector(conn: Connection, connector_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    row = conn.execute(text(f"""
        SELECT {_CONNECTOR_COLS} FROM connectors
        WHERE tenant_id = :tenant_id AND id = :id
    """), {"tenant_id": tenant_id, "id": connector_id}).mappings().first()
    return dict(row) if row else None


def list_connectors(conn: Connection, *, tenant_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(text(f"""
        SELECT {_CONNECTOR_COLS} FROM connectors
        WHERE tenant_id = :tenant_id ORDER BY created_at, id
    """), {"tenant_id": tenant_id}).mappings().all()
    return [dict(r) for r in rows]


def count_connectors(conn: Connection, *, tenant_id: str) -> int:
    return conn.execute(text("SELECT COUNT(*) FROM connectors WHERE tenant_id = :tenant_id"),
                        {"tenant_id": tenant_id}).scalar_one()


def update_connector(
    conn: Connection, connector_id: str, fields: dict[str, Any], *, tenant_id: str,
) -> dict[str, Any] | None:
    """Update configuration columns (a fixed whitelist; anything else is a ValueError)."""
    unknown = set(fields) - set(_CONNECTOR_UPDATABLE)
    if unknown:
        raise ValueError(f"update_connector: cannot update {sorted(unknown)}")
    sets, params = [], {"tenant_id": tenant_id, "id": connector_id}
    for key, value in fields.items():
        if _CONNECTOR_UPDATABLE[key]:
            sets.append(f"{key} = CAST(:{key} AS JSONB)")
            params[key] = json.dumps(value)
        else:
            sets.append(f"{key} = :{key}")
            params[key] = value
    sets.append(f"updated_at = {_UTC_NOW}")
    row = conn.execute(text(f"""
        UPDATE connectors SET {', '.join(sets)}
        WHERE tenant_id = :tenant_id AND id = :id
        RETURNING {_CONNECTOR_COLS}
    """), params).mappings().first()
    return dict(row) if row else None


def delete_connector(conn: Connection, connector_id: str, *, tenant_id: str) -> bool:
    """Deletes the connector and (ON DELETE CASCADE) its ledger and run history.
    Uploads/jobs it already queued stay."""
    # connector_objects hold FKs to uploads/jobs, not the reverse, so those are untouched.
    return conn.execute(text("DELETE FROM connectors WHERE tenant_id = :tenant_id AND id = :id"),
                        {"tenant_id": tenant_id, "id": connector_id}).rowcount > 0


def request_connector_run(
    conn: Connection, connector_id: str, *, tenant_id: str,
) -> dict[str, Any] | None:
    """Run-now: make an ENABLED connector due immediately (the scheduler picks it up)."""
    row = conn.execute(text(f"""
        UPDATE connectors SET next_run_at = {_UTC_NOW}, run_requested = TRUE, updated_at = {_UTC_NOW}
        WHERE tenant_id = :tenant_id AND id = :id AND enabled
        RETURNING {_CONNECTOR_COLS}
    """), {"tenant_id": tenant_id, "id": connector_id}).mappings().first()
    return dict(row) if row else None


def touch_connector_claim(conn: Connection, connector_id: str, *, tenant_id: str) -> None:
    """Heartbeat for a running pull: keeps the lease from expiring mid-run."""
    conn.execute(text(f"""
        UPDATE connectors SET claimed_at = {_UTC_NOW}
        WHERE tenant_id = :tenant_id AND id = :id AND claimed_at IS NOT NULL
    """), {"tenant_id": tenant_id, "id": connector_id})


def finish_connector_state(
    conn: Connection, connector_id: str, *, last_status: str, last_message: str | None,
    next_run_at: datetime | None, list_watermark: datetime | None, tenant_id: str,
) -> None:
    """End of a pull: record the outcome, release the lease, schedule the next run.
    The listing watermark only ever moves forward."""
    conn.execute(text(f"""
        UPDATE connectors SET last_status = :last_status, last_message = :last_message,
               last_run_at = {_UTC_NOW}, claimed_at = NULL, run_requested = FALSE, next_run_at = :next_run_at,
               list_watermark = GREATEST(list_watermark, CAST(:wm AS TIMESTAMP))
        WHERE tenant_id = :tenant_id AND id = :id
    """), {"tenant_id": tenant_id, "id": connector_id, "last_status": last_status,
           "last_message": (last_message or None) and last_message[:300],
           "next_run_at": next_run_at, "wm": list_watermark})


def create_connector_run(
    conn: Connection, run_id: str, connector_id: str, trigger: str, *, tenant_id: str,
) -> None:
    conn.execute(text("""
        INSERT INTO connector_runs (id, tenant_id, connector_id, trigger)
        VALUES (:id, :tenant_id, :connector_id, :trigger)
    """), {"id": run_id, "tenant_id": tenant_id, "connector_id": connector_id, "trigger": trigger})


def finish_connector_run(
    conn: Connection, run_id: str, *, status: str, objects_seen: int, objects_new: int,
    objects_queued: int, objects_failed: int, message: str | None, tenant_id: str,
) -> None:
    conn.execute(text(f"""
        UPDATE connector_runs SET status = :status, finished_at = {_UTC_NOW},
               objects_seen = :seen, objects_new = :new, objects_queued = :queued,
               objects_failed = :failed, message = :message
        WHERE tenant_id = :tenant_id AND id = :id
    """), {"tenant_id": tenant_id, "id": run_id, "status": status, "seen": objects_seen,
           "new": objects_new, "queued": objects_queued, "failed": objects_failed,
           "message": (message or None) and message[:300]})


def list_connector_runs(
    conn: Connection, connector_id: str, limit: int = 20, offset: int = 0, *, tenant_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(text("""
        SELECT id, connector_id, trigger, status, started_at, finished_at, objects_seen,
               objects_new, objects_queued, objects_failed, message
        FROM connector_runs
        WHERE tenant_id = :tenant_id AND connector_id = :connector_id
        ORDER BY started_at DESC, id LIMIT :limit OFFSET :offset
    """), {"tenant_id": tenant_id, "connector_id": connector_id, "limit": limit,
           "offset": offset}).mappings().all()
    return [dict(r) for r in rows]


def get_connector_objects(
    conn: Connection, connector_id: str, keys: list[str], *, tenant_id: str,
) -> list[dict[str, Any]]:
    """Ledger rows for these object keys (any version)."""
    if not keys:
        return []
    rows = conn.execute(text("""
        SELECT object_key, object_version, status, attempts FROM connector_objects
        WHERE tenant_id = :tenant_id AND connector_id = :connector_id
          AND object_key = ANY(:keys)
    """), {"tenant_id": tenant_id, "connector_id": connector_id, "keys": list(keys)}).mappings().all()
    return [dict(r) for r in rows]


def mark_connector_object_queued(
    conn: Connection, connector_id: str, *, object_key: str, object_version: str,
    size_bytes: int | None, object_updated: datetime | None, upload_id: str, job_id: str,
    tenant_id: str,
) -> bool:
    """Record that this object version's job exists. True if this call did it; False if the
    version was already queued/rejected (another scheduler won the race): the caller must
    then discard what it created. A 'retry' row (earlier failed download) is upgraded."""
    row = conn.execute(text(f"""
        INSERT INTO connector_objects (tenant_id, connector_id, object_key, object_version,
               size_bytes, object_updated, status, upload_id, job_id, attempts)
        VALUES (:tenant_id, :connector_id, :key, :version, :size, :updated, 'queued',
                :upload_id, :job_id, 0)
        ON CONFLICT (tenant_id, connector_id, object_key, object_version) DO UPDATE
           SET status = 'queued', upload_id = EXCLUDED.upload_id, job_id = EXCLUDED.job_id,
               size_bytes = EXCLUDED.size_bytes, last_error = NULL, updated_at = {_UTC_NOW}
         WHERE connector_objects.status = 'retry'
        RETURNING id
    """), {"tenant_id": tenant_id, "connector_id": connector_id, "key": object_key,
           "version": object_version, "size": size_bytes, "updated": object_updated,
           "upload_id": upload_id, "job_id": job_id}).first()
    return row is not None


def record_connector_object_failure(
    conn: Connection, connector_id: str, *, object_key: str, object_version: str,
    size_bytes: int | None, object_updated: datetime | None, error: str, permanent: bool,
    max_attempts: int, tenant_id: str,
) -> str | None:
    """Count a failed attempt. Returns the row's status ('retry' until `max_attempts`, then
    'rejected'; a permanent failure is 'rejected' at once), or None if it is already resolved."""
    first = "rejected" if permanent or max_attempts <= 1 else "retry"
    row = conn.execute(text(f"""
        INSERT INTO connector_objects (tenant_id, connector_id, object_key, object_version,
               size_bytes, object_updated, status, attempts, last_error)
        VALUES (:tenant_id, :connector_id, :key, :version, :size, :updated, :first, 1, :error)
        ON CONFLICT (tenant_id, connector_id, object_key, object_version) DO UPDATE
           SET attempts = connector_objects.attempts + 1,
               status = CASE WHEN :permanent OR connector_objects.attempts + 1 >= :max
                             THEN 'rejected' ELSE 'retry' END,
               last_error = EXCLUDED.last_error, updated_at = {_UTC_NOW}
         WHERE connector_objects.status = 'retry'
        RETURNING status
    """), {"tenant_id": tenant_id, "connector_id": connector_id, "key": object_key,
           "version": object_version, "size": size_bytes, "updated": object_updated,
           "first": first, "error": error[:300], "permanent": permanent,
           "max": max_attempts}).first()
    return row[0] if row else None


# ---------------------------------------------------------------------
# Phase 6: dashboard reads (fleet summary, reading series), stored investigations,
# sensor-registry edits
# ---------------------------------------------------------------------

def count_machines(conn: Connection, *, tenant_id: str) -> int:
    return conn.execute(text("SELECT COUNT(*) FROM machines WHERE tenant_id = :tenant_id"),
                        {"tenant_id": tenant_id}).scalar() or 0


def get_fleet_summary_page(
    conn: Connection, limit: int, offset: int, *, tenant_id: str,
) -> list[dict[str, Any]]:
    """One page of machines (ordered by id), each with its latest sensor_summary window
    (NULLs for a machine that has none yet)."""
    rows = conn.execute(text("""
        SELECT m.machine_id, m.production_line, m.name, m.location,
               s.window_start, s.window_end, s.health_status, s.max_anomaly_score,
               s.anomalous_reading_count, s.reading_count
        FROM machines m
        LEFT JOIN LATERAL (
            SELECT window_start, window_end, health_status, max_anomaly_score,
                   anomalous_reading_count, reading_count
            FROM sensor_summary x
            WHERE x.tenant_id = m.tenant_id AND x.machine_id = m.machine_id
            ORDER BY x.window_start DESC LIMIT 1
        ) s ON TRUE
        WHERE m.tenant_id = :tenant_id
        ORDER BY m.machine_id LIMIT :limit OFFSET :offset
    """), {"tenant_id": tenant_id, "limit": limit, "offset": offset}).mappings().all()
    return [dict(r) for r in rows]


def get_machines_trend(
    conn: Connection, machine_ids: list[str], windows: int = 24, *, tenant_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """get_fleet_sensor_trend restricted to the given machines (one page of the fleet)."""
    if not machine_ids:
        return {}
    rows = conn.execute(text("""
        SELECT machine_id, window_start, window_end, health_status,
               max_anomaly_score, anomalous_reading_count, reading_count
        FROM (
            SELECT s.*, ROW_NUMBER() OVER (
                       PARTITION BY s.machine_id ORDER BY s.window_start DESC) AS rn
            FROM sensor_summary s
            WHERE s.tenant_id = :tenant_id AND s.machine_id = ANY(:ids)
        ) ranked
        WHERE rn <= :windows
        ORDER BY machine_id, window_start DESC
    """), {"tenant_id": tenant_id, "ids": list(machine_ids), "windows": windows}).mappings().all()
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["machine_id"], []).append(dict(r))
    return out


def get_machines_recent_high_anomalies(
    conn: Connection, machine_ids: list[str], windows: int = 24, *, tenant_id: str,
) -> set[str]:
    """get_fleet_recent_high_anomalies restricted to the given machines."""
    if not machine_ids:
        return set()
    rows = conn.execute(text("""
        WITH trend AS (
            SELECT machine_id, window_start,
                   ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY window_start DESC) AS rn
            FROM sensor_summary
            WHERE tenant_id = :tenant_id AND machine_id = ANY(:ids)
        ),
        trend_floor AS (
            SELECT machine_id, MIN(window_start) AS floor_ts FROM trend
            WHERE rn <= :windows GROUP BY machine_id
        )
        SELECT DISTINCT a.machine_id
        FROM machine_anomalies a
        JOIN trend_floor t ON t.machine_id = a.machine_id
        WHERE a.tenant_id = :tenant_id AND a.machine_id = ANY(:ids)
          AND a.severity = 'HIGH' AND a.detected_at >= t.floor_ts
    """), {"tenant_id": tenant_id, "ids": list(machine_ids), "windows": windows}).mappings().all()
    return {r["machine_id"] for r in rows}


def get_latest_reading_ts(
    conn: Connection, machine_id: str, sensor_name: str, *, tenant_id: str,
) -> datetime | None:
    return conn.execute(text("""
        SELECT MAX(ts) FROM sensor_readings
        WHERE tenant_id = :tenant_id AND machine_id = :machine_id AND sensor_name = :sensor_name
    """), {"tenant_id": tenant_id, "machine_id": machine_id, "sensor_name": sensor_name}).scalar()


def get_reading_series(
    conn: Connection, machine_id: str, sensor_name: str, since: datetime, until: datetime,
    bucket_seconds: int, limit: int, *, tenant_id: str,
) -> list[dict[str, Any]]:
    """Readings of one sensor bucketed into `bucket_seconds` (UTC) windows: avg/min/max/count."""
    rows = conn.execute(text("""
        SELECT (to_timestamp(floor(extract(epoch FROM ts) / :bucket) * :bucket)
                    AT TIME ZONE 'UTC') AS bucket,
               AVG(value) AS avg_value, MIN(value) AS min_value, MAX(value) AS max_value,
               COUNT(*) AS n
        FROM sensor_readings
        WHERE tenant_id = :tenant_id AND machine_id = :machine_id AND sensor_name = :sensor_name
          AND ts >= :since AND ts <= :until
        GROUP BY 1 ORDER BY 1 LIMIT :limit
    """), {"tenant_id": tenant_id, "machine_id": machine_id, "sensor_name": sensor_name,
           "since": since, "until": until, "bucket": bucket_seconds, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def insert_investigation(
    conn: Connection, investigation_id: str, *, created_by: str | None, created_by_label: str,
    endpoint: str, question: str, machine_id: str | None, intent: str | None,
    risk_level: str | None, result: dict[str, Any], tenant_id: str,
) -> None:
    conn.execute(text("""
        INSERT INTO investigations
            (id, tenant_id, created_by, created_by_label, endpoint, question, machine_id,
             intent, risk_level, result)
        VALUES (:id, :tenant_id, :created_by, :label, :endpoint, :question, :machine_id,
                :intent, :risk_level, CAST(:result AS JSONB))
    """), {"id": investigation_id, "tenant_id": tenant_id, "created_by": created_by,
           "label": created_by_label, "endpoint": endpoint, "question": question,
           "machine_id": machine_id, "intent": intent, "risk_level": risk_level,
           "result": json.dumps(result, default=str)})


_INV_LIGHT = ("id, created_by_label, endpoint, question, machine_id, intent, risk_level, created_at")


def list_investigations(
    conn: Connection, limit: int, offset: int, machine_id: str | None = None, *, tenant_id: str,
) -> list[dict[str, Any]]:
    """Newest first; summary columns only (the result JSON is fetched per investigation)."""
    query = f"SELECT {_INV_LIGHT} FROM investigations WHERE tenant_id = :tenant_id"
    params: dict[str, Any] = {"tenant_id": tenant_id, "limit": limit, "offset": offset}
    if machine_id is not None:
        query += " AND machine_id = :machine_id"
        params["machine_id"] = machine_id
    query += " ORDER BY created_at DESC, id LIMIT :limit OFFSET :offset"
    return [dict(r) for r in conn.execute(text(query), params).mappings().all()]


def get_investigation(conn: Connection, investigation_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    row = conn.execute(text(f"""
        SELECT {_INV_LIGHT}, result FROM investigations
        WHERE tenant_id = :tenant_id AND id = :id
    """), {"tenant_id": tenant_id, "id": investigation_id}).mappings().first()
    return dict(row) if row else None


def update_sensor_registry_entry(
    conn: Connection, sensor_name: str, fields: dict[str, Any], *, tenant_id: str,
) -> dict[str, Any] | None:
    """Change normal_min / normal_max / z_score_threshold / enabled of ONE existing sensor."""
    allowed = ("normal_min", "normal_max", "z_score_threshold", "enabled")
    sets = [f"{k} = :{k}" for k in fields if k in allowed]
    if not sets:
        raise ValueError("no editable field given")
    params = {k: v for k, v in fields.items() if k in allowed}
    params.update({"tenant_id": tenant_id, "sensor_name": sensor_name})
    row = conn.execute(text(f"""
        UPDATE sensor_registry SET {', '.join(sets)}
        WHERE tenant_id = :tenant_id AND sensor_name = :sensor_name
        RETURNING sensor_name, unit, normal_min, normal_max, z_score_threshold, enabled
    """), params).mappings().first()
    return dict(row) if row else None
