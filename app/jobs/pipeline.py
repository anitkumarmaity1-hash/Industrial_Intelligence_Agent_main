"""The ingest job: validate -> map -> normalize -> load -> detect -> store results.

Runs inside the worker (app/jobs/worker.py), as the `iia_worker` role, with the
job's tenant bound to every transaction (Postgres RLS), never as the owner.

Nothing here re-implements a rule. Normalisation is the Phase 1
`normalize_sensor_readings` (same mapping format, same per-row error report);
detection is the Phase 1 Spark `spark_jobs.long_format.detect` run in local mode,
so results are the Spark path's results by construction.

Idempotency (re-running the same upload, or retrying after a crash, never
duplicates anything):
  * readings        INSERT .. ON CONFLICT DO NOTHING on the Phase 1 primary key
  * machines        upsert that never erases a known line/type
  * sensor_summary  upsert on (tenant, machine, window_start)
  * anomalies       upsert on (tenant, machine, detected_at)   [migration 0007]
  * rejected_rows   replaced wholesale on every attempt
Every stage commits on its own, so a crash leaves only idempotent work behind.

Detection scope: detection does not run on the file alone. For each machine in
the upload it re-reads the STORED readings from that machine's earliest
uploaded timestamp onward, plus `rolling_window_readings` earlier timestamps as
baseline context, so a second upload continues the first one's rolling window
exactly as one big file would (tests/test_uploads_pipeline.py proves it), and
a back-filled upload re-detects the later readings it now precedes.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
from sqlalchemy.engine import Connection, Engine

from app.core.anomaly_rules import load_anomaly_rules
from app.core.ingest_settings import ingest_settings
from app.database import queries
from app.database.tenant_context import bind_tenant
from app.onboarding.columns import missing_source_columns
from app.onboarding.sensors import (
    UNMAPPED_TYPE,
    NormalizedSensorData,
    SensorMapping,
    SensorMappingError,
    normalize_sensor_readings,
)
from app.storage import StorageError, StorageKeyError, StorageProvider, assert_key_for_tenant, get_storage

logger = logging.getLogger(__name__)

CHECKPOINT_SOURCE = "sensor_summary"   # same source scripts/incremental_load.py advances
MAX_ID_LEN, MAX_LINE_LEN = 64, 100     # machines.machine_id / machines.production_line
CHUNK = 20_000
TOO_LONG_MACHINE_ID = "machine_id_too_long"
TOO_LONG_LINE = "production_line_too_long"
FILE_PROBLEM = "file_problem"


class PermanentJobError(Exception):
    """The job cannot succeed by retrying (bad mapping, bad file, no usable
    rows). Its message is safe to show to the tenant."""


@contextmanager
def tenant_conn(engine: Engine, tenant_id: str) -> Iterator[Connection]:
    """One committed transaction with the tenant bound for RLS."""
    with engine.begin() as conn:
        bind_tenant(conn, tenant_id)
        yield conn


def _stage(engine: Engine, tenant_id: str, job_id: str, name: str) -> None:
    logger.info("job %s: stage %s", job_id, name)
    with tenant_conn(engine, tenant_id) as conn:
        queries.set_job_stage(conn, job_id, name, tenant_id=tenant_id)


# ---------------------------------------------------------------------
# Stages (module-level so tests can inject a crash into any of them)
# ---------------------------------------------------------------------

def stage_validate(engine: Engine, storage: StorageProvider, tenant_id: str, job_id: str,
                   workdir: Path) -> tuple[dict[str, Any], Path]:
    _stage(engine, tenant_id, job_id, "validate")
    with tenant_conn(engine, tenant_id) as conn:
        job = queries.get_job(conn, job_id, tenant_id=tenant_id)
        upload = queries.get_upload(conn, job["upload_id"], tenant_id=tenant_id) if job else None
    if job is None or upload is None:
        raise PermanentJobError("job or upload record not found")
    if not upload["mapping"]:
        raise PermanentJobError("the upload has no confirmed column mapping")
    try:
        key = assert_key_for_tenant(upload["storage_key"], tenant_id)
    except StorageKeyError as exc:
        raise PermanentJobError(f"invalid storage key: {exc}") from exc
    if not storage.exists(key, tenant_id=tenant_id):
        raise PermanentJobError("the uploaded file is missing from storage")
    if (upload["size_bytes"] or 0) > ingest_settings().upload_max_bytes:
        raise PermanentJobError("the uploaded file exceeds the size limit")
    path = workdir / "data.csv"
    with storage.open(key, tenant_id=tenant_id) as src, open(path, "wb") as dst:
        dst.write(src.read())
    return upload, path


def stage_map(engine: Engine, tenant_id: str, job_id: str, upload: dict[str, Any]) -> SensorMapping:
    _stage(engine, tenant_id, job_id, "map")
    try:
        mapping = SensorMapping.from_dict(upload["mapping"])
    except SensorMappingError as exc:
        raise PermanentJobError(str(exc)) from exc
    # The tenant's sensor vocabulary and rules come from this mapping (same as
    # onboard_tenant.py / load_sensor_readings.py): registry + anomaly settings.
    with tenant_conn(engine, tenant_id) as conn:
        queries.upsert_sensor_registry(conn, mapping.registry_rows(), tenant_id=tenant_id)
        if mapping.anomaly:
            queries.merge_tenant_anomaly_settings(conn, mapping.anomaly, tenant_id=tenant_id)
    return mapping


def _rejects_from_report(data: NormalizedSensorData, raw: pd.DataFrame,
                         extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for e in data.report.errors:
        record = None
        if e.line is not None and 2 <= e.line < len(raw) + 2:
            record = {k: v for k, v in raw.iloc[e.line - 2].to_dict().items()}
        rows.append({"row_number": e.line, "code": e.code,
                     "reason": e.message + (f" (value {e.value!r})" if e.value is not None else ""),
                     "raw": record})
    return rows + extra


def stage_normalize(engine: Engine, tenant_id: str, job_id: str, path: Path,
                    mapping: SensorMapping) -> tuple[NormalizedSensorData, int, int]:
    """-> (clean data, rows_in, rows_rejected). Persists every rejection."""
    _stage(engine, tenant_id, job_id, "normalize")
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise PermanentJobError(f"the file is not a readable CSV: {exc}") from exc

    missing = missing_source_columns(raw.columns, mapping)
    if missing:
        raise PermanentJobError(f"column(s) named in the mapping were not found in the file: {missing}")

    data = normalize_sensor_readings(raw, mapping)
    extra: list[dict[str, Any]] = []
    dropped = 0

    # Values the database columns cannot hold are rejected up front (a long
    # asset tag would otherwise abort the whole load at INSERT time).
    r, m = data.readings, data.machines
    bad_ids = set(m.loc[m["machine_id"].str.len() > MAX_ID_LEN, "machine_id"])
    bad_lines = set(m.loc[m["production_line"].fillna("").str.len() > MAX_LINE_LEN, "machine_id"])
    for ids, code, limit in ((bad_ids, TOO_LONG_MACHINE_ID, MAX_ID_LEN),
                             (bad_lines - bad_ids, TOO_LONG_LINE, MAX_LINE_LEN)):
        for machine in sorted(ids):
            n = int((r["machine_id"] == machine).sum())
            dropped += n
            extra.append({"row_number": None, "code": code, "raw": {"machine_id": machine},
                          "reason": f"machine {machine[:40]!r}: value longer than {limit} characters; "
                                    f"its {n} reading(s) were rejected"})
    if bad_ids | bad_lines:
        keep = ~r["machine_id"].isin(bad_ids | bad_lines)
        data = NormalizedSensorData(r[keep].reset_index(drop=True),
                                    m[~m["machine_id"].isin(bad_ids | bad_lines)].reset_index(drop=True),
                                    data.report)

    # A reading is "rejected" if the report says it was dropped (every code except
    # an unmapped machine type, which only leaves that machine's type empty).
    rows_rejected = sum(n for code, n in data.report.counts.items() if code != UNMAPPED_TYPE) + dropped
    rejects = _rejects_from_report(data, raw, extra)
    with tenant_conn(engine, tenant_id) as conn:
        queries.replace_rejected_rows(conn, job_id, rejects, tenant_id=tenant_id)
        queries.set_job_counts(conn, job_id, rows_in=len(raw), rows_loaded=0,
                               rows_rejected=rows_rejected, tenant_id=tenant_id)
    if data.readings.empty:
        raise PermanentJobError(
            f"no usable readings in the file ({rows_rejected} problem(s); see the job's rejected rows)")
    return data, len(raw), rows_rejected


def stage_load_readings(engine: Engine, tenant_id: str, job_id: str, data: NormalizedSensorData) -> int:
    _stage(engine, tenant_id, job_id, "load")
    with tenant_conn(engine, tenant_id) as conn:
        queries.upsert_machines_keep_known(conn, data.machines.to_dict("records"), tenant_id=tenant_id)
    r = data.readings
    inserted = 0
    for start in range(0, len(r), CHUNK):
        part = r.iloc[start:start + CHUNK]
        with tenant_conn(engine, tenant_id) as conn:
            inserted += queries.bulk_insert_sensor_readings(
                conn, part["machine_id"].tolist(), part["sensor_name"].tolist(),
                [t.to_pydatetime() for t in part["ts"]], part["value"].astype(float).tolist(),
                tenant_id=tenant_id)
    logger.info("job %s: %d of %d readings were new", job_id, inserted, len(r))
    return inserted


def _db_values(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> dicts of plain Python values, NaN/NaT as None."""
    obj = df.astype(object).where(df.notna(), None)
    return obj.to_dict("records")


def spark_to_pandas_utc(df) -> pd.DataFrame:
    """Collect a Spark DataFrame without letting Spark/Python re-interpret timestamps in
    the host's local time zone (`toPandas` does, which shifted every time by the machine's
    UTC offset on an Asia/Calcutta host). The session time zone is UTC (load_long_readings),
    so timestamp columns are rendered as UTC strings and parsed back as naive UTC."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import TimestampType
    stamps = [f.name for f in df.schema.fields if isinstance(f.dataType, TimestampType)]
    for name in stamps:
        df = df.withColumn(name, F.date_format(F.col(name), "yyyy-MM-dd HH:mm:ss"))
    out = df.toPandas()
    for name in stamps:
        out[name] = pd.to_datetime(out[name], format="%Y-%m-%d %H:%M:%S")
    return out


def stage_detect(engine: Engine, tenant_id: str, job_id: str, data: NormalizedSensorData,
                 workdir: Path, spark_factory: Callable[[], Any] | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spark `detect()` over the stored readings from each machine's first
    uploaded timestamp (+ baseline context). -> (sensor_summary, anomalies)."""
    _stage(engine, tenant_id, job_id, "detect")
    from spark_jobs.ingestion import get_spark_session
    from spark_jobs.long_format import detect, load_long_readings, load_machines

    with tenant_conn(engine, tenant_id) as conn:
        rules = load_anomaly_rules(conn, tenant_id)
        if not rules.active_sensors:
            logger.warning("job %s: tenant has no enabled sensors; skipping detection", job_id)
            return pd.DataFrame(), pd.DataFrame()
        since = data.readings.groupby("machine_id")["ts"].min()
        stored = queries.fetch_readings_for_detection(
            conn, since.index.tolist(), [t.to_pydatetime() for t in since],
            [s.sensor_name for s in rules.active_sensors], rules.rolling_window_readings,
            tenant_id=tenant_id)
        machines = pd.DataFrame(
            [{k: m[k] for k in ("machine_id", "production_line", "type")}
             for m in queries.list_machines(conn, tenant_id=tenant_id)
             if m["machine_id"] in set(since.index)])

    long_df = pd.DataFrame(stored, columns=["machine_id", "sensor_name", "timestamp", "value"])[
        ["machine_id", "timestamp", "sensor_name", "value"]]   # LONG_SCHEMA column order
    readings_csv, machines_csv = workdir / "detect_readings.csv", workdir / "detect_machines.csv"
    long_df.to_csv(readings_csv, index=False)
    machines.to_csv(machines_csv, index=False)

    spark = (spark_factory or get_spark_session)()
    summary, anomalies = detect(load_long_readings(spark, str(readings_csv)),
                                load_machines(spark, str(machines_csv)), rules)
    summary, anomalies = spark_to_pandas_utc(summary), spark_to_pandas_utc(anomalies)
    logger.info("job %s: detect read %d stored readings; spark produced %d summary rows, %d anomalies",
                job_id, len(long_df), len(summary), len(anomalies))

    # Keep only output the upload can have changed: hours from the machine's
    # first uploaded hour on, anomalies from its first uploaded timestamp on.
    first = since.rename("since").reset_index()
    first["hour"] = first["since"].dt.floor("h")
    if not summary.empty:
        summary = summary.merge(first[["machine_id", "hour"]], on="machine_id")
        summary = summary[summary["window_start"] >= summary["hour"]].drop(columns="hour")
    if not anomalies.empty:
        anomalies = anomalies.merge(first[["machine_id", "since"]], on="machine_id")
        anomalies = anomalies[anomalies["detected_at"] >= anomalies["since"]].drop(columns="since")
    return summary, anomalies


def stage_store_results(engine: Engine, tenant_id: str, job_id: str, summary: pd.DataFrame,
                        anomalies: pd.DataFrame) -> tuple[int, int]:
    _stage(engine, tenant_id, job_id, "store")
    s_rows = _db_values(summary.drop(columns=["production_line", "type"], errors="ignore")) \
        if not summary.empty else []
    a_rows = _db_values(anomalies) if not anomalies.empty else []
    for start in range(0, len(s_rows), CHUNK):
        with tenant_conn(engine, tenant_id) as conn:
            queries.upsert_sensor_summary(conn, s_rows[start:start + CHUNK], tenant_id=tenant_id)
    for start in range(0, len(a_rows), CHUNK):
        with tenant_conn(engine, tenant_id) as conn:
            queries.upsert_machine_anomalies(conn, a_rows[start:start + CHUNK], tenant_id=tenant_id)
    if s_rows:
        # The checkpoint is a high-water mark (how far this tenant's summary reaches),
        # not a row filter: uploads can back-fill older data, so filtering by it would
        # skip rows. Idempotency comes from the upserts above.
        with tenant_conn(engine, tenant_id) as conn:
            previous = queries.get_ingestion_checkpoint(conn, CHECKPOINT_SOURCE, tenant_id=tenant_id)
            newest = max(r["window_end"] for r in s_rows)
            if previous and previous["last_window_end"] and previous["last_window_end"] > newest:
                newest = previous["last_window_end"]
            queries.upsert_ingestion_checkpoint(
                conn, CHECKPOINT_SOURCE, pd.Timestamp(newest).to_pydatetime(), len(s_rows),
                tenant_id=tenant_id)
    return len(s_rows), len(a_rows)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_job(engine: Engine, job_id: str, tenant_id: str, *, storage: StorageProvider | None = None,
            spark_factory: Callable[[], Any] | None = None) -> dict[str, int]:
    """Run one claimed job to completion. Raises PermanentJobError (do not
    retry) or any other exception (transient: the worker retries with backoff)."""
    storage = storage or get_storage()
    with tempfile.TemporaryDirectory(prefix=f"iia-job-{job_id[:8]}-") as tmp:
        workdir = Path(tmp)
        upload, path = stage_validate(engine, storage, tenant_id, job_id, workdir)
        mapping = stage_map(engine, tenant_id, job_id, upload)
        data, rows_in, rows_rejected = stage_normalize(engine, tenant_id, job_id, path, mapping)
        stage_load_readings(engine, tenant_id, job_id, data)
        summary, anomalies = stage_detect(engine, tenant_id, job_id, data, workdir, spark_factory)
        n_summary, n_anomalies = stage_store_results(engine, tenant_id, job_id, summary, anomalies)
        accepted = len(data.readings)
        with tenant_conn(engine, tenant_id) as conn:
            queries.set_job_counts(conn, job_id, rows_in=rows_in, rows_loaded=accepted,
                                   rows_rejected=rows_rejected, tenant_id=tenant_id)
            queries.finish_job(conn, job_id, status="succeeded", error=None, tenant_id=tenant_id)
    logger.info("job %s succeeded: %d in, %d loaded, %d rejected, %d summary rows, %d anomalies",
                job_id, rows_in, accepted, rows_rejected, n_summary, n_anomalies)
    return {"rows_in": rows_in, "rows_loaded": accepted, "rows_rejected": rows_rejected,
            "summary_rows": n_summary, "anomalies": n_anomalies}
