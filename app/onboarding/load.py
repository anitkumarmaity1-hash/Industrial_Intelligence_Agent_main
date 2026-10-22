"""
Load an onboarded sensor bundle (app.onboarding.sensors.write_bundle) into
Postgres: registry -> settings -> machines -> readings. Every step is an
upsert / ON CONFLICT DO NOTHING, so re-running a load, or loading an
overlapping file, never duplicates anything (readings: first write wins).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.engine import Connection

from app.database import queries

CHUNK = 10_000


def read_bundle(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    d = Path(directory)
    readings = pd.read_csv(d / "sensor_readings.csv", parse_dates=["timestamp"]).rename(
        columns={"timestamp": "ts"})
    machines = pd.read_csv(d / "machines.csv", dtype=str, keep_default_na=False)
    machines["type"] = machines["type"].replace("", None)
    registry = json.loads((d / "sensor_registry.json").read_text())
    return readings, machines, registry


def apply_registry(conn: Connection, tenant_id: str, registry: dict[str, Any]) -> None:
    queries.upsert_sensor_registry(conn, registry["sensors"], tenant_id=tenant_id)
    if registry.get("anomaly"):
        queries.merge_tenant_anomaly_settings(conn, registry["anomaly"], tenant_id=tenant_id)


def load_readings(conn: Connection, tenant_id: str, readings: pd.DataFrame) -> int:
    """Insert long-format readings in chunks; returns rows newly inserted."""
    inserted = 0
    for start in range(0, len(readings), CHUNK):
        chunk = readings.iloc[start:start + CHUNK]
        rows = [
            {"machine_id": m, "sensor_name": s, "ts": t.to_pydatetime(), "value": float(v)}
            for m, s, t, v in zip(chunk["machine_id"], chunk["sensor_name"], chunk["ts"], chunk["value"])
        ]
        inserted += queries.insert_sensor_readings(conn, rows, tenant_id=tenant_id)
    return inserted


def load_bundle(
    conn: Connection, tenant_id: str, readings: pd.DataFrame, machines: pd.DataFrame,
    registry: dict[str, Any] | None = None,
) -> dict[str, int]:
    if registry is not None:
        apply_registry(conn, tenant_id, registry)
    n_machines = queries.upsert_machines(
        conn, machines.to_dict("records"), tenant_id=tenant_id)
    n_readings = load_readings(conn, tenant_id, readings)
    return {"machines": n_machines, "readings_inserted": n_readings,
            "readings_in_file": len(readings)}
