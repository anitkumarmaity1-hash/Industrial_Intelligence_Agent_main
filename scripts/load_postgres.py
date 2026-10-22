"""
Load Phase 1's processed Parquet output into PostgreSQL.

Usage:
    python scripts/load_postgres.py [--truncate] [--tenant ID] [--processed-dir DIR]

--tenant (default "default", the demo tenant) says whose data this is; the
tenant must already exist (scripts/manage_tenants.py create ...). Every row
is stamped with it. --truncate deletes ONLY that tenant's rows — it never
touches other tenants. The shared ai4i_reference table is loaded/cleared
only when loading the demo tenant. --processed-dir points at a different
pipeline output directory (e.g. one produced from onboarded company data).

Reads DATABASE_URL from the environment (see .env.example). Requires
sql/schema.sql to have already been applied.

Design notes:
- machines has no dedicated source file — machine_id/production_line/type
  are derived by taking the distinct combination out of sensor_summary,
  which is where the Phase 1 pipeline actually put them.
- Loading is idempotent: pass --truncate to clear all tables (in FK-safe
  order) before reloading, so this script can be re-run after a pipeline
  re-run without manual cleanup or duplicate rows.
- Uses pandas.DataFrame.to_sql with method="multi" in chunks rather than
  row-by-row inserts — 18,912 rows (machine_anomalies) row-by-row would be
  needlessly slow for a script meant to be re-run often during development.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id  # noqa: E402
from app.database import queries  # noqa: E402

# Order matters: children before parents when truncating (reverse of load
# order), parents before children when loading (FK constraints).
LOAD_ORDER = ["machines", "sensor_summary", "machine_anomalies", "maintenance_records", "ai4i_reference"]


def get_engine() -> Engine:
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)
    return create_engine(database_url)


TENANT_TABLES = ["machines", "sensor_summary", "machine_anomalies", "maintenance_records"]


def ensure_tenant_exists(engine: Engine, tenant_id: str) -> None:
    with engine.connect() as conn:
        found = conn.execute(
            text("SELECT 1 FROM tenants WHERE tenant_id = :t"), {"t": tenant_id}).first()
    if not found:
        logger.error("Tenant %r does not exist. Create it first: "
                     "python scripts/manage_tenants.py create %s --name ...", tenant_id, tenant_id)
        sys.exit(1)


def truncate_tenant(engine: Engine, tenant_id: str) -> None:
    """Delete this tenant's rows (children first). Never TRUNCATE: that would
    wipe every tenant's data. ai4i_reference is shared and only cleared for
    the demo tenant."""
    with engine.begin() as conn:
        for table in reversed(TENANT_TABLES):
            conn.execute(text(f"DELETE FROM {table} WHERE tenant_id = :t"), {"t": tenant_id})
        if tenant_id == DEFAULT_TENANT_ID:
            conn.execute(text("TRUNCATE TABLE ai4i_reference RESTART IDENTITY"))
    logger.info("Cleared tenant %s's rows", tenant_id)


def _stamp(df: pd.DataFrame, tenant_id: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "tenant_id", tenant_id)
    return df


def load_machines(engine: Engine, tenant_id: str, processed_dir: Path) -> int:
    """Derive the machine dimension from sensor_summary's distinct
    (machine_id, production_line, type) — there is no separate source file."""
    df = pd.read_parquet(processed_dir / "sensor_summary.parquet", columns=["machine_id", "production_line", "type"])
    machines = df.drop_duplicates().sort_values("machine_id").reset_index(drop=True)
    machines["machine_id"] = machines["machine_id"].astype(str)
    machines = machines.astype(object).where(machines.notna(), None)   # NaN type -> NULL
    # Upsert (not a plain append): a tenant onboarded through
    # scripts/load_sensor_readings.py already has its machine rows.
    with engine.begin() as conn:
        queries.upsert_machines(conn, machines.to_dict("records"), tenant_id=tenant_id)
    return len(machines)


def load_sensor_summary(engine: Engine, tenant_id: str, processed_dir: Path) -> int:
    df = pd.read_parquet(processed_dir / "sensor_summary.parquet")
    df = df.drop(columns=["production_line", "type"])  # normalized out, lives in machines
    df["machine_id"] = df["machine_id"].astype(str)
    _stamp(df, tenant_id).to_sql("sensor_summary", engine, if_exists="append", index=False, method="multi", chunksize=2000)
    return len(df)


def load_machine_anomalies(engine: Engine, tenant_id: str, processed_dir: Path) -> int:
    df = pd.read_parquet(processed_dir / "machine_anomalies.parquet")
    df["machine_id"] = df["machine_id"].astype(str)
    _stamp(df, tenant_id).to_sql("machine_anomalies", engine, if_exists="append", index=False, method="multi", chunksize=2000)
    return len(df)


def load_maintenance_records(engine: Engine, tenant_id: str, processed_dir: Path) -> int:
    if not (processed_dir / "maintenance_clean.parquet").exists():
        # Tenants onboarded with sensor data only (scripts/run_generic_pipeline.py)
        # have no maintenance history; that is fine.
        return 0
    df = pd.read_parquet(processed_dir / "maintenance_clean.parquet")
    _stamp(df, tenant_id).to_sql("maintenance_records", engine, if_exists="append", index=False, method="multi")
    return len(df)


def load_ai4i_reference(engine: Engine, tenant_id: str, processed_dir: Path) -> int:
    """Shared public reference data — not tenant-scoped, so tenant_id is unused."""
    df = pd.read_parquet(processed_dir / "ai4i_clean.parquet")
    df.to_sql("ai4i_reference", engine, if_exists="append", index=False, method="multi", chunksize=2000)
    return len(df)


LOADERS = {
    "machines": load_machines,
    "sensor_summary": load_sensor_summary,
    "machine_anomalies": load_machine_anomalies,
    "maintenance_records": load_maintenance_records,
    "ai4i_reference": load_ai4i_reference,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truncate", action="store_true",
                        help="Delete this tenant's existing rows before loading")
    parser.add_argument("--tenant", default=DEFAULT_TENANT_ID,
                        help="Tenant these rows belong to (default: the demo tenant)")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR,
                        help="Directory holding the pipeline's parquet output")
    args = parser.parse_args()

    try:
        tenant_id = validate_tenant_id(args.tenant)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    engine = get_engine()
    ensure_tenant_exists(engine, tenant_id)

    if args.truncate:
        truncate_tenant(engine, tenant_id)

    tables = [t for t in LOAD_ORDER if t != "ai4i_reference" or tenant_id == DEFAULT_TENANT_ID]
    total_start = time.time()
    for table in tables:
        start = time.time()
        row_count = LOADERS[table](engine, tenant_id, args.processed_dir)
        logger.info("Loaded %-20s %6d rows in %.2fs (tenant=%s)", table, row_count, time.time() - start, tenant_id)

    logger.info("Total load time: %.2fs", time.time() - total_start)


if __name__ == "__main__":
    main()
