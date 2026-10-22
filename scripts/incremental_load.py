"""
Incremental, checkpointed loading of sensor_summary — production-
readiness fix 17.

scripts/load_postgres.py is a one-shot, whole-file loader: every run reads
the full Parquet output and appends it (or, with --truncate, replaces a
tenant's rows wholesale). That's the right tool for "load this pipeline
run's output once". It is NOT the right tool for "a company's sensors
keep producing new hourly rollups and we want to ingest just the new
ones" — the audit's own "Continuous machine telemetry — not supported at
all today" finding (Part 2.1).

This script is the micro-batch answer to that: it reads
`ingestion_checkpoints` (migration 0003) for (tenant, "sensor_summary"),
loads only rows whose window_start is newer than the checkpoint, inserts
them with ON CONFLICT DO NOTHING (sensor_summary's own
UNIQUE(tenant_id, machine_id, window_start) makes this idempotent — safe
to re-run after a partial failure without double-counting), and advances
the checkpoint to the newest window_end it saw. Run on a schedule (cron,
a scheduled Action, Airflow — whatever the deployment already uses) each
time new Parquet output lands.

This does NOT attempt true streaming (Spark Structured Streaming, Kafka,
etc.) — see docs/streaming_ingestion_design.md for why a checkpointed
micro-batch is the right next step before that, and what the actual
streaming design would look like if continuous telemetry volume ever
justifies it.

Usage:
    python scripts/incremental_load.py --tenant acme

Reads DATABASE_URL from the environment / .env.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id  # noqa: E402
from app.database import queries  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SOURCE = "sensor_summary"


def _engine() -> Engine:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set.")
        sys.exit(1)
    return create_engine(url)


def run(engine: Engine, tenant_id: str, processed_dir: Path) -> int:
    with engine.begin() as conn:
        checkpoint = queries.get_ingestion_checkpoint(conn, SOURCE, tenant_id=tenant_id)

    since = checkpoint["last_window_end"] if checkpoint else None
    logger.info("Checkpoint for tenant=%s source=%s: %s", tenant_id, SOURCE,
                since or "none (first run — loading everything)")

    df = pd.read_parquet(processed_dir / "sensor_summary.parquet")
    df = df.drop(columns=["production_line", "type"])
    df["machine_id"] = df["machine_id"].astype(str)
    if since is not None:
        df = df[df["window_start"] > since]

    if df.empty:
        logger.info("No new rows since checkpoint. Nothing to do.")
        return 0

    rows = df.to_dict(orient="records")
    for row in rows:
        row["tenant_id"] = tenant_id

    columns = list(rows[0].keys())
    insert_sql = text(f"""
        INSERT INTO sensor_summary ({", ".join(columns)})
        VALUES ({", ".join(f":{c}" for c in columns)})
        ON CONFLICT (tenant_id, machine_id, window_start) DO NOTHING
    """)

    with engine.begin() as conn:
        result = conn.execute(insert_sql, rows)
        inserted = result.rowcount if result.rowcount is not None else len(rows)
        new_checkpoint = df["window_end"].max().to_pydatetime()
        queries.upsert_ingestion_checkpoint(
            conn, SOURCE, new_checkpoint, inserted, tenant_id=tenant_id)

    skipped = len(rows) - inserted
    logger.info(
        "Inserted %d new row(s)%s; checkpoint advanced to %s.",
        inserted, f" ({skipped} already present, skipped)" if skipped else "", new_checkpoint,
    )
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default=DEFAULT_TENANT_ID,
                        help="Tenant to load for (default: the demo tenant).")
    parser.add_argument("--processed-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed",
                        help="Directory holding the pipeline's parquet output.")
    args = parser.parse_args()

    try:
        tenant_id = validate_tenant_id(args.tenant)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    run(_engine(), tenant_id, args.processed_dir)


if __name__ == "__main__":
    main()
