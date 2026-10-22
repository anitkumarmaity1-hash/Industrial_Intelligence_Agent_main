"""
Create dedicated monthly partitions for sensor_summary / machine_anomalies
ahead of time (production-readiness fix 13).

Both tables are partitioned by RANGE (migrations/versions/
0004_partition_timeseries_tables.py) with a single DEFAULT partition
holding everything that existed when the migration ran. New rows land in
DEFAULT until a dedicated partition exists for their time range — this
script creates one.

Usage:
    # Create next month's partition for both tables (the common case —
    # run this from cron/a scheduled job a few days before month-end):
    python scripts/manage_partitions.py ensure-next

    # Create a specific partition:
    python scripts/manage_partitions.py create sensor_summary 2026-11
    python scripts/manage_partitions.py create machine_anomalies 2026-11

    # See what partitions exist today:
    python scripts/manage_partitions.py list

Creating a partition for a month that already has rows sitting in
DEFAULT does NOT move them — Postgres only lets you attach a new range
partition once it has scanned DEFAULT and confirmed no existing row
there falls in that range (an error here means "that month's data is
already in DEFAULT"; this script does not split DEFAULT, see the
migration's own docstring for why that is a deliberate follow-up step,
not this script's job).

Reads DATABASE_URL from the environment / .env.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PARTITION_COLUMN = {
    "sensor_summary": "window_start",
    "machine_anomalies": "detected_at",
    "sensor_readings": "ts",   # migration 0005
}


def _engine():
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set.")
        sys.exit(1)
    return create_engine(url)


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    days = calendar.monthrange(year, month)[1]
    end = date(year, month, days) + __import__("datetime").timedelta(days=1)
    return start, end


def _next_month(today: date) -> tuple[int, int]:
    return (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)


def create_partition(engine, table: str, year: int, month: int) -> None:
    if table not in PARTITION_COLUMN:
        logger.error("Unknown partitioned table %r (expected one of %s)",
                     table, sorted(PARTITION_COLUMN))
        sys.exit(1)
    start, end = _month_bounds(year, month)
    partition_name = f"{table}_y{year}m{month:02d}"
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE {partition_name} PARTITION OF {table}
            FOR VALUES FROM (:start) TO (:end)
        """), {"start": start, "end": end})
        # Phase 2: new partitions get the tenant RLS policy too (defence in
        # depth; the app role has no grants on partitions in any case).
        conn.execute(text("SELECT iia_enable_tenant_rls(CAST(:p AS regclass))"),
                     {"p": partition_name})
    logger.info(
        "Created %s for %s covering [%s, %s).", partition_name, table, start, end)


def ensure_next(engine) -> None:
    year, month = _next_month(date.today())
    for table in PARTITION_COLUMN:
        try:
            create_partition(engine, table, year, month)
        except Exception as exc:  # noqa: BLE001 - likely "already exists", report and continue
            logger.warning("Skipped %s %04d-%02d: %s", table, year, month, exc)


def list_partitions(engine) -> None:
    with engine.connect() as conn:
        for table in PARTITION_COLUMN:
            rows = conn.execute(text("""
                SELECT child.relname,
                       pg_get_expr(child.relpartbound, child.oid) AS bounds
                FROM pg_inherits
                JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
                JOIN pg_class child ON pg_inherits.inhrelid = child.oid
                WHERE parent.relname = :table
                ORDER BY child.relname
            """), {"table": table}).fetchall()
            logger.info("%s:", table)
            for name, bounds in rows:
                logger.info("  %-40s %s", name, bounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "ensure-next", help="Create next month's partition for both tables.")
    sub.add_parser("list", help="Show existing partitions for both tables.")

    create = sub.add_parser(
        "create", help="Create one table's partition for one month.")
    create.add_argument("table", choices=sorted(PARTITION_COLUMN))
    create.add_argument("year_month", help="YYYY-MM, e.g. 2026-11")

    args = parser.parse_args()
    engine = _engine()

    if args.command == "ensure-next":
        ensure_next(engine)
    elif args.command == "list":
        list_partitions(engine)
    elif args.command == "create":
        year_str, month_str = args.year_month.split("-")
        create_partition(engine, args.table, int(year_str), int(month_str))


if __name__ == "__main__":
    main()
