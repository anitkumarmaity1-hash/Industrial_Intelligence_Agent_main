"""
The 1M-row scenario from the audit's Part 2.1 (production-readiness fix
16): "likely ready, unverified... load test to confirm". This generates
~1M sensor_summary rows directly in Postgres (bulk SQL, not the full
Spark pipeline — that's a pipeline-correctness test, this is a database-
at-scale test) and checks the query plans app/database/queries.py's own
queries actually get, not just that they return in some acceptable time
on this one machine.

Marked `load` (pytest.ini) — not run by default or in CI's main job; see
.github/workflows/tests.yml's separate scheduled/manual load-test job.
Skips without DATABASE_URL/CREATEDB rights, same pattern as
tests/test_migrations.py.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

pytestmark = pytest.mark.load

ROWS_PER_MACHINE = 60_000   # ~18 machines * 60,000 = ~1,080,000 rows
MACHINE_COUNT = 18


@pytest.fixture()
def scratch_db_with_schema():
    """A throwaway database with migrations applied — same scratch_db
    pattern as tests/test_migrations.py, plus `alembic upgrade head` so
    this runs against the real (partitioned, fix-13) schema."""
    from alembic import command
    from alembic.config import Config
    from pathlib import Path

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    base = make_url(url)
    name = f"load_{uuid.uuid4().hex[:10]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot CREATE DATABASE here: {exc}")
    scratch = base.set(database=name).render_as_string(hide_password=False)

    root = Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    cfg.set_main_option("sqlalchemy.url", scratch.replace("%", "%%"))
    command.upgrade(cfg, "head")

    try:
        yield scratch
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _seed_machines_and_rows(engine) -> None:
    with engine.begin() as conn:
        for i in range(1, MACHINE_COUNT + 1):
            conn.execute(text("""
                INSERT INTO machines (tenant_id, machine_id, production_line, type)
                VALUES ('default', :mid, 'LINE-1', 'M')
            """), {"mid": f"M-{i:02d}"})

        # Bulk, server-side generation — a Python-side loop of 1M inserts
        # would measure Python/network overhead, not the database. This
        # measures what load_postgres.py's real bulk COPY path produces:
        # one row per (machine, hour) going back ROWS_PER_MACHINE hours.
        conn.execute(text("""
            INSERT INTO sensor_summary (
                tenant_id, machine_id, window_start, window_end,
                avg_air_temp_k, reading_count, anomalous_reading_count,
                max_anomaly_score, health_status
            )
            SELECT
                'default',
                m.machine_id,
                TIMESTAMP '2020-01-01' + (h || ' hours')::interval,
                TIMESTAMP '2020-01-01' + ((h + 1) || ' hours')::interval,
                300 + (h % 10),
                60,
                (h % 50 = 0)::int,
                CASE WHEN h % 50 = 0 THEN 70 ELSE 5 END,
                CASE WHEN h % 50 = 0 THEN 'WATCH' ELSE 'HEALTHY' END
            FROM machines m
            CROSS JOIN generate_series(0, :rows_per_machine - 1) AS h
            WHERE m.tenant_id = 'default'
        """), {"rows_per_machine": ROWS_PER_MACHINE})


def _explain(conn, sql: str, params: dict) -> str:
    plan = conn.execute(text(f"EXPLAIN {sql}"), params).fetchall()
    return "\n".join(row[0] for row in plan)


def test_1m_rows_load_and_index_scan_plans_hold_at_scale(scratch_db_with_schema):
    engine = create_engine(scratch_db_with_schema)
    _seed_machines_and_rows(engine)

    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM sensor_summary")).scalar()
        assert total >= 1_000_000, f"expected >=1,000,000 rows, got {total}"

        # The exact query app.database.queries.get_sensor_summary issues
        # for one machine's recent history — must use the composite
        # index, not a sequential scan over a million rows.
        plan = _explain(conn, """
            SELECT machine_id, window_start, window_end, health_status
            FROM sensor_summary
            WHERE tenant_id = :tenant_id AND machine_id = :machine_id
            ORDER BY window_start DESC LIMIT 24
        """, {"tenant_id": "default", "machine_id": "M-01"})
        assert "Seq Scan" not in plan, f"expected an index scan, got:\n{plan}"

        # The fleet-wide "latest window per machine" pattern
        # (get_fleet_status) — a plain index on (tenant_id, window_start)
        # should serve this without scanning the whole table either.
        plan = _explain(conn, """
            SELECT DISTINCT ON (machine_id) machine_id, window_start, health_status
            FROM sensor_summary
            WHERE tenant_id = :tenant_id
            ORDER BY machine_id, window_start DESC
        """, {"tenant_id": "default"})
        assert "Seq Scan on sensor_summary" not in plan, f"expected an index scan, got:\n{plan}"

    engine.dispose()
