"""
scripts/incremental_load.py — production-readiness fix 17. Proves the
checkpoint behavior itself (idempotent re-run, only-new-rows-loaded,
checkpoint advances correctly) against a real scratch database, using a
small synthetic Parquet file rather than the full demo fleet's — this is
a test of the checkpointing logic, not of scripts/run_pipeline.py's
output (that's tests/test_pipeline.py's job).

Skips without DATABASE_URL/CREATEDB rights, same pattern as
tests/test_migrations.py.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def scratch_db_with_schema():
    """Throwaway DB with migrations applied and a demo tenant + two
    machines seeded — enough for sensor_summary's FK to resolve."""
    from alembic import command
    from alembic.config import Config

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    base = make_url(url)
    name = f"incr_{uuid.uuid4().hex[:10]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot CREATE DATABASE here: {exc}")
    scratch_url = base.set(database=name).render_as_string(hide_password=False)

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", scratch_url.replace("%", "%%"))
    command.upgrade(cfg, "head")

    engine = create_engine(scratch_url)
    with engine.begin() as conn:
        # The 'default' tenant is already seeded by sql/schema.sql's own
        # baseline INSERT (applied via migration 0001) — only the two
        # machines sensor_summary's FK needs are missing here.
        for mid in ("M-01", "M-02"):
            conn.execute(text(
                "INSERT INTO machines (tenant_id, machine_id, production_line, type) "
                "VALUES ('default', :mid, 'LINE-1', 'M')"), {"mid": mid})
    engine.dispose()

    try:
        yield scratch_url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / "sensor_summary.parquet", index=False)


def _row(machine_id: str, hour: int, reading_count: int = 60) -> dict:
    start = datetime(2026, 1, 1, hour % 24)
    return {
        "machine_id": machine_id,
        "production_line": "LINE-1",
        "type": "M",
        "window_start": start,
        "window_end": start,
        "avg_air_temp_k": 300.0,
        "avg_process_temp_k": 310.0,
        "avg_rotational_speed_rpm": 1500.0,
        "avg_torque_nm": 40.0,
        "tool_wear_min": 10.0,
        "avg_production_rate": 1.0,
        "avg_energy_consumption_kwh": 5.0,
        "avg_defect_rate": 0.01,
        "reading_count": reading_count,
        "anomalous_reading_count": 0,
        "max_anomaly_score": 0,
        "health_status": "HEALTHY",
    }


def test_first_run_loads_everything_and_sets_checkpoint(tmp_path, scratch_db_with_schema):
    import scripts.incremental_load as incremental_load

    _write_parquet(tmp_path, [_row("M-01", h) for h in range(5)])
    engine = create_engine(scratch_db_with_schema)

    inserted = incremental_load.run(engine, "default", tmp_path)
    assert inserted == 5

    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sensor_summary")).scalar() == 5
        checkpoint = conn.execute(text(
            "SELECT rows_processed FROM ingestion_checkpoints "
            "WHERE tenant_id = 'default' AND source = 'sensor_summary'"
        )).scalar()
        assert checkpoint == 5
    engine.dispose()


def test_rerun_with_no_new_rows_is_a_noop(tmp_path, scratch_db_with_schema):
    import scripts.incremental_load as incremental_load

    _write_parquet(tmp_path, [_row("M-01", h) for h in range(5)])
    engine = create_engine(scratch_db_with_schema)

    incremental_load.run(engine, "default", tmp_path)
    second_run_inserted = incremental_load.run(engine, "default", tmp_path)

    assert second_run_inserted == 0
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sensor_summary")).scalar() == 5
    engine.dispose()


def test_only_rows_newer_than_checkpoint_are_loaded_on_a_later_run(tmp_path, scratch_db_with_schema):
    import scripts.incremental_load as incremental_load

    engine = create_engine(scratch_db_with_schema)

    _write_parquet(tmp_path, [_row("M-01", h) for h in range(3)])
    first = incremental_load.run(engine, "default", tmp_path)
    assert first == 3

    # Simulate a new pipeline run that appended 2 more hours of data.
    _write_parquet(tmp_path, [_row("M-01", h) for h in range(5)])
    second = incremental_load.run(engine, "default", tmp_path)
    assert second == 2  # only hours 3 and 4 are newer than the checkpoint

    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sensor_summary")).scalar() == 5
    engine.dispose()


def test_rerun_after_a_partial_failure_does_not_double_count(tmp_path, scratch_db_with_schema):
    """ON CONFLICT DO NOTHING means re-running over an overlapping window
    (e.g. after a crash mid-run, with the checkpoint not yet advanced
    that far) is safe — no duplicate rows, no unique-constraint error."""
    import scripts.incremental_load as incremental_load

    _write_parquet(tmp_path, [_row("M-01", h) for h in range(5)])
    engine = create_engine(scratch_db_with_schema)

    incremental_load.run(engine, "default", tmp_path)

    # Rewind the checkpoint to simulate "we're not sure how far the last
    # run got" and re-run over the same (now overlapping) window.
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE ingestion_checkpoints SET last_window_end = :t "
            "WHERE tenant_id = 'default' AND source = 'sensor_summary'"
        ), {"t": datetime(2026, 1, 1, 0)})

    incremental_load.run(engine, "default", tmp_path)

    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sensor_summary")).scalar() == 5
    engine.dispose()
