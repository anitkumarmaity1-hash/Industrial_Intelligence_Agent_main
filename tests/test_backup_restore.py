"""
A backup nobody has ever restored is an unverified assumption, not a
backup (production-readiness fix 15 — the audit's "no backup/restore
strategy... beyond the Docker named volume" finding, and its own "a
tested restore procedure" follow-up item). This test actually runs
scripts/backup_db.py against the live (seeded) database, restores the
result into a second, empty scratch database via scripts/restore_db.py,
and checks that every row made the round trip — not just that the
scripts exit 0.

Skips cleanly without DATABASE_URL, pg_dump or pg_restore, matching the
rest of this suite's pattern for anything that needs real Postgres
tooling (see tests/test_migrations.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} not installed")
    return path


@pytest.fixture()
def scratch_db():
    """An empty scratch database on the same server as DATABASE_URL —
    same pattern as tests/test_migrations.py's fixture of the same name."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    base = make_url(url)
    name = f"restore_drill_{uuid.uuid4().hex[:10]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot CREATE DATABASE here: {exc}")
    scratch = base.set(database=name)
    try:
        yield scratch.render_as_string(hide_password=False)
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _row_counts(url: str) -> dict[str, int]:
    engine = create_engine(url)
    tables = [
        "tenants", "machines", "sensor_summary", "machine_anomalies",
        "maintenance_records", "ai4i_reference", "tenant_schema_mappings",
        "tenant_settings", "investigation_audit_log", "ingestion_checkpoints",
    ]
    with engine.connect() as conn:
        counts = {}
        for table in tables:
            exists = conn.execute(text(
                "SELECT 1 FROM information_schema.tables WHERE table_name = :t"),
                {"t": table}).first()
            counts[table] = (
                conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
                if exists else None
            )
    engine.dispose()
    return counts


def test_backup_then_restore_preserves_every_row(tmp_path, scratch_db):
    source_url = os.environ.get("DATABASE_URL")
    if not source_url:
        pytest.skip("DATABASE_URL not set")
    _require_tool("pg_dump")
    _require_tool("pg_restore")

    source_counts = _row_counts(source_url)
    assert sum(c or 0 for c in source_counts.values()) > 0, (
        "Source database looks empty — this test needs the seeded demo "
        "database (sql/seed.sql.gz loaded), not a fresh/empty one."
    )

    dump_path = tmp_path / "drill.dump"
    backup_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "backup_db.py"), "--out", str(dump_path)],
        capture_output=True, text=True,
        env={**os.environ, "DATABASE_URL": source_url},
    )
    assert backup_result.returncode == 0, backup_result.stderr
    assert dump_path.exists() and dump_path.stat().st_size > 0

    restore_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "restore_db.py"),
         str(dump_path), "--no-clean"],
        capture_output=True, text=True,
        env={**os.environ, "DATABASE_URL": scratch_db},
    )
    assert restore_result.returncode == 0, restore_result.stderr

    restored_counts = _row_counts(scratch_db)
    assert restored_counts == source_counts, (
        f"Row counts after restore don't match the source:\n"
        f"  source:   {source_counts}\n  restored: {restored_counts}"
    )


def test_backup_keep_prunes_older_dumps(tmp_path, monkeypatch):
    """--keep N: the prune step itself, independent of a live database —
    exercises scripts/backup_db.py's prune() directly."""
    sys.path.insert(0, str(ROOT))
    import importlib
    backup_db = importlib.import_module("scripts.backup_db")

    for i in range(5):
        path = tmp_path / f"backup_{i}.dump"
        path.write_bytes(b"x")
        os.utime(path, (i, i))  # distinct, increasing mtimes

    backup_db.prune(tmp_path, keep=2)

    remaining = sorted(p.name for p in tmp_path.glob("*.dump"))
    assert remaining == ["backup_3.dump", "backup_4.dump"]
