"""Migration 0010 (Phase 6): up/down round trip, RLS flags, grants, downgrade guard."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import create_engine, text

from tests.test_migrations import _alembic, scratch_db  # noqa: F401  (fixture)


def test_0010_investigations_round_trip_grants_and_guard(scratch_db):
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, "0010")
    engine = create_engine(scratch_db)
    one = lambda sql: engine.connect().execute(text(sql)).scalar()   # noqa: E731
    assert one("SELECT relrowsecurity AND relforcerowsecurity FROM pg_class WHERE relname = 'investigations'")
    assert one("SELECT has_table_privilege('iia_app', 'investigations', 'INSERT')")
    assert one("SELECT has_table_privilege('iia_app', 'investigations', 'SELECT')")
    assert not one("SELECT has_table_privilege('iia_app', 'investigations', 'UPDATE')")
    assert not one("SELECT has_table_privilege('iia_app', 'investigations', 'DELETE')")
    assert not one("SELECT has_table_privilege('iia_worker', 'investigations', 'SELECT')")
    # registry: only the four editable columns, nothing that identifies the sensor
    for col in ("normal_min", "normal_max", "z_score_threshold", "enabled"):
        assert one(f"SELECT has_column_privilege('iia_app', 'sensor_registry', '{col}', 'UPDATE')")
    for col in ("sensor_name", "unit", "tenant_id"):
        assert not one(f"SELECT has_column_privilege('iia_app', 'sensor_registry', '{col}', 'UPDATE')")
    assert not one("SELECT has_table_privilege('iia_app', 'sensor_registry', 'INSERT')")
    with engine.begin() as c:
        c.execute(text("INSERT INTO investigations (id, tenant_id, created_by_label, endpoint, question, result) "
                       "VALUES ('00000000-0000-0000-0000-0000000000a1', 'default', 'x', 'chat', 'q', '{}')"))
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0009")
    with engine.begin() as c:
        c.execute(text("DELETE FROM investigations"))
    command.downgrade(cfg, "0009")
    assert one("SELECT to_regclass('investigations')") is None
    assert not one("SELECT has_column_privilege('iia_app', 'sensor_registry', 'enabled', 'UPDATE')")
    command.upgrade(cfg, "head")
    engine.dispose()
