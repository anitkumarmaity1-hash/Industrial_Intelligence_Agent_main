"""Migration 0009 (Phase 5): up/down round trip, RLS flags, grants, guards."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from tests.test_migrations import _alembic, scratch_db  # noqa: F401  (fixture)


def test_0009_connectors_round_trip_grants_and_guards(scratch_db):
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, "0008")
    command.upgrade(cfg, "0009")
    engine = create_engine(scratch_db)
    one = lambda sql: engine.connect().execute(text(sql)).scalar()   # noqa: E731
    # RLS: connectors ENABLEd not FORCEd (the claim function runs as the owner); the rest FORCEd
    assert one("SELECT relrowsecurity AND NOT relforcerowsecurity FROM pg_class WHERE relname = 'connectors'")
    for t in ("connector_objects", "connector_runs"):
        assert one(f"SELECT relrowsecurity AND relforcerowsecurity FROM pg_class WHERE relname = '{t}'")
    # the API role manages config only; the worker role runs pulls and queues jobs
    assert one("SELECT has_column_privilege('iia_app', 'connectors', 'name', 'UPDATE')")
    assert not one("SELECT has_column_privilege('iia_app', 'connectors', 'last_status', 'UPDATE')")
    assert not one("SELECT has_column_privilege('iia_app', 'connectors', 'claimed_at', 'UPDATE')")
    assert not one("SELECT has_table_privilege('iia_app', 'connector_runs', 'INSERT')")
    assert one("SELECT has_table_privilege('iia_worker', 'connector_objects', 'INSERT')")
    assert one("SELECT has_table_privilege('iia_worker', 'jobs', 'INSERT')")
    assert not one("SELECT has_table_privilege('iia_worker', 'connectors', 'DELETE')")
    assert one("SELECT has_function_privilege('iia_worker', 'iia_claim_due_connectors(integer, integer)', 'EXECUTE')")
    assert not one("SELECT has_function_privilege('iia_app', 'iia_claim_due_connectors(integer, integer)', 'EXECUTE')")
    # no credential-shaped column anywhere in the connector tables
    cols = {r[0] for r in engine.connect().execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name IN "
        "('connectors', 'connector_objects', 'connector_runs')"))}
    assert not {c for c in cols if any(w in c for w in ("password", "token", "key_json", "credential"))}
    # audit_log accepts the system principal and still rejects unknown ones
    with engine.begin() as c:
        c.execute(text("INSERT INTO audit_log (tenant_id, principal_kind, action, outcome) "
                       "VALUES ('default', 'system', 'connector.pull', 'success')"))
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(text("INSERT INTO audit_log (tenant_id, principal_kind, action, outcome) "
                       "VALUES ('default', 'bogus', 'x', 'success')"))
    # idempotency key: one ledger row per (connector, key, version)
    with engine.begin() as c:
        c.execute(text("INSERT INTO connectors (id, tenant_id, name, type, config, schedule, mapping) VALUES "
                       "('00000000-0000-0000-0000-0000000000c1', 'default', 'c', 'gcs', '{}', '0 * * * *', '{}')"))
        c.execute(text("INSERT INTO connector_objects (tenant_id, connector_id, object_key, object_version, status) "
                       "VALUES ('default', '00000000-0000-0000-0000-0000000000c1', 'k', '1', 'queued')"))
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(text("INSERT INTO connector_objects (tenant_id, connector_id, object_key, object_version, status) "
                       "VALUES ('default', '00000000-0000-0000-0000-0000000000c1', 'k', '1', 'queued')"))
    # guards: stored connectors, then system audit rows, block the downgrade
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0008")
    with engine.begin() as c:
        c.execute(text("DELETE FROM connectors"))
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0008")
    with engine.begin() as c:
        c.execute(text("DELETE FROM audit_log WHERE principal_kind = 'system'"))
    command.downgrade(cfg, "0008")
    assert one("SELECT to_regclass('connectors')") is None
    assert one("SELECT to_regproc('iia_claim_due_connectors')") is None
    assert not one("SELECT has_table_privilege('iia_worker', 'jobs', 'INSERT')")
    command.upgrade(cfg, "head")
    engine.dispose()
