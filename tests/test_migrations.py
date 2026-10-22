"""
Alembic migrations (production-readiness fix 9).

Two guarantees, each on a throwaway database:
  1. `alembic upgrade head` produces exactly the schema in sql/schema.sql
     (which docker-compose and CI apply directly) — so the two can't drift.
  2. Upgrading a database that already holds the pre-tenancy seed data
     preserves every row and puts it in the demo tenant. That is the real
     production path: a running deployment upgrading in place.
Needs DATABASE_URL with CREATEDB rights (the compose/CI superuser has it);
skips cleanly otherwise. The psql binary is needed to load the seed.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SQL = ROOT / "migrations" / "sql" / "0001_baseline_single_tenant.sql"
SCHEMA_SQL = ROOT / "sql" / "schema.sql"
SEED = ROOT / "sql" / "seed.sql.gz"


@pytest.fixture()
def scratch_db():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    base = make_url(url)
    name = f"mig_{uuid.uuid4().hex[:10]}"
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
            conn.execute(
                text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _alembic(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def _psql_file(url: str, sql_text: str) -> None:
    psql = shutil.which("psql")
    if not psql:
        pytest.skip("psql not installed")
    subprocess.run([psql, url, "-q", "-v", "ON_ERROR_STOP=1"], input=sql_text.encode(),
                   check=True, capture_output=True)


def _describe(url: str) -> dict[str, set]:
    engine = create_engine(url)
    with engine.connect() as c:
        cols = {tuple(r) for r in c.execute(text("""
            SELECT table_name, column_name, data_type, is_nullable, column_default
            FROM information_schema.columns WHERE table_schema = 'public'
            AND table_name <> 'alembic_version'"""))}
        idx = {tuple(r) for r in c.execute(text(
            "SELECT tablename, indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"))}
        cons = {tuple(r) for r in c.execute(text("""
            SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid)
            FROM pg_constraint WHERE connamespace = 'public'::regnamespace
            AND conrelid::regclass::text <> 'alembic_version'
            AND contype <> 'n'"""))}  # PG18 lists NOT NULL as named constraints; is_nullable above already covers them
        # Phase 2 (RLS): flags, policies, helper functions and the app role's grants.
        rls = {tuple(r) for r in c.execute(text("""
            SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
            WHERE relnamespace = 'public'::regnamespace AND relkind IN ('r', 'p')
            AND relname <> 'alembic_version'"""))}
        policies = {tuple(r) for r in c.execute(text(
            "SELECT tablename, policyname, cmd, qual, with_check FROM pg_policies "
            "WHERE schemaname = 'public'"))}
        functions = {tuple(r) for r in c.execute(text(
            "SELECT proname, prosecdef, pg_get_functiondef(oid) FROM pg_proc "
            "WHERE pronamespace = 'public'::regnamespace"))}
        grants = {tuple(r) for r in c.execute(text(
            "SELECT c.relname, has_table_privilege('iia_app', c.oid, p) FROM pg_class c, "
            "unnest(ARRAY['SELECT','INSERT','UPDATE','DELETE']) p "
            "WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p') "
            "AND c.relname <> 'alembic_version'"))}
        seed_tenants = {tuple(r) for r in c.execute(text("SELECT tenant_id, name FROM tenants"))} \
            if any(t[0] == "tenants" for t in cols) else set()
    engine.dispose()
    return {"columns": cols, "indexes": idx, "constraints": cons, "tenants": seed_tenants,
            "rls_flags": rls, "policies": policies, "functions": functions, "app_grants": grants}


def test_alembic_head_matches_schema_sql(scratch_db, tmp_path):
    command.upgrade(_alembic(scratch_db), "head")
    from_migrations = _describe(scratch_db)

    base = make_url(scratch_db)
    other_name = base.database + "_b"
    admin = create_engine(base.set(database=make_url(os.environ["DATABASE_URL"]).database),
                          isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{other_name}"'))
    try:
        other_url = base.set(database=other_name).render_as_string(
            hide_password=False)
        _psql_file(other_url, SCHEMA_SQL.read_text())
        from_schema_sql = _describe(other_url)
    finally:
        with admin.connect() as conn:
            conn.execute(
                text(f'DROP DATABASE IF EXISTS "{other_name}" WITH (FORCE)'))
        admin.dispose()

    for kind in ("columns", "indexes", "constraints", "tenants",
                 "rls_flags", "policies", "functions", "app_grants"):
        assert from_migrations[kind] == from_schema_sql[kind], (
            f"{kind} differ between `alembic upgrade head` and sql/schema.sql:\n"
            f"only in migrations: {sorted(from_migrations[kind] - from_schema_sql[kind])}\n"
            f"only in schema.sql: {sorted(from_schema_sql[kind] - from_migrations[kind])}")


def test_upgrading_a_populated_pre_tenancy_database_keeps_every_row(scratch_db):
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, "0001")
    _psql_file(scratch_db, gzip.decompress(SEED.read_bytes()).decode())

    tables = ("machines", "sensor_summary",
              "machine_anomalies", "maintenance_records")
    engine = create_engine(scratch_db)
    with engine.connect() as c:
        before = {t: c.execute(
            text(f"SELECT COUNT(*) FROM {t}")).scalar() for t in tables}
    assert before["sensor_summary"] > 0

    command.upgrade(cfg, "head")
    with engine.connect() as c:
        for t in tables:
            assert c.execute(
                text(f"SELECT COUNT(*) FROM {t}")).scalar() == before[t]
            assert c.execute(text(
                f"SELECT COUNT(*) FROM {t} WHERE tenant_id <> 'default'")).scalar() == 0
    engine.dispose()


def test_downgrade_refuses_when_other_tenants_hold_data(scratch_db):
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, "head")
    engine = create_engine(scratch_db)
    with engine.begin() as c:
        c.execute(
            text("INSERT INTO tenants (tenant_id, name) VALUES ('acme', 'Acme')"))
        c.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line, type) "
                       "VALUES ('acme', 'X-1', 'L1', 'L')"))
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0001")
    with engine.begin() as c:
        c.execute(text("DELETE FROM machines WHERE tenant_id = 'acme'"))
    command.downgrade(cfg, "0001")   # single-tenant again: allowed
    engine.dispose()


def test_0005_keeps_rows_widens_ids_and_downgrades_cleanly(scratch_db):
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, "0004")
    engine = create_engine(scratch_db)
    with engine.begin() as c:
        c.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line, type) "
                       "VALUES ('default', 'M1', 'L1', 'L')"))
    command.upgrade(cfg, "0005")
    with engine.connect() as c:
        assert c.execute(
            text("SELECT COUNT(*) FROM machines WHERE machine_id='M1'")).scalar() == 1
        assert c.execute(text(
            "SELECT COUNT(*) FROM sensor_registry WHERE tenant_id='default'")).scalar() == 8
        assert c.execute(text("SELECT config->'anomaly'->'rule_packs' ? 'ai4i' FROM tenant_settings "
                              "WHERE tenant_id='default'")).scalar() is True
        width = c.execute(text("SELECT character_maximum_length FROM information_schema.columns "
                               "WHERE table_name='machines' AND column_name='machine_id'")).scalar()
        assert width == 64

    # guard 1: a long id the old schema cannot hold
    with engine.begin() as c:
        c.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line, type) "
                       "VALUES ('default', 'PUMP-LONG-TAG-001', 'L1', NULL)"))
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0004")
    with engine.begin() as c:
        c.execute(text("DELETE FROM machines WHERE machine_id='PUMP-LONG-TAG-001'"))

    # guard 2: stored readings
    with engine.begin() as c:
        c.execute(text("INSERT INTO sensor_readings VALUES ('default','M1','torque_nm',"
                       "'2026-01-01 00:00:00', 40.0)"))
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0004")
    with engine.begin() as c:
        c.execute(text("DELETE FROM sensor_readings"))

    command.downgrade(cfg, "0004")
    with engine.connect() as c:
        assert c.execute(text("SELECT character_maximum_length FROM information_schema.columns "
                              "WHERE table_name='machines' AND column_name='machine_id'")).scalar() == 10
        assert c.execute(
            text("SELECT to_regclass('sensor_readings')")).scalar() is None
        assert c.execute(
            text("SELECT COUNT(*) FROM machines WHERE machine_id='M1'")).scalar() == 1
    engine.dispose()


def test_0006_rls_upgrade_and_downgrade_round_trip(scratch_db):
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, "0006")
    engine = create_engine(scratch_db)
    with engine.connect() as c:
        on = c.execute(text(
            "SELECT COUNT(*) FROM pg_class WHERE relrowsecurity AND relforcerowsecurity")).scalar()
        assert on == 13   # 10 tenant tables + 3 default partitions
    command.downgrade(cfg, "0005")
    with engine.connect() as c:
        assert c.execute(
            text("SELECT COUNT(*) FROM pg_class WHERE relrowsecurity")).scalar() == 0
        assert c.execute(
            text("SELECT COUNT(*) FROM pg_policies")).scalar() == 0
        assert c.execute(
            text("SELECT to_regproc('auth_lookup_tenant')")).scalar() is None
        assert c.execute(text(
            "SELECT has_table_privilege('iia_app', 'machines', 'SELECT')")).scalar() is False
    command.upgrade(cfg, "head")   # and back again
    engine.dispose()


def test_0007_uploads_jobs_round_trip_and_guards(scratch_db):
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, "0006")
    engine = create_engine(scratch_db)
    with engine.begin() as c:
        c.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line, type) "
                       "VALUES ('default', 'M1', 'L1', 'L')"))
        c.execute(text("INSERT INTO machine_anomalies (tenant_id, machine_id, detected_at, anomaly_score, severity) "
                       "VALUES ('default','M1','2026-01-01',1,'MEDIUM'),('default','M1','2026-01-01',2,'HIGH')"))
    # guard: duplicate anomalies block the unique key
    with pytest.raises(RuntimeError, match="Refusing to upgrade"):
        command.upgrade(cfg, "0007")
    with engine.begin() as c:
        c.execute(text("DELETE FROM machine_anomalies WHERE anomaly_score = 2"))
    command.upgrade(cfg, "0007")
    with engine.connect() as c:
        assert c.execute(text("SELECT character_maximum_length FROM information_schema.columns "
                              "WHERE table_name='machines' AND column_name='production_line'")).scalar() == 100
        for t in ("uploads", "jobs", "rejected_rows"):
            assert c.execute(text(
                "SELECT relrowsecurity FROM pg_class WHERE relname = :t"), {"t": t}).scalar()
        assert c.execute(text(
            "SELECT has_table_privilege('iia_worker', 'sensor_readings', 'INSERT')")).scalar()
        assert not c.execute(text(
            "SELECT has_table_privilege('iia_app', 'sensor_readings', 'INSERT')")).scalar()
    # guard: stored uploads block the downgrade
    with engine.begin() as c:
        c.execute(text("INSERT INTO uploads (upload_id, tenant_id, filename, content_type, storage_key) VALUES "
                       "('00000000-0000-0000-0000-000000000001','default','a.csv','text/csv','tenants/default/uploads/x/data.csv')"))
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0006")
    with engine.begin() as c:
        c.execute(text("DELETE FROM uploads"))
    # guard: a production_line the old width cannot hold
    with engine.begin() as c:
        c.execute(text("UPDATE machines SET production_line = :p"),
                  {"p": "L" * 40})
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0006")
    with engine.begin() as c:
        c.execute(text("UPDATE machines SET production_line = 'L1'"))
    command.downgrade(cfg, "0006")
    with engine.connect() as c:
        assert c.execute(text("SELECT to_regclass('jobs')")).scalar() is None
        assert c.execute(
            text("SELECT to_regproc('iia_claim_job')")).scalar() is None
    command.upgrade(cfg, "head")
    engine.dispose()


def test_0008_users_audit_round_trip_grants_and_guard(scratch_db):
    cfg = _alembic(scratch_db)
    command.upgrade(cfg, "0007")
    command.upgrade(cfg, "0008")
    engine = create_engine(scratch_db)
    with engine.connect() as c:
        for t in ("users", "audit_log"):
            assert c.execute(text(
                "SELECT relrowsecurity FROM pg_class WHERE relname = :t"), {"t": t}).scalar()
        # users: ENABLE only (the SECURITY DEFINER login lookup runs as the owner); audit_log: FORCEd
        assert not c.execute(text(
            "SELECT relforcerowsecurity FROM pg_class WHERE relname = 'users'")).scalar()
        assert c.execute(text(
            "SELECT relforcerowsecurity FROM pg_class WHERE relname = 'audit_log'")).scalar()
        # the API role can never read a password hash, and cannot rewrite one
        assert not c.execute(text(
            "SELECT has_column_privilege('iia_app', 'users', 'password_hash', 'SELECT')")).scalar()
        assert not c.execute(text(
            "SELECT has_column_privilege('iia_app', 'users', 'password_hash', 'UPDATE')")).scalar()
        assert c.execute(text(
            "SELECT has_column_privilege('iia_app', 'users', 'role', 'UPDATE')")).scalar()
        assert c.execute(
            text("SELECT has_table_privilege('iia_app', 'audit_log', 'INSERT')")).scalar()
        assert not c.execute(
            text("SELECT has_table_privilege('iia_app', 'audit_log', 'SELECT')")).scalar()
        assert not c.execute(
            text("SELECT has_table_privilege('iia_worker', 'users', 'SELECT')")).scalar()
        assert c.execute(text(
            "SELECT has_function_privilege('iia_app', 'auth_lookup_user(text, text)', 'EXECUTE')")).scalar()
        assert c.execute(text(
            "SELECT has_function_privilege('iia_app', 'auth_check_user(text, uuid)', 'EXECUTE')")).scalar()
    # guard: stored users block the downgrade
    with engine.begin() as c:
        c.execute(text("INSERT INTO users (id, tenant_id, email, password_hash, role) VALUES "
                       "('00000000-0000-0000-0000-000000000009', 'default', 'a@example.com', 'x', 'admin')"))
    with pytest.raises(RuntimeError, match="Refusing to downgrade"):
        command.downgrade(cfg, "0007")
    with engine.begin() as c:
        c.execute(text("DELETE FROM users"))
    command.downgrade(cfg, "0007")
    with engine.connect() as c:
        assert c.execute(text("SELECT to_regclass('users')")).scalar() is None
        assert c.execute(
            text("SELECT to_regproc('auth_lookup_user')")).scalar() is None
        assert c.execute(
            text("SELECT to_regproc('auth_check_user')")).scalar() is None
        # 0007 grants restored
        assert c.execute(
            text("SELECT has_table_privilege('iia_app', 'uploads', 'INSERT')")).scalar()
    command.upgrade(cfg, "head")   # and back again
    engine.dispose()
