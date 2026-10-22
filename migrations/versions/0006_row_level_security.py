"""Row-level security: tenant isolation enforced by Postgres (Phase 2).

* ENABLE + FORCE ROW LEVEL SECURITY and one `tenant_isolation` policy
  (USING and WITH CHECK) on every tenant-scoped table and on every partition
  of the three partitioned ones. The policy compares tenant_id with the
  transaction-local setting `app.tenant_id`; unset/empty means no rows and
  failed writes (fail closed).
* `tenants` and `ai4i_reference` stay outside RLS. The API role gets no access
  to `tenants`; it authenticates through SECURITY DEFINER `auth_lookup_tenant`.
* Least-privilege role `iia_app` (NOLOGIN here; scripts/provision_app_role.py
  sets LOGIN + password from APP_DB_PASSWORD). Its grants live in
  `iia_grant_app_privileges()`, which is idempotent and is re-run after a
  restore (pg_restore --no-privileges drops grants).
* `iia_enable_tenant_rls(regclass)` is what scripts/manage_partitions.py calls
  for each new partition.

FORCE means even the table owner is subject to the policies; a superuser or
BYPASSRLS role (seed load, pg_dump, admin scripts) still bypasses them.
Downgrade removes policies, grants and functions but leaves the NOLOGIN role
(roles are cluster-wide; other databases may still reference it).

Revision ID: 0006
Revises: 0005
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

TENANT_TABLES = (
    "machines", "sensor_summary", "machine_anomalies", "maintenance_records",
    "sensor_registry", "sensor_readings", "tenant_settings",
    "tenant_schema_mappings", "investigation_audit_log", "ingestion_checkpoints",
)
PARTITIONED = ("sensor_summary", "machine_anomalies", "sensor_readings")

_TENANT_EXPR = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')"

_ENABLE_FN = f"""
CREATE FUNCTION iia_enable_tenant_rls(tbl regclass) RETURNS void
LANGUAGE plpgsql AS $fn$
BEGIN
    EXECUTE format('ALTER TABLE %s ENABLE ROW LEVEL SECURITY', tbl);
    EXECUTE format('ALTER TABLE %s FORCE ROW LEVEL SECURITY', tbl);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %s', tbl);
    EXECUTE format($q$CREATE POLICY tenant_isolation ON %s
                       USING ({_TENANT_EXPR}) WITH CHECK ({_TENANT_EXPR})$q$, tbl);
END
$fn$;
"""

_AUTH_FN = """
CREATE FUNCTION auth_lookup_tenant(p_hash text)
RETURNS TABLE (tenant_id varchar, name varchar)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public AS $fn$
    SELECT t.tenant_id, t.name FROM public.tenants t
    WHERE t.api_key_hash = p_hash AND t.active
$fn$;
"""

_GRANT_FN = """
CREATE FUNCTION iia_grant_app_privileges() RETURNS void
LANGUAGE plpgsql AS $fn$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_app') THEN
        CREATE ROLE iia_app NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
    REVOKE ALL ON ALL TABLES IN SCHEMA public FROM iia_app;
    REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM iia_app;
    GRANT USAGE ON SCHEMA public TO iia_app;
    GRANT SELECT ON machines, sensor_summary, machine_anomalies, maintenance_records,
        sensor_registry, sensor_readings, tenant_settings, ai4i_reference TO iia_app;
    GRANT INSERT ON investigation_audit_log TO iia_app;
    GRANT USAGE ON SEQUENCE investigation_audit_log_id_seq TO iia_app;
    REVOKE ALL ON FUNCTION auth_lookup_tenant(text) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION auth_lookup_tenant(text) TO iia_app;
    REVOKE ALL ON FUNCTION iia_enable_tenant_rls(regclass) FROM PUBLIC;
    REVOKE ALL ON FUNCTION iia_grant_app_privileges() FROM PUBLIC;
END
$fn$;
"""


def _run(sql: str) -> None:
    # Raw driver SQL (no text() bind parsing); psycopg2 needs literal % doubled.
    op.get_bind().exec_driver_sql(sql.replace("%", "%%"))


def upgrade() -> None:
    for ddl in (_ENABLE_FN, _AUTH_FN, _GRANT_FN):
        _run(ddl)
    for table in TENANT_TABLES:
        _run(f"SELECT iia_enable_tenant_rls('{table}')")
    for parent in PARTITIONED:
        _run("SELECT iia_enable_tenant_rls(relid) FROM pg_partition_tree("
             f"'{parent}') WHERE isleaf")
    _run("SELECT iia_grant_app_privileges()")


def downgrade() -> None:
    conn = op.get_bind()
    rels = [row[0] for row in conn.exec_driver_sql(
        "SELECT relid::regclass::text FROM (VALUES "
        + ", ".join(f"('{p}')" for p in PARTITIONED)
        + ") v(p), LATERAL pg_partition_tree(v.p::regclass) WHERE isleaf")]
    for rel in [*TENANT_TABLES, *rels]:
        _run(f"DROP POLICY IF EXISTS tenant_isolation ON {rel}")
        _run(f"ALTER TABLE {rel} NO FORCE ROW LEVEL SECURITY")
        _run(f"ALTER TABLE {rel} DISABLE ROW LEVEL SECURITY")
    _run("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_app') THEN
                REVOKE ALL ON ALL TABLES IN SCHEMA public FROM iia_app;
                REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM iia_app;
                REVOKE ALL ON FUNCTION auth_lookup_tenant(text) FROM iia_app;
                REVOKE ALL ON SCHEMA public FROM iia_app;
            END IF;
        END $$
    """)
    _run("DROP FUNCTION iia_grant_app_privileges()")
    _run("DROP FUNCTION auth_lookup_tenant(text)")
    _run("DROP FUNCTION iia_enable_tenant_rls(regclass)")
