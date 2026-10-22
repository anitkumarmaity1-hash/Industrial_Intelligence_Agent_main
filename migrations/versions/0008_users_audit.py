"""Users, roles and a general audit log (Phase 4: users, roles, login).

  * users      - human accounts. Email is stored lower-case and is unique PER TENANT
    (login names the tenant, so one address can exist in two companies and a create
    in one tenant never reveals another's users). argon2id hash in password_hash,
    role admin|viewer. RLS is ENABLEd but not FORCEd (like `jobs`): the SECURITY
    DEFINER login lookup runs as the owner before any tenant is bound.
  * audit_log  - general audit trail (login, denial, role change, writes), separate
    from investigation_audit_log whose CHECK only allows investigate/chat. FORCEd RLS.
  * auth_lookup_user(tenant, email) - SECURITY DEFINER; returns the tenant row (if
    active) LEFT JOINed to the user, including the hash. The API role cannot SELECT
    password_hash (column-level grants), so hashes are reachable only through this.
  * auth_check_user(tenant, user_id) - SECURITY DEFINER; the per-request JWT check
    (user exists, active, tenant active; returns the CURRENT role, never the hash).
  * iia_grant_app_privileges() gains the users/audit_log grants (idempotent, so
    restore_db.py still needs no change).

Revision ID: 0008
Revises: 0007
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

_TABLES = """
CREATE TABLE users (
    id            UUID         PRIMARY KEY,
    tenant_id     VARCHAR(48)  NOT NULL REFERENCES tenants(tenant_id),
    email         VARCHAR(254) NOT NULL CHECK (email = lower(email)),
    password_hash TEXT         NOT NULL,
    role          VARCHAR(10)  NOT NULL CHECK (role IN ('admin', 'viewer')),
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP    NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    last_login_at TIMESTAMP,
    UNIQUE (tenant_id, id),
    UNIQUE (tenant_id, email)
);

CREATE TABLE audit_log (
    id             BIGSERIAL    PRIMARY KEY,
    tenant_id      VARCHAR(48)  NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    request_id     VARCHAR(64),
    principal_kind VARCHAR(10)  NOT NULL CHECK (principal_kind IN ('jwt', 'api_key', 'demo', 'anonymous')),
    principal_id   VARCHAR(64),
    actor          VARCHAR(254),
    action         VARCHAR(40)  NOT NULL,
    resource       VARCHAR(200),
    outcome        VARCHAR(10)  NOT NULL CHECK (outcome IN ('success', 'failure', 'denied')),
    detail         JSONB,
    created_at     TIMESTAMP    NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);
CREATE INDEX idx_audit_log_tenant_created ON audit_log (tenant_id, created_at);
"""

_LOOKUP_FN = """
CREATE FUNCTION auth_lookup_user(p_tenant text, p_email text)
RETURNS TABLE (tenant_id varchar, user_id uuid, password_hash text, role varchar, is_active boolean)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public AS $fn$
    SELECT t.tenant_id, u.id, u.password_hash, u.role, u.is_active
    FROM public.tenants t
    LEFT JOIN public.users u ON u.tenant_id = t.tenant_id AND u.email = lower(p_email)
    WHERE t.tenant_id = p_tenant AND t.active
$fn$;

CREATE FUNCTION auth_check_user(p_tenant text, p_user uuid)
RETURNS TABLE (user_id uuid, email varchar, role varchar, is_active boolean)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public AS $fn$
    SELECT u.id, u.email, u.role, u.is_active
    FROM public.users u JOIN public.tenants t ON t.tenant_id = u.tenant_id
    WHERE u.tenant_id = p_tenant AND u.id = p_user AND t.active
$fn$;
"""

_GRANT_FN = """
CREATE OR REPLACE FUNCTION iia_grant_app_privileges() RETURNS void
LANGUAGE plpgsql AS $fn$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_app') THEN
        CREATE ROLE iia_app NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_worker') THEN
        CREATE ROLE iia_worker NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
    REVOKE ALL ON ALL TABLES IN SCHEMA public FROM iia_app, iia_worker;
    REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM iia_app, iia_worker;
    GRANT USAGE ON SCHEMA public TO iia_app, iia_worker;
    -- API role: reads, the audit log, and the upload/job records it creates.
    GRANT SELECT ON machines, sensor_summary, machine_anomalies, maintenance_records,
        sensor_registry, sensor_readings, tenant_settings, ai4i_reference,
        rejected_rows TO iia_app;
    GRANT SELECT, INSERT, UPDATE ON uploads, jobs TO iia_app;
    GRANT INSERT ON investigation_audit_log TO iia_app;
    GRANT USAGE ON SEQUENCE investigation_audit_log_id_seq TO iia_app;
    REVOKE ALL ON FUNCTION auth_lookup_tenant(text) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION auth_lookup_tenant(text) TO iia_app;
    -- Phase 4: users + audit_log. password_hash is deliberately not selectable by the
    -- API role: login reads it through SECURITY DEFINER auth_lookup_user() only.
    GRANT SELECT (id, tenant_id, email, role, is_active, created_at, last_login_at)
        ON users TO iia_app;
    GRANT INSERT ON users TO iia_app;
    GRANT UPDATE (role, is_active, last_login_at) ON users TO iia_app;
    GRANT INSERT ON audit_log TO iia_app;
    GRANT USAGE ON SEQUENCE audit_log_id_seq TO iia_app;
    REVOKE ALL ON FUNCTION auth_lookup_user(text, text) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION auth_lookup_user(text, text) TO iia_app;
    REVOKE ALL ON FUNCTION auth_check_user(text, uuid) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION auth_check_user(text, uuid) TO iia_app;
    -- Worker role: writes the ingest pipeline's outputs, nothing else.
    GRANT SELECT ON ai4i_reference TO iia_worker;
    GRANT SELECT, UPDATE ON uploads, jobs TO iia_worker;
    GRANT SELECT, INSERT, DELETE ON rejected_rows TO iia_worker;
    GRANT SELECT, INSERT, UPDATE ON machines, sensor_registry, tenant_settings,
        sensor_summary, machine_anomalies, ingestion_checkpoints TO iia_worker;
    GRANT SELECT, INSERT ON sensor_readings TO iia_worker;
    GRANT USAGE ON SEQUENCE sensor_summary_id_seq, machine_anomalies_id_seq,
        rejected_rows_id_seq TO iia_worker;
    REVOKE ALL ON FUNCTION iia_claim_job() FROM PUBLIC;
    REVOKE ALL ON FUNCTION iia_reap_stale_jobs(integer) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION iia_claim_job() TO iia_worker;
    GRANT EXECUTE ON FUNCTION iia_reap_stale_jobs(integer) TO iia_worker;
    REVOKE ALL ON FUNCTION iia_enable_tenant_rls(regclass) FROM PUBLIC;
    REVOKE ALL ON FUNCTION iia_grant_app_privileges() FROM PUBLIC;
END
$fn$;
"""

# The 0007 definition, restored on downgrade.
_GRANT_FN_0007 = """
CREATE OR REPLACE FUNCTION iia_grant_app_privileges() RETURNS void
LANGUAGE plpgsql AS $fn$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_app') THEN
        CREATE ROLE iia_app NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_worker') THEN
        CREATE ROLE iia_worker NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
    REVOKE ALL ON ALL TABLES IN SCHEMA public FROM iia_app, iia_worker;
    REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM iia_app, iia_worker;
    GRANT USAGE ON SCHEMA public TO iia_app, iia_worker;
    -- API role: reads, the audit log, and the upload/job records it creates.
    GRANT SELECT ON machines, sensor_summary, machine_anomalies, maintenance_records,
        sensor_registry, sensor_readings, tenant_settings, ai4i_reference,
        rejected_rows TO iia_app;
    GRANT SELECT, INSERT, UPDATE ON uploads, jobs TO iia_app;
    GRANT INSERT ON investigation_audit_log TO iia_app;
    GRANT USAGE ON SEQUENCE investigation_audit_log_id_seq TO iia_app;
    REVOKE ALL ON FUNCTION auth_lookup_tenant(text) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION auth_lookup_tenant(text) TO iia_app;
    -- Worker role: writes the ingest pipeline's outputs, nothing else.
    GRANT SELECT ON ai4i_reference TO iia_worker;
    GRANT SELECT, UPDATE ON uploads, jobs TO iia_worker;
    GRANT SELECT, INSERT, DELETE ON rejected_rows TO iia_worker;
    GRANT SELECT, INSERT, UPDATE ON machines, sensor_registry, tenant_settings,
        sensor_summary, machine_anomalies, ingestion_checkpoints TO iia_worker;
    GRANT SELECT, INSERT ON sensor_readings TO iia_worker;
    GRANT USAGE ON SEQUENCE sensor_summary_id_seq, machine_anomalies_id_seq,
        rejected_rows_id_seq TO iia_worker;
    REVOKE ALL ON FUNCTION iia_claim_job() FROM PUBLIC;
    REVOKE ALL ON FUNCTION iia_reap_stale_jobs(integer) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION iia_claim_job() TO iia_worker;
    GRANT EXECUTE ON FUNCTION iia_reap_stale_jobs(integer) TO iia_worker;
    REVOKE ALL ON FUNCTION iia_enable_tenant_rls(regclass) FROM PUBLIC;
    REVOKE ALL ON FUNCTION iia_grant_app_privileges() FROM PUBLIC;
END
$fn$;
"""


def _run(sql: str) -> None:
    # Raw driver SQL (no text() bind parsing); psycopg2 needs literal % doubled.
    op.get_bind().exec_driver_sql(sql.replace("%", "%%"))


def upgrade() -> None:
    _run(_TABLES)
    _run(_LOOKUP_FN)
    _run("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
    _run("CREATE POLICY tenant_isolation ON users "
         "USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')) "
         "WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))")
    _run("SELECT iia_enable_tenant_rls('audit_log')")
    _run(_GRANT_FN)
    _run("SELECT iia_grant_app_privileges()")


def downgrade() -> None:
    conn = op.get_bind()
    stored = conn.exec_driver_sql(
        "SELECT (SELECT COUNT(*) FROM users) + (SELECT COUNT(*) FROM audit_log)").scalar()
    if stored:
        raise RuntimeError(
            f"Refusing to downgrade: {stored} user/audit_log row(s) would be dropped with the "
            "users and audit_log tables. Export or delete them first.")
    _run("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_app') THEN
                REVOKE ALL ON users, audit_log FROM iia_app;
                REVOKE ALL ON SEQUENCE audit_log_id_seq FROM iia_app;
                REVOKE ALL ON FUNCTION auth_lookup_user(text, text) FROM iia_app;
                REVOKE ALL ON FUNCTION auth_check_user(text, uuid) FROM iia_app;
            END IF;
        END $$
    """)
    _run("DROP FUNCTION auth_check_user(text, uuid)")
    _run("DROP FUNCTION auth_lookup_user(text, text)")
    _run("DROP TABLE audit_log")
    _run("DROP TABLE users")
    _run(_GRANT_FN_0007)
    _run("SELECT iia_grant_app_privileges()")
