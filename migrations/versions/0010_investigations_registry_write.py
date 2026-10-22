"""Phase 6: stored investigation results + sensor-registry editing.

  * investigations - one row per /investigate or /chat call: the question, target, intent,
    risk level and the full result JSON (the audit log keeps only the question and verdict).
    `created_by` is the user id for a JWT principal (no FK: a user row is never hard-deleted
    here, and an API-key/demo caller has none); `created_by_label` is what the UI shows.
    Tenant-scoped, ENABLEd + FORCEd RLS via iia_enable_tenant_rls().
  * iia_grant_app_privileges() gains SELECT/INSERT on investigations for iia_app and a
    column-limited UPDATE on sensor_registry (normal_min, normal_max, z_score_threshold, enabled).

Revision ID: 0010
Revises: 0009
"""
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

_NOW = "(NOW() AT TIME ZONE 'UTC')"

_TABLE = f"""
CREATE TABLE investigations (
    id               UUID          PRIMARY KEY,
    tenant_id        VARCHAR(48)   NOT NULL REFERENCES tenants(tenant_id),
    created_by       UUID,
    created_by_label VARCHAR(254)  NOT NULL,
    endpoint         VARCHAR(12)   NOT NULL CHECK (endpoint IN ('investigate', 'chat')),
    question         TEXT          NOT NULL,
    machine_id       VARCHAR(64),
    intent           VARCHAR(30),
    risk_level       VARCHAR(10),
    result           JSONB         NOT NULL,
    created_at       TIMESTAMP     NOT NULL DEFAULT {_NOW},
    UNIQUE (tenant_id, id)
);
CREATE INDEX idx_investigations_tenant_time ON investigations (tenant_id, created_at DESC);
CREATE INDEX idx_investigations_tenant_machine ON investigations (tenant_id, machine_id, created_at DESC);
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
    -- Phase 5: connectors. The API role manages configuration (it cannot write run state
    -- columns); the worker role runs pulls: it records runs/objects and, as the system
    -- principal, creates the uploads/jobs it queues and writes audit rows.
    GRANT SELECT, INSERT, DELETE ON connectors TO iia_app;
    GRANT UPDATE (name, config, schedule, mapping, secret_ref, enabled, run_requested, next_run_at, list_watermark, updated_at)
        ON connectors TO iia_app;
    GRANT SELECT ON connector_objects, connector_runs TO iia_app;
    -- Phase 6: stored investigation results (API writes + reads them) and the sensor-registry
    -- editor (API may change a sensor's range/threshold/enabled flag, nothing else).
    GRANT SELECT, INSERT ON investigations TO iia_app;
    GRANT UPDATE (normal_min, normal_max, z_score_threshold, enabled) ON sensor_registry TO iia_app;
    GRANT SELECT, UPDATE ON connectors TO iia_worker;
    GRANT SELECT, INSERT, UPDATE ON connector_objects, connector_runs TO iia_worker;
    GRANT INSERT ON uploads, jobs, audit_log TO iia_worker;
    GRANT USAGE ON SEQUENCE connector_objects_id_seq, audit_log_id_seq TO iia_worker;
    REVOKE ALL ON FUNCTION iia_claim_due_connectors(integer, integer) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION iia_claim_due_connectors(integer, integer) TO iia_worker;
    REVOKE ALL ON FUNCTION iia_enable_tenant_rls(regclass) FROM PUBLIC;
    REVOKE ALL ON FUNCTION iia_grant_app_privileges() FROM PUBLIC;
END
$fn$;
"""

# The 0009 definition, restored on downgrade.
_GRANT_FN_0009 = """
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
    -- Phase 5: connectors. The API role manages configuration (it cannot write run state
    -- columns); the worker role runs pulls: it records runs/objects and, as the system
    -- principal, creates the uploads/jobs it queues and writes audit rows.
    GRANT SELECT, INSERT, DELETE ON connectors TO iia_app;
    GRANT UPDATE (name, config, schedule, mapping, secret_ref, enabled, run_requested, next_run_at, list_watermark, updated_at)
        ON connectors TO iia_app;
    GRANT SELECT ON connector_objects, connector_runs TO iia_app;
    GRANT SELECT, UPDATE ON connectors TO iia_worker;
    GRANT SELECT, INSERT, UPDATE ON connector_objects, connector_runs TO iia_worker;
    GRANT INSERT ON uploads, jobs, audit_log TO iia_worker;
    GRANT USAGE ON SEQUENCE connector_objects_id_seq, audit_log_id_seq TO iia_worker;
    REVOKE ALL ON FUNCTION iia_claim_due_connectors(integer, integer) FROM PUBLIC;
    GRANT EXECUTE ON FUNCTION iia_claim_due_connectors(integer, integer) TO iia_worker;
    REVOKE ALL ON FUNCTION iia_enable_tenant_rls(regclass) FROM PUBLIC;
    REVOKE ALL ON FUNCTION iia_grant_app_privileges() FROM PUBLIC;
END
$fn$;
"""


def _run(sql: str) -> None:
    # Raw driver SQL (no text() bind parsing); psycopg2 needs literal % doubled.
    op.get_bind().exec_driver_sql(sql.replace("%", "%%"))


def upgrade() -> None:
    _run(_TABLE)
    _run("SELECT iia_enable_tenant_rls('investigations')")
    _run(_GRANT_FN)
    _run("SELECT iia_grant_app_privileges()")


def downgrade() -> None:
    stored = op.get_bind().exec_driver_sql("SELECT COUNT(*) FROM investigations").scalar()
    if stored:
        raise RuntimeError(
            f"Refusing to downgrade: {stored} stored investigation(s) would be dropped with the "
            "investigations table. Export or delete them first.")
    _run("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_app') THEN
                REVOKE ALL ON investigations FROM iia_app;
                REVOKE UPDATE (normal_min, normal_max, z_score_threshold, enabled)
                    ON sensor_registry FROM iia_app;
            END IF;
        END $$
    """)
    _run("DROP TABLE investigations")
    _run(_GRANT_FN_0009)
    _run("SELECT iia_grant_app_privileges()")
