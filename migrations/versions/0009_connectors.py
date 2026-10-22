"""Pull connectors (Phase 5): configuration, per-object ingest ledger, run history.

  * connectors        - one row per configured pull source (type 'gcs'). `config` holds only
    bucket/prefix/pattern; credentials are NEVER stored: `secret_ref` is a NAME resolved by a
    SecretsProvider at run time. `mapping` is a snapshot of a confirmed Phase 1 mapping.
    `list_watermark` is the listing cursor; `claimed_at` is the lease that stops two
    scheduler processes from running the same connector. RLS is ENABLEd but not FORCEd
    (like `jobs`): the SECURITY DEFINER claim function runs as the owner and must see
    every tenant's due connectors. iia_app / iia_worker are not owners, so the policy binds them.
  * connector_objects - one row per (connector, object key, object version): the dedupe
    ledger. UNIQUE (tenant, connector, key, version) is the idempotency key: a file is
    queued at most once, even with two schedulers racing. Independent of the
    `ingestion_checkpoints` timestamp, which stays a data high-water mark.
  * connector_runs    - history shown by GET /connectors/{id}/runs (counts + a secret-free message).
  * iia_claim_due_connectors(lease, limit) - SECURITY DEFINER, ids only (like iia_claim_job).
  * audit_log.principal_kind gains 'system' (the scheduler has no human or API-key principal).
  * iia_grant_app_privileges() gains the connector grants and the worker's INSERT on
    uploads/jobs/audit_log (so restore_db.py still needs no change).

Revision ID: 0009
Revises: 0008
"""
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

_NOW = "(NOW() AT TIME ZONE 'UTC')"

_TABLES = f"""
CREATE TABLE connectors (
    id             UUID          PRIMARY KEY,
    tenant_id      VARCHAR(48)   NOT NULL REFERENCES tenants(tenant_id),
    name           VARCHAR(100)  NOT NULL,
    type           VARCHAR(10)   NOT NULL CHECK (type IN ('gcs')),
    config         JSONB         NOT NULL,
    schedule       VARCHAR(100)  NOT NULL,
    mapping        JSONB         NOT NULL,
    secret_ref     VARCHAR(100),
    enabled        BOOLEAN       NOT NULL DEFAULT TRUE,
    run_requested  BOOLEAN       NOT NULL DEFAULT FALSE,
    next_run_at    TIMESTAMP,
    claimed_at     TIMESTAMP,
    list_watermark TIMESTAMP,
    last_run_at    TIMESTAMP,
    last_status    VARCHAR(10)   NOT NULL DEFAULT 'never'
                   CHECK (last_status IN ('never', 'ok', 'partial', 'failed')),
    last_message   VARCHAR(300),
    created_at     TIMESTAMP     NOT NULL DEFAULT {_NOW},
    updated_at     TIMESTAMP     NOT NULL DEFAULT {_NOW},
    UNIQUE (tenant_id, id),
    UNIQUE (tenant_id, name)
);
CREATE INDEX idx_connectors_due ON connectors (next_run_at) WHERE enabled;

CREATE TABLE connector_objects (
    id             BIGSERIAL     PRIMARY KEY,
    tenant_id      VARCHAR(48)   NOT NULL REFERENCES tenants(tenant_id),
    connector_id   UUID          NOT NULL,
    object_key     VARCHAR(1024) NOT NULL,
    object_version VARCHAR(64)   NOT NULL,
    size_bytes     BIGINT,
    object_updated TIMESTAMP,
    status         VARCHAR(10)   NOT NULL CHECK (status IN ('queued', 'retry', 'rejected')),
    upload_id      UUID,
    job_id         UUID,
    attempts       INTEGER       NOT NULL DEFAULT 0,
    last_error     VARCHAR(300),
    first_seen_at  TIMESTAMP     NOT NULL DEFAULT {_NOW},
    updated_at     TIMESTAMP     NOT NULL DEFAULT {_NOW},
    UNIQUE (tenant_id, connector_id, object_key, object_version),
    FOREIGN KEY (tenant_id, connector_id) REFERENCES connectors (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, upload_id) REFERENCES uploads (tenant_id, upload_id),
    FOREIGN KEY (tenant_id, job_id) REFERENCES jobs (tenant_id, job_id)
);
CREATE INDEX idx_connector_objects_connector ON connector_objects (tenant_id, connector_id, status);

CREATE TABLE connector_runs (
    id              UUID         PRIMARY KEY,
    tenant_id       VARCHAR(48)  NOT NULL REFERENCES tenants(tenant_id),
    connector_id    UUID         NOT NULL,
    trigger         VARCHAR(10)  NOT NULL CHECK (trigger IN ('schedule', 'manual')),
    status          VARCHAR(10)  NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'ok', 'partial', 'failed')),
    started_at      TIMESTAMP    NOT NULL DEFAULT {_NOW},
    finished_at     TIMESTAMP,
    objects_seen    INTEGER      NOT NULL DEFAULT 0,
    objects_new     INTEGER      NOT NULL DEFAULT 0,
    objects_queued  INTEGER      NOT NULL DEFAULT 0,
    objects_failed  INTEGER      NOT NULL DEFAULT 0,
    message         VARCHAR(300),
    FOREIGN KEY (tenant_id, connector_id) REFERENCES connectors (tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX idx_connector_runs_connector ON connector_runs (tenant_id, connector_id, started_at);
"""

_CLAIM_FN = f"""
CREATE FUNCTION iia_claim_due_connectors(p_lease_seconds integer, p_limit integer)
RETURNS TABLE (connector_id uuid, tenant_id varchar, manual boolean)
LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog, public AS $fn$
    UPDATE public.connectors c SET claimed_at = {_NOW}
     WHERE c.id IN (
        SELECT q.id FROM public.connectors q
          JOIN public.tenants t ON t.tenant_id = q.tenant_id
         WHERE q.enabled AND t.active AND q.next_run_at <= {_NOW}
           AND (q.claimed_at IS NULL
                OR q.claimed_at < {_NOW} - make_interval(secs => p_lease_seconds))
         ORDER BY q.next_run_at
         FOR UPDATE OF q SKIP LOCKED LIMIT p_limit)
    RETURNING c.id, c.tenant_id, c.run_requested
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

# The 0008 definition, restored on downgrade.
_GRANT_FN_0008 = """
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


def _run(sql: str) -> None:
    # Raw driver SQL (no text() bind parsing); psycopg2 needs literal % doubled.
    op.get_bind().exec_driver_sql(sql.replace("%", "%%"))


def upgrade() -> None:
    _run(_TABLES)
    _run(_CLAIM_FN)
    _run("ALTER TABLE connectors ENABLE ROW LEVEL SECURITY")
    _run("CREATE POLICY tenant_isolation ON connectors "
         "USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')) "
         "WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))")
    _run("SELECT iia_enable_tenant_rls('connector_objects')")
    _run("SELECT iia_enable_tenant_rls('connector_runs')")
    _run("ALTER TABLE audit_log DROP CONSTRAINT audit_log_principal_kind_check")
    _run("ALTER TABLE audit_log ADD CONSTRAINT audit_log_principal_kind_check CHECK "
         "(principal_kind IN ('jwt', 'api_key', 'demo', 'anonymous', 'system'))")
    _run(_GRANT_FN)
    _run("SELECT iia_grant_app_privileges()")


def downgrade() -> None:
    conn = op.get_bind()
    stored = conn.exec_driver_sql(
        "SELECT (SELECT COUNT(*) FROM connectors) + (SELECT COUNT(*) FROM connector_objects) "
        "+ (SELECT COUNT(*) FROM connector_runs)").scalar()
    if stored:
        raise RuntimeError(
            f"Refusing to downgrade: {stored} connector/object/run row(s) would be dropped with "
            "the connectors, connector_objects and connector_runs tables. Export or delete them first.")
    system = conn.exec_driver_sql("SELECT COUNT(*) FROM audit_log WHERE principal_kind = 'system'").scalar()
    if system:
        raise RuntimeError(
            f"Refusing to downgrade: {system} audit_log row(s) have principal_kind 'system', which "
            "the previous schema's CHECK does not allow. Export or delete them first.")
    _run("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_worker') THEN
                REVOKE ALL ON connectors, connector_objects, connector_runs FROM iia_worker;
                REVOKE ALL ON SEQUENCE connector_objects_id_seq FROM iia_worker;
                REVOKE ALL ON FUNCTION iia_claim_due_connectors(integer, integer) FROM iia_worker;
                REVOKE INSERT ON uploads, jobs, audit_log FROM iia_worker;
                REVOKE ALL ON SEQUENCE audit_log_id_seq FROM iia_worker;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iia_app') THEN
                REVOKE ALL ON connectors, connector_objects, connector_runs FROM iia_app;
            END IF;
        END $$
    """)
    _run("DROP TABLE connector_runs")
    _run("DROP TABLE connector_objects")
    _run("DROP TABLE connectors")
    _run("DROP FUNCTION iia_claim_due_connectors(integer, integer)")
    _run("ALTER TABLE audit_log DROP CONSTRAINT audit_log_principal_kind_check")
    _run("ALTER TABLE audit_log ADD CONSTRAINT audit_log_principal_kind_check CHECK "
         "(principal_kind IN ('jwt', 'api_key', 'demo', 'anonymous'))")
    _run(_GRANT_FN_0008)
    _run("SELECT iia_grant_app_privileges()")
