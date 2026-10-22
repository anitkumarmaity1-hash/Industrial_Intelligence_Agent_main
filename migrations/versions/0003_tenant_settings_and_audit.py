"""Per-tenant calibration, investigation audit log, ingestion checkpoints.

Production-readiness fixes 11, 12, 17, 19. All three tables are purely
additive (no ALTER on any existing table), so this migration carries none
of the risk 0002's PK/FK restructuring did.

  * tenant_settings — one JSONB config row per tenant: risk thresholds,
    RAG coverage floor, machine-ID scheme (app.core.tenant_settings).
    No row for a tenant means "use the documented defaults" — nothing
    changes for an existing tenant until an operator opts in.
  * investigation_audit_log — one row per successful /investigate or
    /chat call: which tenant, which credential state, which machine,
    what the agent concluded. Append-only, never read by the API.
    ON DELETE CASCADE on tenant_id: an audit trail is scoped to its
    tenant, so removing a tenant removes its audit history with it
    rather than permanently blocking the tenant's own deletion.
  * ingestion_checkpoints — one row per (tenant, source) recording how
    far an incremental ingestion run has gotten (see
    scripts/incremental_load.py).

Revision ID: 0003
Revises: 0002
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE tenant_settings (
            tenant_id   VARCHAR(48) PRIMARY KEY REFERENCES tenants(tenant_id),
            config      JSONB       NOT NULL DEFAULT '{}'::jsonb,
            updated_at  TIMESTAMP   NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
        )
    """)

    op.execute("""
        CREATE TABLE investigation_audit_log (
            id            BIGSERIAL PRIMARY KEY,
            tenant_id     VARCHAR(48) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            request_id    VARCHAR(64),
            endpoint      VARCHAR(20) NOT NULL CHECK (endpoint IN ('investigate', 'chat')),
            machine_id    VARCHAR(10),
            intent        VARCHAR(30),
            risk_level    VARCHAR(10),
            authenticated BOOLEAN     NOT NULL,
            question      TEXT        NOT NULL,
            created_at    TIMESTAMP   NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
        )
    """)
    op.execute(
        "CREATE INDEX idx_investigation_audit_log_tenant_created "
        "ON investigation_audit_log (tenant_id, created_at)"
    )

    op.execute("""
        CREATE TABLE ingestion_checkpoints (
            tenant_id        VARCHAR(48) NOT NULL REFERENCES tenants(tenant_id),
            source           VARCHAR(50) NOT NULL,
            last_window_end  TIMESTAMP,
            last_run_at      TIMESTAMP   NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
            rows_processed   BIGINT      NOT NULL DEFAULT 0,
            PRIMARY KEY (tenant_id, source)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE ingestion_checkpoints")
    op.execute("DROP TABLE investigation_audit_log")
    op.execute("DROP TABLE tenant_settings")
