"""Multi-tenancy: tenants table, tenant_id on every company-data table,
composite keys, tenant-leading indexes, tenant_schema_mappings.

Existing rows are backfilled into the demo tenant ('default'). ai4i_reference
is public reference data and stays global.

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

CHILDREN = ("sensor_summary", "machine_anomalies", "maintenance_records")


def upgrade() -> None:
    op.execute("""
        CREATE TABLE tenants (
            tenant_id      VARCHAR(48)  PRIMARY KEY CHECK (tenant_id ~ '^[a-z0-9][a-z0-9_-]{0,47}$'),
            name           VARCHAR(200) NOT NULL,
            api_key_hash   CHAR(64)     UNIQUE,
            active         BOOLEAN      NOT NULL DEFAULT TRUE,
            created_at     TIMESTAMP    NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
        )""")
    op.execute("INSERT INTO tenants (tenant_id, name) VALUES ('default', 'Demo tenant (synthetic fleet)')")

    # Children first drop their FK to machines(machine_id) — it can't survive
    # the primary-key change.
    for table in CHILDREN:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_machine_id_fkey")

    op.execute("ALTER TABLE machines ADD COLUMN tenant_id VARCHAR(48) NOT NULL DEFAULT 'default' REFERENCES tenants(tenant_id)")
    op.execute("ALTER TABLE machines DROP CONSTRAINT machines_pkey")
    op.execute("ALTER TABLE machines ADD PRIMARY KEY (tenant_id, machine_id)")
    op.execute("DROP INDEX idx_machines_production_line")
    op.execute("CREATE INDEX idx_machines_production_line ON machines (tenant_id, production_line)")

    for table in CHILDREN:
        op.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id VARCHAR(48) NOT NULL DEFAULT 'default'")
        op.execute(f"ALTER TABLE {table} ADD FOREIGN KEY (tenant_id, machine_id) REFERENCES machines (tenant_id, machine_id)")

    op.execute("ALTER TABLE sensor_summary DROP CONSTRAINT sensor_summary_machine_id_window_start_key")
    op.execute("ALTER TABLE sensor_summary ADD UNIQUE (tenant_id, machine_id, window_start)")
    for old, new in (
        ("idx_sensor_summary_machine_window", "ON sensor_summary (tenant_id, machine_id, window_start)"),
        ("idx_sensor_summary_window_start", "ON sensor_summary (tenant_id, window_start)"),
        ("idx_machine_anomalies_machine_detected", "ON machine_anomalies (tenant_id, machine_id, detected_at)"),
        ("idx_machine_anomalies_severity_detected", "ON machine_anomalies (tenant_id, severity, detected_at)"),
        ("idx_maintenance_records_machine_date", "ON maintenance_records (tenant_id, machine_id, event_date)"),
    ):
        op.execute(f"DROP INDEX {old}")
        op.execute(f"CREATE INDEX {old} {new}")

    op.execute("""
        CREATE TABLE tenant_schema_mappings (
            id            SERIAL PRIMARY KEY,
            tenant_id     VARCHAR(48) NOT NULL REFERENCES tenants(tenant_id),
            dataset_kind  VARCHAR(30) NOT NULL DEFAULT 'operational' CHECK (dataset_kind IN ('operational', 'maintenance')),
            config        JSONB       NOT NULL,
            created_at    TIMESTAMP   NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
            UNIQUE (tenant_id, dataset_kind)
        )""")


def downgrade() -> None:
    # Only safe while the database holds a single tenant's data: collapsing
    # tenants would merge or orphan rows, so refuse rather than guess.
    conn = op.get_bind()
    extra = conn.exec_driver_sql(
        "SELECT (SELECT COUNT(*) FROM machines WHERE tenant_id <> 'default')"
        " + (SELECT COUNT(*) FROM sensor_summary WHERE tenant_id <> 'default')"
        " + (SELECT COUNT(*) FROM machine_anomalies WHERE tenant_id <> 'default')"
        " + (SELECT COUNT(*) FROM maintenance_records WHERE tenant_id <> 'default')").scalar()
    if extra:
        raise RuntimeError(
            f"Refusing to downgrade: {extra} row(s) belong to non-default tenants. "
            "Delete or export them first.")

    op.execute("DROP TABLE tenant_schema_mappings")
    for old, orig in (
        ("idx_maintenance_records_machine_date", "ON maintenance_records (machine_id, event_date)"),
        ("idx_machine_anomalies_severity_detected", "ON machine_anomalies (severity, detected_at)"),
        ("idx_machine_anomalies_machine_detected", "ON machine_anomalies (machine_id, detected_at)"),
        ("idx_sensor_summary_window_start", "ON sensor_summary (window_start)"),
        ("idx_sensor_summary_machine_window", "ON sensor_summary (machine_id, window_start)"),
    ):
        op.execute(f"DROP INDEX {old}")
        op.execute(f"CREATE INDEX {old} {orig}")
    op.execute("ALTER TABLE sensor_summary DROP CONSTRAINT sensor_summary_tenant_id_machine_id_window_start_key")
    op.execute("ALTER TABLE sensor_summary ADD UNIQUE (machine_id, window_start)")
    for table in CHILDREN:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_tenant_id_machine_id_fkey")
        op.execute(f"ALTER TABLE {table} DROP COLUMN tenant_id")
    op.execute("DROP INDEX idx_machines_production_line")
    op.execute("CREATE INDEX idx_machines_production_line ON machines (production_line)")
    op.execute("ALTER TABLE machines DROP CONSTRAINT machines_pkey")
    op.execute("ALTER TABLE machines DROP COLUMN tenant_id")
    op.execute("ALTER TABLE machines ADD PRIMARY KEY (machine_id)")
    for table in CHILDREN:
        op.execute(f"ALTER TABLE {table} ADD FOREIGN KEY (machine_id) REFERENCES machines(machine_id)")
    op.execute("DROP TABLE tenants")
