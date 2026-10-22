"""Generic sensor data model (Phase 1 of the self-serve platform).

  * sensor_registry  - per-tenant sensor vocabulary: unit, normal range,
    optional z-score threshold, enabled flag. Read by the anomaly engine
    (app.core.anomaly_rules). Seeded for the demo tenant with the eight AI4I
    sensors (five z-scored + three that only feed the `ai4i` rule pack).
  * sensor_readings  - long-format raw readings (tenant, machine, sensor,
    ts, value), RANGE-partitioned on ts like migration 0004 (single DEFAULT
    partition to start; scripts/manage_partitions.py adds monthly ones). The
    primary key (tenant_id, machine_id, sensor_name, ts) is the idempotency
    key: loads use ON CONFLICT on it. Composite FK to machines, and an FK to
    sensor_registry so a reading can only use a sensor its tenant declared.
  * tenant_settings seed for the demo tenant: enables the `ai4i` rule pack so
    the AI4I physical rules keep running exactly as before.

Two EXISTING columns are relaxed (backward compatible, but not purely
additive - called out in CHANGES_PHASE1.md):
  * machines.type loses NOT NULL. Its CHECK (L/M/H) stays, so a NULL is
    allowed and any non-NULL value must still be L/M/H. "Quality variant"
    is an AI4I concept; a pump fleet has no honest value for it.
  * machine_id widens VARCHAR(10) -> VARCHAR(64) everywhere it appears, so
    real asset tags fit.

Revision ID: 0005
Revises: 0004
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

_MACHINE_ID_TABLES = (
    "machines", "sensor_summary", "machine_anomalies",
    "maintenance_records", "investigation_audit_log",
)

_DEMO_SENSORS = (
    ("torque_nm", "Nm", True),
    ("tool_wear_min", "min", True),
    ("production_rate", "units/h", True),
    ("defect_rate", "fraction", True),
    ("energy_consumption_kwh", "kWh", True),
    ("air_temperature_k", "K", False),
    ("process_temperature_k", "K", False),
    ("rotational_speed_rpm", "rpm", False),
)


def upgrade() -> None:
    # machines first: it is the referenced side of every composite FK.
    for table in _MACHINE_ID_TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN machine_id TYPE VARCHAR(64)")
    op.execute("ALTER TABLE machines ALTER COLUMN type DROP NOT NULL")

    op.execute("""
        CREATE TABLE sensor_registry (
            tenant_id          VARCHAR(48)      NOT NULL REFERENCES tenants(tenant_id),
            sensor_name        VARCHAR(64)      NOT NULL CHECK (sensor_name ~ '^[a-z][a-z0-9_]{0,63}$'),
            unit               VARCHAR(32)      NOT NULL,
            normal_min         DOUBLE PRECISION,
            normal_max         DOUBLE PRECISION,
            z_score_threshold  DOUBLE PRECISION CHECK (z_score_threshold > 0),
            enabled            BOOLEAN          NOT NULL DEFAULT TRUE,
            PRIMARY KEY (tenant_id, sensor_name),
            CHECK (normal_min IS NULL OR normal_max IS NULL OR normal_min <= normal_max)
        )
    """)
    values = ", ".join(
        f"('default', '{n}', '{u}', {'TRUE' if e else 'FALSE'})" for n, u, e in _DEMO_SENSORS)
    op.execute(f"INSERT INTO sensor_registry (tenant_id, sensor_name, unit, enabled) VALUES {values}")

    op.execute("""
        CREATE TABLE sensor_readings (
            tenant_id    VARCHAR(48)      NOT NULL,
            machine_id   VARCHAR(64)      NOT NULL,
            sensor_name  VARCHAR(64)      NOT NULL,
            ts           TIMESTAMP        NOT NULL,
            value        DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (tenant_id, machine_id, sensor_name, ts),
            FOREIGN KEY (tenant_id, machine_id) REFERENCES machines (tenant_id, machine_id),
            FOREIGN KEY (tenant_id, sensor_name) REFERENCES sensor_registry (tenant_id, sensor_name)
        ) PARTITION BY RANGE (ts)
    """)
    op.execute("CREATE TABLE sensor_readings_default PARTITION OF sensor_readings DEFAULT")
    op.execute("CREATE INDEX idx_sensor_readings_tenant_ts ON sensor_readings (tenant_id, ts)")

    op.execute("""
        INSERT INTO tenant_settings (tenant_id, config)
        VALUES ('default', '{"anomaly": {"rule_packs": {"ai4i": {}}}}'::jsonb)
        ON CONFLICT (tenant_id) DO UPDATE
            SET config = tenant_settings.config || EXCLUDED.config
    """)


def downgrade() -> None:
    conn = op.get_bind()
    long_ids = conn.exec_driver_sql(
        "SELECT (SELECT COUNT(*) FROM machines WHERE length(machine_id) > 10)"
        " + (SELECT COUNT(*) FROM machines WHERE type IS NULL)").scalar()
    if long_ids:
        raise RuntimeError(
            f"Refusing to downgrade: {long_ids} machine(s) have a machine_id longer than "
            "10 characters or a NULL type, which the pre-0005 schema cannot hold. "
            "Delete or export them first.")

    stored = conn.exec_driver_sql(
        "SELECT (SELECT COUNT(*) FROM sensor_readings)"
        " + (SELECT COUNT(*) FROM sensor_registry WHERE tenant_id <> 'default')").scalar()
    if stored:
        raise RuntimeError(
            f"Refusing to downgrade: {stored} sensor reading / non-demo registry row(s) would be "
            "dropped with the sensor_readings and sensor_registry tables. Export or delete them first.")

    op.execute("""
        UPDATE tenant_settings SET config = config - 'anomaly' WHERE tenant_id = 'default'
    """)
    op.execute("DELETE FROM tenant_settings WHERE tenant_id = 'default' AND config = '{}'::jsonb")

    op.execute("DROP TABLE sensor_readings")
    op.execute("DROP TABLE sensor_registry")

    op.execute("ALTER TABLE machines ALTER COLUMN type SET NOT NULL")
    for table in reversed(_MACHINE_ID_TABLES):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN machine_id TYPE VARCHAR(10)")
