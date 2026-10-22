"""Partition sensor_summary and machine_anomalies by time (production-
readiness fix 13).

Postgres cannot ALTER a regular table into a partitioned one in place —
partitioning is a property of the table set at CREATE TABLE time — so this
recreates each table as PARTITION BY RANGE and copies every existing row
across. Both tables get exactly one partition to start, a DEFAULT
partition holding everything that exists today. That is deliberate: the
committed synthetic seed's timestamps are relative to whenever it was
generated (~60 days ending "now"), not a fixed calendar range, so this
migration has no fixed date to carve real range partitions at. What it
DOES buy immediately: the catalog-level structure a growing deployment
needs — CREATE INDEX/CHECK on the parent already applies to every future
partition, and `scripts/manage_partitions.py` can now create dedicated
monthly partitions for new data going forward (Postgres validates that no
existing DEFAULT-partition row would belong to a new range before
attaching it, scanning DEFAULT once to prove that — expected to be fast
today, and exactly the trigger for actually splitting old rows out of
DEFAULT once that scan starts being the slow part). The audit's own
framing (Part 2.1 / P2) was "before scaling past a handful of pilots" —
this is that foundation, not a claim that pruning already helps today's
25,920/28,855-row demo tables.

Both tables' primary keys change to include the partition column
(sensor_summary: window_start; machine_anomalies: detected_at) because
Postgres requires every unique constraint on a partitioned table to
include the partition key. Neither `id` column is referenced anywhere in
app/ or scripts/ (grepped — only ever used as an opaque surrogate key),
so this is safe: nothing selects, joins or ON CONFLICTs on it.

Revision ID: 0004
Revises: 0003
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


_CONSTRAINTS = {
    "sensor_summary": [
        "sensor_summary_pkey",
        "sensor_summary_tenant_id_machine_id_window_start_key",
        "sensor_summary_health_status_check",
        "sensor_summary_tenant_id_machine_id_fkey",
    ],
    "machine_anomalies": [
        "machine_anomalies_pkey",
        "machine_anomalies_severity_check",
        "machine_anomalies_tenant_id_machine_id_fkey",
    ],
}


def _park_old_constraint_names(table: str, old: str, names: list[str]) -> None:
    """Move the old table's auto-named constraints out of the way BEFORE
    the partitioned replacement is created. Constraint names (and the
    indexes behind PK/UNIQUE) are unique per schema, so without this
    Postgres suffixes the new table's with "1" and the result no longer
    matches a from-scratch sql/schema.sql."""
    for name in names:
        op.execute(f"ALTER TABLE {old} RENAME CONSTRAINT {name} TO {name}_old")


def _restore_fk_name(table: str) -> None:
    """Postgres suffixes a re-created FK with "1" if a same-named one still
    existed on the (then-not-yet-dropped) partitioned table; restore the
    canonical name once it is gone. No-op if it already has it."""
    op.execute(f"""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_constraint
                       WHERE conrelid = '{table}'::regclass
                         AND conname = '{table}_tenant_id_machine_id_fkey1') THEN
                ALTER TABLE {table} RENAME CONSTRAINT
                    {table}_tenant_id_machine_id_fkey1 TO {table}_tenant_id_machine_id_fkey;
            END IF;
        END $$
    """)


def upgrade() -> None:
    # ---- sensor_summary -------------------------------------------------
    op.execute("ALTER TABLE sensor_summary RENAME TO sensor_summary_unpartitioned")
    # Detach the BIGSERIAL sequence from the old table's column first, or
    # DROP TABLE below cascades into dropping the sequence with it.
    op.execute("ALTER SEQUENCE sensor_summary_id_seq OWNED BY NONE")
    _park_old_constraint_names("sensor_summary", "sensor_summary_unpartitioned", [
        "sensor_summary_pkey",
        "sensor_summary_tenant_id_machine_id_window_start_key",
        "sensor_summary_health_status_check",
        "sensor_summary_tenant_id_machine_id_fkey",
    ])

    op.execute("""
        CREATE TABLE sensor_summary (
            id                          BIGINT      NOT NULL DEFAULT nextval('sensor_summary_id_seq'),
            tenant_id                   VARCHAR(48) NOT NULL DEFAULT 'default',
            machine_id                  VARCHAR(10) NOT NULL,
            window_start                TIMESTAMP   NOT NULL,
            window_end                  TIMESTAMP   NOT NULL,
            avg_air_temp_k               DOUBLE PRECISION,
            avg_process_temp_k           DOUBLE PRECISION,
            avg_rotational_speed_rpm     DOUBLE PRECISION,
            avg_torque_nm                DOUBLE PRECISION,
            tool_wear_min                DOUBLE PRECISION,
            avg_production_rate          DOUBLE PRECISION,
            avg_energy_consumption_kwh   DOUBLE PRECISION,
            avg_defect_rate               DOUBLE PRECISION,
            reading_count                INTEGER NOT NULL,
            anomalous_reading_count      INTEGER NOT NULL DEFAULT 0,
            max_anomaly_score            INTEGER NOT NULL DEFAULT 0,
            health_status                VARCHAR(10) NOT NULL CHECK (health_status IN ('HEALTHY', 'WATCH', 'AT_RISK')),
            PRIMARY KEY (window_start, id),
            UNIQUE (tenant_id, machine_id, window_start),
            FOREIGN KEY (tenant_id, machine_id) REFERENCES machines (tenant_id, machine_id)
        ) PARTITION BY RANGE (window_start)
    """)
    op.execute("CREATE TABLE sensor_summary_default PARTITION OF sensor_summary DEFAULT")

    op.execute("""
        INSERT INTO sensor_summary (
            id, tenant_id, machine_id, window_start, window_end,
            avg_air_temp_k, avg_process_temp_k, avg_rotational_speed_rpm, avg_torque_nm,
            tool_wear_min, avg_production_rate, avg_energy_consumption_kwh, avg_defect_rate,
            reading_count, anomalous_reading_count, max_anomaly_score, health_status
        )
        SELECT
            id, tenant_id, machine_id, window_start, window_end,
            avg_air_temp_k, avg_process_temp_k, avg_rotational_speed_rpm, avg_torque_nm,
            tool_wear_min, avg_production_rate, avg_energy_consumption_kwh, avg_defect_rate,
            reading_count, anomalous_reading_count, max_anomaly_score, health_status
        FROM sensor_summary_unpartitioned
    """)
    op.execute("DROP TABLE sensor_summary_unpartitioned")
    op.execute("ALTER SEQUENCE sensor_summary_id_seq OWNED BY sensor_summary.id")

    op.execute(
        "CREATE INDEX idx_sensor_summary_machine_window "
        "ON sensor_summary (tenant_id, machine_id, window_start)"
    )
    op.execute(
        "CREATE INDEX idx_sensor_summary_window_start "
        "ON sensor_summary (tenant_id, window_start)"
    )

    # ---- machine_anomalies -----------------------------------------------
    op.execute("ALTER TABLE machine_anomalies RENAME TO machine_anomalies_unpartitioned")
    op.execute("ALTER SEQUENCE machine_anomalies_id_seq OWNED BY NONE")
    _park_old_constraint_names("machine_anomalies", "machine_anomalies_unpartitioned", [
        "machine_anomalies_pkey",
        "machine_anomalies_severity_check",
        "machine_anomalies_tenant_id_machine_id_fkey",
    ])

    op.execute("""
        CREATE TABLE machine_anomalies (
            id                  BIGINT      NOT NULL DEFAULT nextval('machine_anomalies_id_seq'),
            tenant_id           VARCHAR(48) NOT NULL DEFAULT 'default',
            machine_id          VARCHAR(10) NOT NULL,
            detected_at         TIMESTAMP   NOT NULL,
            anomaly_score       INTEGER     NOT NULL,
            severity            VARCHAR(10) NOT NULL CHECK (severity IN ('MEDIUM', 'HIGH')),
            triggered_reasons   TEXT,
            PRIMARY KEY (detected_at, id),
            FOREIGN KEY (tenant_id, machine_id) REFERENCES machines (tenant_id, machine_id)
        ) PARTITION BY RANGE (detected_at)
    """)
    op.execute("CREATE TABLE machine_anomalies_default PARTITION OF machine_anomalies DEFAULT")

    op.execute("""
        INSERT INTO machine_anomalies (
            id, tenant_id, machine_id, detected_at, anomaly_score, severity, triggered_reasons
        )
        SELECT id, tenant_id, machine_id, detected_at, anomaly_score, severity, triggered_reasons
        FROM machine_anomalies_unpartitioned
    """)
    op.execute("DROP TABLE machine_anomalies_unpartitioned")
    op.execute("ALTER SEQUENCE machine_anomalies_id_seq OWNED BY machine_anomalies.id")

    op.execute(
        "CREATE INDEX idx_machine_anomalies_machine_detected "
        "ON machine_anomalies (tenant_id, machine_id, detected_at)"
    )
    op.execute(
        "CREATE INDEX idx_machine_anomalies_severity_detected "
        "ON machine_anomalies (tenant_id, severity, detected_at)"
    )


def downgrade() -> None:
    for partitioned, plain, seq, pk_extra in (
        ("machine_anomalies", "machine_anomalies_unpartitioned",
         "machine_anomalies_id_seq", "id"),
        ("sensor_summary", "sensor_summary_unpartitioned",
         "sensor_summary_id_seq", "id"),
    ):
        op.execute(f"ALTER TABLE {partitioned} RENAME TO {plain}")
        op.execute(f"ALTER SEQUENCE {seq} OWNED BY NONE")
        _park_old_constraint_names(partitioned, plain, _CONSTRAINTS[partitioned])

    op.execute("""
        CREATE TABLE sensor_summary (
            id                          BIGINT      NOT NULL DEFAULT nextval('sensor_summary_id_seq') PRIMARY KEY,
            tenant_id                   VARCHAR(48) NOT NULL DEFAULT 'default',
            machine_id                  VARCHAR(10) NOT NULL,
            window_start                TIMESTAMP   NOT NULL,
            window_end                  TIMESTAMP   NOT NULL,
            avg_air_temp_k               DOUBLE PRECISION,
            avg_process_temp_k           DOUBLE PRECISION,
            avg_rotational_speed_rpm     DOUBLE PRECISION,
            avg_torque_nm                DOUBLE PRECISION,
            tool_wear_min                DOUBLE PRECISION,
            avg_production_rate          DOUBLE PRECISION,
            avg_energy_consumption_kwh   DOUBLE PRECISION,
            avg_defect_rate               DOUBLE PRECISION,
            reading_count                INTEGER NOT NULL,
            anomalous_reading_count      INTEGER NOT NULL DEFAULT 0,
            max_anomaly_score            INTEGER NOT NULL DEFAULT 0,
            health_status                VARCHAR(10) NOT NULL CHECK (health_status IN ('HEALTHY', 'WATCH', 'AT_RISK')),
            UNIQUE (tenant_id, machine_id, window_start),
            FOREIGN KEY (tenant_id, machine_id) REFERENCES machines (tenant_id, machine_id)
        )
    """)
    op.execute("""
        INSERT INTO sensor_summary SELECT
            id, tenant_id, machine_id, window_start, window_end,
            avg_air_temp_k, avg_process_temp_k, avg_rotational_speed_rpm, avg_torque_nm,
            tool_wear_min, avg_production_rate, avg_energy_consumption_kwh, avg_defect_rate,
            reading_count, anomalous_reading_count, max_anomaly_score, health_status
        FROM sensor_summary_unpartitioned
    """)
    op.execute("DROP TABLE sensor_summary_unpartitioned")
    _restore_fk_name("sensor_summary")
    op.execute("ALTER SEQUENCE sensor_summary_id_seq OWNED BY sensor_summary.id")
    op.execute(
        "CREATE INDEX idx_sensor_summary_machine_window "
        "ON sensor_summary (tenant_id, machine_id, window_start)"
    )
    op.execute(
        "CREATE INDEX idx_sensor_summary_window_start "
        "ON sensor_summary (tenant_id, window_start)"
    )

    op.execute("""
        CREATE TABLE machine_anomalies (
            id                  BIGINT      NOT NULL DEFAULT nextval('machine_anomalies_id_seq') PRIMARY KEY,
            tenant_id           VARCHAR(48) NOT NULL DEFAULT 'default',
            machine_id          VARCHAR(10) NOT NULL,
            detected_at         TIMESTAMP   NOT NULL,
            anomaly_score       INTEGER     NOT NULL,
            severity            VARCHAR(10) NOT NULL CHECK (severity IN ('MEDIUM', 'HIGH')),
            triggered_reasons   TEXT,
            FOREIGN KEY (tenant_id, machine_id) REFERENCES machines (tenant_id, machine_id)
        )
    """)
    op.execute("""
        INSERT INTO machine_anomalies SELECT
            id, tenant_id, machine_id, detected_at, anomaly_score, severity, triggered_reasons
        FROM machine_anomalies_unpartitioned
    """)
    op.execute("DROP TABLE machine_anomalies_unpartitioned")
    _restore_fk_name("machine_anomalies")
    op.execute("ALTER SEQUENCE machine_anomalies_id_seq OWNED BY machine_anomalies.id")
    op.execute(
        "CREATE INDEX idx_machine_anomalies_machine_detected "
        "ON machine_anomalies (tenant_id, machine_id, detected_at)"
    )
    op.execute(
        "CREATE INDEX idx_machine_anomalies_severity_detected "
        "ON machine_anomalies (tenant_id, severity, detected_at)"
    )
