"""Baseline: the original single-tenant schema (Phase 2).

Revision ID: 0001
Revises:
"""
from pathlib import Path

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_SQL = Path(__file__).resolve().parent.parent / "sql" / "0001_baseline_single_tenant.sql"


def upgrade() -> None:
    # The pre-tenancy sql/schema.sql, verbatim, so a database created before
    # multi-tenancy can be stamped `alembic stamp 0001` and upgraded in place.
    op.execute(_SQL.read_text())


def downgrade() -> None:
    for table in ("maintenance_records", "machine_anomalies", "sensor_summary",
                  "machines", "ai4i_reference"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
