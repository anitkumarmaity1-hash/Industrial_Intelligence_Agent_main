"""Where a tenant's onboarded raw files live on disk.

Split out from spark_jobs/ingestion.py so anything that only needs these
paths (scripts/onboard_tenant.py, tests) doesn't have to import pyspark.
spark_jobs.ingestion re-exports the same three names for pipeline code.
"""

from __future__ import annotations

import os

from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def raw_dir_for(tenant_id: str) -> str:
    """The demo tenant keeps using DATA_RAW_DIR directly so nothing already
    generated has to move; every other tenant gets its own subdirectory."""
    validate_tenant_id(tenant_id)
    if tenant_id == DEFAULT_TENANT_ID:
        return DATA_RAW_DIR
    return os.path.join(DATA_RAW_DIR, "tenants", tenant_id)


def operational_path_for(tenant_id: str) -> str:
    return os.path.join(raw_dir_for(tenant_id), "synthetic_operational.csv")


def maintenance_path_for(tenant_id: str) -> str:
    return os.path.join(raw_dir_for(tenant_id), "synthetic_maintenance.csv")


# Phase 1 (generic sensors): the onboarded bundle - long-format readings,
# machine dimension and registry/anomaly config - lives next to the legacy
# operational file for the tenant.
def sensor_readings_path_for(tenant_id: str) -> str:
    return os.path.join(raw_dir_for(tenant_id), "sensor_readings.csv")


def machines_path_for(tenant_id: str) -> str:
    return os.path.join(raw_dir_for(tenant_id), "machines.csv")


def sensor_registry_path_for(tenant_id: str) -> str:
    return os.path.join(raw_dir_for(tenant_id), "sensor_registry.json")
