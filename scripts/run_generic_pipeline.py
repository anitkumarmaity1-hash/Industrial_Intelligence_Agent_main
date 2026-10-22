"""
Run anomaly detection for a tenant that onboarded its own sensors
(scripts/onboard_tenant.py --sensor-csv ...), using that tenant's rules from
sensor_registry / tenant_settings in Postgres.

    python scripts/run_generic_pipeline.py --tenant acme
    python scripts/load_postgres.py --tenant acme --processed-dir data/processed/tenants/acme

Input : data/raw/tenants/<tenant>/sensor_readings.csv + machines.csv
Output: data/processed/tenants/<tenant>/sensor_summary.parquet,
        machine_anomalies.parquet  (same shape load_postgres.py already loads)

The demo tenant keeps using scripts/run_pipeline.py (wide AI4I CSV); both
call the same engine.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from app.core.anomaly_rules import load_anomaly_rules  # noqa: E402
from app.core.tenancy import validate_tenant_id  # noqa: E402
from app.onboarding.paths import machines_path_for, sensor_readings_path_for  # noqa: E402
from spark_jobs.ingestion import DATA_PROCESSED_DIR, get_spark_session  # noqa: E402
from spark_jobs.long_format import detect, load_long_readings, load_machines  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", required=True)
    args = parser.parse_args()
    tenant_id = validate_tenant_id(args.tenant)

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set (needed to read the tenant's sensor_registry).", file=sys.stderr)
        return 1
    engine = create_engine(url)
    with engine.connect() as conn:
        rules = load_anomaly_rules(conn, tenant_id)
    if not rules.active_sensors:
        print(f"Tenant {tenant_id!r} has no enabled sensors in sensor_registry; nothing to detect.",
              file=sys.stderr)
        return 1

    spark = get_spark_session()
    try:
        long_df = load_long_readings(spark, sensor_readings_path_for(tenant_id))
        machines = load_machines(spark, machines_path_for(tenant_id))
        summary, anomalies = detect(long_df, machines, rules)
        out = os.path.join(DATA_PROCESSED_DIR, "tenants", tenant_id)
        os.makedirs(out, exist_ok=True)
        summary.write.mode("overwrite").partitionBy("machine_id").parquet(
            os.path.join(out, "sensor_summary.parquet"))
        anomalies.write.mode("overwrite").partitionBy("machine_id").parquet(
            os.path.join(out, "machine_anomalies.parquet"))
        print(f"tenant={tenant_id}: sensor_summary={summary.count()} rows, "
              f"machine_anomalies={anomalies.count()} rows -> {out}")
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
