"""
End-to-end Phase 1 pipeline: ingestion -> preprocessing -> feature
engineering -> anomaly detection -> Parquet output.

Run:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --tenant acme     # a company's own onboarded data

--tenant (default "default", the demo tenant) reads from
spark_jobs.ingestion.operational_path_for/maintenance_path_for (produced by
scripts/onboard_tenant.py) and writes Parquet output under
data/processed/tenants/<tenant>/, so scripts/load_postgres.py --tenant
<tenant> --processed-dir data/processed/tenants/<tenant> can load it without
touching any other tenant's data. AI4I is not tenant-scoped (see
app.database.queries) so it is still read from the shared AI4I_PATH and
written once, only when loading the demo tenant.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F  # noqa: E402

from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id  # noqa: E402
from spark_jobs.anomaly_detection import (  # noqa: E402
    MACHINE_ANOMALIES_PATH,
    SENSOR_SUMMARY_PATH,
    run as run_anomaly_detection,
)
from spark_jobs.feature_engineering import engineer_features  # noqa: E402
from spark_jobs.ingestion import (  # noqa: E402
    AI4I_PATH,
    DATA_PROCESSED_DIR,
    get_spark_session,
    load_ai4i,
    load_synthetic_maintenance,
    load_synthetic_operational,
    maintenance_path_for,
    operational_path_for,
)
from spark_jobs.preprocessing import (  # noqa: E402
    clean_ai4i,
    clean_maintenance,
    clean_synthetic_operational,
)


def timed(label: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            print(f"[timing] {label}: {elapsed:.2f}s")
            return result

        return wrapper

    return decorator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", default=DEFAULT_TENANT_ID,
                        help="Tenant whose onboarded raw data to process (default: the demo tenant).")
    args = parser.parse_args()
    tenant_id = validate_tenant_id(args.tenant)
    is_demo = tenant_id == DEFAULT_TENANT_ID
    processed_dir = DATA_PROCESSED_DIR if is_demo else os.path.join(
        DATA_PROCESSED_DIR, "tenants", tenant_id)
    sensor_summary_path = os.path.join(processed_dir, "sensor_summary.parquet") if not is_demo else SENSOR_SUMMARY_PATH
    machine_anomalies_path = os.path.join(processed_dir, "machine_anomalies.parquet") if not is_demo else MACHINE_ANOMALIES_PATH

    pipeline_start = time.time()
    spark = get_spark_session()

    try:
        print("=" * 70)
        print(f"PHASE 1 PIPELINE: industrial-intelligence-agent (tenant={tenant_id})")
        print("=" * 70)

        # --- ingestion ---
        # AI4I is shared, public reference data (see app.database.queries) —
        # read once regardless of tenant; only written to Parquet for the
        # demo tenant's load (see below).
        t0 = time.time()
        ai4i_raw = load_ai4i(spark, AI4I_PATH)
        synth_raw = load_synthetic_operational(spark, operational_path_for(tenant_id))
        maint_raw = load_synthetic_maintenance(spark, maintenance_path_for(tenant_id))
        print(f"[timing] ingestion: {time.time() - t0:.2f}s")

        # --- preprocessing ---
        t0 = time.time()
        ai4i_clean = clean_ai4i(ai4i_raw)
        synth_clean = clean_synthetic_operational(synth_raw)
        maint_clean = clean_maintenance(maint_raw)
        print(f"[timing] preprocessing: {time.time() - t0:.2f}s")

        # --- feature engineering (synthetic operational only — AI4I is a
        # separate, one-shot reference dataset without a time dimension to
        # window over) ---
        t0 = time.time()
        featured = engineer_features(synth_clean).cache()
        n_features = len(featured.columns)
        print(f"[timing] feature engineering: {time.time() - t0:.2f}s ({n_features} columns)")

        # --- anomaly detection + aggregation ---
        # Audit F9: sensor_summary/machine_anomalies are each hit by a write
        # AND a count() AND (for sensor_summary) a final groupBy().show()
        # below — three-plus actions on the same lineage. Caching them
        # right after they're built means only the first action actually
        # recomputes from `featured`; the audit's "15+ actions re-reading
        # the CSV" was this exact pattern, repeated at every stage.
        t0 = time.time()
        sensor_summary, machine_anomalies = run_anomaly_detection(featured)
        sensor_summary = sensor_summary.cache()
        machine_anomalies = machine_anomalies.cache()

        os.makedirs(processed_dir, exist_ok=True)
        sensor_summary.write.mode("overwrite").partitionBy("machine_id").parquet(sensor_summary_path)
        machine_anomalies.write.mode("overwrite").partitionBy("machine_id").parquet(machine_anomalies_path)

        n_summary_rows = sensor_summary.count()
        n_anomaly_rows = machine_anomalies.count()
        print(f"[timing] anomaly detection + write: {time.time() - t0:.2f}s")

        # --- also persist cleaned maintenance as Parquet for Phase 2 (Postgres load) ---
        # AI4I is only (re-)written for the demo tenant: it is shared
        # reference data loaded once, not per tenant (scripts/load_postgres.py
        # skips ai4i_reference entirely for any other --tenant).
        maint_out = os.path.join(processed_dir, "maintenance_clean.parquet")
        maint_clean.write.mode("overwrite").parquet(maint_out)
        if is_demo:
            ai4i_out = os.path.join(processed_dir, "ai4i_clean.parquet")
            ai4i_clean.write.mode("overwrite").parquet(ai4i_out)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"AI4I rows (real data):            {ai4i_clean.count():,}")
        print(f"Synthetic operational rows:        {synth_clean.count():,}")
        print(f"Maintenance records:                {maint_clean.count():,}")
        print(f"sensor_summary rows (hourly):       {n_summary_rows:,}")
        print(f"machine_anomalies rows:             {n_anomaly_rows:,}")
        print(f"Total pipeline time:                 {time.time() - pipeline_start:.2f}s")

        print("\nTop machines by total anomalous readings:")
        sensor_summary.groupBy("machine_id").agg(
            F.sum("anomalous_reading_count").alias("total_anomalous_readings")
        ).orderBy(F.desc("total_anomalous_readings")).show(18)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
