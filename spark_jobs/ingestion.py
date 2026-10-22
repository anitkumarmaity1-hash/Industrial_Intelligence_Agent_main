"""
Ingestion layer: load the real AI4I 2020 dataset and the synthetic
operational/maintenance data into validated Spark DataFrames.

Why PySpark here (not pandas): this module is the entry point for a
pipeline meant to scale past the 311K-row MVP dataset to a full
fleet-telemetry workload. Doing schema validation and null-handling as
Spark operations (rather than pandas) means the exact same code scales to
a multi-file/partitioned data lake without a rewrite — that's the point
we're demonstrating, not just "processed a CSV."
"""

from __future__ import annotations

import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from app.onboarding.paths import (  # noqa: E402
    maintenance_path_for,
    operational_path_for,
    raw_dir_for,
)

# Re-exported for backward compatibility with anything importing them from
# here; the implementations live in app.onboarding.paths (see its docstring
# for why: it avoids a pyspark import for callers that only need a path).
__all__ = ["raw_dir_for", "operational_path_for", "maintenance_path_for"]

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

AI4I_PATH = os.path.join(DATA_RAW_DIR, "ai4i2020.csv")
SYNTHETIC_OPERATIONAL_PATH = os.path.join(
    DATA_RAW_DIR, "synthetic_operational.csv")
SYNTHETIC_MAINTENANCE_PATH = os.path.join(
    DATA_RAW_DIR, "synthetic_maintenance.csv")


def get_spark_session(app_name: str = "industrial-intelligence-agent") -> SparkSession:
    """Local-mode Spark session — no cluster needed for this MVP's data volume."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


SYNTHETIC_OPERATIONAL_SCHEMA = StructType(
    [
        StructField("machine_id", StringType(), nullable=False),
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("production_line", StringType(), nullable=False),
        StructField("type", StringType(), nullable=False),
        StructField("air_temperature_k", DoubleType(), nullable=False),
        StructField("process_temperature_k", DoubleType(), nullable=False),
        StructField("rotational_speed_rpm", DoubleType(), nullable=False),
        StructField("torque_nm", DoubleType(), nullable=False),
        StructField("tool_wear_min", DoubleType(), nullable=False),
        StructField("production_rate", DoubleType(), nullable=False),
        StructField("energy_consumption_kwh", DoubleType(), nullable=False),
        StructField("defect_rate", DoubleType(), nullable=False),
        StructField("injected_scenario_active", StringType(), nullable=True),
    ]
)


def load_ai4i(spark: SparkSession, path: str = AI4I_PATH) -> DataFrame:
    """
    Load the real AI4I 2020 Predictive Maintenance dataset (10,000 rows,
    UCI ML Repository, CC BY 4.0) and normalize column names to snake_case.

    Cached before validation (audit F9): `_validate_row_count` is a Spark
    action, and without a cache Spark re-walks the whole read-from-CSV
    lineage every time a downstream `.count()` is called on this frame
    (here, and again in preprocessing.py, and again in run_pipeline.py's
    summary stats) — one of the "15+ actions that re-read the CSV" the
    audit flagged. Caching once, right after the source read, means every
    later action on this same lineage hits memory instead of disk.
    """
    df = spark.read.csv(path, header=True, inferSchema=True).cache()

    rename_map = {
        "UDI": "udi",
        "Product ID": "product_id",
        "Type": "type",
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
        "TWF": "twf",
        "HDF": "hdf",
        "PWF": "pwf",
        "OSF": "osf",
        "RNF": "rnf",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    _validate_row_count(df, "AI4I", expected_min=9000)
    return df


def load_synthetic_operational(spark: SparkSession, path: str = SYNTHETIC_OPERATIONAL_PATH) -> DataFrame:
    """Load the synthetic 5-minute operational readings with an explicit schema.

    Cached before validation (audit F9) — same reasoning as load_ai4i.
    This is the 29 MB / ~311K-row file, so it's the one where repeated
    unmemoized re-reads cost the most wall-clock time.
    """
    df = spark.read.csv(path, header=True,
                        schema=SYNTHETIC_OPERATIONAL_SCHEMA).cache()
    _validate_row_count(df, "synthetic_operational", expected_min=100_000)
    _validate_no_nulls(df, "synthetic_operational", critical_cols=[
        "machine_id", "timestamp", "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    ])
    return df


def load_synthetic_maintenance(spark: SparkSession, path: str = SYNTHETIC_MAINTENANCE_PATH) -> DataFrame:
    """Load the synthetic maintenance event log."""
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("event_date", F.to_date("event_date"))
    df = df.withColumn("resolved", F.col("resolved").cast("boolean"))
    return df


def _validate_row_count(df: DataFrame, name: str, expected_min: int) -> None:
    n = df.count()
    if n < expected_min:
        raise ValueError(
            f"{name}: expected at least {expected_min:,} rows, got {n:,}")
    print(f"[ingestion] {name}: {n:,} rows OK")


def _validate_no_nulls(df: DataFrame, name: str, critical_cols: list[str]) -> None:
    """Audit F9: used to be one `.count()` action per critical column (7
    actions for the synthetic-operational check alone). A single
    aggregation collects all the null counts in one action instead."""
    null_counts = df.agg(
        *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in critical_cols]
    ).first()
    bad_cols = {c: null_counts[c]
                for c in critical_cols if (null_counts[c] or 0) > 0}
    if bad_cols:
        detail = ", ".join(
            f"'{c}' has {n} null values" for c, n in bad_cols.items())
        raise ValueError(f"{name}: {detail}")
    print(f"[ingestion] {name}: no nulls in critical columns OK")


if __name__ == "__main__":
    spark = get_spark_session()
    try:
        ai4i = load_ai4i(spark)
        ai4i.printSchema()
        ai4i.show(5)

        synth = load_synthetic_operational(spark)
        synth.printSchema()
        synth.show(5)

        maint = load_synthetic_maintenance(spark)
        maint.show(5)
    finally:
        spark.stop()
