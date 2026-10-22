"""
Long-format (tenant-defined sensors) input to the anomaly engine.

A tenant's readings arrive as rows of (machine_id, timestamp, sensor_name,
value). `pivot_to_wide` turns them into the one-column-per-sensor frame the
engine (feature_engineering.engineer_features + anomaly_detection.run)
already works on, using the tenant's registered sensor names as the pivot
columns. After that there is NO tenant- or vocabulary-specific code: the
same z-score / range rules run on whatever columns the registry declares.

Sparse data: a timestamp where a sensor has no reading gets NULL in that
column. Rolling windows count ROWS (distinct timestamps) not wall-clock
time, and NULLs are ignored by avg/stddev and never flag - so irregular or
sparse sampling is tolerated but the "24 h baseline" is only literally 24 h
when the sensor reports at a steady rate (see README Limitations).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType

from app.core.anomaly_rules import AnomalyRules
from spark_jobs.anomaly_detection import run as run_anomaly_detection
from spark_jobs.feature_engineering import engineer_features

LONG_SCHEMA = StructType([
    StructField("machine_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("sensor_name", StringType(), False),
    StructField("value", DoubleType(), False),
])
MACHINES_SCHEMA = StructType([
    StructField("machine_id", StringType(), False),
    StructField("production_line", StringType(), True),
    StructField("type", StringType(), True),
])


def load_long_readings(spark: SparkSession, path: str) -> DataFrame:
    """Reads onboarded readings. Onboarding stores timestamps as naive UTC,
    so this pins the SESSION time zone to UTC: otherwise Spark would read
    them as local time and shift every window/detected_at by the machine's
    offset (observed on an Asia/Calcutta host). The legacy wide-AI4I path
    deliberately does not do this - its output is pinned by tests."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    return spark.read.csv(path, header=True, schema=LONG_SCHEMA)


def load_machines(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, schema=MACHINES_SCHEMA)


def pivot_to_wide(long_df: DataFrame, rules: AnomalyRules, machines_df: DataFrame | None = None) -> DataFrame:
    names = [s.sensor_name for s in rules.active_sensors]
    wide = (long_df.filter(F.col("sensor_name").isin(names))
            .groupBy("machine_id", "timestamp")
            .pivot("sensor_name", names)
            .agg(F.first("value")))
    if machines_df is not None:
        wide = wide.join(machines_df.select("machine_id", "production_line", "type"),
                         on="machine_id", how="left")
        wide = wide.withColumn(
            "production_line", F.coalesce(F.col("production_line"), F.lit("UNASSIGNED")))
    return wide


def detect(long_df: DataFrame, machines_df: DataFrame | None, rules: AnomalyRules) -> tuple[DataFrame, DataFrame]:
    """Long readings -> (sensor_summary, machine_anomalies) DataFrames."""
    wide = pivot_to_wide(long_df, rules, machines_df).cache()
    return run_anomaly_detection(engineer_features(wide, rules), rules)
