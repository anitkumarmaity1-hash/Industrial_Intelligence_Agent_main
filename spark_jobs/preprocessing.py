"""
Preprocessing: cleaning and validation applied after ingestion, before
feature engineering.

Kept deliberately simple — both datasets are already well-formed (AI4I has
no missing values by construction; the synthetic generator has no missing
values either) — but a real operational pipeline WILL see duplicates,
out-of-range sensor glitches, and late/duplicate maintenance entries, so
these checks are not decorative.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Physically-plausible sensor ranges (generous bounds — catches sensor
# glitches / bad data, not meant to be a failure-detection rule).
VALID_RANGES = {
    "air_temperature_k": (250.0, 350.0),
    "process_temperature_k": (250.0, 360.0),
    "rotational_speed_rpm": (0.0, 5000.0),
    "torque_nm": (0.0, 200.0),
    "tool_wear_min": (0.0, 10_000.0),
}


def clean_synthetic_operational(df: DataFrame) -> DataFrame:
    """Dedup, drop out-of-range readings, cast types.

    Audit F9: `df` arrives already cached by ingestion.py, so `n_before`
    doesn't re-read the CSV. The cleaned frame is cached again here before
    it's returned — every stage downstream (feature engineering, anomaly
    detection, and run_pipeline.py's repeated `.count()`/`.show()` calls)
    reuses this same DataFrame, and without a cache each of those actions
    would re-walk the full read -> dedup -> filter chain from scratch.
    """
    n_before = df.count()

    df = df.dropDuplicates(["machine_id", "timestamp"])

    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df = df.filter((F.col(col) >= lo) & (F.col(col) <= hi))

    df = df.withColumn("timestamp", F.col("timestamp").cast("timestamp"))
    df = df.cache()

    n_after = df.count()
    dropped = n_before - n_after
    pct = (dropped / n_before * 100) if n_before else 0
    print(f"[preprocessing] synthetic_operational: dropped {dropped:,} rows ({pct:.2f}%) "
          f"as duplicates/out-of-range; {n_after:,} remain")
    return df


def clean_ai4i(df: DataFrame) -> DataFrame:
    """Cast AI4I numeric columns to double for consistency with the synthetic schema."""
    numeric_cols = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("double"))
    n = df.count()
    print(f"[preprocessing] AI4I: {n:,} rows after type casting (no rows dropped — dataset has no missing values)")
    return df


def clean_maintenance(df: DataFrame) -> DataFrame:
    """Dedup maintenance events (same machine + same event_date + same type)."""
    n_before = df.count()
    df = df.dropDuplicates(["machine_id", "event_date", "event_type"])
    n_after = df.count()
    if n_before != n_after:
        print(f"[preprocessing] maintenance: dropped {n_before - n_after} duplicate events")
    return df


if __name__ == "__main__":
    from spark_jobs.ingestion import (
        get_spark_session,
        load_ai4i,
        load_synthetic_maintenance,
        load_synthetic_operational,
    )

    spark = get_spark_session()
    try:
        synth = clean_synthetic_operational(load_synthetic_operational(spark))
        synth.show(5)

        ai4i = clean_ai4i(load_ai4i(spark))
        ai4i.show(5)

        maint = clean_maintenance(load_synthetic_maintenance(spark))
        maint.show(5)
    finally:
        spark.stop()
