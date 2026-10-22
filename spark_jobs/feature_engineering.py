"""
Feature engineering: derived features, rolling-window statistics, and
deterministic failure-mode flags computed from the AI4I 2020 dataset's own
published threshold definitions (see spark_jobs/config.py for the sources).

This is where PySpark's window functions do real work: every feature here
is computed per-machine, ordered by time, over a trailing window — the
kind of computation that's awkward to hand-roll in pandas at scale and is
exactly what Spark's Window API is for.
"""

from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from app.core.anomaly_rules import AI4I_DEFAULT_RULES, Ai4iPack, AnomalyRules

RAD_PER_RPM = 2 * math.pi / 60  # rpm -> rad/s conversion factor


def add_derived_features(df: DataFrame) -> DataFrame:
    """
    Add:
      - power_w: torque[Nm] * rotational_speed[rad/s]  (AI4I's PWF metric)
      - temp_diff_k: process_temperature - air_temperature  (AI4I's HDF metric)
      - wear_torque_product: tool_wear[min] * torque[Nm]  (AI4I's OSF metric)
    """
    return (
        df.withColumn(
            "power_w",
            F.col("torque_nm") * (F.col("rotational_speed_rpm") * F.lit(RAD_PER_RPM)),
        )
        .withColumn("temp_diff_k", F.col("process_temperature_k") - F.col("air_temperature_k"))
        .withColumn("wear_torque_product", F.col("tool_wear_min") * F.col("torque_nm"))
    )


def add_ai4i_failure_flags(df: DataFrame, pack: Ai4iPack | None = None) -> DataFrame:
    """
    Deterministic, per-reading boolean flags using AI4I's own published
    thresholds. These are NOT statistical anomalies — they are direct
    physical-threshold checks, kept separate from the statistical anomaly
    layer in anomaly_detection.py so the two signal types stay distinguishable
    in the final evidence ("a known failure-mode threshold was crossed" vs.
    "this machine is behaving unlike its own recent history").

    Phase 1: the thresholds come from `pack` (a tenant's `ai4i` rule pack);
    omitted -> the original spark_jobs/config.py values, so behaviour is
    unchanged for every caller that does not pass one.
    """
    pack = pack or Ai4iPack()
    osf_threshold = F.when(F.col("type") == "L", pack.osf_threshold_nm_min["L"]) \
        .when(F.col("type") == "M", pack.osf_threshold_nm_min["M"]) \
        .otherwise(pack.osf_threshold_nm_min["H"])

    return (
        df.withColumn(
            "flag_twf",
            (F.col("tool_wear_min") >= pack.twf_wear_min_minutes)
            & (F.col("tool_wear_min") <= pack.twf_wear_max_minutes),
        )
        .withColumn(
            "flag_hdf",
            (F.col("temp_diff_k") < pack.hdf_temp_diff_k)
            & (F.col("rotational_speed_rpm") < pack.hdf_rotational_speed_rpm),
        )
        .withColumn(
            "flag_pwf",
            (F.col("power_w") < pack.pwf_min_power_w) | (F.col("power_w") > pack.pwf_max_power_w),
        )
        .withColumn("flag_osf", F.col("wear_torque_product") > osf_threshold)
    )


def add_rolling_features(df: DataFrame, rules: AnomalyRules | None = None) -> DataFrame:
    """
    Trailing rolling mean/std per machine over the ROLLING_WINDOW_READINGS
    readings *preceding* each row (24 hours at 5-min granularity), plus
    reading-over-reading deltas. Window is partitioned by machine and
    ordered by time — this is the core "meaningful PySpark usage" of the
    pipeline.

    Audit F9: the window used to be `rowsBetween(-(N-1), 0)`, which
    includes the current row in its own baseline. A reading is then being
    compared against a mean/std that it itself contributed to, which
    mechanically pulls the z-score toward zero and, with sample std, caps
    the statistic below levels a real deviation could reach. Excluding the
    current row (`rowsBetween(-N, -1)`) makes each reading's z-score a
    comparison against what came before it, not against itself.

    Phase 1: the window length and the set of sensors come from `rules`
    (a tenant's registry); omitted -> the AI4I defaults (same five sensors,
    288 readings). The torque/tool-wear delta columns are an AI4I-only
    extra, added only when those columns exist.
    """
    rules = rules or AI4I_DEFAULT_RULES
    w = (
        Window.partitionBy("machine_id")
        .orderBy("timestamp")
        .rowsBetween(-rules.rolling_window_readings, -1)
    )
    w_prev = Window.partitionBy("machine_id").orderBy("timestamp")

    for sensor in rules.active_sensors:
        col = sensor.sensor_name
        df = df.withColumn(f"{col}_roll_mean", F.avg(col).over(w))
        df = df.withColumn(f"{col}_roll_std", F.stddev(col).over(w))

    if rules.ai4i is not None:
        df = df.withColumn("torque_nm_delta", F.col("torque_nm") - F.lag("torque_nm", 1).over(w_prev))
        df = df.withColumn(
            "tool_wear_min_delta", F.col("tool_wear_min") - F.lag("tool_wear_min", 1).over(w_prev)
        )

    return df


def engineer_features(df: DataFrame, rules: AnomalyRules | None = None) -> DataFrame:
    """Full feature-engineering pipeline, applied in order.

    `rules=None` is the AI4I demo behaviour, unchanged. A rules object
    without the `ai4i` pack (any other tenant) skips the AI4I-specific
    derived columns and flags entirely.
    """
    rules = rules or AI4I_DEFAULT_RULES
    if rules.ai4i is not None:
        df = add_derived_features(df)
        df = add_ai4i_failure_flags(df, rules.ai4i)
    df = add_rolling_features(df, rules)
    return df


if __name__ == "__main__":
    from spark_jobs.ingestion import get_spark_session, load_synthetic_operational
    from spark_jobs.preprocessing import clean_synthetic_operational

    spark = get_spark_session()
    try:
        df = clean_synthetic_operational(load_synthetic_operational(spark))
        df = engineer_features(df)
        df.select(
            "machine_id", "timestamp", "power_w", "temp_diff_k", "wear_torque_product",
            "flag_twf", "flag_hdf", "flag_pwf", "flag_osf",
            "torque_nm_roll_mean", "torque_nm_roll_std",
        ).filter(F.col("machine_id") == "M-01").show(20)

        print("\nFlag counts by machine (top 10):")
        df.groupBy("machine_id").agg(
            F.sum(F.col("flag_twf").cast("int")).alias("twf_count"),
            F.sum(F.col("flag_hdf").cast("int")).alias("hdf_count"),
            F.sum(F.col("flag_pwf").cast("int")).alias("pwf_count"),
            F.sum(F.col("flag_osf").cast("int")).alias("osf_count"),
        ).orderBy(F.desc("osf_count")).show(18)
    finally:
        spark.stop()
