"""
Anomaly detection: combines two independent signal types into a
deterministic, explainable risk score — no ML model, no LLM involved here
(per the project's "deterministic computation outside the LLM" principle).

  1. AI4I threshold flags (flag_twf/hdf/pwf/osf from feature_engineering.py)
     — direct physical-threshold checks grounded in the real dataset's own
     published failure-mode definitions.
  2. Statistical z-score anomalies — is this specific reading unlike this
     specific machine's own recent rolling history? Catches drift that
     threshold checks miss (e.g. business-metric degradation that hasn't
     yet crossed a hard physical limit).

Output: two Parquet datasets, matching the Phase-0 PostgreSQL schema:
  - data/processed/sensor_summary.parquet   (hourly, one row per machine-hour)
  - data/processed/machine_anomalies.parquet (one row per detected anomaly)
"""

from __future__ import annotations

import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.core.anomaly_rules import AI4I_DEFAULT_RULES, AnomalyRules

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SENSOR_SUMMARY_PATH = os.path.join(DATA_PROCESSED_DIR, "sensor_summary.parquet")
MACHINE_ANOMALIES_PATH = os.path.join(DATA_PROCESSED_DIR, "machine_anomalies.parquet")

# Columns the AI4I demo's sensor_summary carries as per-sensor hourly
# averages. A tenant whose frame lacks them simply gets no such columns
# (they load as NULL).
_AVG_COLUMNS = [
    ("air_temperature_k", "avg_air_temp_k"),
    ("process_temperature_k", "avg_process_temp_k"),
    ("rotational_speed_rpm", "avg_rotational_speed_rpm"),
    ("torque_nm", "avg_torque_nm"),
    ("production_rate", "avg_production_rate"),
    ("energy_consumption_kwh", "avg_energy_consumption_kwh"),
    ("defect_rate", "avg_defect_rate"),
]

AI4I_FLAG_DESCRIPTIONS = {
    "flag_twf": "Tool wear in AI4I TWF failure window (200-240 min)",
    "flag_hdf": "Heat dissipation failure pattern: low temp differential + low rotational speed",
    "flag_pwf": "Power outside safe operating band (torque x rotational speed)",
    "flag_osf": "Overstrain: tool_wear x torque exceeds type-specific threshold",
}


def add_zscore_flags(df: DataFrame, rules: AnomalyRules | None = None) -> DataFrame:
    """
    Flag readings where a metric deviates more than its z-score threshold
    (per-sensor override, else the tenant default; 3.0 unless configured)
    standard deviations from that machine's own trailing rolling mean.
    Guards against divide-by-zero on the (rare) zero-variance window.

    Phase 1: also flags readings outside a sensor's registered
    normal_min/normal_max (`flag_range_<sensor>`), when set. A null reading
    (sparse sensor) never flags: flags are coalesced to False so one missing
    value cannot null-out the row's anomaly_score.
    """
    rules = rules or AI4I_DEFAULT_RULES
    for sensor in rules.active_sensors:
        metric = sensor.sensor_name
        mean_col = f"{metric}_roll_mean"
        std_col = f"{metric}_roll_std"
        z_col = f"{metric}_zscore"
        flag_col = f"flag_stat_{metric}"
        threshold = rules.z_threshold_for(sensor)

        if sensor.normal_min is not None or sensor.normal_max is not None:
            out_of_range = F.lit(False)
            if sensor.normal_min is not None:
                out_of_range = out_of_range | (F.col(metric) < sensor.normal_min)
            if sensor.normal_max is not None:
                out_of_range = out_of_range | (F.col(metric) > sensor.normal_max)
            df = df.withColumn(f"flag_range_{metric}", F.coalesce(out_of_range, F.lit(False)))

        df = df.withColumn(
            z_col,
            F.when(
                (F.col(std_col).isNotNull()) & (F.col(std_col) > 0.01),
                (F.col(metric) - F.col(mean_col)) / F.col(std_col),
            ).otherwise(F.lit(0.0)),
        )
        df = df.withColumn(
            flag_col, F.coalesce(F.abs(F.col(z_col)) > F.lit(threshold), F.lit(False)))

    return df


def _flag_columns(df: DataFrame, rules: AnomalyRules) -> list[str]:
    """Every flag column present, in the stable order used for the score and
    for triggered_reasons: AI4I pack flags, then z-score flags, then range."""
    cols = [c for c in AI4I_FLAG_DESCRIPTIONS if c in df.columns]
    cols += [f"flag_stat_{s.sensor_name}" for s in rules.active_sensors]
    cols += [f"flag_range_{s.sensor_name}" for s in rules.active_sensors
             if f"flag_range_{s.sensor_name}" in df.columns]
    return cols


def add_risk_score(df: DataFrame, rules: AnomalyRules | None = None) -> DataFrame:
    """
    Deterministic risk score = count of all triggered flags (AI4I threshold
    flags + statistical flags). Severity is a simple, explainable mapping
    from that count — never assigned by an LLM.
    """
    rules = rules or AI4I_DEFAULT_RULES
    flag_cols = _flag_columns(df, rules)
    score_expr = sum(F.col(c).cast("int") for c in flag_cols)
    df = df.withColumn("anomaly_score", score_expr)

    df = df.withColumn(
        "severity",
        F.when(F.col("anomaly_score") >= 3, "HIGH")
        .when(F.col("anomaly_score") >= 1, "MEDIUM")
        .otherwise("LOW"),
    )
    return df


def build_sensor_summary(df: DataFrame, rules: AnomalyRules | None = None) -> DataFrame:
    """
    Hourly aggregation per machine — matches the sensor_summary table from
    the Phase-0 PostgreSQL schema. This is the pre-computed, query-ready
    output the FastAPI/agent layer will read; the LLM will only ever see
    rows from THIS table, never the raw 5-minute readings.
    """
    hourly = df.withColumn("window_start", F.date_trunc("hour", "timestamp"))

    # Phase 1: per-sensor averages exist only for the AI4I vocabulary; a
    # tenant with other sensors gets the generic count/score columns only
    # (per-sensor aggregates for arbitrary sensors are a later phase).
    aggs = []
    for src, alias in _AVG_COLUMNS[:4]:
        if src in df.columns:
            aggs.append(F.avg(src).alias(alias))
    if "tool_wear_min" in df.columns:
        aggs.append(F.max("tool_wear_min").alias("tool_wear_min"))
    for src, alias in _AVG_COLUMNS[4:]:
        if src in df.columns:
            aggs.append(F.avg(src).alias(alias))
    aggs += [
        F.max("anomaly_score").alias("max_anomaly_score"),
        F.sum((F.col("anomaly_score") > 0).cast("int")).alias("anomalous_reading_count"),
        F.count("*").alias("reading_count"),
    ]
    keys = [k for k in ("machine_id", "production_line", "type") if k in df.columns]
    summary = hourly.groupBy(*keys, "window_start").agg(*aggs)
    summary = summary.withColumn("window_end", F.col("window_start") + F.expr("INTERVAL 1 HOUR"))
    summary = summary.withColumn(
        "health_status",
        F.when(F.col("max_anomaly_score") >= 3, "AT_RISK")
        .when(F.col("max_anomaly_score") >= 1, "WATCH")
        .otherwise("HEALTHY"),
    )
    return summary


def build_machine_anomalies(df: DataFrame, rules: AnomalyRules | None = None) -> DataFrame:
    """
    One row per detected anomaly, matching machine_anomalies from the
    Phase-0 schema. Only anomaly_score > 0 readings are kept — this is a
    much smaller table than sensor_summary and is what the agent's
    get_machine_anomalies() tool will query directly.
    """
    rules = rules or AI4I_DEFAULT_RULES
    anomalies = df.filter(F.col("anomaly_score") > 0)

    flag_descriptions = {k: v for k, v in AI4I_FLAG_DESCRIPTIONS.items() if k in df.columns}
    for sensor in rules.active_sensors:
        flag_descriptions[f"flag_stat_{sensor.sensor_name}"] = (
            f"{sensor.display.capitalize() if sensor.label is None else sensor.display} "
            "statistically unlike this machine's recent history")
    for sensor in rules.active_sensors:
        if f"flag_range_{sensor.sensor_name}" in df.columns:
            bounds = f"[{sensor.normal_min}, {sensor.normal_max}] {sensor.unit}".strip()
            flag_descriptions[f"flag_range_{sensor.sensor_name}"] = (
                f"{sensor.display.capitalize() if sensor.label is None else sensor.display} "
                f"outside its normal range {bounds}")

    exprs = [
        F.when(F.col(flag), F.lit(desc)).alias(flag)
        for flag, desc in flag_descriptions.items()
    ]
    with_reasons = anomalies.select(
        "machine_id", "timestamp", "anomaly_score", "severity", *exprs
    )

    reason_cols = list(flag_descriptions.keys())
    with_reasons = with_reasons.withColumn(
        "triggered_reasons",
        F.array_join(F.array_compact(F.array(*[F.col(c) for c in reason_cols])), "; "),
    )

    return with_reasons.select(
        "machine_id",
        F.col("timestamp").alias("detected_at"),
        "anomaly_score",
        "severity",
        "triggered_reasons",
    )


def run(df: DataFrame, rules: AnomalyRules | None = None) -> tuple[DataFrame, DataFrame]:
    """`rules=None` is the AI4I demo behaviour, unchanged; pass a tenant's
    AnomalyRules (app.core.anomaly_rules.load_anomaly_rules) for anyone else."""
    df = add_zscore_flags(df, rules)
    df = add_risk_score(df, rules)
    sensor_summary = build_sensor_summary(df, rules)
    machine_anomalies = build_machine_anomalies(df, rules)
    return sensor_summary, machine_anomalies


if __name__ == "__main__":
    from spark_jobs.feature_engineering import engineer_features
    from spark_jobs.ingestion import get_spark_session, load_synthetic_operational
    from spark_jobs.preprocessing import clean_synthetic_operational

    spark = get_spark_session()
    try:
        df = clean_synthetic_operational(load_synthetic_operational(spark))
        df = engineer_features(df)
        summary, anomalies = run(df)
        # Audit F9: two .show() calls plus two .write() calls below all hit
        # this same lineage — cache so only the first one recomputes it.
        summary = summary.cache()
        anomalies = anomalies.cache()

        print("\n=== Health status by machine (most recent window) ===")
        summary.groupBy("machine_id").agg(
            F.max("max_anomaly_score").alias("peak_anomaly_score"),
            F.sum("anomalous_reading_count").alias("total_anomalous_readings"),
        ).orderBy(F.desc("peak_anomaly_score")).show(18)

        print("\n=== Sample anomaly rows for M-01 (TWF scenario) ===")
        anomalies.filter(F.col("machine_id") == "M-01").orderBy("detected_at").show(10, truncate=60)

        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        summary.write.mode("overwrite").partitionBy("machine_id").parquet(SENSOR_SUMMARY_PATH)
        anomalies.write.mode("overwrite").partitionBy("machine_id").parquet(MACHINE_ANOMALIES_PATH)
        print(f"\nWrote sensor_summary -> {SENSOR_SUMMARY_PATH}")
        print(f"Wrote machine_anomalies -> {MACHINE_ANOMALIES_PATH}")
    finally:
        spark.stop()
