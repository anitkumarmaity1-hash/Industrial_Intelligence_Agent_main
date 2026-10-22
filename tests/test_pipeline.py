"""
Phase 1 pipeline tests.

These aren't generic unit tests — they check the pipeline against KNOWN
GROUND TRUTH: we know exactly which machines have injected failure
scenarios (spark_jobs/config.py), so the anomaly detector's output can be
checked against that ground truth directly. This is the most valuable
kind of test for an anomaly-detection pipeline: does it actually find the
things we know are there, and does it stay quiet on the things we know
are healthy?

Audit F9 caveat, stated plainly rather than left implicit: the injected
scenarios in spark_jobs/config.py and the AI4I threshold flags in
spark_jobs/feature_engineering.py are built from the same published AI4I
formulas (TWF/HDF/PWF/OSF). So these tests confirm the pipeline correctly
detects the failure modes it was told about and shaped around — they
validate the plumbing (ingestion -> features -> flags -> aggregation is
wired correctly end to end), not the detector's ability to generalize to
a failure pattern it wasn't designed against. That's a legitimate thing
for a portfolio pipeline test to do; it just isn't the same claim as
"detects unknown anomalies," and the two shouldn't be conflated when
describing this project.

Run:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pytest
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spark_jobs import config as cfg  # noqa: E402
from spark_jobs.anomaly_detection import run as run_anomaly_detection  # noqa: E402
from spark_jobs.feature_engineering import engineer_features  # noqa: E402
from spark_jobs.ingestion import (  # noqa: E402
    get_spark_session,
    load_ai4i,
    load_synthetic_maintenance,
    load_synthetic_operational,
)
from spark_jobs.preprocessing import clean_maintenance, clean_synthetic_operational

from pathlib import Path  # noqa: E402

ROOT_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SCENARIO_MACHINE_IDS = {s.machine_id for s in cfg.INJECTED_SCENARIOS}
HEALTHY_MACHINE_IDS = {f"M-{i:02d}" for i in range(1, cfg.N_MACHINES + 1)} - SCENARIO_MACHINE_IDS
SUSTAINED_SCENARIO_MACHINES = SCENARIO_MACHINE_IDS - {"M-05"}  # M-05/RNF is deliberately a one-off, not sustained


@pytest.fixture(scope="module")
def spark():
    s = get_spark_session("pytest-industrial-intelligence-agent")
    yield s
    s.stop()


@pytest.fixture(scope="module")
def summary_and_anomalies(spark):
    synth = clean_synthetic_operational(load_synthetic_operational(spark))
    featured = engineer_features(synth)
    summary, anomalies = run_anomaly_detection(featured)
    summary.cache()
    anomalies.cache()
    return summary, anomalies


def test_ai4i_has_expected_row_count(spark):
    df = load_ai4i(spark)
    assert df.count() == 10_000, "AI4I dataset should have exactly 10,000 rows"
    assert "machine_failure" in df.columns


def test_ai4i_pwf_threshold_matches_real_labels(spark):
    """
    Sanity check the AI4I-derived PWF threshold logic (used in
    feature_engineering.py) against the REAL dataset's own PWF labels —
    this is what caught the power-model bug during development.
    """
    import math

    df = load_ai4i(spark)
    df = df.withColumn(
        "power_w", F.col("torque_nm") * F.col("rotational_speed_rpm") * (2 * math.pi / 60)
    )
    predicted_pwf = df.filter(
        (F.col("power_w") < cfg.PWF_MIN_POWER_W) | (F.col("power_w") > cfg.PWF_MAX_POWER_W)
    ).count()
    actual_pwf = df.filter(F.col("pwf") == 1).count()
    assert predicted_pwf == actual_pwf == 95, (
        f"Threshold-derived PWF count ({predicted_pwf}) should match the real dataset's "
        f"labeled PWF=1 count ({actual_pwf})"
    )


def test_synthetic_data_has_no_nulls_in_critical_columns(spark):
    df = load_synthetic_operational(spark)
    critical = ["machine_id", "timestamp", "torque_nm", "tool_wear_min", "rotational_speed_rpm"]
    for col in critical:
        assert df.filter(F.col(col).isNull()).count() == 0


def test_synthetic_data_row_count(spark):
    df = load_synthetic_operational(spark)
    expected = cfg.N_MACHINES * cfg.SIMULATION_DAYS * 24 * 60 // cfg.READING_INTERVAL_MINUTES
    assert df.count() == expected


def test_tool_wear_is_bounded_for_healthy_machines(spark):
    """
    Regression test for the wear-accumulation bug found during development
    (wear reached ~50,000 minutes before routine-replacement logic was
    added). Healthy machines must stay near the routine-replacement ceiling.
    """
    df = load_synthetic_operational(spark)
    healthy = df.filter(F.col("machine_id").isin(list(HEALTHY_MACHINE_IDS)))
    max_wear = healthy.agg(F.max("tool_wear_min")).collect()[0][0]
    assert max_wear < cfg.ROUTINE_TOOL_REPLACEMENT_THRESHOLD_MIN + 5, (
        f"Healthy machine tool wear should stay near the routine-replacement threshold "
        f"({cfg.ROUTINE_TOOL_REPLACEMENT_THRESHOLD_MIN} min); got max {max_wear}"
    )


def test_feature_engineering_adds_expected_columns(spark):
    df = clean_synthetic_operational(load_synthetic_operational(spark))
    df = engineer_features(df)
    expected = {"power_w", "temp_diff_k", "wear_torque_product", "flag_twf", "flag_hdf", "flag_pwf", "flag_osf"}
    assert expected.issubset(set(df.columns))


def test_sustained_scenario_machines_rank_above_all_healthy_machines(summary_and_anomalies):
    """
    Core ground-truth test: every machine with a SUSTAINED injected
    degradation (TWF, HDF, PWF, OSF, TWF_RECURRING) must show strictly
    more anomalous readings than every healthy machine. This is the
    detector doing its actual job.
    """
    summary, _ = summary_and_anomalies
    totals = (
        summary.groupBy("machine_id")
        .agg(F.sum("anomalous_reading_count").alias("total"))
        .collect()
    )
    totals_by_machine = {r["machine_id"]: r["total"] for r in totals}

    healthy_max = max(totals_by_machine[m] for m in HEALTHY_MACHINE_IDS)
    for machine_id in SUSTAINED_SCENARIO_MACHINES:
        assert totals_by_machine[machine_id] > healthy_max, (
            f"{machine_id} (injected scenario) should have more anomalous readings "
            f"than the healthiest-looking healthy machine ({healthy_max}), "
            f"got {totals_by_machine[machine_id]}"
        )


def test_rnf_blip_is_detected_near_its_onset_even_if_not_sustained(summary_and_anomalies):
    """
    M-05's RNF scenario is a deliberate one-off, unexplained blip with no
    precursor — by design it should NOT be distinguishable in aggregate
    anomaly counts (see the sustained-scenario test above, which excludes
    it). But the single injected event itself must still show up as a
    detected anomaly at the time it happened.
    """
    _, anomalies = summary_and_anomalies
    rnf_cfg = next(s for s in cfg.INJECTED_SCENARIOS if s.machine_id == "M-05")
    onset_ts = (datetime(2026, 1, 1) + timedelta(days=rnf_cfg.onset_day)).strftime("%Y-%m-%d")

    hits = anomalies.filter(
        (F.col("machine_id") == "M-05") & (F.col("detected_at").cast("string").startswith(onset_ts))
    ).count()
    assert hits > 0, "The RNF blip should be detected as an anomaly on the day it was injected"


def test_maintenance_records_reference_valid_scenario_machines(spark):
    maint = clean_maintenance(load_synthetic_maintenance(spark))
    machine_ids = {r["machine_id"] for r in maint.select("machine_id").distinct().collect()}
    assert machine_ids.issubset(SCENARIO_MACHINE_IDS)
    # the unresolved OSF machine should have a resolved=false record
    unresolved = maint.filter((F.col("machine_id") == "M-04") & (F.col("resolved") == False)).count()  # noqa: E712
    assert unresolved == 1
