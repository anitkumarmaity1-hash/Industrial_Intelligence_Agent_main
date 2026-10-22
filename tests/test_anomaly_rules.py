"""Per-tenant anomaly rules (Phase 1, task 2): defaults, overrides, validation,
and the guarantee that the demo tenant's DB-loaded rules ARE the old constants."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from dotenv import load_dotenv
from pyspark.sql import functions as F
from sqlalchemy import create_engine

from app.core.anomaly_rules import (
    AI4I_DEFAULT_RULES, Ai4iPack, RuleConfigError, ai4i_registry_rows, build_rules, load_anomaly_rules)
from spark_jobs import config as cfg
from spark_jobs.anomaly_detection import run as run_anomaly_detection
from spark_jobs.feature_engineering import engineer_features
from spark_jobs.ingestion import get_spark_session, load_synthetic_operational
from spark_jobs.preprocessing import clean_synthetic_operational
from tests.test_pipeline import ROOT_PROCESSED  # noqa: F401  (import guard: same suite)

load_dotenv()
DEMO_SETTINGS = {"anomaly": {"rule_packs": {"ai4i": {}}}}


def test_defaults_are_exactly_the_old_constants():
    r = AI4I_DEFAULT_RULES
    assert r.z_score_threshold == cfg.Z_SCORE_ANOMALY_THRESHOLD == 3.0
    assert r.rolling_window_readings == cfg.ROLLING_WINDOW_READINGS == 288
    p = r.ai4i
    assert (p.hdf_temp_diff_k, p.hdf_rotational_speed_rpm) == (8.6, 1380)
    assert (p.pwf_min_power_w, p.pwf_max_power_w) == (3500.0, 9000.0)
    assert p.osf_threshold_nm_min == {"L": 11000, "M": 12000, "H": 13000}
    assert (p.twf_wear_min_minutes, p.twf_wear_max_minutes) == (200, 240)
    assert [s.sensor_name for s in r.active_sensors] == [
        "torque_nm", "tool_wear_min", "production_rate", "defect_rate", "energy_consumption_kwh"]


def test_registry_plus_demo_settings_rebuild_the_default_rules():
    built = build_rules(ai4i_registry_rows(), DEMO_SETTINGS)
    assert built.active_sensors == AI4I_DEFAULT_RULES.active_sensors
    assert built.ai4i == AI4I_DEFAULT_RULES.ai4i
    assert (built.z_score_threshold, built.rolling_window_readings) == (3.0, 288)


def test_overrides_and_tenant_default_threshold():
    rows = [{"sensor_name": "a", "unit": "A", "z_score_threshold": 5.0},
            {"sensor_name": "b", "unit": "A", "enabled": False},
            {"sensor_name": "c", "unit": "A", "normal_min": 1, "normal_max": 2}]
    r = build_rules(
        rows, {"anomaly": {"z_score_threshold": 2.5, "rolling_window_readings": 36}})
    assert r.rolling_window_readings == 36 and r.ai4i is None
    assert [s.sensor_name for s in r.active_sensors] == ["a", "c"]
    a, c = r.active_sensors
    assert r.z_threshold_for(a) == 5.0 and r.z_threshold_for(c) == 2.5


@pytest.mark.parametrize("rows,settings", [
    ([{"sensor_name": "a", "unit": "A", "normal_min": 5, "normal_max": 1}], {}),
    ([{"sensor_name": "a", "unit": "A", "z_score_threshold": 0}], {}),
    ([], {"anomaly": {"z_score_threshold": -1}}),
    ([], {"anomaly": {"rolling_window_readings": "many"}}),
])
def test_nonsense_config_is_rejected_not_silently_ignored(rows, settings):
    with pytest.raises(RuleConfigError):
        build_rules(rows, settings)


def test_ai4i_pack_parameters_can_be_overridden():
    r = build_rules([], {"anomaly": {"rule_packs": {
                    "ai4i": {"pwf_max_power_w": 8000, "osf_threshold_nm_min": {"L": 1}}}}})
    assert r.ai4i.pwf_max_power_w == 8000.0
    assert r.ai4i.osf_threshold_nm_min == {"L": 1, "M": 12000, "H": 13000}


def test_demo_tenant_rules_loaded_from_postgres_equal_the_defaults():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    with create_engine(url).connect() as conn:
        loaded = load_anomaly_rules(conn, "default")
    assert loaded.active_sensors == AI4I_DEFAULT_RULES.active_sensors
    assert loaded.ai4i == AI4I_DEFAULT_RULES.ai4i
    assert loaded.z_score_threshold == 3.0 and loaded.rolling_window_readings == 288


def test_demo_output_from_db_rules_is_identical_to_the_committed_pipeline_output():
    """The 'byte-for-byte' proof: run the engine with the rules the DATABASE
    holds for the demo tenant over the full 311,040-row dataset and compare
    every value of both outputs with the parquet committed in data/processed
    (produced by the pre-Phase-1 code)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    with create_engine(url).connect() as conn:
        rules = load_anomaly_rules(conn, "default")
    spark = get_spark_session("pytest-rules-equivalence")
    try:
        df = engineer_features(clean_synthetic_operational(
            load_synthetic_operational(spark)), rules)
        summary, anomalies = run_anomaly_detection(df, rules)
        got_s = summary.orderBy("machine_id", "window_start").toPandas()
        got_a = anomalies.orderBy(
            "machine_id", "detected_at", "anomaly_score", "triggered_reasons").toPandas()
    finally:
        spark.stop()
    want_s = pd.read_parquet(ROOT_PROCESSED / "sensor_summary.parquet").sort_values(
        ["machine_id", "window_start"]).reset_index(drop=True)
    want_a = pd.read_parquet(ROOT_PROCESSED / "machine_anomalies.parquet").sort_values(
        ["machine_id", "detected_at", "anomaly_score", "triggered_reasons"]).reset_index(drop=True)
    want_s["machine_id"] = want_s["machine_id"].astype(str)
    want_a["machine_id"] = want_a["machine_id"].astype(str)
    assert len(got_s) == len(want_s) == 25920 and len(
        got_a) == len(want_a) == 28855
    got_s, got_a = got_s[want_s.columns].copy(), got_a[want_a.columns].copy()
    _align_timezone_shift(got_s, want_s)
    _align_timezone_shift(got_a, want_a)
    pd.testing.assert_frame_equal(
        got_s, want_s, check_exact=True, check_dtype=False)
    pd.testing.assert_frame_equal(
        got_a, want_a, check_exact=True, check_dtype=False)


def _align_timezone_shift(got, want):
    """toPandas() renders Spark timestamps in the machine's local timezone, and
    the committed parquet was written on a machine with a different one, so
    every timestamp can differ by ONE constant offset (e.g. 5.5 h) on some
    platforms. Accept exactly that - a single uniform shift per column - and
    nothing else: every other value is still compared exactly."""
    for col in want.columns:
        if pd.api.types.is_datetime64_any_dtype(want[col]):
            delta = (got[col] - want[col]).dropna().unique()
            assert len(
                delta) <= 1, f"{col}: timestamps differ by more than a uniform shift"
            if len(delta) == 1 and delta[0] != pd.Timedelta(0):
                got[col] = got[col] - delta[0]
