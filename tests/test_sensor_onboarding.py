"""Onboarding for arbitrary sensors (Phase 1, task 3): unit catalog, mapping
validation, and the per-row error report. Pure pandas - no DB or Spark."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from app.onboarding import units
from app.onboarding.sensors import (
    BAD_TIMESTAMP, CONFLICTING_DUPLICATE, DUPLICATE_ROW, MISSING_COLUMN, MISSING_MACHINE_ID,
    MISSING_TIMESTAMP, NON_FINITE, NON_NUMERIC, UNKNOWN_SENSOR, UNMAPPED_TYPE,
    SensorMapping, SensorMappingError, ai4i_wide_to_long, normalize_sensor_readings, write_bundle)


def mapping(**over):
    raw = {
        "dataset_kind": "sensor_readings", "format": "wide",
        "machine_id": "Tag", "timestamp": "T", "production_line": "Line",
        "sensors": {"pressure": {"source": "P", "unit": "kPa", "target_unit": "bar"},
                    "temp": {"source": "Tmp", "unit": "F", "target_unit": "C"}},
    }
    raw.update(over)
    return SensorMapping.from_dict(raw)


def frame(rows):
    return pd.DataFrame(rows, columns=["Tag", "T", "Line", "P", "Tmp"])


# ---- units -----------------------------------------------------------
def test_unit_conversions_are_exact_where_known():
    assert units.convert(100.0, "C", "K") == pytest.approx(373.15)
    assert units.convert(212.0, "F", "C") == pytest.approx(100.0)
    assert units.convert(100.0, "kPa", "bar") == pytest.approx(1.0)
    assert units.convert(14.5037738, "psi", "bar") == pytest.approx(1.0, rel=1e-6)
    assert units.convert(1500.0, "mA", "A") == pytest.approx(1.5)
    assert units.convert(60.0, "rpm", "rps") == pytest.approx(1.0)
    assert units.convert(1.0, "rps", "rpm") == 60.0
    assert units.convert(50.0, "percent", "fraction") == pytest.approx(0.5)
    assert units.convert(1.0, "celsius", "kelvin") == pytest.approx(274.15)   # aliases


def test_round_trip_and_cross_dimension_refusal():
    for a, b in (("C", "F"), ("bar", "psi"), ("A", "mA"), ("h", "s")):
        assert units.convert(units.convert(12.3, a, b), b, a) == pytest.approx(12.3)
    with pytest.raises(ValueError, match="cannot convert"):
        units.convert(1.0, "bar", "C")
    with pytest.raises(ValueError, match="unknown unit"):
        units.convert(1.0, "furlongs", "C")


# ---- mapping validation ---------------------------------------------
def test_unknown_unit_is_rejected_listing_every_problem():
    with pytest.raises(SensorMappingError) as e:
        SensorMapping.from_dict({
            "dataset_kind": "sensor_readings", "machine_id": "Tag", "timestamp": "T",
            "sensors": {"a": {"unit": "furlongs"}, "b": {"unit": "bar", "target_unit": "C"},
                        "Bad Name": {"unit": "A"}, "c": {"unit": "A", "normal_min": 9, "normal_max": 1}}})
    msg = str(e.value)
    assert "unknown unit 'furlongs'" in msg and "cannot convert 'bar' to 'C'" in msg
    assert "Bad Name" in msg and "normal_min 9.0 > normal_max 1.0" in msg


@pytest.mark.parametrize("drop", ["machine_id", "timestamp"])
def test_mapping_without_machine_id_or_timestamp_is_rejected(drop):
    raw = {"dataset_kind": "sensor_readings", "machine_id": "Tag", "timestamp": "T",
           "sensors": {"a": {"unit": "A"}}}
    del raw[drop]
    with pytest.raises(SensorMappingError, match=drop):
        SensorMapping.from_dict(raw)


def test_ai4i_is_one_valid_preset_and_overridable():
    m = SensorMapping.from_dict({"dataset_kind": "sensor_readings", "preset": "ai4i"})
    assert set(m.sensors) >= {"torque_nm", "tool_wear_min", "air_temperature_k", "rotational_speed_rpm"}
    assert m.anomaly == {"rule_packs": {"ai4i": {}}}
    renamed = SensorMapping.from_dict({"dataset_kind": "sensor_readings", "preset": "ai4i",
                                       "machine_id": "AssetID",
                                       "sensors": {"torque_nm": {"source": "TorqueNm", "unit": "Nm"}}})
    assert renamed.machine_col == "AssetID" and renamed.sensors["torque_nm"].source == "TorqueNm"
    with pytest.raises(SensorMappingError, match="unknown preset"):
        SensorMapping.from_dict({"dataset_kind": "sensor_readings", "preset": "nope"})


# ---- row-level report -------------------------------------------------
def test_clean_file_converts_units_and_builds_machines():
    out = normalize_sensor_readings(frame([
        ["P-1", "2026-01-01T00:00:00", "L1", "200", "212"],
        ["P-1", "2026-01-01T00:05:00", "L1", "", "32"],        # blank cell = no reading, not an error
    ]), mapping())
    assert out.report.ok and out.report.readings_accepted == 3
    r = out.readings.set_index(["ts", "sensor_name"])["value"]
    assert r[(pd.Timestamp("2026-01-01 00:00"), "pressure")] == pytest.approx(2.0)
    assert r[(pd.Timestamp("2026-01-01 00:00"), "temp")] == pytest.approx(100.0)
    assert r[(pd.Timestamp("2026-01-01 00:05"), "temp")] == pytest.approx(0.0, abs=1e-9)
    assert out.machines.to_dict("records") == [{"machine_id": "P-1", "production_line": "L1", "type": None}]


def test_every_kind_of_bad_row_is_reported_with_line_and_column():
    out = normalize_sensor_readings(frame([
        ["P-1", "2026-01-01T00:00:00", "L1", "100", "50"],      # line 2  good
        ["",    "2026-01-01T00:05:00", "L1", "100", "50"],      # line 3  missing machine
        ["P-1", "",                    "L1", "100", "50"],      # line 4  missing timestamp
        ["P-1", "yesterday",           "L1", "100", "50"],      # line 5  bad timestamp
        ["P-1", "2026-01-01T00:10:00", "L1", "ERR", "50"],      # line 6  non numeric
        ["P-1", "2026-01-01T00:15:00", "L1", "inf", "50"],      # line 7  non finite
        ["P-1", "2026-01-01T00:20:00", "L1", "nan", "50"],      # line 8  'nan' is not a reading
        ["P-1", "2026-01-01T00:00:00", "L1", "100", "50"],      # line 9  exact duplicate of line 2
        ["P-1", "2026-01-01T00:00:00", "L1", "999", "50"],      # line 10 conflicting duplicate
    ]), mapping())
    by_code = {}
    for e in out.report.errors:
        by_code.setdefault(e.code, []).append(e)
    assert [e.line for e in by_code[MISSING_MACHINE_ID]] == [3]
    assert [e.line for e in by_code[MISSING_TIMESTAMP]] == [4]
    assert [e.line for e in by_code[BAD_TIMESTAMP]] == [5] and by_code[BAD_TIMESTAMP][0].value == "yesterday"
    assert sorted(e.line for e in by_code[NON_NUMERIC]) == [6, 8]
    assert all(e.column == "P" for e in by_code[NON_NUMERIC])
    assert [e.line for e in by_code[NON_FINITE]] == [7]
    assert [e.line for e in by_code[DUPLICATE_ROW]] == [9, 9, 10]   # line 9 repeats both sensors; line 10 repeats temp exactly
    assert [e.line for e in by_code[CONFLICTING_DUPLICATE]] == [10]
    # first occurrence wins; the conflicting 999 never lands
    assert 999 not in out.readings["value"].round().tolist()
    assert "line 6" in out.report.as_text() and out.report.total_errors == sum(out.report.counts.values())
    # the good rows still come through (lines 2, and temp readings of lines 6-8)
    assert out.report.readings_accepted == len(out.readings) >= 4


def test_missing_source_columns_are_a_file_level_error():
    out = normalize_sensor_readings(pd.DataFrame({"Tag": ["x"], "T": ["2026-01-01"]}), mapping())
    assert out.report.counts == {MISSING_COLUMN: 2} and out.readings.empty
    assert all(e.line is None for e in out.report.errors)


def test_custom_timestamp_format_offsets_and_unmapped_type():
    m = mapping(timestamp_format="%d/%m/%Y %H:%M", type="Class", type_map={"A": "L"})
    df = pd.DataFrame({"Tag": ["P-1", "P-1"], "T": ["31/01/2026 14:05", "01/02/2026 00:00"],
                       "Line": ["L1", "L1"], "P": ["100", "100"], "Tmp": ["50", "50"], "Class": ["A", "Z"]})
    out = normalize_sensor_readings(df, m)
    assert sorted(out.readings["ts"].unique()) == [pd.Timestamp("2026-01-31 14:05"), pd.Timestamp("2026-02-01")]
    assert out.report.counts == {UNMAPPED_TYPE: 1}
    assert out.machines.loc[0, "type"] == "L"
    iso = normalize_sensor_readings(frame([["P-1", "2026-01-01T05:30:00+05:30", "L1", "100", "50"]]), mapping())
    assert iso.readings["ts"].iloc[0] == pd.Timestamp("2026-01-01 00:00")      # converted to UTC


def test_long_format_with_unknown_sensor_name():
    m = SensorMapping.from_dict({
        "dataset_kind": "sensor_readings", "format": "long", "machine_id": "m", "timestamp": "t",
        "sensor_name_column": "s", "value_column": "v",
        "sensors": {"flow": {"source": "FLOW_LPM", "unit": "L_min", "target_unit": "m3_h"}}})
    df = pd.DataFrame({"m": ["a", "a"], "t": ["2026-01-01T00:00:00"] * 2,
                       "s": ["FLOW_LPM", "MYSTERY"], "v": ["100", "1"]})
    out = normalize_sensor_readings(df, m)
    assert out.readings["value"].tolist() == [pytest.approx(6.0)]
    assert [e.code for e in out.report.errors] == [UNKNOWN_SENSOR]


def test_report_error_list_is_capped_but_counts_are_exact():
    n = 1500
    df = pd.DataFrame({"Tag": ["p"] * n, "T": [f"2026-01-01T00:00:{i % 60:02d}" for i in range(n)],
                       "Line": "L", "P": ["bad"] * n, "Tmp": [""] * n})
    rep = normalize_sensor_readings(df, mapping()).report
    assert rep.counts[NON_NUMERIC] == n and len(rep.errors) == 1000
    assert rep.to_dict()["errors_truncated"] is True


def test_bundle_round_trip_and_ai4i_unpivot(tmp_path):
    m = mapping(anomaly={"z_score_threshold": 4.0})
    data = normalize_sensor_readings(frame([["P-1", "2026-01-01T00:00:00", "L1", "100", "50"]]), m)
    paths = write_bundle(tmp_path, data, m)
    reg = json.loads(paths["registry"].read_text())
    assert {s["sensor_name"]: s["unit"] for s in reg["sensors"]} == {"pressure": "bar", "temp": "C"}
    assert reg["anomaly"] == {"z_score_threshold": 4.0}
    wide = pd.DataFrame({"machine_id": ["M-01"], "timestamp": ["2026-01-01 00:00:00"],
                         "torque_nm": [40.0], "tool_wear_min": [3.0], "air_temperature_k": [300.0],
                         "process_temperature_k": [310.0], "rotational_speed_rpm": [1500.0],
                         "production_rate": [100.0], "energy_consumption_kwh": [12.0], "defect_rate": [0.02]})
    long = ai4i_wide_to_long(wide)
    assert len(long) == 8 and set(long["sensor_name"]) == set(m.__class__.from_dict(
        {"dataset_kind": "sensor_readings", "preset": "ai4i"}).sensors)
