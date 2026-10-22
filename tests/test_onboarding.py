"""Onboarding layer (production-readiness fix 10): mapping validation,
unit-converting normalization, and end-to-end CLI. No Spark or Postgres
needed for the mapping/normalize tests below (they read/write plain CSV);
the CLI's tenant-creation step needs DATABASE_URL and skips without it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.onboarding.mapping import MappingConfig
from app.onboarding.normalize import (
    NormalizationError,
    normalize_maintenance,
    normalize_operational,
)
from app.onboarding.paths import maintenance_path_for, operational_path_for, raw_dir_for

OP_MAPPING = {
    "dataset_kind": "operational",
    "columns": {
        "machine_id": "AssetID", "timestamp": "ts", "production_line": "Line", "type": "Grade",
        "air_temperature_k": {"source_column": "AirTempC", "unit": "C"},
        "process_temperature_k": {"source_column": "ProcTempC", "unit": "C"},
        "rotational_speed_rpm": {"source_column": "RPM", "unit": "rpm"},
        "torque_nm": {"source_column": "TorqueLbFt", "unit": "lbf_ft"},
        "tool_wear_min": {"source_column": "WearHrs", "unit": "hours"},
    },
    "machine_type_map": {"Standard": "L", "Premium": "H"},
}
MAINT_MAPPING = {
    "dataset_kind": "maintenance",
    "columns": {"machine_id": "Asset", "event_date": "Date", "event_type": "Kind",
                "technician_notes": "Notes", "resolved": "Status"},
}


# ---------------------------------------------------------------------
# MappingConfig validation
# ---------------------------------------------------------------------

def test_valid_operational_mapping_is_accepted():
    cfg = MappingConfig.from_dict(OP_MAPPING)
    assert cfg.dataset_kind == "operational"
    assert cfg.columns["torque_nm"].unit == "lbf_ft"


def test_missing_required_column_is_rejected():
    bad = {**OP_MAPPING, "columns": {k: v for k,
                                     v in OP_MAPPING["columns"].items() if k != "torque_nm"}}
    with pytest.raises(ValueError, match="torque_nm"):
        MappingConfig.from_dict(bad)


def test_missing_unit_on_a_unit_required_column_is_rejected():
    bad = json.loads(json.dumps(OP_MAPPING))
    bad["columns"]["torque_nm"] = "TorqueLbFt"  # bare string: no unit
    with pytest.raises(ValueError, match="unit.*is required"):
        MappingConfig.from_dict(bad)


def test_unsupported_unit_is_rejected():
    bad = json.loads(json.dumps(OP_MAPPING))
    bad["columns"]["torque_nm"]["unit"] = "furlong_stone"
    with pytest.raises(ValueError, match="not supported"):
        MappingConfig.from_dict(bad)


def test_operational_mapping_requires_machine_type_map():
    bad = {k: v for k, v in OP_MAPPING.items() if k != "machine_type_map"}
    with pytest.raises(ValueError, match="machine_type_map"):
        MappingConfig.from_dict(bad)


def test_machine_type_map_must_target_valid_ai4i_types():
    bad = {**OP_MAPPING, "machine_type_map": {"Standard": "Gold"}}
    with pytest.raises(ValueError, match=r"must be one of \['H', 'L', 'M'\]"):
        MappingConfig.from_dict(bad)


def test_unknown_dataset_kind_is_rejected():
    with pytest.raises(ValueError, match="operational.*maintenance"):
        MappingConfig.from_dict({"dataset_kind": "sensors", "columns": {}})


def test_all_problems_are_reported_together_not_just_the_first():
    bad = json.loads(json.dumps(OP_MAPPING))
    del bad["columns"]["torque_nm"]
    del bad["columns"]["tool_wear_min"]
    bad["machine_type_map"] = {}
    with pytest.raises(ValueError) as exc_info:
        MappingConfig.from_dict(bad)
    msg = str(exc_info.value)
    assert "torque_nm" in msg and "tool_wear_min" in msg and "machine_type_map" in msg


def test_from_json_file_round_trips(tmp_path):
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps(OP_MAPPING))
    cfg = MappingConfig.from_json_file(p)
    assert cfg.dataset_kind == "operational"


# ---------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------

@pytest.mark.parametrize("column,unit,value,expected", [
    ("air_temperature_k", "C", 0.0, 273.15),
    ("air_temperature_k", "K", 300.0, 300.0),
    ("air_temperature_k", "F", 32.0, 273.15),
    ("torque_nm", "Nm", 40.0, 40.0),
    ("torque_nm", "lbf_ft", 1.0, pytest.approx(1.355818)),
    ("tool_wear_min", "hours", 1.0, 60.0),
    ("tool_wear_min", "min", 5.0, 5.0),
])
def test_unit_conversions(column, unit, value, expected):
    mapping = MappingConfig.from_dict({
        "dataset_kind": "operational",
        "columns": {**OP_MAPPING["columns"], column: {"source_column": "x", "unit": unit}},
        "machine_type_map": {"Standard": "L"},
    })
    assert mapping.convert(column, value) == pytest.approx(expected)


# ---------------------------------------------------------------------
# normalize_operational / normalize_maintenance
# ---------------------------------------------------------------------

@pytest.fixture()
def raw_operational_csv(tmp_path):
    df = pd.DataFrame({
        "AssetID": ["X-1", "X-1"], "ts": ["2026-01-01 00:00", "2026-01-01 00:05"],
        "Line": ["L1", "L1"], "Grade": ["Standard", "Premium"],
        "AirTempC": [25.0, 26.0], "ProcTempC": [35.0, 36.0],
        "RPM": [1500, 1520], "TorqueLbFt": [30.0, 31.0], "WearHrs": [0.1, 0.2],
    })
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


def test_normalize_operational_produces_the_canonical_schema(raw_operational_csv):
    out = normalize_operational(
        raw_operational_csv, MappingConfig.from_dict(OP_MAPPING))
    assert list(out.columns) == [
        "machine_id", "timestamp", "production_line", "type",
        "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
        "torque_nm", "tool_wear_min", "production_rate", "energy_consumption_kwh",
        "defect_rate", "injected_scenario_active"]
    assert out["air_temperature_k"].iloc[0] == pytest.approx(298.15)
    assert out["type"].tolist() == ["L", "H"]
    assert out["tool_wear_min"].iloc[0] == pytest.approx(6.0)


def test_normalize_operational_rejects_unmapped_type_value(raw_operational_csv):
    df = pd.read_csv(raw_operational_csv)
    df.loc[0, "Grade"] = "Mystery"
    df.to_csv(raw_operational_csv, index=False)
    with pytest.raises(NormalizationError, match="Mystery"):
        normalize_operational(raw_operational_csv,
                              MappingConfig.from_dict(OP_MAPPING))


def test_normalize_operational_rejects_non_numeric_reading(raw_operational_csv):
    df = pd.read_csv(raw_operational_csv, dtype={"RPM": object})
    df.loc[0, "RPM"] = "fast"
    df.to_csv(raw_operational_csv, index=False)
    with pytest.raises(NormalizationError, match="rotational_speed_rpm"):
        normalize_operational(raw_operational_csv,
                              MappingConfig.from_dict(OP_MAPPING))


def test_normalize_operational_rejects_missing_source_column(raw_operational_csv):
    df = pd.read_csv(raw_operational_csv).drop(columns=["RPM"])
    df.to_csv(raw_operational_csv, index=False)
    with pytest.raises(NormalizationError, match="RPM"):
        normalize_operational(raw_operational_csv,
                              MappingConfig.from_dict(OP_MAPPING))


def test_normalize_maintenance_parses_dates_and_booleans(tmp_path):
    df = pd.DataFrame({"Asset": ["X-1", "X-1"], "Date": ["2026-02-01", "2026-02-15"],
                       "Kind": ["pm", "corrective"], "Notes": ["ok", "fixed belt"],
                       "Status": ["Resolved", "false"]})
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False)
    out = normalize_maintenance(path, MappingConfig.from_dict(MAINT_MAPPING))
    assert list(out.columns) == [
        "machine_id", "event_date", "event_type", "technician_notes", "resolved"]
    assert out["resolved"].tolist() == [True, False]


def test_normalize_maintenance_rejects_unrecognized_resolved_value(tmp_path):
    df = pd.DataFrame({"Asset": ["X-1"], "Date": ["2026-02-01"], "Kind": ["pm"],
                       "Notes": ["ok"], "Status": ["maybe"]})
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False)
    with pytest.raises(NormalizationError, match="resolved"):
        normalize_maintenance(path, MappingConfig.from_dict(MAINT_MAPPING))


def test_wrong_dataset_kind_mapping_is_rejected(raw_operational_csv):
    with pytest.raises(ValueError, match="maintenance.*not.*operational"):
        normalize_operational(raw_operational_csv,
                              MappingConfig.from_dict(MAINT_MAPPING))


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def test_demo_tenant_keeps_the_original_raw_paths():
    assert "tenants" not in raw_dir_for("default")
    assert Path(operational_path_for("default")).resolve(
    ).as_posix().endswith("data/raw/synthetic_operational.csv")


def test_other_tenants_get_their_own_raw_subdirectory():
    assert Path(operational_path_for("acme")).resolve().as_posix().endswith(
        "data/raw/tenants/acme/synthetic_operational.csv")
    assert maintenance_path_for("acme") != maintenance_path_for("globex")


# ---------------------------------------------------------------------
# CLI (needs DATABASE_URL for tenant creation)
# ---------------------------------------------------------------------

needs_db = pytest.mark.skipif(not os.environ.get(
    "DATABASE_URL"), reason="DATABASE_URL not set")


@needs_db
def test_onboard_cli_end_to_end(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, text

    op = pd.DataFrame({
        "AssetID": ["ONB-1"], "ts": ["2026-01-01 00:00"], "Line": ["L1"], "Grade": ["Standard"],
        "AirTempC": [25.0], "ProcTempC": [35.0], "RPM": [1500], "TorqueLbFt": [30.0], "WearHrs": [0.1],
    })
    op_csv = tmp_path / "op.csv"
    op.to_csv(op_csv, index=False)
    op_map = tmp_path / "op_map.json"
    op_map.write_text(json.dumps(OP_MAPPING))

    tenant = "onboard-cli-test"
    engine = create_engine(os.environ["DATABASE_URL"])
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "onboard_tenant.py"),
             "--tenant", tenant, "--name", "Onboard CLI Test",
             "--operational-csv", str(op_csv), "--operational-mapping", str(op_map)],
            capture_output=True, text=True, cwd=root,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "run_pipeline.py --tenant onboard-cli-test" in (
            result.stdout + result.stderr)
        written = Path(operational_path_for(tenant))
        assert written.exists()
        assert pd.read_csv(written)["machine_id"].tolist() == ["ONB-1"]
        with engine.connect() as conn:
            assert conn.execute(text("SELECT 1 FROM tenants WHERE tenant_id = :t"),
                                {"t": tenant}).first() is not None
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM tenants WHERE tenant_id = :t"), {"t": tenant})
        import shutil
        shutil.rmtree(Path(raw_dir_for(tenant)), ignore_errors=True)
