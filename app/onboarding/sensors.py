"""
Onboarding for ARBITRARY sensors (Phase 1, task 3).

`mapping.py` (the original layer) only accepts the five AI4I columns. This
module lets a tenant declare any sensors instead:

    {
      "dataset_kind": "sensor_readings",
      "format": "wide",                       # one column per sensor, or "long"
      "machine_id": "AssetTag", "timestamp": "ReadTime",
      "timestamp_format": "%d/%m/%Y %H:%M",   # optional; default ISO 8601
      "production_line": "Line",              # optional
      "type": "Class", "type_map": {"A": "L"},# optional (AI4I L/M/H only)
      "sensors": {
        "discharge_pressure": {"source": "PressKPa", "unit": "kPa",
                               "target_unit": "bar",       # default = unit
                               "normal_min": 1.5, "normal_max": 6.0,
                               "z_score_threshold": 3.5, "enabled": true}
      },
      "anomaly": {"z_score_threshold": 3.0, "rolling_window_readings": 288}
    }

  * `unit` is the unit the FILE uses; `target_unit` is what is stored (and
    what normal_min/normal_max are written in). Both must be in
    app.onboarding.units and of the same dimension, else the mapping is
    rejected. Values are converted at ingest.
  * long format adds "sensor_name_column" and "value_column"; each sensor's
    "source" is then the value found in the sensor-name column.
  * {"preset": "ai4i"} expands to the AI4I vocabulary (the original demo
    schema) - one valid mapping among others; the legacy `operational`
    mapping in mapping.py still works unchanged.
  * Timestamps are ISO 8601 unless `timestamp_format` is given. Ones with a
    UTC offset are converted to UTC; ones without are taken as UTC.

`normalize_sensor_readings` never raises on bad DATA: every problem becomes
a row in `NormalizationReport` (line number, column, code, message), bad
readings are dropped, good ones are returned. Callers decide whether a
non-empty report is fatal (the CLI says yes unless --allow-partial).
Mapping problems (unknown unit, missing machine_id/timestamp column, ...)
raise `SensorMappingError` listing every problem found.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.anomaly_rules import RuleConfigError, build_rules
from app.onboarding import units as unit_catalog

SENSOR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
VALID_TYPES = frozenset({"L", "M", "H"})
MAX_REPORTED_ERRORS = 1000

# Error codes (stable; tests and API consumers can rely on them).
MISSING_COLUMN = "missing_column"
MISSING_MACHINE_ID = "missing_machine_id"
MISSING_TIMESTAMP = "missing_timestamp"
BAD_TIMESTAMP = "bad_timestamp"
NON_NUMERIC = "non_numeric"
NON_FINITE = "non_finite"
DUPLICATE_ROW = "duplicate_row"
CONFLICTING_DUPLICATE = "conflicting_duplicate"
UNKNOWN_SENSOR = "unknown_sensor"
UNMAPPED_TYPE = "unmapped_type"


class SensorMappingError(ValueError):
    """The mapping config itself is invalid (all problems listed)."""


# ---------------------------------------------------------------------
# Mapping config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SensorSpec:
    name: str
    source: str
    unit: str
    target_unit: str
    normal_min: float | None = None
    normal_max: float | None = None
    z_score_threshold: float | None = None
    enabled: bool = True

    def registry_row(self) -> dict[str, Any]:
        return {
            "sensor_name": self.name, "unit": self.target_unit,
            "normal_min": self.normal_min, "normal_max": self.normal_max,
            "z_score_threshold": self.z_score_threshold, "enabled": self.enabled,
        }


@dataclass(frozen=True)
class SensorMapping:
    fmt: str
    machine_col: str
    timestamp_col: str
    sensors: dict[str, SensorSpec]
    timestamp_format: str | None = None
    line_col: str | None = None
    type_col: str | None = None
    type_map: dict[str, str] = field(default_factory=dict)
    sensor_name_col: str | None = None
    value_col: str | None = None
    anomaly: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SensorMapping":
        raw = _expand_preset(raw)
        errors: list[str] = []
        if raw.get("dataset_kind") != "sensor_readings":
            raise SensorMappingError("dataset_kind must be 'sensor_readings'")

        fmt = raw.get("format", "wide")
        if fmt not in ("wide", "long"):
            errors.append(f"format must be 'wide' or 'long', got {fmt!r}")
        for key in ("machine_id", "timestamp"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                errors.append(f"'{key}' (the source column name) is required")
        if fmt == "long":
            for key in ("sensor_name_column", "value_column"):
                if not isinstance(raw.get(key), str) or not raw[key].strip():
                    errors.append(f"'{key}' is required for long format")

        raw_sensors = raw.get("sensors")
        sensors: dict[str, SensorSpec] = {}
        if not isinstance(raw_sensors, dict) or not raw_sensors:
            errors.append("'sensors' must declare at least one sensor")
            raw_sensors = {}
        sources: dict[str, str] = {}
        for name, spec in raw_sensors.items():
            where = f"sensors.{name}"
            if not SENSOR_NAME_RE.match(str(name)):
                errors.append(f"{where}: name must match {SENSOR_NAME_RE.pattern}")
                continue
            if isinstance(spec, str):
                spec = {"source": spec}
            if not isinstance(spec, dict):
                errors.append(f"{where}: must be an object")
                continue
            source = spec.get("source", name)
            unit = spec.get("unit")
            if not unit:
                errors.append(f"{where}: 'unit' is required (one of {unit_catalog.known_units()})")
                continue
            target = spec.get("target_unit", unit)
            bad = [u for u in (unit, target) if unit_catalog.canonical_unit(u) is None]
            if bad:
                errors.append(f"{where}: unknown unit {bad[0]!r} (known: {unit_catalog.known_units()})")
                continue
            if not unit_catalog.compatible(unit, target):
                errors.append(f"{where}: cannot convert {unit!r} to {target!r} "
                              f"({unit_catalog.dimension_of(unit)} vs {unit_catalog.dimension_of(target)})")
                continue
            lo, hi = spec.get("normal_min"), spec.get("normal_max")
            z = spec.get("z_score_threshold")
            try:
                lo = None if lo is None else float(lo)
                hi = None if hi is None else float(hi)
                z = None if z is None else float(z)
            except (TypeError, ValueError):
                errors.append(f"{where}: normal_min/normal_max/z_score_threshold must be numbers")
                continue
            if lo is not None and hi is not None and lo > hi:
                errors.append(f"{where}: normal_min {lo} > normal_max {hi}")
                continue
            if z is not None and z <= 0:
                errors.append(f"{where}: z_score_threshold must be > 0")
                continue
            if source in sources:
                errors.append(f"{where}: source {source!r} is already used by sensor {sources[source]!r}")
                continue
            sources[source] = name
            sensors[name] = SensorSpec(
                name=name, source=source,
                unit=unit_catalog.canonical_unit(unit), target_unit=unit_catalog.canonical_unit(target),
                normal_min=lo, normal_max=hi, z_score_threshold=z,
                enabled=bool(spec.get("enabled", True)))

        type_map = raw.get("type_map") or {}
        if raw.get("type") and not type_map:
            errors.append("'type' needs a 'type_map' onto the AI4I quality classes L/M/H")
        bad_t = sorted({v for v in type_map.values() if v not in VALID_TYPES})
        if bad_t:
            errors.append(f"type_map values must be one of {sorted(VALID_TYPES)}, got {bad_t}")

        anomaly = raw.get("anomaly") or {}
        try:
            build_rules([s.registry_row() for s in sensors.values()], {"anomaly": anomaly})
        except (RuleConfigError, ValueError, TypeError) as exc:
            errors.append(f"anomaly: {exc}")

        if errors:
            raise SensorMappingError("Invalid sensor mapping:\n  - " + "\n  - ".join(errors))
        return cls(
            fmt=fmt, machine_col=raw["machine_id"], timestamp_col=raw["timestamp"],
            sensors=sensors, timestamp_format=raw.get("timestamp_format"),
            line_col=raw.get("production_line"), type_col=raw.get("type"),
            type_map=dict(type_map), sensor_name_col=raw.get("sensor_name_column"),
            value_col=raw.get("value_column"), anomaly=dict(anomaly))

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SensorMapping":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def registry_rows(self) -> list[dict[str, Any]]:
        return [s.registry_row() for s in self.sensors.values()]


def ai4i_preset_dict() -> dict[str, Any]:
    """The AI4I vocabulary as a sensor mapping (wide format, canonical column
    names). The same eight sensors migration 0005 seeds for the demo tenant."""
    sensors = {
        "torque_nm": {"unit": "Nm"}, "tool_wear_min": {"unit": "min"},
        "production_rate": {"unit": "units/h"}, "defect_rate": {"unit": "fraction"},
        "energy_consumption_kwh": {"unit": "kWh"},
        "air_temperature_k": {"unit": "K", "enabled": False},
        "process_temperature_k": {"unit": "K", "enabled": False},
        "rotational_speed_rpm": {"unit": "rpm", "enabled": False},
    }
    return {
        "dataset_kind": "sensor_readings", "format": "wide",
        "machine_id": "machine_id", "timestamp": "timestamp",
        "production_line": "production_line", "type": "type",
        "type_map": {"L": "L", "M": "M", "H": "H"},
        "sensors": sensors,
        "anomaly": {"rule_packs": {"ai4i": {}}},
    }


def _expand_preset(raw: dict[str, Any]) -> dict[str, Any]:
    preset = raw.get("preset")
    if preset is None:
        return raw
    if preset != "ai4i":
        raise SensorMappingError(f"unknown preset {preset!r} (available: 'ai4i')")
    base = ai4i_preset_dict()
    merged = {**base, **{k: v for k, v in raw.items() if k not in ("preset", "sensors", "anomaly")}}
    merged["sensors"] = {**base["sensors"], **(raw.get("sensors") or {})}
    merged["anomaly"] = {**base["anomaly"], **(raw.get("anomaly") or {})}
    return merged


# ---------------------------------------------------------------------
# Error report
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RowError:
    line: int | None        # 1-based line in the file (header = line 1); None = file-level
    column: str | None
    code: str
    message: str
    value: str | None = None


@dataclass
class NormalizationReport:
    rows_read: int = 0
    readings_accepted: int = 0
    errors: list[RowError] = field(default_factory=list)   # capped at MAX_REPORTED_ERRORS
    counts: dict[str, int] = field(default_factory=dict)   # exact, per code

    @property
    def ok(self) -> bool:
        return not self.counts

    @property
    def total_errors(self) -> int:
        return sum(self.counts.values())

    def add(self, lines, column: str | None, code: str, message: str, values=None) -> None:
        lines = list(lines)
        if not lines:
            return
        self.counts[code] = self.counts.get(code, 0) + len(lines)
        vals = list(values) if values is not None else [None] * len(lines)
        for line, val in zip(lines, vals):
            if len(self.errors) >= MAX_REPORTED_ERRORS:
                break
            self.errors.append(RowError(
                None if line is None else int(line), column, code, message,
                None if val is None else str(val)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read, "readings_accepted": self.readings_accepted,
            "total_errors": self.total_errors, "counts": dict(self.counts),
            "errors": [e.__dict__ for e in self.errors],
            "errors_truncated": self.total_errors > len(self.errors),
        }

    def as_text(self, max_lines: int = 25) -> str:
        head = (f"{self.rows_read} row(s) read, {self.readings_accepted} reading(s) accepted, "
                f"{self.total_errors} problem(s)")
        if self.ok:
            return head
        lines = [head, "  by code: " + ", ".join(f"{c}={n}" for c, n in sorted(self.counts.items()))]
        for e in self.errors[:max_lines]:
            where = "file" if e.line is None else f"line {e.line}"
            col = f" [{e.column}]" if e.column else ""
            val = f" (value {e.value!r})" if e.value is not None else ""
            lines.append(f"  - {where}{col}: {e.code}: {e.message}{val}")
        if self.total_errors > max_lines:
            lines.append(f"  ... {self.total_errors - max_lines} more (full list in the report object)")
        return "\n".join(lines)


@dataclass
class NormalizedSensorData:
    readings: pd.DataFrame     # machine_id, ts, sensor_name, value  (target units)
    machines: pd.DataFrame     # machine_id, production_line, type
    report: NormalizationReport


# ---------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------

def _clean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _parse_numeric(raw: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """-> (values, non_numeric_mask, non_finite_mask) over NON-blank cells."""
    blank = raw == ""
    num = pd.to_numeric(raw.where(~blank), errors="coerce").astype(float)
    non_numeric = ~blank & num.isna()
    non_finite = num.notna() & ~np.isfinite(num)
    return num, non_numeric, non_finite


def _read(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.astype(str).reset_index(drop=True)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_sensor_readings(source, mapping: SensorMapping) -> NormalizedSensorData:
    """Validate and convert a raw export (path or DataFrame) to long-format
    readings in each sensor's target unit. See the module docstring."""
    df = _read(source)
    report = NormalizationReport(rows_read=len(df))
    empty = NormalizedSensorData(
        pd.DataFrame(columns=["machine_id", "ts", "sensor_name", "value"]),
        pd.DataFrame(columns=["machine_id", "production_line", "type"]), report)

    needed = [mapping.machine_col, mapping.timestamp_col]
    if mapping.fmt == "wide":
        needed += [s.source for s in mapping.sensors.values()]
    else:
        needed += [mapping.sensor_name_col, mapping.value_col]
    missing = [c for c in dict.fromkeys(needed) if c not in df.columns]
    for c in missing:
        report.add([None], c, MISSING_COLUMN, f"column {c!r} not found in the file")
    if missing:
        return empty

    lines = pd.Series(np.arange(len(df)) + 2, index=df.index)   # header is line 1
    machine = _clean(df[mapping.machine_col])
    ts_raw = _clean(df[mapping.timestamp_col])
    miss_m, miss_t = machine == "", ts_raw == ""
    parsed = pd.to_datetime(ts_raw.where(~miss_t), errors="coerce", utc=True,
                            format=mapping.timestamp_format or "ISO8601")
    ts = parsed.dt.tz_convert(None)
    bad_t = ts.isna() & ~miss_t
    report.add(lines[miss_m], mapping.machine_col, MISSING_MACHINE_ID, "machine_id is empty")
    report.add(lines[miss_t], mapping.timestamp_col, MISSING_TIMESTAMP, "timestamp is empty")
    fmt_hint = mapping.timestamp_format or "ISO 8601 (e.g. 2026-01-31T14:05:00)"
    report.add(lines[bad_t], mapping.timestamp_col, BAD_TIMESTAMP,
               f"timestamp does not match {fmt_hint}", ts_raw[bad_t])
    row_ok = ~miss_m & ~miss_t & ~bad_t

    pieces: list[pd.DataFrame] = []

    def collect(mask, name_values, raw_values, spec: SensorSpec) -> None:
        num, non_num, non_fin = _parse_numeric(raw_values)
        report.add(lines[mask & non_num], spec.source, NON_NUMERIC,
                   f"{spec.name}: not a number", raw_values[mask & non_num])
        report.add(lines[mask & non_fin], spec.source, NON_FINITE,
                   f"{spec.name}: infinite value", raw_values[mask & non_fin])
        keep = mask & num.notna() & ~non_fin
        if not keep.any():
            return
        values = unit_catalog.convert_many(num[keep], spec.unit, spec.target_unit)
        pieces.append(pd.DataFrame({
            "machine_id": machine[keep], "ts": ts[keep], "sensor_name": spec.name,
            "value": values.astype(float), "line": lines[keep]}))

    if mapping.fmt == "wide":
        for spec in mapping.sensors.values():
            collect(row_ok, spec.name, _clean(df[spec.source]), spec)
    else:
        by_source = {s.source: s for s in mapping.sensors.values()}
        names = _clean(df[mapping.sensor_name_col])
        values_raw = _clean(df[mapping.value_col])
        unknown = row_ok & (names != "") & ~names.isin(by_source)
        report.add(lines[unknown], mapping.sensor_name_col, UNKNOWN_SENSOR,
                   "sensor is not declared in the mapping", names[unknown])
        for source, spec in by_source.items():
            collect(row_ok & (names == source), spec.name, values_raw, spec)

    if not pieces:
        return empty
    long = pd.concat(pieces, ignore_index=True).sort_values("line", kind="stable")

    key = ["machine_id", "ts", "sensor_name"]
    dup = long.duplicated(key, keep="first")
    if dup.any():
        first = long[~dup].set_index(key)["value"]
        dups = long[dup]
        first_vals = first.reindex(pd.MultiIndex.from_frame(dups[key])).to_numpy()
        same = np.isclose(dups["value"].to_numpy(), first_vals, rtol=0, atol=0, equal_nan=False)
        for is_same, code, msg in ((True, DUPLICATE_ROW, "exact duplicate of an earlier reading; ignored"),
                                   (False, CONFLICTING_DUPLICATE,
                                    "same machine/sensor/timestamp as an earlier reading with a "
                                    "DIFFERENT value; the earlier one is kept")):
            sel = dups[same == is_same]
            report.add(sel["line"], None, code, f"{msg}",
                       (sel["sensor_name"] + "@" + sel["ts"].astype(str)))
    long = long[~dup]
    readings = (long[["machine_id", "ts", "sensor_name", "value"]]
                .sort_values(["machine_id", "ts", "sensor_name"]).reset_index(drop=True))
    report.readings_accepted = len(readings)

    # Machine dimension: first non-blank line / mapped type per machine.
    meta = pd.DataFrame({"machine_id": machine[row_ok]})
    meta["production_line"] = (_clean(df[mapping.line_col])[row_ok] if mapping.line_col else "")
    if mapping.type_col:
        raw_type = _clean(df[mapping.type_col])[row_ok]
        mapped = raw_type.map(mapping.type_map)
        unmapped = (raw_type != "") & mapped.isna()
        report.add(lines[row_ok][unmapped], mapping.type_col, UNMAPPED_TYPE,
                   "value has no entry in type_map; machine type left empty", raw_type[unmapped])
        meta["type"] = mapped
    else:
        meta["type"] = None
    meta = meta[meta["machine_id"].isin(readings["machine_id"].unique())]
    machines = (meta.groupby("machine_id", sort=True)
                .agg(production_line=("production_line", lambda s: next((v for v in s if v), "UNASSIGNED")),
                     type=("type", lambda s: next((v for v in s if isinstance(v, str)), None)))
                .reset_index())
    machines["type"] = machines["type"].astype(object).where(machines["type"].notna(), None)
    return NormalizedSensorData(readings, machines, report)


# ---------------------------------------------------------------------
# AI4I demo path: wide operational frame -> long format
# ---------------------------------------------------------------------

AI4I_SENSOR_COLUMNS = tuple(ai4i_preset_dict()["sensors"])


def ai4i_wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Unpivot the demo tenant's wide operational frame (the shape of
    data/raw/synthetic_operational.csv) into sensor_readings rows. Values are
    already in canonical units, so nothing is converted; nulls are dropped."""
    cols = [c for c in AI4I_SENSOR_COLUMNS if c in wide.columns]
    long = wide.melt(id_vars=["machine_id", "timestamp"], value_vars=cols,
                     var_name="sensor_name", value_name="value")
    long = long.dropna(subset=["value"]).rename(columns={"timestamp": "ts"})
    long["ts"] = pd.to_datetime(long["ts"])
    long["value"] = long["value"].astype(float)
    return (long[["machine_id", "ts", "sensor_name", "value"]]
            .sort_values(["machine_id", "ts", "sensor_name"]).reset_index(drop=True))


# ---------------------------------------------------------------------
# On-disk bundle (what the Spark job and the DB loader read)
# ---------------------------------------------------------------------

def write_bundle(directory: str | Path, data: NormalizedSensorData, mapping: SensorMapping) -> dict[str, Path]:
    """Write long-format readings, machines and the registry/anomaly config."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths = {
        "readings": d / "sensor_readings.csv",
        "machines": d / "machines.csv",
        "registry": d / "sensor_registry.json",
    }
    data.readings.rename(columns={"ts": "timestamp"}).to_csv(paths["readings"], index=False)
    data.machines.to_csv(paths["machines"], index=False)
    paths["registry"].write_text(json.dumps(
        {"sensors": mapping.registry_rows(), "anomaly": mapping.anomaly}, indent=2))
    return paths
