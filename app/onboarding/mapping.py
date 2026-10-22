"""
The schema a tenant's mapping config must resolve to, and the config's own
validation.

The target is deliberately the EXACT columns spark_jobs/ingestion.py's
SYNTHETIC_OPERATIONAL_SCHEMA / SYNTHETIC_MAINTENANCE_PATH already validate
(see spark_jobs/ingestion.py) — onboarding a tenant means producing a file
in that shape, not a parallel schema the pipeline has to learn. That keeps
the "no arbitrary sensor types without generalising the anomaly rules"
limit explicit rather than silently promised away: a tenant whose fleet
doesn't report these specific AI4I-style readings (air/process temperature,
rotational speed, torque, tool wear) cannot be onboarded through this
layer, full stop — the thresholds in spark_jobs/config.py (e.g.
ROTATIONAL_SPEED_MIN_RPM) are calibrated to exactly these units and would
silently misfire on a different sensor vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------
# Canonical target schema (must match spark_jobs/ingestion.py exactly)
# ---------------------------------------------------------------------

OPERATIONAL_REQUIRED = (
    "machine_id", "timestamp", "production_line", "type",
    "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min",
)
# Present in the demo data and read by downstream feature engineering, but
# not required from a new tenant: filled with a documented default if absent.
OPERATIONAL_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "production_rate": None,
    "energy_consumption_kwh": None,
    "defect_rate": None,
    "injected_scenario_active": None,
}
MAINTENANCE_REQUIRED = (
    "machine_id", "event_date", "event_type", "technician_notes", "resolved",
)

VALID_MACHINE_TYPES = frozenset({"L", "M", "H"})  # AI4I quality variant, see spark_jobs/config.py

# ---------------------------------------------------------------------
# Unit conversions -> the canonical unit each column must arrive in
# ---------------------------------------------------------------------

CELSIUS_TO_KELVIN = 273.15
INCH_LB_TO_NM = 0.112985
LBF_FT_TO_NM = 1.355818
HOUR_TO_MIN = 60.0


def _identity(x: float) -> float:
    return x


UNIT_CONVERTERS: dict[str, dict[str, Callable[[float], float]]] = {
    "air_temperature_k": {
        "K": _identity, "kelvin": _identity,
        "C": lambda c: c + CELSIUS_TO_KELVIN, "celsius": lambda c: c + CELSIUS_TO_KELVIN,
        "F": lambda f: (f - 32) * 5 / 9 + CELSIUS_TO_KELVIN, "fahrenheit": lambda f: (f - 32) * 5 / 9 + CELSIUS_TO_KELVIN,
    },
    "process_temperature_k": {
        "K": _identity, "kelvin": _identity,
        "C": lambda c: c + CELSIUS_TO_KELVIN, "celsius": lambda c: c + CELSIUS_TO_KELVIN,
        "F": lambda f: (f - 32) * 5 / 9 + CELSIUS_TO_KELVIN, "fahrenheit": lambda f: (f - 32) * 5 / 9 + CELSIUS_TO_KELVIN,
    },
    "torque_nm": {
        "Nm": _identity, "N*m": _identity, "newton_meter": _identity,
        "lbf_ft": lambda x: x * LBF_FT_TO_NM, "in_lb": lambda x: x * INCH_LB_TO_NM,
    },
    "tool_wear_min": {
        "min": _identity, "minutes": _identity,
        "hour": lambda x: x * HOUR_TO_MIN, "hours": lambda x: x * HOUR_TO_MIN,
    },
    "rotational_speed_rpm": {"rpm": _identity},
}
# Columns whose unit MUST be declared (no unit-less default) because a
# wrong guess would silently corrupt the anomaly thresholds.
UNIT_REQUIRED_COLUMNS = frozenset(UNIT_CONVERTERS)


@dataclass(frozen=True)
class ColumnMapping:
    """One canonical column's source in the tenant's raw file."""
    source_column: str
    unit: str | None = None  # required for columns in UNIT_REQUIRED_COLUMNS


@dataclass(frozen=True)
class MappingConfig:
    """A validated tenant column mapping for one dataset kind.

    Construct via `MappingConfig.from_dict` / `from_json_file`, never the
    dataclass directly — that's what runs validation.
    """
    dataset_kind: str  # "operational" | "maintenance"
    columns: dict[str, ColumnMapping]
    machine_type_map: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _required_columns(dataset_kind: str) -> tuple[str, ...]:
        if dataset_kind == "operational":
            return OPERATIONAL_REQUIRED
        if dataset_kind == "maintenance":
            return MAINTENANCE_REQUIRED
        raise ValueError(f"Unknown dataset_kind {dataset_kind!r}: must be 'operational' or 'maintenance'.")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MappingConfig":
        """Validate `raw` (as parsed from the tenant's mapping JSON) and
        return a MappingConfig. Raises ValueError, with every problem found
        (not just the first), on anything that would silently corrupt data
        downstream: a missing required column, a declared unit this layer
        doesn't know how to convert, or an unrequested unit on a column that
        needs one.
        """
        errors: list[str] = []

        dataset_kind = raw.get("dataset_kind")
        try:
            required = cls._required_columns(dataset_kind)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        raw_columns = raw.get("columns")
        if not isinstance(raw_columns, dict):
            raise ValueError("mapping config must have a 'columns' object")

        columns: dict[str, ColumnMapping] = {}
        for canonical, spec in raw_columns.items():
            if isinstance(spec, str):
                spec = {"source_column": spec}
            if not isinstance(spec, dict) or "source_column" not in spec:
                errors.append(f"columns.{canonical}: must be a string or "
                              "{'source_column': ..., 'unit': ...}")
                continue
            unit = spec.get("unit")
            if canonical in UNIT_REQUIRED_COLUMNS:
                if not unit:
                    errors.append(f"columns.{canonical}: 'unit' is required "
                                  f"(one of {sorted(UNIT_CONVERTERS[canonical])})")
                elif unit not in UNIT_CONVERTERS[canonical]:
                    errors.append(f"columns.{canonical}: unit {unit!r} not supported "
                                  f"(one of {sorted(UNIT_CONVERTERS[canonical])})")
            columns[canonical] = ColumnMapping(source_column=spec["source_column"], unit=unit)

        missing = [c for c in required if c not in columns]
        if missing:
            errors.append(f"missing required column mapping(s): {sorted(missing)}")

        unknown = [c for c in columns if c not in required and c not in OPERATIONAL_OPTIONAL_DEFAULTS]
        if unknown:
            errors.append(f"unrecognised canonical column(s) (not part of the "
                          f"target schema): {sorted(unknown)}")

        machine_type_map = raw.get("machine_type_map", {})
        if dataset_kind == "operational":
            if not isinstance(machine_type_map, dict) or not machine_type_map:
                errors.append("operational mappings need 'machine_type_map': how the "
                              "tenant's own type/quality labels map onto AI4I's L/M/H")
            else:
                bad_targets = {v for v in machine_type_map.values() if v not in VALID_MACHINE_TYPES}
                if bad_targets:
                    errors.append(f"machine_type_map values must be one of "
                                  f"{sorted(VALID_MACHINE_TYPES)}, got {sorted(bad_targets)}")

        if errors:
            raise ValueError("Invalid mapping config:\n  - " + "\n  - ".join(errors))

        return cls(dataset_kind=dataset_kind, columns=columns, machine_type_map=machine_type_map)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "MappingConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def convert(self, canonical_column: str, value: float) -> float:
        """Apply this mapping's declared unit conversion for one value."""
        mapping = self.columns[canonical_column]
        if canonical_column not in UNIT_CONVERTERS:
            return value
        return UNIT_CONVERTERS[canonical_column][mapping.unit](value)
