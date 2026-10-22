"""Which source columns a Phase 1 sensor mapping needs from the file.

`normalize_sensor_readings` reports a missing machine/timestamp/sensor column
but would raise KeyError for a missing optional one (production_line / type);
callers check this first so that is a clear, permanent "column not found"
instead of a crash.
"""

from __future__ import annotations

from typing import Iterable

from app.onboarding.sensors import SensorMapping


def required_source_columns(mapping: SensorMapping) -> list[str]:
    needed = [mapping.machine_col, mapping.timestamp_col, mapping.line_col, mapping.type_col]
    if mapping.fmt == "wide":
        needed += [s.source for s in mapping.sensors.values()]
    else:
        needed += [mapping.sensor_name_col, mapping.value_col]
    return list(dict.fromkeys(c for c in needed if c))


def missing_source_columns(columns: Iterable[str], mapping: SensorMapping) -> list[str]:
    have = set(columns)
    return [c for c in required_source_columns(mapping) if c not in have]
