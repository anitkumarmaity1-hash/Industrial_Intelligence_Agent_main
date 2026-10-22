"""
Unit catalog for tenant sensors (Phase 1, task 3).

Each unit belongs to one *dimension* and is defined by an affine map to that
dimension's base unit:  base = value * scale + offset.  Converting between
two units of the same dimension is therefore exact and symmetric; converting
across dimensions (say a pressure column declared as a temperature) is an
error, not a guess.

This is a deliberately small, explicit table - the units we can convert and
test. An unlisted unit is rejected at mapping-validation time ("unknown
unit"), so nothing is ever converted by assumption. Adding a unit is one
line here.
"""

from __future__ import annotations

import math
from typing import Iterable

# unit -> (dimension, scale, offset)  [base = value * scale + offset]
_F_SCALE = 5.0 / 9.0
UNITS: dict[str, tuple[str, float, float]] = {
    # temperature (base K)
    "K": ("temperature", 1.0, 0.0),
    "C": ("temperature", 1.0, 273.15),
    "F": ("temperature", _F_SCALE, 273.15 - 32.0 * _F_SCALE),
    # pressure (base Pa)
    "Pa": ("pressure", 1.0, 0.0),
    "kPa": ("pressure", 1e3, 0.0),
    "MPa": ("pressure", 1e6, 0.0),
    "bar": ("pressure", 1e5, 0.0),
    "psi": ("pressure", 6894.757293168, 0.0),
    # electrical current (base A)
    "A": ("current", 1.0, 0.0),
    "mA": ("current", 1e-3, 0.0),
    "kA": ("current", 1e3, 0.0),
    # voltage (base V)
    "V": ("voltage", 1.0, 0.0),
    "mV": ("voltage", 1e-3, 0.0),
    "kV": ("voltage", 1e3, 0.0),
    # power (base W)
    "W": ("power", 1.0, 0.0),
    "kW": ("power", 1e3, 0.0),
    "MW": ("power", 1e6, 0.0),
    "hp": ("power", 745.699872, 0.0),
    # rotational speed (base rpm)
    "rpm": ("rotational_speed", 1.0, 0.0),
    "rps": ("rotational_speed", 60.0, 0.0),
    "rad_s": ("rotational_speed", 60.0 / (2.0 * math.pi), 0.0),
    # torque (base Nm)
    "Nm": ("torque", 1.0, 0.0),
    "lbf_ft": ("torque", 1.355818, 0.0),
    "in_lb": ("torque", 0.112985, 0.0),
    # vibration velocity (base mm/s) and acceleration (base m/s^2)
    "mm_s": ("vibration_velocity", 1.0, 0.0),
    "in_s": ("vibration_velocity", 25.4, 0.0),
    "m_s2": ("vibration_accel", 1.0, 0.0),
    "g": ("vibration_accel", 9.80665, 0.0),
    # volumetric flow (base m3/h)
    "m3_h": ("flow", 1.0, 0.0),
    "L_min": ("flow", 0.06, 0.0),
    "L_s": ("flow", 3.6, 0.0),
    "gpm": ("flow", 0.2271247, 0.0),   # US gallons per minute
    # length (base mm)
    "mm": ("length", 1.0, 0.0),
    "m": ("length", 1e3, 0.0),
    "in": ("length", 25.4, 0.0),
    # time / duration (base min)
    "s": ("duration", 1.0 / 60.0, 0.0),
    "min": ("duration", 1.0, 0.0),
    "h": ("duration", 60.0, 0.0),
    # energy (base kWh)
    "Wh": ("energy", 1e-3, 0.0),
    "kWh": ("energy", 1.0, 0.0),
    "MWh": ("energy", 1e3, 0.0),
    "MJ": ("energy", 1.0 / 3.6, 0.0),
    # frequency (base Hz)
    "Hz": ("frequency", 1.0, 0.0),
    "kHz": ("frequency", 1e3, 0.0),
    # ratios (base fraction)
    "fraction": ("ratio", 1.0, 0.0),
    "percent": ("ratio", 0.01, 0.0),
    # throughput / dimensionless counts (each its own dimension: no conversion)
    "units/h": ("count_rate", 1.0, 0.0),
    "count": ("count", 1.0, 0.0),
}

ALIASES = {
    "kelvin": "K", "celsius": "C", "degC": "C", "fahrenheit": "F", "degF": "F",
    "minutes": "min", "minute": "min", "hours": "h", "hour": "h", "sec": "s", "seconds": "s",
    "N*m": "Nm", "newton_meter": "Nm", "%": "percent", "pct": "percent",
    "amp": "A", "amps": "A", "volt": "V", "volts": "V", "watt": "W",
    "gal/min": "gpm", "l/min": "L_min", "L/min": "L_min", "m3/h": "m3_h", "mm/s": "mm_s",
    "m/s2": "m_s2", "hz": "Hz", "KPa": "kPa", "Bar": "bar", "PSI": "psi",
}


def canonical_unit(unit: str | None) -> str | None:
    """The catalog name for `unit` (resolving aliases), or None if unknown."""
    if unit is None:
        return None
    unit = unit.strip()
    if unit in UNITS:
        return unit
    return ALIASES.get(unit)


def known_units() -> list[str]:
    return sorted(UNITS)


def dimension_of(unit: str) -> str:
    return UNITS[canonical_unit(unit)][0]


def compatible(unit_a: str, unit_b: str) -> bool:
    a, b = canonical_unit(unit_a), canonical_unit(unit_b)
    return a is not None and b is not None and UNITS[a][0] == UNITS[b][0]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert one number between two units of the same dimension."""
    f, t = canonical_unit(from_unit), canonical_unit(to_unit)
    if f is None or t is None:
        raise ValueError(f"unknown unit: {from_unit if f is None else to_unit!r}")
    (dim_f, s_f, o_f), (dim_t, s_t, o_t) = UNITS[f], UNITS[t]
    if dim_f != dim_t:
        raise ValueError(f"cannot convert {f} ({dim_f}) to {t} ({dim_t})")
    if f == t:
        return value
    return ((value * s_f + o_f) - o_t) / s_t


def convert_many(values: Iterable[float], from_unit: str, to_unit: str):
    """Vectorised `convert` for a pandas Series / numpy array."""
    f, t = canonical_unit(from_unit), canonical_unit(to_unit)
    if f is None or t is None:
        raise ValueError(f"unknown unit: {from_unit if f is None else to_unit!r}")
    (dim_f, s_f, o_f), (dim_t, s_t, o_t) = UNITS[f], UNITS[t]
    if dim_f != dim_t:
        raise ValueError(f"cannot convert {f} ({dim_f}) to {t} ({dim_t})")
    if f == t:
        return values
    return ((values * s_f + o_f) - o_t) / s_t
