"""
Per-tenant anomaly rules (Phase 1, task 2).

The numbers that used to be module constants in spark_jobs/config.py
(Z_SCORE_ANOMALY_THRESHOLD, ROLLING_WINDOW_READINGS and the AI4I physical
failure-mode thresholds) are now an `AnomalyRules` value built per tenant
from two places:

  * `sensor_registry`   - one row per sensor: unit, normal_min/normal_max,
                          optional z_score_threshold, enabled.
  * `tenant_settings`   - `config["anomaly"]`, tenant-wide:
        {"z_score_threshold": 3.0,
         "rolling_window_readings": 288,
         "rule_packs": {"ai4i": {"hdf_temp_diff_k": 8.6, ...}}}

spark_jobs/config.py still holds the original constants; they are the
*defaults* here, so a tenant that overrides nothing gets exactly the
previous behaviour. This module is pure Python (no pyspark, no DB) so both
the API and the Spark jobs can import it.

Rule types a registry sensor can trigger (all per reading):
  * z-score   - |value - trailing mean| / trailing std > threshold
  * range     - value < normal_min or value > normal_max (only if set)
The optional "ai4i" rule pack adds the four AI4I composite failure-mode
checks (TWF/HDF/PWF/OSF). It needs the AI4I columns and is only enabled for
a tenant whose settings say so (the demo tenant, seeded by migration 0005).
Composite rules for other sensor vocabularies are NOT supported (see README
Limitations).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from spark_jobs import config as cfg


@dataclass(frozen=True)
class SensorRule:
    sensor_name: str
    unit: str = ""
    normal_min: float | None = None
    normal_max: float | None = None
    z_score_threshold: float | None = None   # None -> tenant default
    enabled: bool = True
    label: str | None = None                 # human text in triggered_reasons

    @property
    def display(self) -> str:
        return self.label or self.sensor_name.replace("_", " ")


@dataclass(frozen=True)
class Ai4iPack:
    """The AI4I failure-mode thresholds, defaults = spark_jobs/config.py."""
    hdf_temp_diff_k: float = cfg.HDF_TEMP_DIFF_THRESHOLD_K
    hdf_rotational_speed_rpm: float = cfg.HDF_ROTATIONAL_SPEED_THRESHOLD_RPM
    pwf_min_power_w: float = cfg.PWF_MIN_POWER_W
    pwf_max_power_w: float = cfg.PWF_MAX_POWER_W
    osf_threshold_nm_min: Mapping[str, float] = field(
        default_factory=lambda: dict(cfg.OSF_THRESHOLD_NM_MIN))
    twf_wear_min_minutes: float = cfg.TWF_WEAR_MIN_MINUTES
    twf_wear_max_minutes: float = cfg.TWF_WEAR_MAX_MINUTES


@dataclass(frozen=True)
class AnomalyRules:
    z_score_threshold: float = cfg.Z_SCORE_ANOMALY_THRESHOLD
    rolling_window_readings: int = cfg.ROLLING_WINDOW_READINGS
    sensors: tuple[SensorRule, ...] = ()
    ai4i: Ai4iPack | None = None

    @property
    def active_sensors(self) -> tuple[SensorRule, ...]:
        return tuple(s for s in self.sensors if s.enabled)

    def z_threshold_for(self, sensor: SensorRule) -> float:
        return (sensor.z_score_threshold
                if sensor.z_score_threshold is not None else self.z_score_threshold)


# AI4I demo vocabulary. The five z-scored names + labels reproduce the
# original Z_SCORE_METRICS and flag descriptions exactly; the other three
# are registered but disabled (they were never z-scored; they only feed the
# ai4i pack).
_AI4I_Z_SENSORS = (
    ("torque_nm", "Nm", "Torque"),
    ("tool_wear_min", "min", "Tool wear"),
    ("production_rate", "units/h", "Production rate"),
    ("defect_rate", "fraction", "Defect rate"),
    ("energy_consumption_kwh", "kWh", "Energy consumption"),
)
_AI4I_PACK_ONLY = (
    ("air_temperature_k", "K"),
    ("process_temperature_k", "K"),
    ("rotational_speed_rpm", "rpm"),
)
AI4I_LABELS = {n: label for n, _u, label in _AI4I_Z_SENSORS}
# Registry rows come back in no particular order; this keeps the AI4I
# sensors in their original Z_SCORE_METRICS order (it fixes the order of
# the reasons text), then everything else alphabetically.
_CANONICAL_ORDER = {n: i for i, (n, _u, _l) in enumerate(_AI4I_Z_SENSORS)}

AI4I_DEFAULT_RULES = AnomalyRules(
    sensors=tuple(SensorRule(n, u, label=lbl) for n, u, lbl in _AI4I_Z_SENSORS),
    ai4i=Ai4iPack(),
)


def ai4i_registry_rows() -> list[dict[str, Any]]:
    """The demo tenant's sensor_registry rows (migration 0005 seeds these)."""
    rows = [{"sensor_name": n, "unit": u, "enabled": True} for n, u, _l in _AI4I_Z_SENSORS]
    rows += [{"sensor_name": n, "unit": u, "enabled": False} for n, u in _AI4I_PACK_ONLY]
    return rows


class RuleConfigError(ValueError):
    pass


def _num(raw: Mapping[str, Any], key: str, kind=float, positive=True):
    if key not in raw or raw[key] is None:
        return None
    try:
        value = kind(raw[key])
    except (TypeError, ValueError) as exc:
        raise RuleConfigError(f"anomaly.{key}: {raw[key]!r} is not a number") from exc
    if positive and value <= 0:
        raise RuleConfigError(f"anomaly.{key} must be > 0, got {value}")
    return value


def build_rules(
    registry_rows: Sequence[Mapping[str, Any]],
    settings_config: Mapping[str, Any] | None,
) -> AnomalyRules:
    """Registry rows + tenant_settings config -> AnomalyRules.

    Raises RuleConfigError on nonsense (non-positive thresholds, min > max)
    rather than silently detecting nothing.
    """
    anomaly = (settings_config or {}).get("anomaly") or {}
    z = _num(anomaly, "z_score_threshold")
    window = _num(anomaly, "rolling_window_readings", kind=int)

    sensors = []
    for row in registry_rows:
        lo, hi = row.get("normal_min"), row.get("normal_max")
        if lo is not None and hi is not None and lo > hi:
            raise RuleConfigError(
                f"sensor {row['sensor_name']!r}: normal_min {lo} > normal_max {hi}")
        zt = row.get("z_score_threshold")
        if zt is not None and zt <= 0:
            raise RuleConfigError(
                f"sensor {row['sensor_name']!r}: z_score_threshold must be > 0")
        sensors.append(SensorRule(
            sensor_name=row["sensor_name"], unit=row.get("unit") or "",
            normal_min=lo, normal_max=hi, z_score_threshold=zt,
            enabled=bool(row.get("enabled", True)),
            label=AI4I_LABELS.get(row["sensor_name"]),
        ))
    sensors.sort(key=lambda s: (_CANONICAL_ORDER.get(s.sensor_name, len(_CANONICAL_ORDER)), s.sensor_name))

    pack = None
    packs = anomaly.get("rule_packs") or {}
    if packs.get("ai4i") is not None:
        raw = packs["ai4i"] or {}
        base = Ai4iPack()
        pack = replace(
            base,
            hdf_temp_diff_k=float(raw.get("hdf_temp_diff_k", base.hdf_temp_diff_k)),
            hdf_rotational_speed_rpm=float(raw.get("hdf_rotational_speed_rpm", base.hdf_rotational_speed_rpm)),
            pwf_min_power_w=float(raw.get("pwf_min_power_w", base.pwf_min_power_w)),
            pwf_max_power_w=float(raw.get("pwf_max_power_w", base.pwf_max_power_w)),
            osf_threshold_nm_min={**base.osf_threshold_nm_min, **(raw.get("osf_threshold_nm_min") or {})},
            twf_wear_min_minutes=float(raw.get("twf_wear_min_minutes", base.twf_wear_min_minutes)),
            twf_wear_max_minutes=float(raw.get("twf_wear_max_minutes", base.twf_wear_max_minutes)),
        )

    return AnomalyRules(
        z_score_threshold=z if z is not None else cfg.Z_SCORE_ANOMALY_THRESHOLD,
        rolling_window_readings=window if window is not None else cfg.ROLLING_WINDOW_READINGS,
        sensors=tuple(sensors),
        ai4i=pack,
    )


def load_anomaly_rules(conn, tenant_id: str) -> AnomalyRules:
    """Read this tenant's rules from Postgres (registry + tenant_settings)."""
    from app.database import queries  # local import: keep module DB-free at import time
    registry = queries.list_sensor_registry(conn, tenant_id=tenant_id)
    settings = queries.get_tenant_settings(conn, tenant_id=tenant_id)
    return build_rules(registry, settings)
