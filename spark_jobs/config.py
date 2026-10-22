"""
Configuration for the synthetic operational data generator and the
PySpark processing pipeline.

Every "magic number" used across scripts/generate_synthetic_data.py and
the pyspark/ modules lives here, so the injected scenarios are inspectable
in one place rather than scattered through generator code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Fleet composition
# ---------------------------------------------------------------------------
N_MACHINES = 18
N_LINES = 3
MACHINES_PER_LINE = N_MACHINES // N_LINES

# AI4I-style quality-variant mix (L=low, M=medium, H=high). Real AI4I data is
# ~60% L / 30% M / 10% H (measured from the actual downloaded dataset) — we
# mirror that so our synthetic layer stays consistent with the real one.
TYPE_WEIGHTS = {"L": 0.55, "M": 0.30, "H": 0.15}

# Tool-wear accrual rate (minutes of wear added per hour of operation),
# matching AI4I's documented per-type wear multipliers (H/M/L -> 5/3/2).
TOOL_WEAR_RATE_PER_HOUR = {"H": 5.0, "M": 3.0, "L": 2.0}

# ---------------------------------------------------------------------------
# Time range / granularity
# ---------------------------------------------------------------------------
SIMULATION_DAYS = 60
READING_INTERVAL_MINUTES = 5

# ---------------------------------------------------------------------------
# Baseline sensor generation (matches AI4I's own documented generation
# process, so the synthetic layer is statistically consistent with the real
# dataset rather than an arbitrary invention):
#   air_temperature: random walk, SD 2K around 300K
#   process_temperature: air_temperature + 10K, random walk, SD 1K
#   rotational_speed: derived from ~2860W power, with noise
#   torque: Normal(40, 10), no negative values
# ---------------------------------------------------------------------------
AIR_TEMP_BASE_K = 300.0
AIR_TEMP_WALK_SD = 0.15          # per-step random-walk increment
AIR_TEMP_NOISE_SD = 0.4
PROCESS_TEMP_OFFSET_K = 10.0
PROCESS_TEMP_WALK_SD = 0.1
PROCESS_TEMP_NOISE_SD = 0.3

# Power, torque and rotational speed are calibrated directly against the
# *actual downloaded* AI4I dataset (measured, not assumed).
#
# Two earlier drafts of this constant were wrong, caught by verification
# against the real data rather than trusted from memory/documentation:
#   1. A "2860W-derived" formula produced power permanently below the PWF
#      band (0% baseline flag rate on healthy machines — signal dead).
#   2. Sampling torque and rotational_speed as *independent* Normals
#      (matching their individual real marginal distributions) produced a
#      12.4% baseline PWF flag rate on HEALTHY machines, because real
#      torque and speed are physically coupled through power (a motor's
#      speed depends on the load/torque it's under) — independent sampling
#      loses that coupling and inflates power's variance far past reality.
#
# Fix: sample power directly (mean/std measured from the real dataset:
# implied power = torque * angular_speed, mean 6280W, std 1067W — and the
# published PWF thresholds [3500W, 9000W] correctly reproduce exactly the
# 95 PWF=1 rows the real dataset itself labels), then DERIVE rotational
# speed from power / torque. This reproduces the real ~1% baseline
# crossing rate instead of an independent-sampling artifact.
POWER_MEAN_W = 6280.0
POWER_SD_W = 1067.0
TORQUE_MEAN_NM = 40.0
TORQUE_SD_NM = 10.0
ROTATIONAL_SPEED_MIN_RPM = 1100.0
ROTATIONAL_SPEED_MAX_RPM = 2950.0
PWF_SCENARIO_POWER_SHIFT_W = 4200.0  # additive mean shift at full ramp severity

# ---------------------------------------------------------------------------
# Business-layer metrics (NOT part of AI4I — clearly synthetic, generated to
# correlate with injected degradation so business-facing questions have a
# grounded answer).
# ---------------------------------------------------------------------------
BASE_PRODUCTION_RATE = 100.0      # units/hour, healthy baseline
BASE_ENERGY_KWH = 12.0            # kWh per hour, healthy baseline
BASE_DEFECT_RATE = 0.02           # 2% defects, healthy baseline

# ---------------------------------------------------------------------------
# AI4I's own published deterministic failure-mode thresholds. We reuse these
# exact thresholds (rather than inventing new ones) so the anomaly-detection
# rules are grounded in the same domain logic as the real dataset:
#   HDF: (process_temp - air_temp) < 8.6K AND rotational_speed < 1380 rpm
#   PWF: power = torque[Nm] * rotational_speed[rad/s]; power <3500W or >9000W
#   OSF: tool_wear[min] * torque[Nm] > type-specific threshold
#   TWF: tool_wear in [200, 240] minutes
# ---------------------------------------------------------------------------
HDF_TEMP_DIFF_THRESHOLD_K = 8.6
HDF_ROTATIONAL_SPEED_THRESHOLD_RPM = 1380
PWF_MIN_POWER_W = 3500.0
PWF_MAX_POWER_W = 9000.0
OSF_THRESHOLD_NM_MIN = {"L": 11000, "M": 12000, "H": 13000}
TWF_WEAR_MIN_MINUTES = 200
TWF_WEAR_MAX_MINUTES = 240

# ---------------------------------------------------------------------------
# Injected failure scenarios (Phase-1 sanity-check design).
# `onset_day` = day the degradation trend begins (0-indexed).
# `severity_slope` = how quickly the metric drifts toward failure territory.
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    machine_id: str
    scenario: str                  # "TWF" | "HDF" | "PWF" | "OSF" | "RNF" | "TWF_RECURRING" | "HEALTHY"
    onset_day: int = 0
    severity_slope: float = 1.0
    ramp_days: float = 12.0        # days to ramp from onset to full severity
    repair_day: float | None = None    # day degradation is corrected (None = stays broken)
    second_onset_day: float | None = None      # only for TWF_RECURRING
    second_repair_day: float | None = None     # only for TWF_RECURRING


INJECTED_SCENARIOS: list[ScenarioConfig] = [
    ScenarioConfig(machine_id="M-01", scenario="TWF", onset_day=5, repair_day=17),
    ScenarioConfig(machine_id="M-02", scenario="HDF", onset_day=15, repair_day=27),
    ScenarioConfig(machine_id="M-03", scenario="PWF", onset_day=25, repair_day=37),
    ScenarioConfig(machine_id="M-04", scenario="OSF", onset_day=35, repair_day=None),  # unresolved
    ScenarioConfig(machine_id="M-05", scenario="RNF", onset_day=40, repair_day=None),  # one-off blip
    ScenarioConfig(
        machine_id="M-06",
        scenario="TWF_RECURRING",
        onset_day=8,
        repair_day=14,
        second_onset_day=38,
        second_repair_day=48,
    ),
]

# Every machine_id not listed above runs as a healthy baseline machine.

# Routine preventive tool replacement: any machine's tool wear is reset by a
# (simulated) technician once it crosses this point during NORMAL operation.
# Only machines in an active, not-yet-repaired TWF/OSF-type degradation
# window are allowed to keep accruing wear past it — that's precisely what
# makes those machines anomalous. Kept safely under AI4I's TWF window
# (200-240 min) so routine replacement never itself looks like a failure.
ROUTINE_TOOL_REPLACEMENT_THRESHOLD_MIN = 150.0
ACCEL_TWF_WEAR_RATE_PER_HOUR = 9.0   # extra wear/hour at full severity, TWF/TWF_RECURRING
ACCEL_OSF_WEAR_RATE_PER_HOUR = 7.0   # extra wear/hour at full severity, OSF

# ---------------------------------------------------------------------------
# Anomaly-detection / risk-scoring parameters (deterministic, statistical)
# ---------------------------------------------------------------------------
# Audit F9: was 12 readings (1 hour) and the window included the current
# reading itself. Including the point you're testing in its own baseline
# is self-referential: with sample std, it mathematically caps the z-score
# at (n-1)/sqrt(n) ~= 3.18 for n=12, just barely above the 3.0 threshold,
# so almost nothing could ever trip it (measured: 2.3% of flagged rows were
# statistical, the rest were AI4I threshold flags). feature_engineering.py's
# rolling window now excludes the current row, and the window itself is
# widened to a 24-hour trailing baseline (still short enough to adapt to
# slow drift, long enough that one point can't skew its own comparison).
ROLLING_WINDOW_READINGS = 288       # 288 * 5min = 24 hour trailing window, current reading excluded
Z_SCORE_ANOMALY_THRESHOLD = 3.0
RISK_SCORE_THRESHOLDS = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}
