"""
Deterministic risk assessment — Phase 5, revised in Phase 9.

Per the master prompt's RISK ASSESSMENT section: "The exact scoring method
must be deterministic and explainable. Do not allow the LLM to arbitrarily
assign risk." There is no LLM call anywhere in this module, and there
never should be — Phase 6 (Vertex/Gemini) is allowed to *explain* the risk
level this function returns, not compute it.

Phase 9 fix. The Phase 5 rule was "any MEDIUM-severity anomaly among the
latest 20 events => MEDIUM". Running /investigate over the whole fleet
showed that rule is saturated: baseline noise produces ~1% single-rule
MEDIUM events on every machine, so all 18 machines came back MEDIUM —
including the one machine (M-04) whose last 24h were 100% anomalous.
Risk is now driven by the *sustained anomalous-reading rate* over the
recent hourly windows (sensor_trend, already gathered by the graph — no
new tool call, no new state). The old severity rule survives only as a
degraded-mode fallback for when the trend is unavailable.

Rules when at least MIN_TREND_WINDOWS hourly windows are available
(highest wins), where `rate` = anomalous readings / total readings across
the trend windows:

  HIGH    — latest health_status is AT_RISK, OR a HIGH-severity anomaly
            occurred inside the trend window, OR rate >= SUSTAINED_HIGH_RATE.
  MEDIUM  — rate >= SUSTAINED_MEDIUM_RATE.
  LOW     — otherwise. (A lone WATCH snapshot is NOT enough on its own: a
            single hourly window can flag WATCH from statistical noise —
            a known Phase 2 finding.)
  UNKNOWN — there is no evidence of any kind to reason about (no health
            snapshot, no trend, no anomaly log — see below). This is
            checked first and short-circuits everything else.

Phase 9.1 fix (audit finding F2). Before this fix, `assess_risk(None, [])`
returned LOW: no sensor snapshot, no trend and no anomaly log all fell
through to the "otherwise" branch of both the rate rule and the
degraded-mode fallback below, which is indistinguishable from "checked
everything, machine is healthy". A scripted planner run that skipped the
sensor and anomaly tools entirely reproduced this on a machine whose
real last-24h trend was 100% anomalous: the report read LOW, and the
warning that evidence was missing only showed up buried in
`errors`/`confidence_limitations`, not in the risk level a reader
actually scans for. The same failure mode is reachable without any
planner at all — a DB outage during gather_sensor_evidence plus an
already-empty anomaly log in the fixed chain hits it too.

UNKNOWN is returned only when there is truly nothing to go on:
sensor_evidence is None AND sensor_trend is empty AND anomaly_metrics is
empty. Any one of those being non-empty means there is still something
to reason about, so the rate rule / degraded-mode fallback below still
apply — this deliberately does not touch the "few trend windows but a
real snapshot or anomaly log exists" case, which is a legitimate,
already-calibrated degraded mode, not a missing-evidence one.

Threshold calibration (be honest about it): on this project's SYNTHETIC
fleet the healthy-machine noise floor is ~1% of readings (max 1.4% over
the last 24h). SUSTAINED_MEDIUM_RATE is ~5x that floor and
SUSTAINED_HIGH_RATE ~25x. They were chosen by looking at this synthetic
data, so they are tunable starting points, not validated industrial
limits. Change them here and the tests in tests/test_agent.py still pin
the behaviour.

Production-readiness fix 11. These three numbers are calibrated to ONE
fleet's noise floor — a different tenant's machines, sensors and sampling
rate need their own calibration, not this one silently reused. They are
now bundled into `RiskThresholds`, a plain value object: every function in
this module takes an optional `thresholds` argument and falls back to
`DEFAULT_RISK_THRESHOLDS` (this module's original constants, unchanged)
when it's omitted, so every existing caller/test that doesn't pass one
keeps the exact demo-fleet behaviour. The per-tenant override itself is
loaded from the `tenant_settings` table by
`app.core.tenant_settings.load_tenant_settings` and threaded in once, at
graph-build time (see app/agents/graph.py) — this module still has no
database access and no LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.state import RiskLevel

_AT_RISK_STATUSES = {"AT_RISK"}
_WATCH_STATUSES = {"WATCH"}

# Fewer windows than this cannot support a rate estimate (one hourly
# window is exactly the transient-noise case we are trying to avoid).
MIN_TREND_WINDOWS = 6
SUSTAINED_MEDIUM_RATE = 0.05  # ~5x the ~1% healthy-machine noise floor
SUSTAINED_HIGH_RATE = 0.25    # 1 in 4 readings anomalous, sustained


@dataclass(frozen=True)
class RiskThresholds:
    """The tunable numbers behind `assess_risk` — see the module docstring
    (production-readiness fix 11) for why these moved out of bare module
    constants."""

    min_trend_windows: int = MIN_TREND_WINDOWS
    sustained_medium_rate: float = SUSTAINED_MEDIUM_RATE
    sustained_high_rate: float = SUSTAINED_HIGH_RATE


# The original demo-fleet calibration, unchanged — every caller that omits
# `thresholds` gets exactly this, byte-for-byte what the module constants
# above already specified.
DEFAULT_RISK_THRESHOLDS = RiskThresholds()


def _anomalous_rate(
    sensor_trend: list[dict[str, Any]], thresholds: RiskThresholds
) -> float | None:
    """Anomalous readings / total readings across the trend windows, or
    None when there is not enough (or no usable) data to estimate one."""
    if len(sensor_trend) < thresholds.min_trend_windows:
        return None
    total = sum(w.get("reading_count") or 0 for w in sensor_trend)
    if total <= 0:
        return None
    anomalous = sum(w.get("anomalous_reading_count")
                    or 0 for w in sensor_trend)
    return anomalous / total


def _has_recent_high_anomaly(
    anomaly_metrics: list[dict[str, Any]], sensor_trend: list[dict[str, Any]]
) -> bool:
    """HIGH-severity events only count if they fall inside the trend window.

    anomaly_metrics is just "the latest N events" with no time bound, so
    without this a HIGH event from a fault repaired weeks ago would make
    the machine look HIGH-risk today.
    """
    window_start = min(
        (w["window_start"]
         for w in sensor_trend if w.get("window_start") is not None),
        default=None,
    )
    for a in anomaly_metrics:
        if a.get("severity") != "HIGH":
            continue
        detected_at = a.get("detected_at")
        if window_start is None or detected_at is None or detected_at >= window_start:
            return True
    return False


def assess_risk(
    sensor_evidence: dict[str, Any] | None,
    anomaly_metrics: list[dict[str, Any]],
    sensor_trend: list[dict[str, Any]] | None = None,
    thresholds: RiskThresholds | None = None,
) -> RiskLevel:
    """Combine the latest snapshot, recent anomaly events and the hourly
    trend into LOW / MEDIUM / HIGH. See the module docstring for the rules.

    Degraded mode (fewer than MIN_TREND_WINDOWS trend windows, e.g. the
    sensor tool failed): fall back to the conservative Phase 5 rule —
    latest AT_RISK or any HIGH anomaly => HIGH; latest WATCH or any
    MEDIUM anomaly => MEDIUM — because without a rate we cannot tell a
    noisy machine from a failing one and should not under-report.

    No-evidence mode (sensor_evidence is None AND sensor_trend is empty
    AND anomaly_metrics is empty): returns UNKNOWN rather than LOW. LOW
    is a claim that the evidence was checked and looks healthy; with
    nothing gathered at all, that claim isn't available to make.
    """
    thresholds = thresholds or DEFAULT_RISK_THRESHOLDS
    anomaly_metrics = anomaly_metrics or []
    sensor_trend = sensor_trend or []

    if sensor_evidence is None and not sensor_trend and not anomaly_metrics:
        return "UNKNOWN"

    status = (sensor_evidence or {}).get("health_status")

    rate = _anomalous_rate(sensor_trend, thresholds)
    if rate is not None:
        if (
            status in _AT_RISK_STATUSES
            or _has_recent_high_anomaly(anomaly_metrics, sensor_trend)
            or rate >= thresholds.sustained_high_rate
        ):
            return "HIGH"
        if rate >= thresholds.sustained_medium_rate:
            return "MEDIUM"
        return "LOW"

    has_high = any(a.get("severity") == "HIGH" for a in anomaly_metrics)
    has_medium = any(a.get("severity") == "MEDIUM" for a in anomaly_metrics)
    if status in _AT_RISK_STATUSES or has_high:
        return "HIGH"
    if status in _WATCH_STATUSES or has_medium:
        return "MEDIUM"
    return "LOW"
