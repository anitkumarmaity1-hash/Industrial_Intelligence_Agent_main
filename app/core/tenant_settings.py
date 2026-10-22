"""
Per-tenant calibration — production-readiness fixes 11 and 12.

Three things the audit found hardcoded to the demo fleet are now tenant
config, loaded from the `tenant_settings` table (migration 0003) instead
of being read straight off module-level constants:

  * `app.agents.risk.RiskThresholds` — the sustained-rate cutoffs that
    decide LOW/MEDIUM/HIGH.
  * `Settings.rag_min_term_coverage_lexical` — the BM25 relevance floor.
  * `app.agents.query_parser.MachineIdScheme` — how a question's free text
    names a machine.

A tenant with no row (every tenant until an operator sets one, including
the demo tenant today) gets exactly the original hardcoded defaults —
nothing changes for the demo fleet unless someone deliberately overrides
it. `config` is a single small JSONB blob per tenant rather than three
tables: these three numbers are read together, at the same point (once
per investigation / once per retriever build), and a tenant onboarding
their own fleet sets them together too (see scripts/manage_tenants.py
`calibrate`).

Shape of `config` (all keys optional):
    {
      "risk": {"sustained_medium_rate": 0.05, "sustained_high_rate": 0.25,
                "min_trend_windows": 6},
      "rag_min_term_coverage_lexical": 0.3,
      "machine_id": {"id_prefix": "A-", "code_separators": "-_",
                      "digit_width": 0, "word_pattern": null}
    }
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Connection

from app.agents.query_parser import DEFAULT_MACHINE_ID_SCHEME, MachineIdScheme
from app.agents.risk import DEFAULT_RISK_THRESHOLDS, RiskThresholds
from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID
from app.database import queries


@dataclass(frozen=True)
class TenantCalibration:
    """Everything this tenant has overridden, with defaults filled in."""

    risk_thresholds: RiskThresholds
    rag_min_term_coverage_lexical: float
    machine_id_scheme: MachineIdScheme


def _risk_thresholds_from(raw: dict) -> RiskThresholds:
    return RiskThresholds(
        min_trend_windows=int(
            raw.get("min_trend_windows", DEFAULT_RISK_THRESHOLDS.min_trend_windows)),
        sustained_medium_rate=float(
            raw.get("sustained_medium_rate", DEFAULT_RISK_THRESHOLDS.sustained_medium_rate)),
        sustained_high_rate=float(
            raw.get("sustained_high_rate", DEFAULT_RISK_THRESHOLDS.sustained_high_rate)),
    )


def _machine_id_scheme_from(tenant_id: str, raw: dict) -> MachineIdScheme:
    return MachineIdScheme(
        name=tenant_id,
        id_prefix=str(raw["id_prefix"]),
        code_separators=str(raw.get("code_separators", "-_")),
        digit_width=int(raw.get("digit_width", 0)),
        word_pattern=raw.get("word_pattern"),
    )


def load_tenant_calibration(conn: Connection, tenant_id: str = DEFAULT_TENANT_ID) -> TenantCalibration:
    """Read this tenant's overrides (if any) and fill in the documented
    defaults for everything it doesn't set. Never raises on a missing or
    partial config — a typo'd key just falls back to the default for that
    key, which is safer than a 500 on every investigation for a tenant
    whose config is mid-edit."""
    config = queries.get_tenant_settings(conn, tenant_id=tenant_id) or {}

    risk_thresholds = _risk_thresholds_from(config.get("risk") or {})

    rag_floor = config.get("rag_min_term_coverage_lexical")
    rag_floor = float(rag_floor) if rag_floor is not None else get_settings(
    ).rag_min_term_coverage_lexical

    machine_id_cfg = config.get("machine_id") or {}
    scheme = (
        _machine_id_scheme_from(tenant_id, machine_id_cfg)
        if machine_id_cfg.get("id_prefix")
        else DEFAULT_MACHINE_ID_SCHEME
    )

    return TenantCalibration(
        risk_thresholds=risk_thresholds,
        rag_min_term_coverage_lexical=rag_floor,
        machine_id_scheme=scheme,
    )
