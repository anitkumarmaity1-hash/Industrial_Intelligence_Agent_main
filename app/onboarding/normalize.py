"""
Apply a validated MappingConfig to a tenant's raw CSV export, producing a
file in the exact shape spark_jobs/ingestion.py's SYNTHETIC_OPERATIONAL /
SYNTHETIC_MAINTENANCE schemas expect — so the existing Spark pipeline can
process it completely unmodified.

Uses pandas (not Spark): onboarding runs once per tenant on a human-scale
export, and pandas gives clearer per-row error messages for a first-time
integration than a Spark job's stack traces would.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.onboarding.mapping import (
    MAINTENANCE_REQUIRED,
    OPERATIONAL_OPTIONAL_DEFAULTS,
    OPERATIONAL_REQUIRED,
    MappingConfig,
)

logger = logging.getLogger(__name__)


class NormalizationError(ValueError):
    """Raised with every row-level problem collected, not just the first."""


def _read_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_csv(path)


def normalize_operational(raw_csv: str | Path, mapping: MappingConfig) -> pd.DataFrame:
    """Return a DataFrame with exactly OPERATIONAL_REQUIRED (+ optional)
    columns, in canonical units, ready to write next to
    data/raw/synthetic_operational.csv's own format.
    """
    if mapping.dataset_kind != "operational":
        raise ValueError(f"mapping is for {mapping.dataset_kind!r}, not 'operational'")
    df = _read_raw(raw_csv)
    errors: list[str] = []

    for canonical, colmap in mapping.columns.items():
        if colmap.source_column not in df.columns:
            errors.append(f"source column {colmap.source_column!r} (for {canonical!r}) "
                          f"not found in {Path(raw_csv).name}")
    if errors:
        raise NormalizationError("Normalization failed:\n  - " + "\n  - ".join(errors))

    out = pd.DataFrame(index=df.index)
    for canonical, colmap in mapping.columns.items():
        series = df[colmap.source_column]
        if canonical in ("air_temperature_k", "process_temperature_k",
                         "torque_nm", "tool_wear_min", "rotational_speed_rpm"):
            series = pd.to_numeric(series, errors="coerce")
            bad = series.isna() & df[colmap.source_column].notna()
            if bad.any():
                errors.append(f"{canonical}: {int(bad.sum())} row(s) are not numeric "
                              f"(source column {colmap.source_column!r})")
            out[canonical] = series.apply(lambda v: mapping.convert(canonical, v) if pd.notna(v) else v)
        elif canonical == "type":
            out[canonical] = series.map(mapping.machine_type_map)
            unmapped = series[out[canonical].isna() & series.notna()].unique()
            if len(unmapped):
                errors.append(f"type: value(s) {sorted(map(str, unmapped))} have no entry "
                              "in machine_type_map")
        elif canonical == "timestamp":
            out[canonical] = pd.to_datetime(series, errors="coerce")
            bad = out[canonical].isna() & series.notna()
            if bad.any():
                errors.append(f"timestamp: {int(bad.sum())} row(s) could not be parsed")
        else:
            out[canonical] = series

    for col, default in OPERATIONAL_OPTIONAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default

    for col in OPERATIONAL_REQUIRED:
        null_count = out[col].isna().sum()
        if null_count:
            errors.append(f"{col}: {null_count} row(s) are null after mapping "
                          "(required, no default)")

    if errors:
        raise NormalizationError("Normalization failed:\n  - " + "\n  - ".join(errors))

    out = out[list(OPERATIONAL_REQUIRED) + list(OPERATIONAL_OPTIONAL_DEFAULTS)]
    logger.info("Normalized %d operational rows across %d machines.",
               len(out), out["machine_id"].nunique())
    return out


def normalize_maintenance(raw_csv: str | Path, mapping: MappingConfig) -> pd.DataFrame:
    """Return a DataFrame with exactly MAINTENANCE_REQUIRED columns."""
    if mapping.dataset_kind != "maintenance":
        raise ValueError(f"mapping is for {mapping.dataset_kind!r}, not 'maintenance'")
    df = _read_raw(raw_csv)
    errors: list[str] = []

    for canonical, colmap in mapping.columns.items():
        if colmap.source_column not in df.columns:
            errors.append(f"source column {colmap.source_column!r} (for {canonical!r}) "
                          f"not found in {Path(raw_csv).name}")
    if errors:
        raise NormalizationError("Normalization failed:\n  - " + "\n  - ".join(errors))

    out = pd.DataFrame(index=df.index)
    for canonical in MAINTENANCE_REQUIRED:
        source = mapping.columns[canonical].source_column
        out[canonical] = df[source]

    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.date
    bad_dates = out["event_date"].isna()
    if bad_dates.any():
        errors.append(f"event_date: {int(bad_dates.sum())} row(s) could not be parsed")

    resolved = out["resolved"]
    truthy = {"true", "1", "yes", "y", "resolved"}
    falsy = {"false", "0", "no", "n", "open", "unresolved"}
    normalized = resolved.astype(str).str.strip().str.lower().map(
        lambda v: True if v in truthy else (False if v in falsy else None))
    unrecognized = resolved[normalized.isna() & resolved.notna()].unique()
    if len(unrecognized):
        errors.append(f"resolved: value(s) {sorted(map(str, unrecognized))} are not "
                      "recognized as true/false")
    out["resolved"] = normalized

    for col in MAINTENANCE_REQUIRED:
        null_count = out[col].isna().sum()
        if null_count:
            errors.append(f"{col}: {null_count} row(s) are null after mapping (required)")

    if errors:
        raise NormalizationError("Normalization failed:\n  - " + "\n  - ".join(errors))

    out = out[list(MAINTENANCE_REQUIRED)]
    logger.info("Normalized %d maintenance records.", len(out))
    return out
