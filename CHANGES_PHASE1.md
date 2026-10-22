# Phase 1 changes - generic sensors, per-tenant rules, security defaults

## What changed
- **Migration 0005** (+ `sql/schema.sql` mirror): `sensor_registry`, partitioned long-format `sensor_readings` (PK tenant/machine/sensor/ts = idempotency key, composite FKs), demo registry (8 AI4I sensors) and `tenant_settings` seed enabling the `ai4i` rule pack. Downgrade refuses if it would lose data.
- **Per-tenant rules**: `app/core/anomaly_rules.py` builds `AnomalyRules` from registry + `tenant_settings.config["anomaly"]`; `spark_jobs/feature_engineering.py` / `anomaly_detection.py` take rules (default = old behaviour). New `spark_jobs/long_format.py`, `scripts/run_generic_pipeline.py`.
- **Onboarding**: `app/onboarding/units.py`, `sensors.py`, `load.py`; `onboard_tenant.py --sensor-csv/--sensor-mapping/--allow-partial`; `scripts/load_sensor_readings.py`. Per-row error report.
- **Security**: `AUTH_REQUIRED` defaults true for remote DB or set `API_KEY`; compose requires `POSTGRES_PASSWORD`/`API_KEY`, DB bound to 127.0.0.1; `/healthz`, `/readyz`.
- **API**: `GET /sensors`, `GET /machines/{id}/readings`; `MachineOut.type` nullable.

## Verified (actually run)
- Full suite `pytest -m "not dense and not load"`: **358 passed, 0 failed, 10 skipped, 18 deselected** (baseline before: 299 / 10 / 18).
- AI4I outputs equal the committed parquet exactly (`test_anomaly_rules.py`, 25,920 / 28,855 rows).
- Company B fleet (`test_company_b.py`, 16 tests): z-score and range anomalies found with no code change; tenant A's key cannot see B's readings, machines, registry or documents (shared machine id included).
- Migration 0005 upgrade/downgrade and guards (`test_migrations.py`); `schema.sql` == alembic head.

## NOT verified
`docker compose up` (only `docker compose config` checked); `--ai4i-demo` load of ~2.5M rows; `-m dense`; JDBC; CI workflow on GitHub.

## Limitations
See README "Limitations" (AI4I-only composite rules, cold-start false positives, row-count rolling window, AI4I-oriented trends, long-format path pins UTC while legacy does not, no `calibrate` CLI).

## Decisions needing your confirmation (I picked the recommended option; you said to continue)
- D1 `machines.type` made nullable (L/M/H CHECK kept) and `machine_id` widened to VARCHAR(64) - the only non-additive schema changes.
- D2 AI4I composite rules kept as an opt-in rule pack seeded for the demo tenant, rather than generalised.
- D3 Long-format loader pins Spark session TZ to UTC.
- D4 Dirty onboarding files are rejected unless `--allow-partial`; first write wins on duplicate keys.
