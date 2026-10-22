# Phase 5 changes - one scheduled pull connector (GCS)

## What changed
- **Migration 0009** (+ `sql/schema.sql`): `connectors` (config bucket/prefix/pattern, cron schedule, mapping snapshot, `secret_ref` NAME only, lease `claimed_at`, listing `list_watermark`, last status), `connector_objects` (ledger, `UNIQUE (tenant, connector, key, version)`), `connector_runs`; SECURITY DEFINER `iia_claim_due_connectors()` (ids only); `audit_log.principal_kind` gains `system`; grants for `iia_app` (config columns only) and `iia_worker` (run state, INSERT on uploads/jobs/audit_log). `connectors` is ENABLEd not FORCEd (like `jobs`); the other two are FORCEd. Downgrade refuses if connector rows or `system` audit rows exist.
- **`app/connectors/`**: `base.py` (ObjectStore interface, fixed-message errors), `gcs.py` (the one implementation), `validation.py`, `puller.py` (one pull), `scheduler.py` (atomic claim + tick), `settings.py`.
- **`app/core/secrets_provider.py`**: `SecretsProvider` + env/file implementation. Secrets manager NOT implemented.
- **`app/api/connectors.py`** (+ `main.py`, `permissions.py`): admin-only CRUD, enable/disable, test, run-now, runs. New `connectors:read` permission; `connectors:write` now has routes.
- **`app/database/queries.py`**: connector/ledger/run queries, all with required keyword-only `tenant_id`.
- **`app/jobs/worker.py`**: the poll loop also ticks the connector scheduler every `CONNECTOR_TICK_SECONDS` (not in `--once`).
- **Ops**: `docker-compose.yml` (`gcs-emulator`, connector env, secrets mount), `.env.example`, `.gitignore`/`.dockerignore` (`secrets/`).
- **Dependencies** (all needed): `google-cloud-storage==3.16.0` (GCS client), `cronsim==2.7` (cron parser) in `requirements.txt`, `requirements-runtime.txt` (API: test-connection, schedule validation) and `requirements-worker.txt`.

## Decisions taken (as approved)
GCS (not S3); scheduler = DB-driven tick inside the worker loop (not APScheduler); queries in `queries.py`; all-POST writes. My additions: `secret_ref` is mandatory unless an emulator is configured (ADC would give every tenant the worker's cloud identity); endpoint URL is never tenant-configurable (SSRF); `run_requested` flag so run-now runs are labelled `manual`.

## Hazards from your prompt
1. **Checkpoint vs per-object identity**: ledger above; the `sensor_summary` timestamp checkpoint is untouched, no double counting (ledger + the pipeline's `ON CONFLICT`). 2. **No Principal**: `iia_worker` role + `system` principal, RLS bound per connector via `tenant_conn`. 3. **Two schedulers**: atomic lease claim + unique ledger key (both tested, negative-controlled). 4. **Secrets**: name only in DB; fixed-string errors; leak test.

## Verified (actually run; local Postgres 16, Java 21, Python 3.13, `pytest -m "not dense and not load"`)
- **Baseline** (your Phase 4 folder, minus `.env`): 443 passed, 1 failed, 10 skipped, 18 deselected. The 1 failure was `test_anomaly_rules.py::...identical_to_the_committed_pipeline_output`: `ZoneInfoNotFoundError: Asia/Calcutta`, the sandbox lacked the `tzdata` package. Not a code problem; I installed `tzdata` in my test venv (it is NOT added to requirements).
- **After Phase 5: 466 passed, 0 failed, 10 skipped, 18 deselected** (+22 new, +1 baseline test now passing with tzdata): `tests/test_connectors.py` (14), `tests/test_connectors_gcs.py` (7), `tests/test_connectors_migration.py` (1).
- Covered: happy path through a real Spark job; config validation (SSRF/extra fields/cron/secret_ref/mapping); repeated and overlapping pulls create nothing; overwritten object = one new job; one bad file among good (watermark holds before the failed file); permanent rejects don't hold the cursor; bounded retries; young files; per-run cap; bad credentials / missing secret / test-connection; disabled connector not claimed, run-now labelled manual; two schedulers claim disjoint sets, expired lease reclaimable, two concurrent pulls queue each file once and remove the loser's copy; tenant isolation (API 404s on every route, RLS as the worker role, ledgers); viewer/API key 403, anonymous 401, audited; `iia_app` cannot forge run state; planted-secret scan of logs, API responses, job errors, rejected rows, audit and every connector table; `system` audit rows; migration 0009 up/down/guards; `schema.sql` == `alembic upgrade head`.
- The real `GcsObjectStore` (google-cloud-storage) runs against an in-process fake of the GCS HTTP API: pagination, prefix, generation-exact download, 401/403/404/503 mapping, credential handling; plus real adapter + puller + Postgres together.
- **Negative controls** (code temporarily broken, test went red, code restored): logging raw exception text -> leak test red; claim without lease -> race test red; viewer given `connectors:read` -> role test red.
- Edited existing tests: `test_tenant_isolation.py` (static guard knows the 3 new tables). No other existing test changed.

## NOT verified
- `docker compose up`, the `gcs-emulator` service (image tag `latest`, unverified), building the images; fake-gcs-server and real GCS (no Docker daemon / no network to fetch it here); GitHub Actions (add the emulator as a CI service if you want it exercised there); `-m dense`, `-m load`; Python 3.12; behaviour under a real service-account key beyond building the client (no token exchange tested); the Streamlit frontend (no connector UI).
- Not a git repo here, so no commits were made.

## Known limitations
See README "Limitations" (Phase 5 bullet): GCS only; secrets stub; whole-prefix listing each run; a file first appearing with an `updated` time older than the overlap window is missed; pulls run in the worker loop; an orphaned stored copy is possible after a crash; CSV only; no per-object API.

## Heads-up
Your zip contained a `.env` with credentials, backups and `.pytest-basetemp/`; I did not copy them. Rotate anything real, and keep them out of archives you share.
