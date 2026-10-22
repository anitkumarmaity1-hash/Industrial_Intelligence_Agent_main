# Phase 3 changes - upload, mapping, jobs, background worker

## What changed
- **Migration 0007** (+ `sql/schema.sql`, generated from the migration's own SQL): `uploads`, `jobs`, `rejected_rows` with RLS; `iia_claim_job()` / `iia_reap_stale_jobs()` (SECURITY DEFINER, ids only); new role `iia_worker` (grants in `iia_grant_app_privileges()`, so `restore_db.py` is unchanged); `UNIQUE (tenant_id, machine_id, detected_at)` on `machine_anomalies` (without it a re-run duplicated anomalies); `machines.production_line` VARCHAR(20) -> (100). Upgrade refuses on pre-existing duplicate anomalies; downgrade refuses if uploads/jobs exist or a line is > 20 chars.
- **Storage** `app/storage/`: `LocalStorage` (default) and `S3Storage` (MinIO). Keys are `tenants/<tenant>/uploads/<uuid>/data.csv`, built server-side and re-checked on every call.
- **API** `app/api/uploads.py` (+ `include_router` in `app/main.py`): uploads, streamed content, preview, mapping (dry-run / confirm), jobs, rejected rows, retry. Reuses the Phase 1 `SensorMapping` and `NormalizationReport`.
- **Worker** `app/jobs/pipeline.py`, `app/jobs/worker.py`: Postgres queue (`SKIP LOCKED`), stages validate/map/normalize/load/detect/store, backoff retries, stale-job reaping, tenant-safe error messages.
- **Queries** (all keyword-only `tenant_id`): upload/job/rejected-row CRUD, `bulk_insert_sensor_readings`, `upsert_machines_keep_known`, `fetch_readings_for_detection`, `upsert_sensor_summary`, `upsert_machine_anomalies`.
- **Ops**: `Dockerfile.worker`, `requirements-worker.txt`, `docker-compose.yml` (`minio`, `worker`; new required vars `WORKER_DB_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`), `docker/03-app-role.sh`, `scripts/provision_app_role.py` (+`WORKER_DB_PASSWORD`), `.env.example` (the zip had none; this one is new), README section + limitations.
- **Dependencies**: `boto3` (S3/MinIO provider; imported lazily) in `requirements.txt`, `requirements-runtime.txt`, `requirements-worker.txt`; `pandas` added to `requirements-runtime.txt` (the API now previews CSVs and runs the mapping dry-run). No multipart library: the upload is a raw streamed body.

## Decisions taken (as approved)
Spark local mode in a worker image (not a pandas fork of the rules); streamed raw-body POST (not presigned PUT); Postgres queue (not RQ/Redis); separate `iia_worker` role; baseline continuity by re-reading stored readings from each machine's first uploaded timestamp plus `rolling_window_readings` earlier timestamps; summary upsert (not DO NOTHING); mapping stored on the upload row. The ingestion checkpoint (`source = sensor_summary`) is advanced as a monotonic high-water mark, not used as a row filter (uploads can back-fill).

## Bugs found and fixed while testing
- The detect stage fed Spark its columns in the wrong order (found: job produced 0 anomalies).
- `DataFrame.toPandas()` shifted every timestamp by the host's UTC offset (seen on an Asia/Calcutta host); the worker collects timestamps as UTC strings instead (`spark_to_pandas_utc`).
- Phase 1's `normalize_sensor_readings` raises `KeyError` if the mapping's optional `production_line`/`type` column is missing from the file; the API and the job now check all mapped columns first (`app/onboarding/columns.py`). Phase 1 code itself is unchanged.

## Verified (actually run; local Postgres 16, Java 17, Python 3.12, `pytest -m "not dense and not load"`)
- Baseline before any change: 388 passed, 10 skipped, 18 deselected. After: **415 passed, 10 skipped, 18 deselected, 0 failed**.
- New: `tests/test_uploads_pipeline.py` (16), `tests/test_storage.py` (10), `test_0007_*` in `tests/test_migrations.py` (1).
- Company B CSV -> mapping -> job -> readings/anomalies for that tenant only; the job's anomalies and summaries equal Phase 1's `detect()` run directly (Parquet round trip) row for row; faults land at the right absolute times.
- Bad file: exact reject counts per code, row numbers and raw rows; same upload twice leaves counts unchanged; crash after load then retry loads nothing twice; retries are bounded and the stored error never contains the exception text; a dead worker's job is reaped; two uploads split mid-fault equal one upload; tenant A gets 404 on B's uploads, jobs, rejected rows, content, retry and cannot open B's object key; RLS hides rows from both `iia_app` and `iia_worker`; limits (415/413/400/401/429).
- Edited existing tests: `test_migrations.py` (the RLS test now upgrades to 0006 for its 13-table count), `test_tenant_isolation.py` (new tables in the static-check list).
- Skipped (same 10 as baseline): `google.genai` and `pinecone` not installed.

## NOT verified
- `docker compose up`, building `Dockerfile.worker`, or any real MinIO/S3 call (no Docker daemon here; `docker compose config` validates the file; S3 provider tested against a fake client only).
- `-m dense`, `-m load`, the CI workflow on GitHub, and the Streamlit frontend (no upload UI added).
- Not a git repo here, so no commits were made; the changes are one logical set per file listed above.

## Known limitations
See README "Limitations" (Phase 3 bullet). Also: `rows_rejected` counts dropped readings/problems, not CSV rows; viewer/admin rules are Phase 4.
