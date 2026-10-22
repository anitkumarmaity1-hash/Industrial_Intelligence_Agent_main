# Phase 6 changes - dashboard, end-to-end demo, README

## What changed
- **Migration 0010** (+ `sql/schema.sql`): `investigations` (tenant-scoped, RLS ENABLEd + FORCEd, result JSONB, `created_by` / `created_by_label`, two indexes); `iia_grant_app_privileges()` gives `iia_app` SELECT/INSERT on it and a column-limited UPDATE on `sensor_registry` (`normal_min`, `normal_max`, `z_score_threshold`, `enabled`). Downgrade refuses if investigations exist. `investigation_audit_log` is untouched.
- **API** (`app/api/dashboard.py`, wired in `app/main.py`, declared in `app/core/permissions.py`): `GET /fleet/summary` (paginated, `limit` <= 200), `GET /machines/{id}/series` (SQL-bucketed readings + registry range, <= 1000 points, window <= 31 days), `GET /investigations` (paginated summaries) and `GET /investigations/{id}`, `GET /units`, `POST /sensors/{name}` (admin; `registry:write`, audited). The investigation reads reuse `investigate:run`, so the role matrix is unchanged. 11 new query functions in `app/database/queries.py`, all with required keyword-only `tenant_id`.
- **`/investigate` and `/chat`** now also store the result (`routes._store_investigation`, best-effort like `_audit`, result capped at 64 KB).
- **Dashboard** (`frontend/`): `streamlit_app.py` (entry, role-based navigation), `api_client.py` (JWT, uniform errors, token-keyed 20 s cache, 401 -> sign-out), `navigation.py`, `ui.py`, `views/{login,fleet,investigate,jobs,upload,admin}.py`. The sidebar API-key box and the `API_KEY` env var of the frontend are gone (JWT only).
- **Demo**: `scripts/seed_demo.py` (idempotent: Company B tenant, admin + viewer, demo-tenant admin, sample upload through the API), `sample_data/company_b/{readings.csv,mapping.json}` (copies of `demo_readings.csv` / `demo_mapping.json`), compose `seed` service, `Dockerfile` copies the seed script + sample data, `.env.example` gains `DEMO_ADMIN_PASSWORD` / `DEMO_VIEWER_PASSWORD`. `scripts/e2e_demo.py` runs the whole flow with assertions against any API URL.
- **Fixes to claims**: API version 0.7.0 -> 0.6.0 and `/` returns `version` instead of `"phase": 6`; stale phase labels in `main.py`, compose header and the frontend docstring; README sections for architecture, real-vs-stubbed, demo walkthrough, dashboard, design decisions, and corrected limitations (minio service, "no route" claims, test counts).
- **No new dependencies.** (AppTest ships with Streamlit 1.64; Altair comes with Streamlit; pandas was already in `requirements-runtime.txt`.)

## Decisions taken (as approved)
D1 tenant-wide investigation history; D2 `st.navigation` + `frontend/views/`; D3 JWT only in the UI (the API still accepts API keys for machine clients); D4 token-keyed 20 s read cache; D5 seed as an idempotent script + one-shot compose service; D6 version 0.6.0, `phase` key dropped; D7 no separate scheduler container (the worker loop ticks the scheduler). My additions: `days` parameter on `/series`; the sample file is a copy, the originals stay at the repo root.

## Verified (actually run)
Environment: local Postgres 16, Java 21, Python 3.13, `pytest -m "not dense and not load"`.

| Check | Result |
|---|---|
| Baseline before any change | 466 passed, 0 failed, 10 skipped, 18 deselected |
| After Phase 6 | **490 passed, 0 failed, 10 skipped, 18 deselected** (+24: 13 `test_dashboard_api.py`, 1 `test_dashboard_migration.py`, 9 `test_frontend.py`, 1 `test_e2e_demo.py`) |
| Skipped (10) | same as baseline: 6 need `google.genai`, 4 need `pinecone` |
| Migration 0010 | up/down/up, downgrade guard, grants, RLS flags; `alembic upgrade head` == `sql/schema.sql` (`test_alembic_head_matches_schema_sql` passes) |
| Fleet summary | pagination, total, HIGH/LOW/UNKNOWN risk, tenant isolation |
| Series | default window ends at newest reading, `max_points`, 31-day cap (422), unknown machine/sensor (404), another tenant's data never returned |
| Investigations | stored from `/investigate` and `/chat`, listed newest first, filter/pagination, other tenant gets empty list / 404, oversize result truncated, RLS as `iia_app` (zero rows with no tenant, cannot update/delete) |
| Registry editor | admin edit + clear to null, audited, validation (422), viewer/API key 403, tenants separate, `iia_app` cannot change `unit` |
| Dashboard (AppTest, real API) | login, bad password, sign-out, invalid token -> sign-in with notice, viewer vs admin pages/controls (no upload/admin, no Retry), viewer reads fleet and runs an investigation, admin edits a sensor, upload wizard happy path (file -> preview -> map -> validate -> confirm -> job row) and a bad mapping blocked |
| End to end, in-process | `tests/test_e2e_demo.py`: create two tenants, admin login, upload Company B CSV, job succeeds through the real worker + Spark (7,776 readings), fleet shows 3 machines with 48 anomalies, series returned, investigation stored, tenant B sees nothing, viewer cannot upload |
| Whole stack as processes | fresh DB: `alembic upgrade head`, seed SQL, `provision_app_role.py`, `uvicorn`, `python -m app.jobs.worker`, `scripts/seed_demo.py` (twice: second run skips), the queued job succeeded (7,776 rows), viewer login sees PUMP-A101/A102 MEDIUM, A103 LOW; `default` admin sees 18 machines; `python scripts/e2e_demo.py --api http://localhost:8000` printed `E2E OK`; `streamlit run` served `/_stcore/health` ok and HTTP 200 |
| Compose file | `docker compose --env-file ... config` parses and lists postgres, api, frontend, gcs-emulator, seed, worker |

Existing tests edited (cleanup lists only, because `/investigate` now leaves a row): `test_auth.py`, `test_company_b.py`, `test_rls.py`, `test_hardening.py`, `test_tenant_isolation.py` (also: `investigations` added to the static-check table list).

## NOT verified
- **`docker compose up` from a clean clone.** There is no Docker daemon in this sandbox, so no image was built and no container started: the `seed` service, the frontend image's imports (`frontend/views`), the `Dockerfile` COPY of the seed script, the `service_healthy` ordering and `gcs-emulator:latest` are checked only by `compose config` and by running the same programs as processes. To verify on your machine: `cp .env.example .env`, fill the required values, `docker compose up --build`, then `docker compose logs seed`.
- The connector UI against a running emulator; `-m dense`, `-m load`; Python 3.12 (ran 3.13); GitHub Actions; a real browser session (AppTest drives the script, not a browser); behaviour with a large tenant (no load test).
- The "Not a git repo" caveat again: no commits were made.

## Known limitations
See README "Limitations" (Phase 6 bullet) and "What's real vs stubbed". Notably: a page refresh signs you out; editing a sensor range does not recompute past anomalies; investigation history is tenant-wide and never deleted; the question parser only recognises `M-NN` ids unless the machine id field is filled (no calibrate CLI); the `seed` service holds the owner DB credential (demo only).

## Heads-up
Your zip contained a `.env`, `backups/*.dump` and `.pytest-basetemp/`. I did not read or return them (I moved `.env` aside so tests would not pick it up). Rotate anything real and keep them out of shared archives.

## Process-level run (what I executed instead of compose)
```bash
export DATABASE_URL=postgresql://<owner>@localhost:5432/<db> APP_DB_PASSWORD=... WORKER_DB_PASSWORD=... \
  APP_DATABASE_URL=postgresql://iia_app:...@localhost:5432/<db> WORKER_DATABASE_URL=postgresql://iia_worker:...@localhost:5432/<db> \
  JWT_SECRET=<32+ chars> JWT_TTL_MINUTES=60 STORAGE_LOCAL_DIR=/tmp/storage STORAGE_BACKEND=local \
  DEMO_ADMIN_PASSWORD=... DEMO_VIEWER_PASSWORD=... API_BASE_URL=http://localhost:8000
alembic upgrade head && gunzip -c sql/seed.sql.gz | psql "$DATABASE_URL" && python scripts/provision_app_role.py
uvicorn app.main:app --port 8000 &   python -m app.jobs.worker &
python scripts/seed_demo.py          && python scripts/e2e_demo.py --api http://localhost:8000
streamlit run frontend/streamlit_app.py
```
