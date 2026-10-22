# Phase 2 changes - database-level tenant isolation (Postgres RLS)

## What changed
- **Migration 0006** (+ `sql/schema.sql` mirror): `ENABLE` + `FORCE ROW LEVEL SECURITY` and one `tenant_isolation` policy (`USING` and `WITH CHECK`, `tenant_id = NULLIF(current_setting('app.tenant_id', true), '')`) on the 10 tenant-scoped tables and on every existing partition of the 3 partitioned ones. Unset or empty setting = zero rows and failed writes (fail closed). Downgrade removes policies, grants and functions; it leaves the NOLOGIN role (roles are cluster-wide).
- **Inventory.** Tenant-scoped: `machines`, `sensor_summary`*, `machine_anomalies`*, `maintenance_records`, `sensor_registry`, `sensor_readings`*, `tenant_settings`, `tenant_schema_mappings`, `investigation_audit_log`, `ingestion_checkpoints` (* partitioned). Outside RLS: `tenants`, `ai4i_reference`, `alembic_version`.
- **Roles.** `iia_app`: NOLOGIN until `scripts/provision_app_role.py` sets LOGIN + password from required `APP_DB_PASSWORD`; no superuser, no BYPASSRLS, owns nothing. Grants: `SELECT` on 7 tenant tables + `ai4i_reference`, `INSERT` on `investigation_audit_log` (+ its sequence), nothing on `tenants`, nothing on partitions. All grants live in idempotent `iia_grant_app_privileges()`.
- **Login lookup** (hazard 1): `tenants` stays outside RLS but the app role cannot read it; `auth_lookup_tenant(hash)` is a `SECURITY DEFINER` function (pinned `search_path`) returning `tenant_id, name` of an active match only.
- **Tenant context** (hazards 2, 3): `app/database/tenant_context.py`. `bind_tenant()` stores the tenant on the request's `Connection` (execution option); an engine `begin` listener runs `set_config('app.tenant_id', %s, true)` (bound parameter, transaction-local) at the start of every transaction, so the commit inside `routes._audit()` is covered and nothing survives on a pooled connection. Called from `get_tenant` only (demo posture and legacy `API_KEY` path included). `routes.py` and `queries.py` unchanged.
- **Config / startup**: `APP_DATABASE_URL` (API only; falls back to `DATABASE_URL`), `RLS_REQUIRED` (default on for a remote DB). If the API's role bypasses RLS it warns locally and refuses to start when required. `/readyz` no longer reads `tenants`.
- **Partitions** (hazard 5): `iia_enable_tenant_rls(regclass)`; `manage_partitions.py` calls it for every new partition.
- **Backup/restore** (hazard 6): `restore_db.py` keeps `--no-privileges` and then re-runs `iia_grant_app_privileges()`. Roles used: alembic, seed load, backup, restore, `manage_*`, loaders and test fixtures all use `DATABASE_URL` (owner/superuser) on purpose; only the API uses `APP_DATABASE_URL`.
- **Compose / CI**: compose requires `APP_DB_PASSWORD`, mounts `docker/03-app-role.sh`, API gets `APP_DATABASE_URL` only; CI generates a random app-role password per run, provisions the role and exports `APP_DATABASE_URL`, so every API test runs under RLS. New `.env.example` (the zip had none).

## Verified (actually run)
Postgres 16.15 (local, superuser `iia`), Python 3.13, pyspark 3.5.3, Java 17 present. `APP_DATABASE_URL` pointed at `iia_app`, so all TestClient/API tests ran under RLS:

`pytest -m "not dense and not load"`: **388 passed, 0 failed, 10 skipped, 18 deselected** (Phase 1 baseline was 358 passed, so +30: 29 in `tests/test_rls.py` incl. the partition, pooling, mid-request-commit and HTTP-as-app-role cases, +1 migration 0006 upgrade/downgrade round trip).
- Skipped (10): 6 need `google.genai`, 4 need `pinecone` (not installed; same 10 as Phase 1).
- Deselected (18): `dense` (live Pinecone/Vertex) and `load`.
- `alembic upgrade head` vs `sql/schema.sql`: identical columns, indexes, constraints, RLS flags, policies, function bodies and ACLs (also compared by hand; only `alembic_version` differs).
- Negative control: `ALTER TABLE machines DISABLE ROW LEVEL SECURITY` makes 11 `test_rls.py` tests fail; re-enabling restores 29/29.
- Manual backup -> restore into a scratch DB: `iia_app` regained access and RLS stayed forced.
- `EXPLAIN` as `iia_app` with tenant context: per-machine `sensor_summary` lookup still uses the composite index.

## Existing tests changed
`test_security_defaults.py` (compose assertion now checks `APP_DB_PASSWORD:?`; readyz case also unsets `APP_DATABASE_URL`), `test_tenant_isolation.py` (cleanup list, `TENANT_TABLES` now all 10), `test_migrations.py` (also compares RLS state, adds 0006 round trip). Static query check still passes.

## NOT verified
`docker compose up` (the init script and compose wiring were not run; only YAML parsing in a test); GitHub Actions run of the new CI steps; `-m load` and `-m dense`; non-superuser owner role (see limitations); Python 3.12 (sandbox used 3.13); a managed Postgres without `CREATEROLE`.

## Provenance note
The Phase 2 files were already present in the project folder when I started this session (modified Oct 3, 17:21-17:27 UTC) and were not written in it. I reviewed every file and verified them as above; no pre-Phase-2 baseline could be re-run on that folder.

## Limitations (also for README)
- RLS does not cover: Pinecone namespaces, BM25 chunk files on disk, object storage, logs, backups (dumps contain every tenant), or any superuser/BYPASSRLS connection.
- `FORCE` also binds a non-superuser table owner. Seed load (`DISABLE TRIGGER ALL`), `pg_dump` and admin scripts therefore need a superuser or BYPASSRLS role; future data migrations touching tenant tables need the same.
- Role creation needs `CREATEROLE` or superuser.
- Batch scripts use the owner role and do not set tenant context, so RLS does not guard them against a wrong `tenant_id` stamp.
- Foreign-key checks bypass RLS (standard Postgres behaviour); composite keys keep them tenant-consistent.