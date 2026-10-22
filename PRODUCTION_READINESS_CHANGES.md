# Production-readiness fixes — what changed

All ten items from the prioritized list are implemented and covered by
tests (295 passed, 4 skipped, 17 deselected on `pytest -m "not dense"`
against a fresh `sql/schema.sql` + `sql/seed.sql.gz` load).

## 1–3. Multi-tenancy (isolation, RAG, auth)

- `sql/schema.sql`: new `tenants` table; `tenant_id` on `machines`,
  `sensor_summary`, `machine_anomalies`, `maintenance_records`, with
  composite primary/foreign keys so a child row can only reference a
  machine in its own tenant. `ai4i_reference` stays global (public
  reference data).
- `app/database/queries.py`: every function takes a required
  keyword-only `tenant_id` and filters on it. Statically checked by
  `tests/test_tenant_isolation.py::test_every_tenant_table_query_is_tenant_filtered`.
- `app/rag/retriever.py`: one retriever per tenant (own BM25 chunk file
  or Pinecone namespace `<tenant>:<base>`); a tenant with no ingested
  documents gets an empty index, never another tenant's.
- `app/api/dependencies.py::get_tenant`: the tenant comes only from
  `X-API-Key` (SHA-256 hash stored, never the key itself). No request
  field can select a tenant. `AUTH_REQUIRED` (new) enforces a key on
  every data route; unset keeps the zero-config demo posture.
- `scripts/manage_tenants.py` (new): create / rotate-key / activate /
  deactivate / list.
- `scripts/load_postgres.py`, `scripts/ingest_documents.py`: `--tenant`.
- Tests: `tests/test_tenant_isolation.py`, `tests/test_rag_tenancy.py`.

## 4. `.env.example`

Added — every env var `app/core/config.py` reads, with defaults and a
short explanation of the new security/resilience ones.

## 5. CORS

`app/main.py`: `CORS_ALLOW_ORIGINS` allow-list; no middleware installed
at all (safe default) when unset. Credentials never allowed (the API
authenticates via header, not cookies).

## 6. Rate limiting

`app/api/rate_limit.py` (new): in-process sliding window, per tenant
(per client IP for unauthenticated demo traffic), applied to
`/investigate` and `/chat` via `rate_limit_llm`. `RATE_LIMIT_PER_MINUTE`
(default 30, 0 disables). Documented limitation: per-worker, not
cluster-wide — fine for the single-container deployment this repo
ships.

## 7. LLM timeout + retry

`app/core/genai_client.py` (new): one shared `genai.Client` builder used
by synthesis (`app/agents/llm.py`), the planner
(`app/agents/planner.py`), and the embedder (`app/rag/embeddings.py`) —
`LLM_TIMEOUT_SECONDS` (30s) + bounded retry (`LLM_MAX_ATTEMPTS`, default
2) on 408/429/5xx. A failing planner call now degrades to the
rule-based evidence path instead of 500ing the whole request
(`app/agents/nodes.py`).

## 8. Planner/graph loop guard

`planner.MAX_PLANNER_TOOL_ROUNDS` already existed and is enforced
(pinned by a regression test). Added an independent backstop:
`GRAPH_RECURSION_LIMIT` (default 40) on the LangGraph invocation itself,
mapped to a clean 500 in `app/api/routes.py`.

## 9. Alembic migrations

`alembic.ini`, `migrations/env.py`, `migrations/versions/0001_baseline...`
(the original pre-tenancy schema) and `0002_multi_tenancy` (adds
tenants/tenant_id/tenant_schema_mappings, backfills existing rows into
`default`). `tests/test_migrations.py` proves `alembic upgrade head`
produces byte-identical schema to `sql/schema.sql`, and that upgrading a
populated pre-tenancy database preserves every row.

## 10. Onboarding layer

`app/onboarding/`: `mapping.py` (column-mapping + unit-conversion config,
validated against the canonical AI4I-style schema), `normalize.py`
(applies it to a raw CSV), `paths.py` (per-tenant raw-file locations, no
pyspark import). `scripts/onboard_tenant.py` (new CLI) creates the
tenant, normalizes operational/maintenance CSVs, and prints the exact
follow-up commands (`run_pipeline.py --tenant`, `load_postgres.py
--tenant`, `ingest_documents.py --tenant`) — each a separate,
inspectable step. `spark_jobs/ingestion.py` and `scripts/run_pipeline.py`
now take `--tenant` / read tenant-scoped raw and processed paths.

**Explicit scope limit:** only accepts fleets reporting AI4I-style
readings (temperature, rotational speed, torque, tool wear) under
whatever names/units the tenant uses. A genuinely different sensor
vocabulary needs `spark_jobs/config.py`'s anomaly thresholds
generalized first — flagged, not silently papered over.

## Also touched

- `frontend/streamlit_app.py`: sidebar API-key field, sends `X-API-Key`,
  surfaces 401/429 distinctly.
- `docker-compose.yml`: new env vars wired through with the same
  zero-config defaults as everything else in the stack.
- `requirements.txt`: `alembic`.
- `README.md`: new "Multi-tenancy", "Onboarding a tenant's own data",
  "Migrations" sections; updated setup/testing counts.

## Known gaps (not done)

- Rate limiting and the planner LLM timeout are not exercised against a
  *real* Vertex endpoint (no egress in this sandbox) — only unit-tested
  against fakes.
- `docker-compose.yml`'s frontend `API_KEY` is a convenience default,
  same key for every session unless overridden in the sidebar; fine for
  a single-tenant demo, not for a multi-tenant deployment sharing one
  compose stack. (The sidebar's own "API key" field does let a person
  override it per-session without touching the container.)

## Update: CI now runs the migration chain itself

`.github/workflows/tests.yml` applies the schema with `alembic upgrade
head` instead of `psql -f sql/schema.sql` — so CI exercises the actual
migration chain (`migrations/versions/0001...`, `0002...`), not just the
equivalence `tests/test_migrations.py` checks on its own throwaway
database. Verified locally by replaying the same steps (drop schema,
`alembic upgrade head`, load `sql/seed.sql.gz`, full suite): 295 passed,
4 skipped, 17 deselected.

---

# Round 2 — "NEXT 10" (items 11–20)

Everything below was verified against a real, local PostgreSQL 16
instance running the actual migration chain (`alembic upgrade head`
through `0004`) with `sql/seed.sql.gz` loaded — not just read over. Final
count: **286 passed, 10 skipped (Vertex/Pinecone creds), correctly
deselected `dense`/`load`**, plus a full 1M-row load test and a real
backup→restore drill, both run standalone and passing.

## 11. Per-tenant risk thresholds + RAG coverage floor

- New `tenant_settings` table (migration `0003`, additive — no ALTER on
  any existing table): one JSONB config row per tenant.
- `app/agents/risk.py`: the three module constants
  (`SUSTAINED_MEDIUM_RATE`, `SUSTAINED_HIGH_RATE`, `MIN_TREND_WINDOWS`)
  are unchanged as defaults, now wrapped in `RiskThresholds` /
  `DEFAULT_RISK_THRESHOLDS`. `assess_risk()` takes an optional
  `thresholds` — every existing caller that omits it gets byte-identical
  behavior.
- `app/rag/retriever.py`: `build_retriever`/`get_retriever` take an
  optional `min_coverage_override`, overriding
  `Settings.rag_min_term_coverage_lexical` for one tenant; the retriever
  cache rebuilds a tenant's retriever when its override changes.
- `app/core/tenant_settings.py` (new): `load_tenant_calibration(conn,
  tenant_id)` reads the row and fills in documented defaults for
  anything unset.
- Wired into both `/investigate` and `/chat` via a new
  `get_tenant_calibration` FastAPI dependency
  (`app/api/dependencies.py`), degrading to defaults (never 500ing) if
  the lookup fails.
- Verified live: setting a tenant's `machine_id` override changed
  `/investigate`'s actual parsing behavior for that tenant on the next
  request.

## 12. Generalize `query_parser.py` beyond `M-\d{2}`

- `app/agents/query_parser.py`: new `MachineIdScheme` dataclass (prefix,
  separators, zero-padding, optional word-pattern).
  `DEFAULT_MACHINE_ID_SCHEME` is the original hardcoded pattern,
  unchanged — confirmed via standalone regex tests before wiring
  anything else.
- `parse_question`/`resolve_target` take an optional `scheme`; every
  caller that omits it is unaffected.
- Verified parsing the audit's own Company B (`A-118`) and Company C
  (`EQ-042`) ID formats correctly with a custom scheme, live, through
  `/investigate`.
- Known limit, stated honestly in the module docstring: covers any
  "<prefix><separator><digits>" scheme; a genuinely free-form ID format
  needs a custom parser, out of scope here.

## 13. Table partitioning (`sensor_summary`, `machine_anomalies`)

- New migration `0004`: both tables recreated as native Postgres
  `PARTITION BY RANGE` (monthly-ready), with a single `DEFAULT`
  partition holding everything that exists today — see the migration's
  own docstring for why fixed calendar bounds aren't baked in (the seed
  data's timestamps are relative to generation time, not a fixed
  range).
- Both PKs move to `(partition_column, id)` since Postgres requires
  every unique constraint on a partitioned table to include the
  partition key; `id` was confirmed (grep) to be unused anywhere in
  `app/`/`scripts/` as anything but an opaque surrogate.
- `scripts/manage_partitions.py` (new): `ensure-next` / `create` /
  `list` — creates dedicated monthly partitions ahead of time.
- Both `upgrade()` and `downgrade()` are implemented and tested.
- Verified: `alembic upgrade head` → seed load → **full test suite
  passes, including `test_alembic_head_matches_schema_sql`** (migration
  output and `sql/schema.sql` are structurally identical) and
  `test_postgres.py::test_machine_window_query_uses_the_composite_index`
  (the existing index-usage assertion still holds post-partitioning).

## 14. Request/trace-ID propagation through logs

- `app/core/request_context.py` (new): a contextvar (not thread-local —
  correct under Starlette's async request handling) plus a
  `RequestIdLogFilter`.
- `app/main.py`: new middleware sets the ID (from an inbound
  `X-Request-ID` if present, generated otherwise) before anything else
  runs, echoes it back in the response header.
- `app/core/logging_config.py`: every log line now carries
  `[req=<id>]`; `-` outside any request (startup, a standalone script).
- `app/api/routes.py`'s new audit-log insert (item 19) stores the same
  ID, so one ID ties a request's logs to its audit row.
- Verified live: `X-Request-ID` header returned, confirmed present in
  both the log stream and the matching `investigation_audit_log` row
  for the same request.

## 15. Automated backups + a tested restore drill

- `scripts/backup_db.py` (new): wraps `pg_dump --format=custom`;
  `--keep N` prunes older dumps.
- `scripts/restore_db.py` (new): wraps `pg_restore`.
- `tests/test_backup_restore.py` (new): actually runs both scripts
  against the live seeded database, restores into a scratch database,
  and asserts every table's row count matches — a backup that has never
  been restored is an unverified assumption, not a backup. **Ran and
  passed** against the real (now-partitioned) schema.

## 16. Load-test the 1M-row scenario

- `tests/test_load_1m.py` (new, marked `load` — excluded from the
  default/CI-per-push run, see `pytest.ini` and the new `load-test` CI
  job on a weekly schedule / manual dispatch): bulk-generates ~1.08M
  `sensor_summary` rows via server-side SQL (`generate_series`, not a
  Python loop — measures the database, not Python/network overhead),
  then asserts `EXPLAIN` on `get_sensor_summary`'s and
  `get_fleet_status`'s actual query shapes shows an index scan, not a
  sequential scan.
- **Actually run, not just written**: 1,080,018 rows generated and
  queried in ~16s total; both index-scan assertions held, including
  against the new partitioned (item 13) schema.

## 17. Streaming/incremental ingestion path

- New `ingestion_checkpoints` table (migration `0003`): `(tenant_id,
  source)` → `last_window_end`, `rows_processed`.
- `scripts/incremental_load.py` (new): loads only `sensor_summary` rows
  newer than the checkpoint, inserts with `ON CONFLICT ... DO NOTHING`
  (idempotent — safe to re-run from an older checkpoint after a partial
  failure), advances the checkpoint.
- `docs/streaming_ingestion_design.md` (new): what a genuine streaming
  path (Kafka/Pub-Sub + Spark Structured Streaming) would look like on
  top of the same checkpoint/idempotency-key design, and why it isn't
  built now (no current source actually streams).
- `tests/test_incremental_load.py` (new, 4 tests): first-run-loads-all,
  no-op-rerun, only-newer-rows-on-a-later-run, and
  rerun-after-simulated-partial-failure-does-not-double-count. All run
  against a real scratch Postgres database and pass.
- Also manually verified against the repo's actual
  `data/processed/sensor_summary.parquet` (25,920 rows): first run
  loaded correctly, checkpoint advanced, a rewound-checkpoint re-run
  correctly re-identified the same rows via `ON CONFLICT DO NOTHING`
  with zero duplicates.

## 18. Document versioning/dedup in RAG ingestion

- `app/rag/manifest.py` (new): `content_hash` (sha256 of body),
  `find_duplicate_content` (byte-identical bodies under different
  doc_ids), and a small on-disk manifest (`<chunks file>.manifest.json`
  per tenant) diffed against each new `load_documents()` call —
  classifies `new` / `unchanged` / `updated` / `stale_version` (content
  changed but the declared version looks like it went backwards, when
  both versions parse as comparable numbers) / `removed`.
- `scripts/ingest_documents.py`: aborts on a detected `stale_version`
  (exit 1) unless `--force`; logs duplicates and additions/removals
  either way.
- `tests/test_rag_manifest.py` (new, 9 tests) — all pass.
- Verified live against the real demo knowledge base: edited
  `SOP-101-tool-wear-replacement.md`'s body and dropped its version from
  3.1 to 2.0 — ingestion correctly refused (exit 1), then correctly
  proceeded with `--force`. File restored afterward; full suite reran
  clean.

## 19. Audit logging (caller/tenant identity per investigation)

- New `investigation_audit_log` table (migration `0003`;
  `ON DELETE CASCADE` on `tenant_id` — an audit trail is scoped to its
  tenant, so it never permanently blocks that tenant's own deletion).
- `app/api/routes.py`: both `/investigate` and `/chat` write one row
  after a successful call (tenant, request ID, endpoint, machine,
  intent, risk level, auth state, question) via a new `_audit()` helper
  — best-effort (logged and swallowed on failure, never turns a
  successful investigation into a 500).
- Verified live: a real `/investigate` call produced a matching audit
  row with the correct tenant, machine, risk level and request ID.

## 20. RAG evaluation metrics visible in CI

- `tests/test_rag.py` already hard-asserted on `evaluate_retriever`'s
  local-backend report on every run — that assertion was already the
  real regression gate, not something this item needed to add.
- `scripts/report_rag_evaluation.py` (new): prints hit rate / strict
  pass rate / MRR as a Markdown table; wired into
  `.github/workflows/tests.yml` as a step appending to
  `$GITHUB_STEP_SUMMARY`, so a quiet trend (still passing, but drifting)
  is visible on every run's summary, not just when it finally crosses
  the hard threshold.
- Verified by actually running it: `hit_rate=100%, strict_pass_rate=100%,
  MRR=0.950` against the local backend, matching the numbers
  `test_rag.py` asserts.

## Known gaps / what to double-check in your own CI

- No live Vertex AI / Pinecone credentials were available in this
  sandbox, so the dense-backend path and anything gated behind
  `pinecone_configured` were not exercised beyond what was already
  skipped before this round (`-m "not dense"`).
- Migration `0004`'s `upgrade()`/`downgrade()` were run and verified
  repeatedly in this sandbox, including the full test suite both
  directions — but table partitioning is exactly the kind of change
  worth re-confirming in your own CI (which runs the identical
  `alembic upgrade head` step) before merging, as the migration's own
  docstring says.
- `scripts/incremental_load.py` currently only covers `sensor_summary`
  (the table the audit's "continuous telemetry" finding was about);
  `machine_anomalies` would need the same treatment if a tenant's
  anomaly detection also needs incremental loading later.
