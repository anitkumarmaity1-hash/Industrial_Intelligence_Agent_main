#!/bin/sh
# Phase 2 (RLS): give the least-privilege API role `iia_app` a login password.
# Phase 3: same for the background worker's role `iia_worker`.
# Runs once on an empty data directory, after 01-schema.sql created the roles'
# grants (iia_grant_app_privileges). APP_DB_PASSWORD and WORKER_DB_PASSWORD are
# required by compose.
set -e
psql -v ON_ERROR_STOP=1 -v pw="$APP_DB_PASSWORD" -v wpw="$WORKER_DB_PASSWORD" \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
ALTER ROLE iia_app LOGIN PASSWORD :'pw';
ALTER ROLE iia_worker LOGIN PASSWORD :'wpw';
SQL
