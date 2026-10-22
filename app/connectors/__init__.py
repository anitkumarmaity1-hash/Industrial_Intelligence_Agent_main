"""Scheduled pull connectors (Phase 5): fetch new files from a customer's bucket and run
them through the same upload/job pipeline as a manual upload.

  base.py        ObjectStore interface + secret-free error types
  gcs.py         the one implementation: Google Cloud Storage
  validation.py  config / glob / cron validation
  puller.py      one pull of one connector (ledger, dedupe, upload+job creation)
  scheduler.py   claim due connectors (atomic, multi-process safe) and run them

Credentials are never stored in the database: a connector holds only `secret_ref`, a name
resolved by app.core.secrets_provider at run time.
"""
