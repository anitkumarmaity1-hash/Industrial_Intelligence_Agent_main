# Streaming / incremental ingestion — design notes

Production-readiness fix 17. The audit's finding: *"Continuous machine
telemetry — not supported at all today. The pipeline is batch-only."*

This document covers two things: what's actually built today
(`scripts/incremental_load.py`), and what a genuinely continuous path
would look like if telemetry volume ever justifies it. They're presented
together because the second is **not** a rewrite of the first — it's the
same checkpoint, read by a different producer.

## What exists today: checkpointed micro-batch

`scripts/incremental_load.py` reads `ingestion_checkpoints`
(`tenant_id`, `source`, `last_window_end`), loads only `sensor_summary`
rows newer than that checkpoint from the pipeline's Parquet output,
inserts with `ON CONFLICT (tenant_id, machine_id, window_start) DO
NOTHING` (idempotent — a crash mid-run and a naive re-run from an older
checkpoint both land on the same end state, proven by
`tests/test_incremental_load.py`), and advances the checkpoint.

Run it on a schedule — cron, a scheduled CI job, Airflow, whatever the
deployment already has — each time `scripts/run_pipeline.py` produces new
Parquet output. This closes the actual gap for most of the companies the
onboarding layer (Part 4 of the audit) is meant to support: an hourly
cron reading a company's sensor feed and writing Parquet, followed by
this script, is "continuous" for every practical purpose at hourly rollup
granularity, without taking on Kafka/Spark Structured Streaming
operational cost that nothing in this project's actual data volumes
currently justifies — the same "do not over-engineer" principle the
audit applied to Kubernetes/Redis/Celery applies here too.

## What would change at real streaming volume

If a company's telemetry arrives continuously (not "a file shows up every
hour" but literally streaming), the right design is:

```
Machine sensors --> message broker (Kafka / Pub/Sub / Kinesis)
                        |
                        v
          Spark Structured Streaming job
          (windowed aggregation, same hourly rollup
           logic spark_jobs/feature_engineering.py
           already has — the aggregation itself does
           not change, only how it's triggered)
                        |
                        v
          foreachBatch sink -> the SAME insert-with-
          ON-CONFLICT-DO-NOTHING statement
          incremental_load.py already uses, keyed by
          Spark's own micro-batch watermark instead of
          ingestion_checkpoints
                        |
                        v
                 sensor_summary (unchanged)
```

Concretely, this reuses almost everything already built:

- **The aggregation logic** (`spark_jobs/feature_engineering.py`'s
  windowing) doesn't change — Structured Streaming runs the same
  transformations, triggered by arriving data instead of a batch run.
- **The target schema and idempotency key** don't change —
  `sensor_summary`'s `UNIQUE(tenant_id, machine_id, window_start)` is
  exactly the key a streaming `foreachBatch` sink would upsert on.
- **The partitioned table layout** (production-readiness fix 13) is
  already shaped for this: new windows land in whichever time partition
  they belong to, and `scripts/manage_partitions.py` already provides the
  "create tomorrow's partition before tomorrow's data arrives" operation
  a streaming job depends on.
- **What's genuinely new**: a message broker, a long-running Spark
  job (vs. the current scheduled batch job), and watermarking/late-data
  handling (a reading that arrives after its window has already been
  aggregated and written — Structured Streaming's watermark mechanism
  handles this; the current batch pipeline doesn't need to, because a
  batch run only ever sees data that has already fully arrived).

## Why this isn't built now

Two reasons, both concrete rather than "not needed yet" hand-waving:

1. **No current source produces streaming data.** The synthetic
   generator, the onboarding layer, and every example company schema in
   the audit all describe periodic exports (a CSV, a Parquet drop), not
   a live sensor feed. There is nothing to stream from yet.
2. **The migration path is cheap precisely because the checkpoint
   abstraction is already in place.** `ingestion_checkpoints` doesn't
   need to change shape to support a streaming watermark instead of a
   batch-run timestamp — only what writes to it changes. Building the
   Kafka/Structured-Streaming path now, before there's a real streaming
   source to point it at, would mean maintaining untested infrastructure
   against a hypothetical.

The trigger to revisit this: a pilot company whose integration is
genuinely push-based (a webhook, an MQTT feed) rather than file-drop
based. At that point, the broker + Structured Streaming job described
above is the next piece to build — everything downstream of it already
exists.
