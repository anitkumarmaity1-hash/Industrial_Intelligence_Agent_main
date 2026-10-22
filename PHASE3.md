# Phase 3 — RAG layer (maintenance knowledge base)

## What was built

```
documents/*.md                    7 synthetic knowledge-base documents
app/core/config.py                env-driven settings (paths, backend, Pinecone, Vertex)
app/rag/documents.py              loader + frontmatter parsing + strict validation
app/rag/chunking.py               header-aware Markdown chunking -> Chunk -> JSONL
app/rag/embeddings.py             Vertex AI embeddings via the google-genai SDK
app/rag/vector_store.py           Pinecone index management, upsert, filtered query
app/rag/lexical.py                BM25 index (no dependencies, offline backend)
app/rag/retriever.py              one search() contract, two backends, tool entrypoint
app/rag/evaluation.py             10-case relevance set + hit rate / MRR metrics
scripts/ingest_documents.py       load -> validate -> chunk -> JSONL (-> Pinecone)
scripts/verify_dense_retrieval.py dense vs lexical comparison, exit code on regression
tests/test_rag.py                 53 tests, all passing offline
tests/test_rag_dense.py           17 tests, skipped unless credentials are configured
```

## Pipeline

```
documents/*.md
  -> load_documents()        frontmatter validated; unknown doc_type / missing field / duplicate doc_id all raise
  -> chunk_documents()       MarkdownHeaderTextSplitter (h1/h2/h3) then RecursiveCharacterTextSplitter
  -> save_chunks()           data/processed/document_chunks.jsonl  (62 chunks)
  -> [dense path]  VertexEmbedder.embed_documents() -> PineconeVectorStore.upsert_chunks()
  -> [local path]  BM25Index built at query time from the same JSONL
  -> search_maintenance_documents(query, top_k, doc_type, failure_mode) -> [RetrievedChunk]
```

`search_maintenance_documents()` is the tool contract the Phase 5 LangGraph
agent will wrap. It is defined and tested here, before the agent exists.

## Knowledge base

| doc_id  | type            | covers |
|---------|-----------------|--------|
| SOP-101 | sop             | tool wear monitoring / replacement (TWF, OSF) |
| SOP-102 | sop             | heat dissipation and cooling (HDF) |
| SOP-103 | sop             | power delivery and drivetrain (PWF) |
| MAN-200 | manual          | operating envelope, failure modes, reference values |
| SAF-300 | safety          | isolation, lockout/tagout, safe access |
| TRB-400 | troubleshooting | symptom -> candidate causes -> discriminating evidence |
| INC-500 | incident_report | 5 historical incidents (2025), predating the sensor window |

Every document declares `data_class: SYNTHETIC` in frontmatter, repeats the
disclaimer in its body, and a test asserts no document can be loaded without it.
Thresholds (8.6 K, 1380 rpm, 3500-9000 W, 200-240 min wear, 11000/12000/13000
min·Nm overstrain, 150 min routine replacement) are taken from
`spark_jobs/config.py`, so the knowledge base and the anomaly rules agree.

Chunk stats: 62 chunks, min 136 / median 412 / max 895 characters.

## Two backends behind one contract

| | `DenseRetriever` | `LexicalRetriever` |
|---|---|---|
| backend_name | `pinecone-vertex` | `local-bm25` |
| needs | GCP project + Pinecone key + network | nothing |
| strength | paraphrase, synonyms | exact technical terms |
| used by | production / demo | default, CI, all tests |

`RAG_BACKEND=auto` picks dense when both credentials exist, otherwise lexical.
If the dense path fails to construct (missing SDK, bad credentials, unreachable
index) `build_retriever()` logs a WARNING and degrades to BM25 rather than
failing startup — an agent that cannot answer anything is worse than one
answering with a weaker retriever, provided the degradation is observable.
`backend_name` is on every result, so it is.

## Verified vs unverified

Verified in the sandbox:
- ingestion end to end, 7 documents -> 62 chunks
- 51 tests passing, including 10 query-to-document relevance assertions
- Pinecone 10.0.0 and google-genai 2.24.0 import surfaces and method signatures

NOT verified — no network egress to Google or Pinecone from the dev sandbox:
- any live embedding call
- any live Pinecone index creation, upsert or query

## Findings

**1. The old Vertex embedding SDK is past its removal date.**
`vertexai.language_models.TextEmbeddingModel` (from `google-cloud-aiplatform`)
warns at import: deprecated 2025-06-24, removal 2026-06-24. `embeddings.py`
uses the Google Gen AI SDK (`google-genai`, `Client(vertexai=True, ...)`)
instead. Phase 6 should use the same client for Gemini.

**1b. The embedding model matters too.** `text-embedding-005` is legacy;
Google positions `gemini-embedding-001` as the unified replacement for it and
for `text-multilingual-embedding-002` (`text-embedding-004` is already EOL).
The default is now `gemini-embedding-001` at 768 dimensions (a Matryoshka
truncation of its native 3072). One catch that will break naive code:
gemini-embedding-001 accepts **exactly one input text per request**, unlike
the 250-instance limit of other models. `max_batch_size_for()` clamps the
batch size to 1 for it automatically, so 62 chunks means 62 sequential
requests — a few seconds, and a fraction of a cent.

**2. BM25 scores do not support a relevance cutoff.**
"quarterly revenue forecast marketing budget" scores 5.1 against MAN-200
(the word "quarterly" appears in its maintenance-interval table) versus 8.2
for a genuinely on-topic query. That margin is too narrow for an absolute
threshold, so none is implemented and the behaviour is pinned by a test.
Consequence for Phase 5/6: retrieval returning something is weaker evidence
than documentation supporting a claim. Scores must reach the reasoning layer,
and the agent must be able to answer "no applicable procedure found".

**3. Asymmetric task types.** Vertex embedding models require
RETRIEVAL_DOCUMENT for passages and RETRIEVAL_QUERY for queries. Getting this
wrong degrades ranking silently, so it is asserted in tests.

## Evaluation

`app/rag/evaluation.py` holds ten relevance cases — a plausible engineer
question plus the document that should answer it — and computes hit rate @ k,
strict per-case pass rate, MRR and mean latency. The `within_top` values
record *measured* BM25 behaviour, not aspiration: two cases legitimately rank
at 2 and 3, with the reason noted inline.

The same set is asserted case by case in `tests/test_rag.py` (lexical) and
`tests/test_rag_dense.py` (dense). `scripts/verify_dense_retrieval.py` runs
both and prints a per-case comparison, exiting 1 on any case the baseline
passes and dense does not.

## Running it

```bash
pip install -r requirements.txt
python scripts/ingest_documents.py          # writes the chunk JSONL
pytest tests/test_rag.py                    # 51 tests, offline

# dense path
pip install -r requirements-cloud.txt
gcloud auth application-default login
# set PINECONE_API_KEY and GOOGLE_CLOUD_PROJECT in .env
python scripts/ingest_documents.py --backend pinecone --recreate
python scripts/verify_dense_retrieval.py    # dense vs lexical, exit 1 on regression
pytest tests/test_rag_dense.py -v
```

Cost for one full ingestion: 62 chunks of roughly 400 characters each, well
under 10k tokens total. At Vertex embedding prices this rounds to a fraction
of one rupee. The expensive mistake is re-ingesting in a loop, not ingesting.

## Open for Phase 4

- `search_maintenance_documents()` needs a FastAPI endpoint and Pydantic
  response models; `RetrievedChunk` is the natural response shape.
- The retriever is built once per process (`get_retriever()`), so it should be
  wired to FastAPI startup rather than built on first request.
