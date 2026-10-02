# raglab

Six retrieval patterns behind one contract, with access control built into the
search layer, tamper-evident traces, and an evaluation gate that can fail.

It is a small library, not a framework. The retrieval, access-control,
verification and evaluation logic is written here and is meant to be read. It
stands on proven parts: Postgres with pgvector, sqlglot for SQL parsing, and any
OpenAI-compatible model endpoint.

## The six use cases

| # | Use case | Module | What it does |
|---|---|---|---|
| 1 | Permission-aware search | `core/store.py`, `stores/` | Every search takes a principal. The access filter is part of the query. Restricted chunks never leave the store. |
| 2 | Hybrid retrieval | `pipelines/hybrid.py` | Keyword and vector search fused by reciprocal rank fusion, superseded versions excluded, optional reranker. |
| 3 | Structured data | `pipelines/structured.py` | A model writes SQL; a parser-based guard allows only a single read-only SELECT over allow-listed tables; the database is read-only as well. |
| 4 | Agentic retrieval | `pipelines/agentic.py` | A model plans searches, reads and queries within a step limit. Its evidence is then answered and verified like any other. |
| 5 | Graph retrieval | `pipelines/graph.py` | Entities link passages across documents. Walks only pass through passages the asker may see. |
| 6 | Tables and page images | `pipelines/multimodal.py` | Table rows are indexed with their headers; optionally a vision model reads page images and its reading is verified against the page text. |

A seventh pipeline, `pipelines/naive.py`, is the baseline the others are measured
against: vector search only, access filter applied after ranking.

## What every pipeline shares

- **Five verdicts.** `answered`, `insufficient_evidence`, `access_denied`,
  `execution_failed`, `verification_failed`. A provider error is never presented
  as a responsible decline.
- **Verification.** No answer is returned unless a verifier finds every claim
  supported by what it cites.
- **Traces.** Each stage writes an event. Events are hash-chained, so editing,
  removing or reordering one is detectable. Restricted candidates appear in a
  trace as a count and nothing else; redaction happens when the trace is
  written, not when it is displayed.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                 # 85 tests, in-memory and a real Postgres with pgvector
raglab sweep              # try to make every retrieval path leak
raglab eval               # run the golden set, print the matrix and the gate
raglab ask "What does SOP-013 give as the calibration interval for balances and pipettes?" --group commercial
```

## Two modes

**Offline (no key).** A lexical hashing embedder, an extractive generator that
copies sentences, and a lexical verifier. This mode exists so the whole system
runs in tests and CI with no network. It proves the plumbing: access control,
verdict classification, tracing, the SQL guard, the gate. It does not measure
answer quality. The text-to-SQL and agentic pipelines are skipped, not faked.

**With a model.** Set the variables in `.env.example`. Embeddings, generation,
claim-level verification, SQL writing, planning and grading then use real
models.

## What has been verified, and what has not

Verified offline, by the test suite:

- 85 tests pass, with the store tests run against both the in-memory store and
  Postgres 16 with pgvector. (The Postgres tests need `pgserver`; CI runs them
  on Python 3.11.)
- The two stores return identical vector rankings, including for chunks whose
  scores tie.
- The leakage sweep returns 0 leaks in about 6,500 returned chunks, on both
  stores, using the restricted documents' own text as the queries.
- The SQL guard rejects 13 hostile statements, and the database refuses writes
  even with the guard bypassed.
- A deliberately leaky pipeline is caught by the leak metric and rejected by the
  gate.
- Adding an exact-identifier index to the Postgres keyword search raised recall
  on cross-document questions from 0.53 to 0.94, matching BM25.

Verified with real models, in three full evaluation runs over the 42 golden
questions (answers and verification by GPT-5.4 mini, grading by Claude Haiku 4.5,
embeddings by text-embedding-3-small, in-memory store; about $0.20 a run):

| Run | Gate | What it showed |
|---|---|---|
| `model-1` | rejected | A real leak: the SQL pipeline answered a restricted price from the product table (Q-40) and failed all three forbidden questions. |
| `model-2` | rejected | The leak was fixed with column-level access, but the fix over-blocked aggregate queries and SQL dropped to 5 of 6 on numeric questions. |
| `model-3` | pending | Every automated condition passes: 0 leaks, all refusals correct, hybrid 12 of 12 on lookup and version questions, SQL 6 of 6 on numeric. It waits only on a person agreeing with at least 18 of 20 sampled judge grades. |

Results from `model-3`, correct answers per question type:

| Pipeline | Lookup | Versions | Multi-hop | Numeric | Relationships | Tables | Out of scope | Forbidden | Cost | Median time |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 8/8 | 4/4 | 1/6 | 0/6 | 3/6 | 6/6 | 3/3 | 3/3 | $0.025 | 1.7 s |
| hybrid | 8/8 | 4/4 | 2/6 | 0/6 | 3/6 | 6/6 | 3/3 | 3/3 | $0.028 | 1.7 s |
| graph | 8/8 | 4/4 | 1/6 | 0/6 | 3/6 | 6/6 | 3/3 | 3/3 | $0.028 | 1.8 s |
| multimodal | 8/8 | 4/4 | 1/6 | 0/6 | 3/6 | 6/6 | 3/3 | 3/3 | $0.025 | 1.6 s |
| sql | 0/8 | 0/4 | 0/6 | 6/6 | 3/6 | 6/6 | 3/3 | 3/3 | $0.014 | 1.0 s |
| agentic | 8/8 | 4/4 | 4/6 | 5/6 | 3/6 | 6/6 | 3/3 | 3/3 | $0.079 | 4.2 s |

What the runs say:

- The agent wins multi-hop questions (4 of 6) by planning several steps, at about
  three times the cost and latency.
- SQL wins numeric questions (6 of 6); no document pipeline answers any of them.
- Lookup, version and table questions are tied across the document pipelines, so
  the corpus is too easy there to separate them. Harder questions are the next
  step.
- No pipeline gets more than 3 of 6 relationship questions right, graph
  retrieval included.
- Runs are not deterministic: SQL scored 6, then 5, then 6 of 6 across runs for
  reasons unrelated to the change being tested. Compare runs; do not trust one.

Not yet verified:

- Page-image reading with a real vision model and a real page renderer.
- A full real-model run on the Postgres store (the runs above used the in-memory
  store; the stores' rankings are tested to match).
- The human check that would move `model-3` from pending to accepted.

The offline run is rejected by the gate. That is the expected result.

## The live service

`raglab/server.py` puts the pipelines behind a small HTTP API, used by a private
portfolio demo:

- `GET /describe` lists the pipelines and what each workspace can run.
- `POST /workspaces/{id}/files` adds a file (`.txt`, `.md`, `.pdf` or `.csv`, up
  to 2 MB, 20 per workspace). CSV files become tables the SQL pipeline can query.
- `DELETE /workspaces/{id}` clears a workspace.
- `POST /ask` runs a question through the chosen pipelines on the sample corpus or
  a workspace and returns each verdict, answer, sources, trace, time and cost.

Every route except the health check needs a bearer token (`RAGLAB_TOKEN`).
Uploaded files are held in memory per workspace and are lost on restart; their
text is sent to the configured model provider. The `Dockerfile` builds the
service image.

```bash
pip install -e ".[server]"
RAGLAB_TOKEN=change-me uvicorn raglab.server:app --factory --port 8080
```

## The corpus

`corpus/` holds a synthetic document set for an invented company: 60 documents,
four relational tables and 42 questions in eight types, two of which expect a
refusal. Everything in it is invented. `corpus/generate-corpus.mjs` rebuilds it
from a fixed seed and `corpus/check-corpus.mjs` checks it independently.

## Using it on your own data

Documents are markdown with front matter (`id`, `title`, `groups`, `status`).
Point `build_lab` at a folder with `documents/` and `data/`, or assemble
`Components` and the pipelines you want directly; `raglab/build.py` is 100 lines
and shows how.

See `docs/decisions.md` for the design decisions and the alternatives considered.
