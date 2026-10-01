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
pytest -q                 # 65 tests, in-memory and a real Postgres with pgvector
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

Verified by running it:

- 65 tests pass, with the store tests run against both the in-memory store and
  Postgres 16 with pgvector.
- The two stores return identical vector rankings.
- The leakage sweep returns 0 leaks in about 6,500 returned chunks, on both
  stores, using the restricted documents' own text as the queries.
- The SQL guard rejects 13 hostile statements, and the database refuses writes
  even with the guard bypassed.
- A deliberately leaky pipeline is caught by the leak metric and rejected by the
  gate.
- Adding an exact-identifier index to the Postgres keyword search raised recall
  on cross-document questions from 0.53 to 0.94, matching BM25.

Not yet verified, because it needs a provider key:

- `OpenAICompatLLM` and `OpenAICompatEmbedder` against a live endpoint.
- `LLMGenerator`, `LLMVerifier` and `LLMJudge` output quality.
- The text-to-SQL and agentic pipelines with a real model (their control flow is
  tested with a scripted model).
- Page-image reading with a real vision model and a real renderer.
- Any comparison between pipelines. Offline, the embedder is lexical, so vector
  and keyword search behave alike and the baseline looks as good as hybrid
  retrieval. No conclusion about which pattern is better should be drawn until
  a run with real embeddings exists.

The offline run is rejected by the gate. That is the expected result.

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
