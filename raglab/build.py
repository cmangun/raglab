"""Assemble a working lab from a corpus folder and whatever credentials are available.

With no API key everything runs offline on the hashing embedder, the extractive
generator and the lexical verifier. The pipelines that cannot work without a
model (text to SQL, agentic) are left out rather than faked.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .core.embed import HashingEmbedder, OpenAICompatEmbedder
from .core.generate import ExtractiveGenerator, LLMGenerator
from .core.ingest import build_chunks, load_tables
from .core.llm import LLM, OpenAICompatLLM
from .core.types import Chunk
from .core.verify import LexicalVerifier, LLMVerifier
from .eval.metrics import Judge, KeyFactJudge, LLMJudge
from .pipelines.agentic import AgenticPipeline
from .pipelines.base import Components, Pipeline
from .pipelines.graph import GraphPipeline, KnowledgeGraph, PatternExtractor, gazetteer_from_tables
from .pipelines.hybrid import HybridPipeline
from .pipelines.multimodal import MultimodalPipeline
from .pipelines.naive import NaivePipeline
from .pipelines.structured import SqliteDatabase, StructuredPipeline
from .stores.memory import MemoryStore


@dataclass
class Lab:
    comp: Components
    pipelines: dict[str, Pipeline]
    chunks: list[Chunk]
    tables: dict[str, list[dict]]
    graph: KnowledgeGraph
    judge: Judge
    record: dict
    skipped: dict[str, str] = field(default_factory=dict)


SQL_NOTES = (
    "Notes: deviation.opened_date and batch.mfg_date are ISO dates in 2025. deviation.closed_date is NULL while a deviation is open. "
    "batch.status is one of released, rejected, hold. deviation.severity is minor or major. "
    "site.id is a short code (TC, MB, EP); join to site to filter by site name. Join to product to filter by product name."
)


def build_lab(corpus: str | Path = "corpus", *, dsn: str | None = None, api_key: str | None = None, model: str | None = None,
              embed_model: str | None = None, judge_model: str | None = None, vision_model: str | None = None,
              base_url: str | None = None, k: int = 8, reveal_restricted_existence: bool = True, llm: LLM | None = None) -> Lab:
    corpus = Path(corpus)
    api_key = api_key or os.environ.get("RAGLAB_API_KEY")
    base_url = base_url or os.environ.get("RAGLAB_BASE_URL", "https://openrouter.ai/api/v1")
    model = model or os.environ.get("RAGLAB_MODEL")
    embed_model = embed_model or os.environ.get("RAGLAB_EMBED_MODEL")
    judge_model = judge_model or os.environ.get("RAGLAB_JUDGE_MODEL")
    vision_model = vision_model or os.environ.get("RAGLAB_VISION_MODEL")
    dsn = dsn or os.environ.get("RAGLAB_DSN")

    chunks = build_chunks(corpus / "documents")
    tables = load_tables(corpus / "data")
    manifest = json.loads((corpus / "manifest.json").read_text()) if (corpus / "manifest.json").exists() else {}

    embedder = OpenAICompatEmbedder(api_key, embed_model, base_url) if (api_key and embed_model) else HashingEmbedder()
    vectors = embedder.embed([c.embedding_text() for c in chunks])
    if dsn:
        from .stores.postgres import PostgresStore

        store = PostgresStore(dsn, dim=vectors.shape[1])
    else:
        store = MemoryStore()
    if store.count() == 0:
        store.add(chunks, vectors)

    if llm is None and api_key and model:
        llm = OpenAICompatLLM(api_key, model, base_url)
    generator = LLMGenerator(llm) if llm else ExtractiveGenerator.from_chunks(chunks)
    verifier = LLMVerifier(llm) if llm else LexicalVerifier()
    comp = Components(store=store, embedder=embedder, generator=generator, verifier=verifier, k=k, reveal_restricted_existence=reveal_restricted_existence)

    graph = KnowledgeGraph(chunks, PatternExtractor(gazetteer_from_tables(tables)))
    vision = OpenAICompatLLM(api_key, vision_model, base_url) if (api_key and vision_model) else None
    pipelines: dict[str, Pipeline] = {
        "naive": NaivePipeline(comp),
        "hybrid": HybridPipeline(comp),
        "graph": GraphPipeline(comp, graph),
        "multimodal": MultimodalPipeline(comp, vision_llm=vision, renderer=None),
    }
    skipped: dict[str, str] = {}
    if llm:
        structured = StructuredPipeline(llm, SqliteDatabase(tables), LexicalVerifier(), notes=SQL_NOTES)
        pipelines["sql"] = structured
        pipelines["agentic"] = AgenticPipeline(comp, llm, structured=structured, graph=graph)
    else:
        skipped = {"sql": "needs a model to write SQL", "agentic": "needs a model to plan"}

    judge: Judge = LLMJudge(OpenAICompatLLM(api_key, judge_model, base_url)) if (api_key and judge_model) else KeyFactJudge()
    record = {
        "corpus_version": manifest.get("corpus_version"), "chunks": len(chunks), "embedder": embedder.name, "store": store.name,
        "generator": generator.name, "verifier": verifier.name, "answer_model": llm.name if llm else None, "k": k,
        "temperature": 0.0 if llm else None, "reveal_restricted_existence": reveal_restricted_existence,
        "graph": graph.stats(), "skipped_pipelines": skipped,
    }
    return Lab(comp, pipelines, chunks, tables, graph, judge, record, skipped)
