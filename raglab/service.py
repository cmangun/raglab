"""Live question answering over the sample corpus or a visitor's uploaded files.

A workspace is a set of documents and tables with the pipelines built over
them. The sample workspace is the synthetic corpus, with its two access groups.
An upload workspace belongs to one visitor: everything in it is visible to its
owner and to no one else, because it is only ever searched as that owner.

Workspaces live in memory. They are rebuilt from the corpus (sample) or lost
(uploads) when the process restarts; the caller is told when a workspace is gone.
"""

from __future__ import annotations

import csv
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .build import SQL_NOTES, build_lab
from .core.embed import Embedder, HashingEmbedder, OpenAICompatEmbedder
from .core.generate import ExtractiveGenerator, LLMGenerator
from .core.ingest import chunk_text
from .core.llm import LLM, OpenAICompatLLM
from .core.text import slug
from .core.trace import TraceWriter
from .core.types import Chunk, Principal
from .core.verify import LexicalVerifier, LLMVerifier
from .pipelines.agentic import AgenticPipeline
from .pipelines.base import Components, Pipeline, run_pipeline
from .pipelines.graph import GraphPipeline, KnowledgeGraph, PatternExtractor
from .pipelines.hybrid import HybridPipeline
from .pipelines.multimodal import MultimodalPipeline
from .pipelines.naive import NaivePipeline
from .pipelines.structured import SqliteDatabase, StructuredPipeline
from .stores.memory import MemoryStore

PIPELINE_INFO = [
    {"id": "hybrid", "label": "Hybrid search", "summary": "Searches by meaning and by exact words, then merges the two. The dependable default.", "best_for": "Most questions about what a document says"},
    {"id": "agentic", "label": "Agent", "summary": "A model plans several searches, reads whole documents and queries data before answering.", "best_for": "Questions that need facts from more than one place"},
    {"id": "sql", "label": "Data (SQL)", "summary": "Writes a database query over your spreadsheets and answers from the rows.", "best_for": "Counts, totals and filters over CSV data"},
    {"id": "graph", "label": "Linked records", "summary": "Follows references between documents, such as one record citing another.", "best_for": "How records relate across documents"},
    {"id": "multimodal", "label": "Table-aware", "summary": "Indexes each table row with its column headings, so a single cell can be found.", "best_for": "Values that sit inside tables"},
    {"id": "naive", "label": "Basic search", "summary": "Meaning-only search with no extras. Included as the baseline the others are compared with.", "best_for": "Seeing what the simplest approach returns"},
]

MAX_FILES = 20
MAX_FILE_BYTES = 2_000_000
MAX_CHUNKS = 3000
MAX_QUESTION = 600
UPLOAD_GROUP = "owner"


class ServiceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Workspace:
    id: str
    comp: Components
    pipelines: dict[str, Pipeline]
    files: list[dict] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    tables: dict[str, list[dict]] = field(default_factory=dict)
    last_used: float = field(default_factory=time.time)


def _extract_text(name: str, data: bytes) -> str:
    lower = name.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        if not text.strip():
            raise ServiceError(422, f"{name} has no readable text. Scanned PDFs are not supported.")
        return text
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _csv_table(name: str, data: bytes) -> tuple[str, list[dict]]:
    text = _extract_text(name, data)
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ServiceError(422, f"{name} has no data rows.")
    table = re.sub(r"^(\d)", r"t_\1", slug(name.rsplit(".", 1)[0]).replace("-", "_")) or "data"

    def cast(v: str):
        v = (v or "").strip()
        if v == "":
            return None
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        if re.fullmatch(r"-?\d+\.\d+", v):
            return float(v)
        return v

    cols = {c: (re.sub(r"\W+", "_", c.strip().lower()).strip("_") or f"col{i}") for i, c in enumerate(rows[0].keys())}
    return table, [{cols[c]: cast(v) for c, v in r.items() if c in cols} for r in rows[:20000]]


class RagService:
    def __init__(self, corpus: str | Path = "corpus", *, api_key: str | None = None, model: str | None = None,
                 embed_model: str | None = None, base_url: str | None = None, llm: LLM | None = None, embedder: Embedder | None = None):
        api_key = api_key or os.environ.get("RAGLAB_API_KEY")
        base_url = base_url or os.environ.get("RAGLAB_BASE_URL", "https://openrouter.ai/api/v1")
        model = model or os.environ.get("RAGLAB_MODEL")
        embed_model = embed_model or os.environ.get("RAGLAB_EMBED_MODEL")
        self.llm = llm or (OpenAICompatLLM(api_key, model, base_url) if (api_key and model) else None)
        self.embedder = embedder or (OpenAICompatEmbedder(api_key, embed_model, base_url) if (api_key and embed_model) else HashingEmbedder())
        lab = build_lab(corpus, api_key=api_key, model=model, embed_model=embed_model, base_url=base_url, llm=self.llm)
        if "sql" in lab.pipelines:
            lab.pipelines["sql"].phrase = True  # live answers read as sentences; evaluation runs keep the raw result
        self.sample = Workspace("sample", lab.comp, lab.pipelines, chunks=lab.chunks, tables=lab.tables)
        self.sample_groups = sorted({g for c in lab.chunks for g in c.groups})
        self.workspaces: dict[str, Workspace] = {}
        self._pool = ThreadPoolExecutor(max_workers=6)

    # ---------------------------------------------------------------- describe
    @property
    def live(self) -> bool:
        return self.llm is not None

    def describe(self, workspace_id: str | None = None) -> dict:
        ws = self.workspaces.get(workspace_id) if workspace_id else None
        return {
            "live": self.live,
            "answer_model": self.llm.name if self.llm else "offline (extractive)",
            "pipelines": PIPELINE_INFO,
            "sample": {"documents": len({c.document_id for c in self.sample.chunks}), "tables": sorted(self.sample.tables),
                       "groups": self.sample_groups, "available": sorted(self.sample.pipelines)},
            "upload": None if ws is None else {"files": ws.files, "available": sorted(ws.pipelines)},
            "limits": {"files": MAX_FILES, "file_bytes": MAX_FILE_BYTES, "types": [".txt", ".md", ".pdf", ".csv"]},
        }

    # ------------------------------------------------------------------ upload
    def add_file(self, workspace_id: str, name: str, data: bytes) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", workspace_id):
            raise ServiceError(400, "Invalid workspace.")
        name = Path(name).name[:120] or "file"
        if not name.lower().endswith((".txt", ".md", ".pdf", ".csv")):
            raise ServiceError(415, "Upload a .txt, .md, .pdf or .csv file.")
        if len(data) > MAX_FILE_BYTES:
            raise ServiceError(413, "That file is larger than 2 MB.")
        ws = self.workspaces.get(workspace_id)
        files = list(ws.files) if ws else []
        chunks = list(ws.chunks) if ws else []
        tables = dict(ws.tables) if ws else {}
        if any(f["name"] == name for f in files):
            raise ServiceError(409, f"{name} is already uploaded.")
        if len(files) >= MAX_FILES:
            raise ServiceError(409, f"A workspace holds at most {MAX_FILES} files.")

        groups = frozenset({UPLOAD_GROUP})
        if name.lower().endswith(".csv"):
            table, rows = _csv_table(name, data)
            while table in tables:
                table += "_2"
            tables[table] = rows
            files.append({"name": name, "kind": "data", "detail": f"{len(rows)} rows, table '{table}'"})
            new_chunks: list[Chunk] = []
        else:
            new_chunks = chunk_text(name, _extract_text(name, data), groups)
            if not new_chunks:
                raise ServiceError(422, f"{name} has no text.")
            taken = {c.id for c in chunks}
            if any(c.id in taken for c in new_chunks):
                raise ServiceError(409, f"A file with the same name as {name} is already uploaded.")
            files.append({"name": name, "kind": "document", "detail": f"{sum(1 for c in new_chunks if not c.is_table_row)} passages"})
        if len(chunks) + len(new_chunks) > MAX_CHUNKS:
            raise ServiceError(413, "That would exceed the workspace size limit.")

        chunks += new_chunks
        self.workspaces[workspace_id] = self._build(workspace_id, files, chunks, tables)
        return self.describe(workspace_id)

    def clear(self, workspace_id: str) -> dict:
        self.workspaces.pop(workspace_id, None)
        return self.describe(workspace_id)

    def _build(self, workspace_id: str, files: list[dict], chunks: list[Chunk], tables: dict[str, list[dict]]) -> Workspace:
        store = MemoryStore()
        if chunks:
            store.add(chunks, self.embedder.embed([c.embedding_text() for c in chunks]))
        generator = LLMGenerator(self.llm) if self.llm else ExtractiveGenerator.from_chunks(chunks or [Chunk("x", "x", "x", "", frozenset())])
        verifier = LLMVerifier(self.llm) if self.llm else LexicalVerifier()
        comp = Components(store=store, embedder=self.embedder, generator=generator, verifier=verifier, reveal_restricted_existence=False)
        pipelines: dict[str, Pipeline] = {}
        graph = None
        if chunks:
            graph = KnowledgeGraph(chunks, PatternExtractor())
            pipelines = {"naive": NaivePipeline(comp), "hybrid": HybridPipeline(comp), "graph": GraphPipeline(comp, graph), "multimodal": MultimodalPipeline(comp)}
        if self.llm:
            structured = None
            if tables:
                structured = StructuredPipeline(self.llm, SqliteDatabase(tables), LexicalVerifier(), notes="Column names are lower-case with underscores.", comp=comp if chunks else None, phrase=True)
                pipelines["sql"] = structured
            if chunks:
                pipelines["agentic"] = AgenticPipeline(comp, self.llm, structured=structured, graph=graph)
        return Workspace(workspace_id, comp, pipelines, files, chunks, tables)

    # --------------------------------------------------------------------- ask
    def ask(self, question: str, *, source: str, workspace_id: str | None, group: str | None, pipelines: list[str]) -> dict:
        question = (question or "").strip()
        if not question:
            raise ServiceError(400, "Type a question.")
        if len(question) > MAX_QUESTION:
            raise ServiceError(400, f"Keep the question under {MAX_QUESTION} characters.")
        if source == "sample":
            ws = self.sample
            if group not in self.sample_groups:
                raise ServiceError(400, "Choose a group.")
            principal = Principal.of("visitor", group)
        else:
            ws = self.workspaces.get(workspace_id or "")
            if ws is None:
                raise ServiceError(410, "Your uploaded files are no longer loaded. Upload them again.")
            principal = Principal.of(workspace_id, UPLOAD_GROUP)
        ws.last_used = time.time()
        wanted = [p for p in dict.fromkeys(pipelines) if p in ws.pipelines]
        if not wanted:
            raise ServiceError(400, "None of the chosen methods can run on these files.")

        def one(pid: str) -> dict:
            trace = TraceWriter("live", pid)
            answer = run_pipeline(ws.pipelines[pid], question, principal, trace)
            sources = []
            for c in answer.citations:
                entry = {"n": c.n, "ref": c.ref, "kind": c.kind, "title": "", "section": "", "text": ""}
                if c.ref.startswith("table:"):
                    entry["title"] = f"Data table: {c.ref.split(':', 1)[1]}"
                    sql = next((e["payload"].get("sql") for e in trace.events if e["event_type"] == "sql_guard" and e["payload"].get("allowed")), "")
                    entry["text"] = sql or ""
                else:
                    # Fetched through the store as the asker, so a passage they may not see cannot be returned here.
                    found = ws.comp.store.get_by_ref([c.ref], principal)
                    if found:
                        entry.update(title=found[0].title, section=found[0].section, text=found[0].text[:1500])
                sources.append(entry)
            return {"pipeline": pid, "verdict": answer.verdict.value, "text": answer.text, "sources": sources,
                    "ms": answer.usage.ms, "cost_usd": round(answer.usage.cost_usd, 5), "steps": trace.bundle()["events"]}

        results = list(self._pool.map(one, wanted))
        return {"question": question, "source": source, "group": group if source == "sample" else None, "results": results}
