"""Use case 4: a model plans its own retrieval.

The model chooses one tool per step, sees what came back, and decides what to do
next, up to a fixed number of steps. Every tool is permission-aware: the agent
holds the principal and cannot search as anyone else. Whatever it gathers is
then answered and verified by the same tail as every other pipeline, so the
agent decides what to look for but does not get to assert an unverified answer.
"""

from __future__ import annotations

import json
import re

import sqlglot
from sqlglot import exp

from ..core.llm import LLM
from ..core.text import content_tokens
from ..core.trace import TraceWriter, visible_hits
from ..core.types import Answer, Hit, Principal, Usage, Verdict
from .base import MSG_FAILED, Components, answer_from_hits, embed_query
from .graph import KnowledgeGraph
from .hybrid import hybrid_retrieve
from .structured import SqlRejected, StructuredPipeline, render_rows

SYSTEM = """You gather evidence to answer a question. Each turn, reply with one JSON object and nothing else:
{{"action": "<tool>", "input": "<text>"}}

Tools:
- search: find passages. Input is a search query. Use several focused searches rather than one broad one.
- read: fetch every section of one document. Input is a document id such as SOP-008-v2.
{extra}- finish: you have the evidence needed, or have established it is not available. Input is empty.

You have at most {max_steps} steps. Do not answer the question yourself; only gather evidence, then finish."""


class AgenticPipeline:
    id = "agentic"
    label = "Agentic retrieval"
    version = "1"

    def __init__(self, comp: Components, llm: LLM, max_steps: int = 4, max_evidence: int = 12,
                 structured: StructuredPipeline | None = None, graph: KnowledgeGraph | None = None):
        self.comp, self.llm, self.max_steps, self.max_evidence = comp, llm, max_steps, max_evidence
        self.structured, self.graph = structured, graph

    def _system(self) -> str:
        extra = ""
        if self.structured:
            extra += "- sql: answer a counting or totalling question from the database. Input is the question in plain words.\n"
        if self.graph:
            extra += "- related: find passages linked to a record. Input is a record id such as DEV-2025-003 or CC-02.\n"
        return SYSTEM.format(extra=extra, max_steps=self.max_steps)

    @staticmethod
    def _parse(text: str) -> tuple[str, str] | None:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return str(obj["action"]).lower().strip(), str(obj.get("input", "")).strip()
        except (ValueError, KeyError, TypeError):
            return None

    def run(self, question: str, principal: Principal, trace: TraceWriter) -> Answer:
        usage = Usage()
        pool: dict[str, Hit] = {}
        sql_note: str | None = None
        sql_tables: list[str] = []
        transcript = f"Question: {question}\n"
        queries: list[str] = []
        bad = 0
        for step in range(1, self.max_steps + 1):
            r = self.llm.complete(self._system(), transcript + f"\nStep {step} of {self.max_steps}. Your action:", max_tokens=200)
            usage.add(r.usage)
            parsed = self._parse(r.text)
            if parsed is None:
                bad += 1
                trace.event("agent_step", {"step": step, "action": "unparseable"})
                if bad >= 2:
                    return Answer(MSG_FAILED, Verdict.EXECUTION_FAILED, usage=usage, error="AgentProtocolError")
                transcript += f"\nStep {step}: your reply was not valid JSON. Reply with one JSON object.\n"
                continue
            action, arg = parsed
            observation, hits = "", []
            if action == "finish":
                trace.event("agent_step", {"step": step, "action": "finish"})
                break
            if action == "search" and arg:
                queries.append(arg)
                hits = hybrid_retrieve(self.comp, arg, embed_query(self.comp, arg), principal, trace, stage="agent_search")
            elif action == "read" and arg:
                hits = [Hit(c, 1.0, "read") for c in self.comp.store.get_document(arg, principal)]
            elif action == "related" and self.graph and arg:
                seeds = {e for e in self.graph.extractor.extract(arg) if e in self.graph.entity_chunks}
                hits = self.graph.walk(seeds, principal, hops=1)[0][: self.comp.k] if seeds else []
            elif action == "sql" and self.structured and arg:
                try:
                    out = self.structured.query(arg, trace, usage, principal)
                    observation = "The database cannot answer that." if out is None else f"Query result:\n{render_rows(out[1], out[2])}"
                    if out is not None:
                        sql_note = observation
                        sql_tables = sorted({t.name.lower() for t in sqlglot.parse_one(out[0], read=self.structured.db.dialect).find_all(exp.Table)} & self.structured.db.tables())
                except SqlRejected as e:
                    observation = f"The query was rejected: {e}."
            else:
                observation = f"Unknown or empty action '{action}'."
            for h in hits:
                pool.setdefault(h.chunk.id, h)
            if hits:
                observation = "\n".join(f"- {h.chunk.id}: {h.chunk.text[:240]}" for h in hits[: self.comp.k])
            elif not observation:
                observation = "Nothing was found."
            trace.event("agent_step", {"step": step, "action": action, "input": arg, **visible_hits(hits)})
            transcript += f"\nStep {step}: {action}({arg!r})\n{observation}\n"
        else:
            trace.event("agent_step", {"step": self.max_steps, "action": "step_limit_reached"})

        # Keep the gathered passages that best match the question and the agent's own
        # queries, not simply the first ones found: a document read late may hold the answer.
        wanted = set(content_tokens(question + " " + " ".join(queries)))
        order = {cid: i for i, cid in enumerate(pool)}
        ranked = sorted(pool.values(), key=lambda h: (-len(wanted & set(content_tokens(h.chunk.embedding_text()))), order[h.chunk.id]))
        evidence = ranked[: self.max_evidence]
        if sql_note:
            from ..core.types import Chunk

            evidence.insert(0, Hit(Chunk(id="sql-result", document_id="sql", ref=f"table:{sql_tables[0]}" if sql_tables else "table:result", text=sql_note, groups=principal.groups, title="Database query", section="result"), 1.0, "sql"))
        ans = answer_from_hits(question, embed_query(self.comp, question), principal, evidence, self.comp, trace)
        ans.usage.add(usage)
        return ans
