"""Use case 5: questions about how records relate across documents.

Entities are extracted from every chunk and linked through the chunks that
mention them. A question's entities seed a walk of one or two hops, and the
chunks reached are fused with ordinary hybrid results.

Access control applies to the walk itself: a hop may only pass through a chunk
the principal can see. A restricted document therefore cannot connect two
entities for someone who is not allowed to read it.

Extraction here is pattern-based (record identifiers plus a name gazetteer),
which is exact for identifier-rich corpora and costs nothing to build. A model
based extractor can be supplied for prose that names things loosely.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Protocol

from ..core.fusion import rrf
from ..core.text import record_ids
from ..core.trace import TraceWriter, visible_hits
from ..core.types import Answer, Chunk, Hit, Principal
from .base import Components, answer_from_hits, embed_query
from .hybrid import hybrid_retrieve


class EntityExtractor(Protocol):
    name: str

    def extract(self, text: str) -> set[str]: ...


class PatternExtractor:
    """Record identifiers, normalised, plus exact names from a gazetteer."""

    name = "pattern"

    def __init__(self, gazetteer: dict[str, str] | None = None):
        # longest names first so "Marrow Bay" wins over a shorter overlapping alias
        self._names = sorted(((k.lower(), v) for k, v in (gazetteer or {}).items()), key=lambda kv: -len(kv[0]))

    @staticmethod
    def normalise(rid: str) -> str:
        rid = re.sub(r"-v\d+$", "", rid)  # SOP-008-v2 and SOP-008 are the same procedure
        return re.sub(r"^SPEC-", "", rid)  # a specification sheet is about its product

    def extract(self, text: str) -> set[str]:
        found = {self.normalise(r) for r in record_ids(text)}
        low = text.lower()
        for name, entity in self._names:
            if name in low:
                found.add(entity)
        return found


def gazetteer_from_tables(tables: dict[str, list[dict]]) -> dict[str, str]:
    g: dict[str, str] = {}
    for row in tables.get("product", []):
        g[row["name"]] = row["id"]
    for row in tables.get("site", []):
        g[row["name"]] = f"site:{row['id']}"
    return g


class KnowledgeGraph:
    def __init__(self, chunks: list[Chunk], extractor: EntityExtractor, hub_fraction: float = 0.15):
        self.extractor = extractor
        self.chunks = {c.id: c for c in chunks if not c.is_table_row and c.status == "current"}
        self.chunk_entities: dict[str, set[str]] = {}
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        for c in self.chunks.values():
            ents = extractor.extract(f"{c.title} {c.text}")
            ents.add(extractor.extract(c.document_id).pop() if extractor.extract(c.document_id) else c.document_id)
            self.chunk_entities[c.id] = ents
            for e in ents:
                self.entity_chunks[e].add(c.id)
        # Entities mentioned almost everywhere (a procedure every document cites) connect
        # everything to everything; they are not walked through on the second hop.
        limit = max(8, int(len(self.chunks) * hub_fraction))
        self.hubs = {e for e, cs in self.entity_chunks.items() if len(cs) > limit}

    def stats(self) -> dict[str, int]:
        return {"entities": len(self.entity_chunks), "chunks": len(self.chunks), "hubs": len(self.hubs),
                "mentions": sum(len(v) for v in self.entity_chunks.values())}

    def walk(self, seeds: set[str], principal: Principal, hops: int = 2) -> tuple[list[Hit], dict]:
        visible = lambda cid: self.chunks[cid].visible_to(principal)
        score: dict[str, float] = defaultdict(float)
        hop1 = {cid for s in seeds for cid in self.entity_chunks.get(s, ()) if visible(cid)}
        for cid in hop1:
            score[cid] += 1.0 + len(seeds & self.chunk_entities[cid])
        second: set[str] = set()
        if hops >= 2:
            for cid in hop1:
                second |= self.chunk_entities[cid] - seeds - self.hubs
            for e in second:
                for cid in self.entity_chunks[e]:
                    if visible(cid):
                        score[cid] += 0.25
        ranked = sorted(score, key=lambda cid: (-score[cid], cid))
        info = {"seeds": sorted(seeds), "first_hop_chunks": len(hop1), "second_hop_entities": len(second)}
        return [Hit(self.chunks[cid], score[cid], "graph") for cid in ranked], info


class GraphPipeline:
    id = "graph"
    label = "Graph retrieval"
    version = "1"

    def __init__(self, comp: Components, graph: KnowledgeGraph, hops: int = 2, graph_candidates: int = 30):
        self.comp, self.graph, self.hops, self.graph_candidates = comp, graph, hops, graph_candidates

    def retrieve(self, question: str, qvec, principal: Principal, trace: TraceWriter) -> list[Hit]:
        seeds = {e for e in self.graph.extractor.extract(question) if e in self.graph.entity_chunks}
        base = hybrid_retrieve(self.comp, question, qvec, principal, trace, limit=self.graph_candidates)
        if not seeds:
            trace.event("graph_walk", {"seeds": [], "note": "no known entity in the question; hybrid results only"})
            return base[: self.comp.k]
        walked, info = self.graph.walk(seeds, principal, self.hops)
        # Order graph candidates by graph score, breaking ties with their hybrid rank.
        hybrid_rank = {h.chunk.id: i for i, h in enumerate(base)}
        walked.sort(key=lambda h: (-h.score, hybrid_rank.get(h.chunk.id, 10_000), h.chunk.id))
        fused = rrf([walked[: self.graph_candidates], base], weights=[1.5, 1.0])[: self.comp.k]
        trace.event("graph_walk", {**info, "hops": self.hops, **visible_hits(fused)})
        return fused

    def run(self, question: str, principal: Principal, trace: TraceWriter) -> Answer:
        qvec = embed_query(self.comp, question)
        return answer_from_hits(question, qvec, principal, self.retrieve(question, qvec, principal, trace), self.comp, trace)
