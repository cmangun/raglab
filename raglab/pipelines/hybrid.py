"""Use cases 1 and 2: permission-aware hybrid retrieval.

Keyword and vector search each run with the access filter inside the query,
their rankings are fused by reciprocal rank fusion, and superseded documents are
excluded. An optional reranker reorders the fused candidates.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..core.fusion import rrf
from ..core.trace import TraceWriter, visible_hits
from ..core.types import Answer, Hit, Principal
from .base import Components, answer_from_hits, embed_query


class Reranker(Protocol):
    name: str

    def rerank(self, question: str, hits: list[Hit], limit: int) -> list[Hit]: ...


def hybrid_retrieve(
    comp: Components, question: str, qvec: np.ndarray, principal: Principal, trace: TraceWriter, *,
    candidates: int = 30, limit: int | None = None, current_only: bool = True, table_rows: bool = False,
    reranker: Reranker | None = None, stage: str = "hybrid_search",
) -> list[Hit]:
    limit = limit or comp.k
    vec = comp.store.vector_search(qvec, candidates, principal, filter_mode="pre", current_only=current_only, table_rows=table_rows)
    kw = comp.store.keyword_search(question, candidates, principal, current_only=current_only, table_rows=table_rows)
    fused = rrf([vec.hits, kw.hits])
    if reranker is not None:
        fused = reranker.rerank(question, fused[:candidates], limit)
    hits = fused[:limit]
    trace.event(stage, {"candidates_each": candidates, "filter": "inside the query", "current_only": current_only,
                        "vector_hits": len(vec.hits), "keyword_hits": len(kw.hits),
                        "reranker": reranker.name if reranker else None, **visible_hits(hits)})
    return hits


class HybridPipeline:
    id = "hybrid"
    label = "Permission-aware hybrid retrieval"
    version = "1"

    def __init__(self, comp: Components, candidates: int = 30, current_only: bool = True, reranker: Reranker | None = None):
        self.comp, self.candidates, self.current_only, self.reranker = comp, candidates, current_only, reranker

    def run(self, question: str, principal: Principal, trace: TraceWriter) -> Answer:
        qvec = embed_query(self.comp, question)
        hits = hybrid_retrieve(self.comp, question, qvec, principal, trace, candidates=self.candidates, current_only=self.current_only, reranker=self.reranker)
        return answer_from_hits(question, qvec, principal, hits, self.comp, trace)
