"""Baseline: vector search only, access filter applied after ranking.

This is what a first RAG prototype usually looks like, and it is here to be
measured against. It never shows restricted content (the store drops it), but it
loses recall whenever restricted chunks take places in the top k, it has no
keyword match for exact identifiers, and it does not know that a superseded
document should not be cited.
"""

from __future__ import annotations

from ..core.trace import TraceWriter, visible_hits
from ..core.types import Answer, Principal
from .base import Components, answer_from_hits, embed_query


class NaivePipeline:
    id = "naive"
    label = "Naive vector baseline"
    version = "1"

    def __init__(self, comp: Components):
        self.comp = comp

    def run(self, question: str, principal: Principal, trace: TraceWriter) -> Answer:
        qvec = embed_query(self.comp, question)
        result = self.comp.store.vector_search(qvec, self.comp.k, principal, filter_mode="post")
        trace.event("vector_search", {"k": self.comp.k, "filter": "after ranking", **visible_hits(result)})
        return answer_from_hits(question, qvec, principal, result.hits, self.comp, trace)
