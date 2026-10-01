"""Leakage sweep: try to make every retrieval path return something restricted.

Queries are adversarial on purpose. For each group, the sweep searches with the
text of the documents that group must not see, asks for those documents by id
and by reference, and walks the graph from their record ids.
"""

from __future__ import annotations

from ..build import Lab
from ..core.text import record_ids, sentences
from ..core.types import Principal


def leakage_sweep(lab: Lab, groups: list[str], extra_queries: list[str] | None = None) -> dict:
    store, emb = lab.comp.store, lab.comp.embedder
    checks = leaks = 0
    examples: list[str] = []
    all_chunks = [c for c in lab.chunks if not c.is_table_row]

    def see(chunks, principal: Principal, path: str) -> None:
        nonlocal checks, leaks
        for c in chunks:
            checks += 1
            if not c.visible_to(principal):
                leaks += 1
                if len(examples) < 5:
                    examples.append(f"{path}: {c.id} returned to {sorted(principal.groups)}")

    for g in groups:
        principal = Principal.of(f"sweep-{g}", g)
        hidden = [c for c in all_chunks if not c.visible_to(principal)]
        queries = list(extra_queries or [])
        queries += [s for c in hidden for s in sentences(c.text)[:2]]  # the restricted text itself
        queries += [c.title for c in hidden]
        vectors = emb.embed(queries) if queries else []
        for query, vec in zip(queries, vectors):
            for mode in ("pre", "post"):
                see([h.chunk for h in store.vector_search(vec, 10, principal, filter_mode=mode, table_rows=True).hits], principal, f"vector/{mode}")
            see([h.chunk for h in store.keyword_search(query, 10, principal, table_rows=True).hits], principal, "keyword")
        for doc_id in sorted({c.document_id for c in hidden}):
            see(store.get_document(doc_id, principal), principal, "get_document")
        see(store.get_by_ref([c.ref for c in hidden], principal), principal, "get_by_ref")
        seeds = {lab.graph.extractor.normalise(r) for c in hidden for r in record_ids(f"{c.document_id} {c.text}")}
        seeds = {s for s in seeds if s in lab.graph.entity_chunks}
        see([h.chunk for h in lab.graph.walk(seeds, principal, hops=2)[0]], principal, "graph")
    return {"checks": checks, "leaks": leaks, "examples": examples}
