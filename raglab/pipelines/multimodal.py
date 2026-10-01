"""Use case 6: answers that live in tables and figures.

Two steps, and the second is optional.

1. Structure-aware retrieval. Tables are indexed row by row with their column
   headers folded in, so "open-vial stability for the calibrator set" can match
   one row instead of a whole table. Row hits are searched alongside ordinary
   passages.
2. Page-image reading. If a renderer and a vision-capable model are supplied,
   the page images of the best-matching sections are sent to the model, which
   reads the values off the page. This is what handles scanned layouts, merged
   cells and charts that text extraction loses.

Without step 2 this pipeline is a table-aware text pipeline; it is labelled as
such in its trace.
"""

from __future__ import annotations

from typing import Protocol

from ..core.generate import INSUFFICIENT, parse_citations
from ..core.llm import LLM
from ..core.trace import TraceWriter
from ..core.types import Answer, Hit, Principal, Verdict
from .base import MSG_UNVERIFIED, Components, answer_from_hits, embed_query, empty_verdict
from .hybrid import hybrid_retrieve


class PageRenderer(Protocol):
    name: str

    def render(self, hit: Hit) -> bytes | None:
        """PNG of the page or region this hit came from, or None if unavailable."""
        ...


class MultimodalPipeline:
    id = "multimodal"
    label = "Tables and page images"
    version = "1"

    VISION_SYSTEM = (
        "You answer a question by reading the attached page images, which are numbered in order. "
        "Cite the page number in square brackets after every sentence, for example [1]. "
        f"If the pages do not contain the answer, reply with exactly {INSUFFICIENT}. Be brief."
    )

    def __init__(self, comp: Components, vision_llm: LLM | None = None, renderer: PageRenderer | None = None, max_pages: int = 3):
        self.comp, self.vision_llm, self.renderer, self.max_pages = comp, vision_llm, renderer, max_pages

    def run(self, question: str, principal: Principal, trace: TraceWriter) -> Answer:
        qvec = embed_query(self.comp, question)
        hits = hybrid_retrieve(self.comp, question, qvec, principal, trace, table_rows=True, stage="table_aware_search")
        trace.event("mode", {"table_rows_indexed": True, "page_images": bool(self.vision_llm and self.renderer)})
        if not (self.vision_llm and self.renderer):
            return answer_from_hits(question, qvec, principal, hits, self.comp, trace)

        # One image per distinct section, best first.
        pages: list[tuple[Hit, bytes]] = []
        seen: set[str] = set()
        for h in hits:
            if h.chunk.ref in seen:
                continue
            seen.add(h.chunk.ref)
            png = self.renderer.render(h)
            if png:
                pages.append((h, png))
            if len(pages) >= self.max_pages:
                break
        if not pages:
            return answer_from_hits(question, qvec, principal, hits, self.comp, trace)
        page_hits = [h for h, _ in pages]
        r = self.vision_llm.complete(self.VISION_SYSTEM, f"Question: {question}", images=[png for _, png in pages])
        trace.event("read_pages", {"model": self.vision_llm.name, "renderer": self.renderer.name, "pages": [h.chunk.ref for h in page_hits],
                                   "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens})
        text = r.text.strip()
        retrieved = [h.chunk.ref for h in page_hits]
        if INSUFFICIENT in text or not text:
            ans = empty_verdict(question, qvec, principal, self.comp, trace)
            ans.retrieved_refs, ans.usage = retrieved, r.usage
            return ans
        # The page's own text is the evidence the verifier checks the reading against.
        check = self.comp.verifier.verify(text, page_hits)
        trace.event("verify", {"verifier": self.comp.verifier.name, "claims": check.claims, "unsupported": len(check.unsupported), "supported": check.supported})
        r.usage.add(check.usage)
        if not check.supported:
            return Answer(MSG_UNVERIFIED, Verdict.VERIFICATION_FAILED, retrieved_refs=retrieved, verifier_changed=True, usage=r.usage, support=check.support_rate)
        return Answer(text, Verdict.ANSWERED, citations=parse_citations(text, page_hits, kind="page"), retrieved_refs=retrieved, usage=r.usage, support=check.support_rate)
