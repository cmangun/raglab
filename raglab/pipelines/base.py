"""What every pipeline shares: components, the answer-and-verify tail, and failure handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..core.embed import Embedder
from ..core.generate import Generator
from ..core.store import ChunkStore
from ..core.trace import TraceWriter
from ..core.types import Answer, Hit, Principal, Usage, Verdict
from ..core.verify import Verifier

MSG_INSUFFICIENT = "The available documents do not answer this question."
MSG_DENIED = "You do not have access to the documents that answer this question."
MSG_UNVERIFIED = "An answer was drafted but its sources do not support it, so it is not shown."
MSG_FAILED = "The question could not be processed."


@dataclass
class Components:
    store: ChunkStore
    embedder: Embedder
    generator: Generator
    verifier: Verifier
    k: int = 8
    # If True, a question whose best evidence is restricted gets `access_denied`,
    # which tells the asker that something exists. If False it gets
    # `insufficient_evidence`, which reveals nothing. Which is right is a policy
    # decision; see docs/decisions.md.
    reveal_restricted_existence: bool = True


class Pipeline(Protocol):
    id: str
    label: str
    version: str

    def run(self, question: str, principal: Principal, trace: TraceWriter) -> Answer: ...


def embed_query(comp: Components, question: str) -> np.ndarray:
    return comp.embedder.embed([question])[0]


def empty_verdict(question: str, qvec: np.ndarray, principal: Principal, comp: Components, trace: TraceWriter) -> Answer:
    """No usable evidence. Decide between 'nothing exists' and 'exists but not for you'."""
    profile = comp.store.access_profile(qvec, question, comp.k, principal)
    restricted_top = bool(profile) and (not profile[0] or sum(1 for v in profile[:3] if not v) >= 2)
    denied = comp.reveal_restricted_existence and restricted_top
    trace.event(
        "access_check",
        {"restricted_in_top_k": sum(1 for v in profile if not v), "top_candidate_restricted": bool(profile) and not profile[0],
         "verdict": "access_denied" if denied else "insufficient_evidence"},
    )
    if denied:
        return Answer(MSG_DENIED, Verdict.ACCESS_DENIED)
    return Answer(MSG_INSUFFICIENT, Verdict.INSUFFICIENT_EVIDENCE)


def answer_from_hits(question: str, qvec: np.ndarray, principal: Principal, hits: list[Hit], comp: Components, trace: TraceWriter, citation_kind: str = "chunk") -> Answer:
    """Generate, verify, and return. Shared by every document pipeline."""
    usage = Usage()
    retrieved = list(dict.fromkeys(h.chunk.ref for h in hits))
    draft = comp.generator.generate(question, hits)
    usage.add(draft.usage)
    trace.event("generate", {"generator": comp.generator.name, "sources": len(hits), "insufficient": draft.insufficient,
                             "input_tokens": draft.usage.input_tokens, "output_tokens": draft.usage.output_tokens})
    if draft.insufficient:
        ans = empty_verdict(question, qvec, principal, comp, trace)
        ans.retrieved_refs, ans.usage = retrieved, usage
        return ans
    check = comp.verifier.verify(draft.text, hits)
    usage.add(check.usage)
    trace.event("verify", {"verifier": comp.verifier.name, "claims": check.claims, "unsupported": len(check.unsupported), "supported": check.supported})
    if not check.supported:
        return Answer(MSG_UNVERIFIED, Verdict.VERIFICATION_FAILED, retrieved_refs=retrieved, verifier_changed=True, usage=usage, support=check.support_rate)
    for c in draft.citations:
        c.kind = citation_kind
    return Answer(draft.text, Verdict.ANSWERED, citations=draft.citations, retrieved_refs=retrieved, usage=usage, support=check.support_rate)


def run_pipeline(pipeline: Pipeline, question: str, principal: Principal, trace: TraceWriter) -> Answer:
    """Run a pipeline. Any exception becomes `execution_failed`; it is never shown as a decline."""
    t0 = time.perf_counter()
    trace.event("question", {"pipeline": pipeline.id, "version": pipeline.version, "question": question, "groups": sorted(principal.groups)})
    try:
        ans = pipeline.run(question, principal, trace)
    except Exception as exc:  # noqa: BLE001 - the boundary where failures are classified
        trace.event("error", {"type": type(exc).__name__}, admin={"message": str(exc)})
        ans = Answer(MSG_FAILED, Verdict.EXECUTION_FAILED, error=type(exc).__name__)
    ans.usage.ms = round((time.perf_counter() - t0) * 1000, 1)
    trace.event("answer", {"verdict": ans.verdict.value, "citations": [c.ref for c in ans.citations], "ms": ans.usage.ms})
    return ans
