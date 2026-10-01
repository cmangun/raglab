"""Verdicts, the agent loop, graph access, page reading. Models are scripted so the logic is tested, not a provider."""

import json
from dataclasses import replace

from conftest import COMMERCIAL, QUALITY

from raglab.core.generate import INSUFFICIENT, LLMGenerator
from raglab.core.llm import ScriptedLLM
from raglab.core.trace import TraceWriter
from raglab.core.types import Chunk, Principal, Verdict
from raglab.core.verify import LexicalVerifier
from raglab.pipelines.agentic import AgenticPipeline
from raglab.pipelines.base import run_pipeline
from raglab.pipelines.graph import KnowledgeGraph, PatternExtractor
from raglab.pipelines.hybrid import HybridPipeline
from raglab.pipelines.multimodal import MultimodalPipeline
from raglab.pipelines.naive import NaivePipeline


def _comp(lab, reply, **kw):
    """Components whose generator is a scripted model."""
    fn = reply if callable(reply) else (lambda s, u: reply)
    return replace(lab.comp, generator=LLMGenerator(ScriptedLLM(fn)), verifier=LexicalVerifier(), **kw)


def _run(pipeline, question, principal):
    t = TraceWriter("t", pipeline.id)
    return run_pipeline(pipeline, question, principal, t), t


def test_answered_with_citation(lab):
    def reply(system, user):
        n = next(i for i, block in enumerate(user.split("\n\n[")[0:], start=0) if "Maximum bulk hold time" in block)
        return f"Maximum bulk hold time before filling is 72 hours. [{max(n, 1)}]"

    ans, _ = _run(HybridPipeline(_comp(lab, reply)), "How long can reagent bulk be held before filling under SOP-004?", QUALITY)
    assert ans.verdict == Verdict.ANSWERED and ans.citations[0].ref == "SOP-004#parameters" and ans.support == 1.0


def test_forbidden_question_is_access_denied(lab):
    ans, t = _run(HybridPipeline(_comp(lab, INSUFFICIENT)), "What is the maximum discount allowed on distributor contracts?", QUALITY)
    assert ans.verdict == Verdict.ACCESS_DENIED
    assert not any("MEMO" in r for r in ans.retrieved_refs)
    assert "18" not in ans.text


def test_same_question_is_answerable_by_the_owning_group(lab):
    def reply(system, user):
        n = next(i for i, block in enumerate(user.split("\n\n["), start=0) if "up to 18 percent" in block)
        return f"Distributor contracts may carry a discount of up to 18 percent off list price. [{max(n, 1)}]"

    ans, _ = _run(HybridPipeline(_comp(lab, reply)), "What is the maximum discount allowed on distributor contracts?", COMMERCIAL)
    assert ans.verdict == Verdict.ANSWERED and "MEMO-05#details" in [c.ref for c in ans.citations]


def test_existence_can_be_hidden_by_policy(lab):
    comp = _comp(lab, INSUFFICIENT, reveal_restricted_existence=False)
    ans, _ = _run(HybridPipeline(comp), "What is the maximum discount allowed on distributor contracts?", QUALITY)
    assert ans.verdict == Verdict.INSUFFICIENT_EVIDENCE


def test_out_of_scope_is_insufficient_evidence(lab):
    ans, _ = _run(HybridPipeline(_comp(lab, INSUFFICIENT)), "How many people does Calder Ridge Diagnostics employ?", QUALITY)
    assert ans.verdict == Verdict.INSUFFICIENT_EVIDENCE


def test_unsupported_answer_is_suppressed(lab):
    ans, _ = _run(HybridPipeline(_comp(lab, "The maximum bulk hold time is 500 hours. [1]")), "How long can reagent bulk be held?", QUALITY)
    assert ans.verdict == Verdict.VERIFICATION_FAILED and ans.verifier_changed and "500" not in ans.text


def test_provider_failure_is_execution_failed_and_hides_the_message(lab):
    def boom(system, user):
        raise RuntimeError("provider said: key sk-secret is invalid")

    ans, t = _run(HybridPipeline(_comp(lab, boom)), "How long can reagent bulk be held?", QUALITY)
    assert ans.verdict == Verdict.EXECUTION_FAILED and ans.error == "RuntimeError"
    assert "sk-secret" not in t.public_text() and "sk-secret" not in ans.text


def test_hybrid_never_cites_a_superseded_version(lab):
    q = "What is the cumulative excursion limit above 8 °C in SOP-008 cold chain storage?"
    _, t = _run(HybridPipeline(_comp(lab, INSUFFICIENT)), q, QUALITY)
    hybrid_refs = json.dumps([e["payload"] for e in t.events if e["event_type"] == "hybrid_search"])
    assert "SOP-008-v1" not in hybrid_refs and "SOP-008-v2" in hybrid_refs
    _, t2 = _run(NaivePipeline(_comp(lab, INSUFFICIENT)), q, QUALITY)
    assert "SOP-008-v1" in json.dumps([e["payload"] for e in t2.events])  # the baseline has no notion of versions


# ------------------------------------------------------------------ agentic

def _agent(lab, actions, **kw):
    script = iter(actions)
    planner = ScriptedLLM(lambda s, u: next(script, '{"action": "search", "input": "anything else"}'))
    return AgenticPipeline(_comp(lab, INSUFFICIENT), planner, **kw), planner


def test_agent_stops_at_the_step_limit(lab):
    agent, planner = _agent(lab, [], max_steps=3)
    ans, t = _run(agent, "How long can reagent bulk be held?", QUALITY)
    assert len(planner.calls) == 3
    assert any(e["payload"].get("action") == "step_limit_reached" for e in t.events)
    assert ans.verdict in (Verdict.INSUFFICIENT_EVIDENCE, Verdict.ACCESS_DENIED)


def test_agent_protocol_errors_are_execution_failures(lab):
    agent, planner = _agent(lab, ["I think I should search", "still not json"])
    ans, _ = _run(agent, "anything", QUALITY)
    assert ans.verdict == Verdict.EXECUTION_FAILED and len(planner.calls) == 2


def test_agent_cannot_read_a_restricted_document(lab):
    agent, _ = _agent(lab, ['{"action": "read", "input": "MEMO-05"}', '{"action": "finish", "input": ""}'])
    ans, t = _run(agent, "What is the maximum discount allowed on distributor contracts?", QUALITY)
    assert not any("MEMO" in r for r in ans.retrieved_refs)
    step = next(e for e in t.events if e["event_type"] == "agent_step" and e["payload"].get("action") == "read")
    assert step["payload"]["hits"] == []


def test_agent_multi_hop_gathers_both_documents(lab):
    agent, _ = _agent(lab, [
        '{"action": "search", "input": "Lipid Panel Reagent Kit storage procedure"}',
        '{"action": "read", "input": "SOP-008-v2"}',
        '{"action": "finish", "input": ""}',
    ])
    ans, _ = _run(agent, "What cumulative temperature excursion limit applies to storing Lipid Panel Reagent Kit?", QUALITY)
    assert "SOP-008-v2#parameters" in ans.retrieved_refs and any(r.startswith("SPEC-P-101#") for r in ans.retrieved_refs)


# -------------------------------------------------------------------- graph

def test_restricted_chunk_cannot_bridge_two_entities():
    both, secret = frozenset({"a", "b"}), frozenset({"b"})
    chunks = [
        Chunk(id="D1#s", document_id="D1", ref="D1#s", text="DEV-2025-900 was recorded.", groups=both),
        Chunk(id="D2#s", document_id="D2", ref="D2#s", text="DEV-2025-900 led to CC-90.", groups=secret),
        Chunk(id="D3#s", document_id="D3", ref="D3#s", text="CC-90 revised SOP-090.", groups=both),
    ]
    g = KnowledgeGraph(chunks, PatternExtractor())
    insider = {h.chunk.id for h in g.walk({"DEV-2025-900"}, Principal.of("u", "b"))[0]}
    outsider = {h.chunk.id for h in g.walk({"DEV-2025-900"}, Principal.of("u", "a"))[0]}
    assert insider == {"D1#s", "D2#s", "D3#s"}
    assert outsider == {"D1#s"}  # without D2 there is no path to D3


def test_graph_reaches_evidence_two_documents_away(lab):
    g = lab.pipelines["graph"]
    t = TraceWriter("t", "graph")
    q = "Which deviations ultimately led to the current version of SOP-005?"
    refs = {h.chunk.ref for h in g.retrieve(q, lab.comp.embedder.embed([q])[0], QUALITY, t)}
    assert "CC-02#links" in refs and "SOP-005-v2#references-and-history" in refs


# --------------------------------------------------------------- multimodal

class FakeRenderer:
    name = "fake"

    def render(self, hit):
        return b"\x89PNG fake " + hit.chunk.ref.encode()


def test_table_rows_are_retrieved(lab):
    p = MultimodalPipeline(_comp(lab, INSUFFICIENT))
    _, t = _run(p, "In the specification for Hematology Calibrator Set, what is the open-vial stability?", COMMERCIAL)
    hits = next(e for e in t.events if e["event_type"] == "table_aware_search")["payload"]["hits"]
    assert any("@row" in h["ref"] and h["ref"].startswith("SPEC-P-109") for h in hits)


def test_page_reading_is_verified_against_the_page_text(lab):
    seen = {}

    def vision(system, user):
        return "The open-vial stability is 30 days. [1]"

    llm = ScriptedLLM(vision)
    orig = llm.complete
    llm.complete = lambda s, u, images=None, max_tokens=800: (seen.setdefault("images", images), orig(s, u))[1]
    p = MultimodalPipeline(_comp(lab, INSUFFICIENT), vision_llm=llm, renderer=FakeRenderer())
    ans, _ = _run(p, "In the specification for Hematology Calibrator Set, what is the open-vial stability?", COMMERCIAL)
    assert seen["images"] and ans.verdict == Verdict.ANSWERED and ans.citations[0].kind == "page"

    wrong = ScriptedLLM(lambda s, u: "The open-vial stability is 999 days. [1]")
    ans2, _ = _run(MultimodalPipeline(_comp(lab, INSUFFICIENT), vision_llm=wrong, renderer=FakeRenderer()),
                   "In the specification for Hematology Calibrator Set, what is the open-vial stability?", COMMERCIAL)
    assert ans2.verdict == Verdict.VERIFICATION_FAILED
