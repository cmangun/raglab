"""The evaluation must be able to fail. These tests plant faults and check they are caught."""

import copy

from raglab.core.trace import TraceWriter
from raglab.core.types import Answer, Citation, Verdict
from raglab.eval.gate import evaluate_gate
from raglab.eval.golden import access_lists, load_golden
from raglab.eval.metrics import KeyFactJudge, find_leak, score_answer
from raglab.eval.runner import run_eval

QUESTIONS, _ = load_golden("corpus/golden.json")
ACL = access_lists("corpus/documents")
BY_ID = {q.id: q for q in QUESTIONS}
EMPTY = {"events": []}


def test_golden_set_shape():
    assert len(QUESTIONS) == 42
    assert sum(q.type == "forbidden" for q in QUESTIONS) == 3 and sum(q.type == "out_of_scope" for q in QUESTIONS) == 3
    assert all(q.key_facts for q in QUESTIONS if q.expected_verdict == Verdict.ANSWERED)


def test_judge_grades():
    q = BY_ID["Q-12"]  # needs a site and an interval
    j = KeyFactJudge()
    assert j.grade(q, Answer("Elsin Park makes it; line clearance is checked every 4 hours.", Verdict.ANSWERED)) == "correct"
    assert j.grade(q, Answer("It is made at Elsin Park.", Verdict.ANSWERED)) == "partial"
    assert j.grade(q, Answer("Tallow Creek, every 14 hours.", Verdict.ANSWERED)) == "wrong"  # 14 is not 4


def test_leak_detection_catches_each_route():
    q = BY_ID["Q-40"]  # list price, asked as Quality
    assert find_leak(q, Answer("no", Verdict.ACCESS_DENIED), EMPTY, ACL) == ""
    assert "reached the answer" in find_leak(q, Answer("x", Verdict.ANSWERED, retrieved_refs=["MEMO-01#details"]), EMPTY, ACL)
    assert "reached the answer" in find_leak(q, Answer("x", Verdict.ANSWERED, citations=[Citation(1, "chunk", "MEMO-01#details@row3")]), EMPTY, ACL)
    trace = {"events": [{"payload": {"hits": [{"ref": "MEMO-01#details", "score": 1}]}}]}
    assert "public trace" in find_leak(q, Answer("no", Verdict.ACCESS_DENIED), trace, ACL)
    secret = q.secret_facts[0]
    assert "restricted value" in find_leak(q, Answer(f"The price is {secret} dollars.", Verdict.ANSWERED), EMPTY, ACL)


def test_wrong_verdict_scores_as_wrong():
    q = BY_ID["Q-01"]
    s = score_answer(q, Answer("unavailable", Verdict.EXECUTION_FAILED), EMPTY, ACL, KeyFactJudge())
    assert s.grade == "wrong" and s.execution_failed and not s.verdict_ok


class LeakyPipeline:
    """A deliberately broken pipeline: it ignores access control."""

    id, label, version = "leaky", "Leaky", "0"

    def __init__(self, lab):
        self.lab = lab

    def run(self, question, principal, trace: TraceWriter):
        everyone = type(principal)("x", frozenset({"commercial", "quality"}))
        hits = self.lab.comp.store.keyword_search(question, 5, everyone).hits
        return Answer(hits[0].chunk.text if hits else "", Verdict.ANSWERED, retrieved_refs=[h.chunk.ref for h in hits])


def test_gate_rejects_a_leaky_pipeline(lab):
    run = run_eval([LeakyPipeline(lab)], QUESTIONS, ACL, KeyFactJudge(), {})
    assert run["summary"]["leaky"]["leaks"] > 0
    assert run["gate"]["status"] == "rejected"
    assert run["gate"]["conditions"][0]["status"] == "fail"


def test_offline_run_is_clean_on_access_but_not_accepted(lab):
    run = run_eval([lab.pipelines[p] for p in ("naive", "hybrid", "graph", "multimodal")], QUESTIONS, ACL, lab.judge, lab.record, run_id="t")
    assert all(s["leaks"] == 0 and s["execution_failures"] == 0 for s in run["summary"].values())
    assert all(r["trace_chain_ok"] for r in run["results"])
    assert run["gate"]["conditions"][0]["status"] == "pass"
    assert run["gate"]["status"] != "accepted"  # no model, no human check: it must not pass


def test_regression_and_human_check_conditions(lab):
    run = run_eval([lab.pipelines["hybrid"]], QUESTIONS, ACL, lab.judge, lab.record, run_id="t")
    worse = copy.deepcopy(run)
    flipped = 0
    for r in worse["results"]:
        if r["score"]["grade"] == "correct" and flipped < 3:
            r["score"]["grade"], flipped = "wrong", flipped + 1
    gate = evaluate_gate(worse, baseline=run)
    assert next(c for c in gate["conditions"] if "worse than the last accepted" in c["name"])["status"] == "fail"
    run["human_check"] = {"sampled": 20, "agreed": 17}
    assert next(c for c in evaluate_gate(run)["conditions"] if "human" in c["name"])["status"] == "fail"
    run["human_check"] = {"sampled": 20, "agreed": 19}
    assert next(c for c in evaluate_gate(run)["conditions"] if "human" in c["name"])["status"] == "pass"


def test_right_figure_from_wrong_evidence_is_not_correct():
    q = BY_ID["Q-19"]  # a count that only the tables can support
    lucky = Answer(f"There are {int(q.expected_value)} of them. [1]", Verdict.ANSWERED, retrieved_refs=["SOP-001#scope"])
    s = score_answer(q, lucky, EMPTY, ACL, KeyFactJudge())
    assert s.grade == "wrong" and s.coincidental
    grounded = Answer(f"{int(q.expected_value)} [1]", Verdict.ANSWERED, retrieved_refs=["table:batch"])
    assert score_answer(q, grounded, EMPTY, ACL, KeyFactJudge()).grade == "correct"
    # A document question answered from other valid evidence is not penalised.
    doc_q = BY_ID["Q-33"]
    other = Answer("3 business days. [1]", Verdict.ANSWERED, retrieved_refs=["CC-01#change-description"])
    assert score_answer(doc_q, other, EMPTY, ACL, KeyFactJudge()).grade == "correct"
