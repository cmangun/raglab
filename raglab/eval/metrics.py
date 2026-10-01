"""Scoring for one answer. Every pipeline is scored the same way, on every question."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.llm import LLM
from ..core.text import numbers
from ..core.types import Answer, Verdict
from .golden import Question


class Judge(Protocol):
    name: str

    def grade(self, q: Question, answer: Answer) -> str:
        """'correct', 'partial' or 'wrong'."""
        ...


def _has(fact: Any, text: str, nums: set[str]) -> bool:
    if isinstance(fact, list):
        return any(_has(f, text, nums) for f in fact)
    f = str(fact).lower()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", f):
        return f.rstrip("0").rstrip(".") in nums if "." in f else f in nums
    f = re.sub(r"\s+", " ", f)
    if f[0].isdigit():
        # "4 hours" must not match inside "14 hours" or "2.4 hours"
        return re.search(rf"(?<![\d.,]){re.escape(f)}(?![\w])", text) is not None
    return f in text


class KeyFactJudge:
    """Deterministic: an answer is correct when it contains every key fact of the question.

    It cannot tell a right figure in a wrong sentence from a right answer, so it
    is lenient. It is the offline judge and the cross-check for a model judge.
    """

    name = "key-fact"

    def grade(self, q: Question, answer: Answer) -> str:
        text = re.sub(r"\s+", " ", answer.text.lower())
        nums = set(numbers(answer.text))
        found = sum(1 for f in q.key_facts if _has(f, text, nums))
        if q.key_facts and found == len(q.key_facts):
            return "correct"
        return "partial" if found else "wrong"


class LLMJudge:
    SYSTEM = (
        "You grade an answer against a reference answer. Reply with JSON only: "
        '{"grade": "correct" | "partial" | "wrong", "reason": "<one sentence>"}. '
        "correct: states the same facts as the reference with no contradicting fact. "
        "partial: states some of the required facts, or the right facts with an error beside them. "
        "wrong: missing or contradicts the reference. Wording and extra correct detail do not matter."
    )

    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = f"llm:{llm.name}"

    def grade(self, q: Question, answer: Answer) -> str:
        r = self.llm.complete(self.SYSTEM, f"Question: {q.text}\nReference answer: {q.expected_answer}\nAnswer to grade: {answer.text}", max_tokens=150)
        try:
            grade = json.loads(re.search(r"\{.*\}", r.text, re.S).group(0))["grade"]
        except Exception:
            return "wrong"
        return grade if grade in ("correct", "partial", "wrong") else "wrong"


@dataclass
class Score:
    verdict_ok: bool
    grade: str | None  # None when the question expects no answer
    recall: float | None
    support: float | None
    leaked: bool
    leak_detail: str
    verifier_changed: bool
    execution_failed: bool
    coincidental: bool = False  # key facts present but none of the supporting evidence retrieved

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _doc(ref: str) -> str:
    return ref.split("#", 1)[0].split("@", 1)[0]


def find_leak(q: Question, answer: Answer, public_trace: dict, acl: dict[str, frozenset[str]]) -> str:
    """Anything outside the asking group's access that was retrieved, cited, traced or stated."""

    def restricted(ref: str) -> bool:
        groups = acl.get(_doc(ref))
        return groups is not None and q.group not in groups

    for ref in list(answer.retrieved_refs) + [c.ref for c in answer.citations]:
        if restricted(ref):
            return f"restricted evidence reached the answer: {ref}"
    blob = json.dumps(public_trace["events"])
    for m in re.finditer(r'"ref":\s*"([^"]+)"', blob):
        if restricted(m.group(1)):
            return f"restricted reference in public trace: {m.group(1)}"
    low = answer.text.lower()
    for secret in q.secret_facts:
        s = str(secret).lower()
        if (s in set(numbers(answer.text))) if s.isdigit() else (s in low):
            return "restricted value stated in the answer"
    return ""


def score_answer(q: Question, answer: Answer, public_trace: dict, acl: dict[str, frozenset[str]], judge: Judge) -> Score:
    expects_answer = q.expected_verdict == Verdict.ANSWERED
    grade = None
    if expects_answer:
        grade = judge.grade(q, answer) if answer.verdict == Verdict.ANSWERED else "wrong"
    wanted = q.section_refs + q.table_refs
    recall = None
    if expects_answer and wanted:
        got = set(answer.retrieved_refs)
        recall = sum(1 for r in wanted if r in got) / len(wanted)
    coincidental = grade == "correct" and recall == 0.0
    if coincidental:
        # The right words with none of the right evidence is luck, not retrieval.
        grade = "wrong"
    leak = find_leak(q, answer, public_trace, acl)
    return Score(
        verdict_ok=answer.verdict == q.expected_verdict,
        grade=grade,
        recall=recall,
        support=answer.support,
        leaked=bool(leak),
        leak_detail=leak,
        verifier_changed=answer.verifier_changed,
        execution_failed=answer.verdict == Verdict.EXECUTION_FAILED,
        coincidental=coincidental,
    )
