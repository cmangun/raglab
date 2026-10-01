"""Check that an answer is supported by what it cites.

`LexicalVerifier` is deterministic: every figure and record id in a sentence
must appear in a source that sentence cites, and most of its content words must
too. It catches invented numbers and uncited claims; it does not understand
meaning, so it can pass a sentence that reuses the right words wrongly.
`LLMVerifier` asks a model to judge each claim against its cited text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .llm import LLM
from .text import content_tokens, numbers, record_ids, sentences
from .types import Hit, Usage


@dataclass
class Verification:
    supported: bool
    claims: int
    unsupported: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def support_rate(self) -> float:
        return 1.0 if self.claims == 0 else (self.claims - len(self.unsupported)) / self.claims


class Verifier(Protocol):
    name: str

    def verify(self, answer: str, hits: list[Hit]) -> Verification: ...


def _claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into claims, each with the citation markers that follow it.

    "A is 1. B is 2 [3]. C [4][5]" gives (A is 1, [3]), (B is 2, [3]), (C, [4, 5]):
    a sentence with no marker of its own takes the next marker in the answer. Text
    after the last marker is an uncited claim.
    """
    out: list[tuple[str, list[int]]] = []
    pos = 0
    for m in re.finditer(r"((?:\s*\[\d+\])+)", answer):
        cites = [int(n) for n in re.findall(r"\d+", m.group(1))]
        for sent in sentences(answer[pos : m.start()]) or []:
            if sent.strip(" .\n"):
                out.append((sent.strip(), cites))
        pos = m.end()
    for sent in sentences(answer[pos:].lstrip(" .\n")) or []:
        if sent.strip(" .\n"):
            out.append((sent.strip(), []))
    return out


class LexicalVerifier:
    name = "lexical"

    def __init__(self, min_overlap: float = 0.6):
        self._min = min_overlap

    def verify(self, answer: str, hits: list[Hit]) -> Verification:
        claims = _claims(answer)
        bad: list[str] = []
        for text, cites in claims:
            sources = [hits[n - 1].chunk for n in cites if 1 <= n <= len(hits)]
            if not sources:
                bad.append(text)
                continue
            evidence = " ".join(f"{c.context} {c.text}" for c in sources)
            ev_tokens = set(content_tokens(evidence))
            toks = set(content_tokens(text))
            overlap = len(toks & ev_tokens) / len(toks) if toks else 1.0
            ev_numbers, ev_ids = set(numbers(evidence)), set(record_ids(evidence))
            if overlap < self._min or not set(numbers(text)) <= ev_numbers or not set(record_ids(text)) <= ev_ids:
                bad.append(text)
        return Verification(supported=not bad, claims=len(claims), unsupported=bad)


class LLMVerifier:
    SYSTEM = (
        "You check whether each claim is fully supported by the source text given for it. "
        'Reply with JSON only: {"results": [{"claim": 1, "supported": true}, ...]}. '
        "A claim is supported only if the source states it; do not use outside knowledge."
    )

    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = f"llm:{llm.name}"

    def verify(self, answer: str, hits: list[Hit]) -> Verification:
        claims = _claims(answer)
        if not claims:
            return Verification(True, 0)
        blocks, uncited = [], []
        for i, (text, cites) in enumerate(claims, start=1):
            sources = [hits[n - 1].chunk.text for n in cites if 1 <= n <= len(hits)]
            if not sources:
                uncited.append(text)
            blocks.append(f"Claim {i}: {text}\nSource for claim {i}:\n" + ("\n".join(sources) or "(none cited)"))
        r = self.llm.complete(self.SYSTEM, "\n\n".join(blocks))
        try:
            m = re.search(r"\{.*\}", r.text, re.S)
            results = {int(x["claim"]): bool(x["supported"]) for x in json.loads(m.group(0))["results"]}
        except Exception:
            # An unreadable verdict is not a pass.
            return Verification(False, len(claims), [c[0] for c in claims], r.usage)
        bad = [text for i, (text, _) in enumerate(claims, start=1) if not results.get(i, False) or text in uncited]
        return Verification(supported=not bad, claims=len(claims), unsupported=bad, usage=r.usage)
