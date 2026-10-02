"""Answer generation from retrieved evidence.

`LLMGenerator` is the real one. `ExtractiveGenerator` needs no model: it copies
the best-matching sentences out of the evidence and cites them. It exists so the
whole system can be run and tested offline. It cannot paraphrase, combine facts
or reason, so its answers are a floor, not a measure of answer quality.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from .llm import LLM
from .text import content_tokens, sentences
from .types import Chunk, Citation, Hit, Usage

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass
class Draft:
    text: str
    citations: list[Citation] = field(default_factory=list)
    insufficient: bool = False
    usage: Usage = field(default_factory=Usage)


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Draft: ...


def numbered_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.title} / {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits, start=1))


def parse_citations(text: str, hits: list[Hit], kind: str = "chunk") -> list[Citation]:
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", text):
        n = int(m.group(1))
        if 1 <= n <= len(hits) and n not in seen:
            seen.append(n)
    return [Citation(n=n, kind=kind, ref=hits[n - 1].chunk.ref) for n in seen]


class LLMGenerator:
    SYSTEM = (
        "You answer questions using only the numbered sources provided. "
        "Put the source number in square brackets after every sentence, for example [2]. "
        "If two sources give different values for the same thing, use the one marked current. "
        f"If the sources do not contain the answer, reply with exactly {INSUFFICIENT} and nothing else. "
        "Do not use outside knowledge. Answer in one or two complete sentences that say what the value or fact refers to, "
        "not a bare number or name."
    )

    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = f"llm:{llm.name}"

    def generate(self, question: str, hits: list[Hit]) -> Draft:
        if not hits:
            return Draft("", insufficient=True)
        r = self.llm.complete(self.SYSTEM, f"Sources:\n\n{numbered_sources(hits)}\n\nQuestion: {question}")
        text = r.text.strip()
        if INSUFFICIENT in text:
            return Draft("", insufficient=True, usage=r.usage)
        return Draft(text, parse_citations(text, hits), usage=r.usage)


def _units(text: str) -> list[str]:
    """Sentences, with each markdown table row rewritten as "Header: value; Header: value"."""
    out, header = [], None
    for sent in sentences(text):
        if not sent.startswith("|"):
            header = None
            out.append(sent)
            continue
        cells = [c.strip() for c in sent.strip("|").split("|")]
        if header is None:
            header = cells
        elif not all(set(c) <= set("-: ") for c in cells):
            out.append("; ".join(f"{k}: {v}" for k, v in zip(header, cells)))
    return out


class ExtractiveGenerator:
    """Offline baseline. Picks the sentences that best cover the question's rarer terms."""

    name = "extractive"

    def __init__(self, idf: dict[str, float], default_idf: float, min_coverage: float = 0.4, max_sentences: int = 2):
        self._idf = idf
        self._default = default_idf
        self._min = min_coverage
        self._max = max_sentences

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], **kw) -> "ExtractiveGenerator":
        df: Counter[str] = Counter()
        for c in chunks:
            df.update(set(content_tokens(c.embedding_text())))
        n = max(len(chunks), 1)
        idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1) for t, d in df.items()}
        return cls(idf, math.log(n + 1), **kw)

    def _weight(self, tok: str) -> float:
        return self._idf.get(tok, self._default)

    def generate(self, question: str, hits: list[Hit]) -> Draft:
        q = set(content_tokens(question))
        total = sum(self._weight(t) for t in q)
        if not hits or total == 0:
            return Draft("", insufficient=True)
        scored: list[tuple[float, float, int, str]] = []
        for n, h in enumerate(hits, start=1):
            ctx = set(content_tokens(h.chunk.context))
            for sent in _units(h.chunk.text):
                if len(sent) < 12:
                    continue
                own = set(content_tokens(sent))
                # The sentence must carry part of the match itself; context alone is not evidence.
                own_cov = sum(self._weight(t) for t in q & own) / total
                cov = sum(self._weight(t) for t in q & (own | ctx)) / total
                if own_cov > 0:
                    scored.append((cov, own_cov, n, sent.lstrip("-0123456789. ").strip()))
        if not scored:
            return Draft("", insufficient=True)
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        if scored[0][0] < self._min:
            return Draft("", insufficient=True)
        picked = [scored[0]] + [s for s in scored[1:] if s[0] >= max(self._min, scored[0][0] * 0.9)][: self._max - 1]
        text = " ".join(f"{s[3].rstrip('.')}. [{s[2]}]" for s in picked)
        return Draft(text, parse_citations(text, hits))
