"""Shared types. Every pipeline speaks in these and nothing else."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Principal:
    """Who is asking. Every search takes one; there is no anonymous search."""

    user_id: str
    groups: frozenset[str]

    @staticmethod
    def of(user_id: str, *groups: str) -> "Principal":
        return Principal(user_id, frozenset(groups))


@dataclass
class Chunk:
    """One retrievable passage.

    `ref` is the stable evidence reference used by citations and by the golden
    set (document id plus section slug). Several chunks may share a ref: a table
    row chunk points at the section its table lives in.
    """

    id: str
    document_id: str
    ref: str
    text: str
    groups: frozenset[str]
    title: str = ""
    section: str = ""
    doc_type: str = ""
    status: str = "current"  # "current" | "superseded"
    is_table_row: bool = False
    page_no: int | None = None
    context: str = ""  # document context prepended before embedding
    metadata: dict[str, Any] = field(default_factory=dict)

    def visible_to(self, principal: Principal) -> bool:
        return bool(self.groups & principal.groups)

    def embedding_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str  # "vector" | "keyword" | "fused" | "graph" | "rerank"


@dataclass
class SearchResult:
    """Hits the principal may see, plus how many candidates access control removed.

    `removed` is a count and only a count. Nothing about a removed chunk leaves
    the store.
    """

    hits: list[Hit]
    removed: int = 0


class Verdict(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ACCESS_DENIED = "access_denied"
    EXECUTION_FAILED = "execution_failed"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class Citation:
    n: int
    kind: str  # "chunk" | "row" | "page"
    ref: str


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    ms: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd


@dataclass
class Answer:
    text: str
    verdict: Verdict
    citations: list[Citation] = field(default_factory=list)
    retrieved_refs: list[str] = field(default_factory=list)  # everything that reached the model
    verifier_changed: bool = False
    support: float | None = None  # share of claims the verifier found supported
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "verdict": self.verdict.value,
            "citations": [c.__dict__ for c in self.citations],
            "retrieved_refs": self.retrieved_refs,
            "verifier_changed": self.verifier_changed,
            "support": self.support,
            "usage": self.usage.__dict__,
            "error": self.error,
        }
