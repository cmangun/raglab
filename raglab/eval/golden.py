"""The golden set and the access lists it is scored against."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..core.ingest import load_documents
from ..core.types import Principal, Verdict


@dataclass
class Question:
    id: str
    type: str
    text: str
    group: str
    expected_verdict: Verdict
    expected_answer: str
    expected_value: float | None
    key_facts: list
    secret_facts: list[str]
    section_refs: list[str]
    table_refs: list[str]
    hypothesis: str

    @property
    def principal(self) -> Principal:
        return Principal.of(f"eval-{self.group}", self.group)


def load_golden(path: str | Path) -> tuple[list[Question], str]:
    raw = Path(path).read_bytes()
    out = []
    for q in json.loads(raw):
        out.append(Question(
            id=q["id"], type=q["type"], text=q["text"], group=q["ask_as_group"], expected_verdict=Verdict(q["expected_verdict"]),
            expected_answer=q["expected_answer"], expected_value=q.get("expected_value"), key_facts=q.get("key_facts", []),
            secret_facts=q.get("secret_facts", []),
            section_refs=[r["ref"] for r in q["supporting_refs"] if r["kind"] == "section"],
            table_refs=[f"table:{r['table']}" for r in q["supporting_refs"] if r["kind"] == "rows"],
            hypothesis=q.get("hypothesis", ""),
        ))
    return out, hashlib.sha256(raw).hexdigest()[:12]


def access_lists(documents_dir: str | Path) -> dict[str, frozenset[str]]:
    """Document id -> groups, read from the source files, not from the store under test."""
    return {d.id: d.groups for d in load_documents(documents_dir)}
