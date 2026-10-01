"""Turn front-matter markdown into chunks.

One chunk per `##` section. Each chunk carries a short document context (title
and section) that is prepended before embedding and keyword indexing, so a
passage such as "Maximum hold time: 72 hours" stays tied to the procedure it
belongs to.

Markdown tables additionally yield one chunk per row, with the column headers
folded in ("Parameter: Shelf life (months); Limit: 24"). Row chunks point at the
same evidence ref as their section and are only searched when a pipeline asks
for them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .text import slug
from .types import Chunk


@dataclass
class Document:
    id: str
    title: str
    type: str
    version: str
    status: str
    superseded_by: str | None
    groups: frozenset[str]
    effective: str
    sections: list[tuple[str, str]]  # (heading, body)


def parse_document(raw: str) -> Document:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        raise ValueError("document has no front matter")
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    groups = frozenset(g.strip() for g in meta["groups"].strip("[]").split(",") if g.strip())
    if not groups:
        raise ValueError(f"{meta.get('id')}: a document must list at least one group")
    sections = []
    for part in re.split(r"\n## ", m.group(2))[1:]:
        heading, _, body = part.partition("\n")
        sections.append((heading.strip(), body.strip()))
    return Document(
        id=meta["id"], title=meta["title"].strip('"'), type=meta["type"], version=meta.get("version", "1"),
        status=meta.get("status", "current"), superseded_by=meta.get("superseded_by") or None, groups=groups,
        effective=meta.get("effective", ""), sections=sections,
    )


def load_documents(folder: str | Path) -> list[Document]:
    return [parse_document(p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("*.md"))]


def _table_rows(body: str) -> list[dict[str, str]]:
    lines = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    cells = lambda ln: [c.strip() for c in ln.strip("|").split("|")]
    header = cells(lines[0])
    return [dict(zip(header, cells(ln))) for ln in lines[2:]]


def chunk_document(doc: Document, *, enrich: bool = True, table_rows: bool = True) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in doc.sections:
        ref = f"{doc.id}#{slug(heading)}"
        context = f"{doc.title}. Section: {heading}." if enrich else ""
        common = dict(document_id=doc.id, ref=ref, groups=doc.groups, title=doc.title, section=heading, doc_type=doc.type, status=doc.status)
        chunks.append(Chunk(id=ref, text=body, context=context, **common))
        if table_rows:
            for i, row in enumerate(_table_rows(body)):
                text = "; ".join(f"{k}: {v}" for k, v in row.items())
                chunks.append(Chunk(id=f"{ref}@row{i + 1}", text=text, context=context, is_table_row=True, metadata={"row": row}, **common))
    return chunks


def build_chunks(folder: str | Path, **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in load_documents(folder):
        out.extend(chunk_document(doc, **kw))
    return out


def load_tables(folder: str | Path) -> dict[str, list[dict]]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("*.json"))}
