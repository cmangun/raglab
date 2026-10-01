"""Small text helpers shared by the keyword index, the embedder and the checks."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")
_NUMBER = re.compile(r"(?<![\w.-])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\w-])")
# Record identifiers such as SOP-008, DEV-2025-033, CC-03, P-101, B-25-0042.
_RECORD_ID = re.compile(r"\b(?:SOP|DEV|CC|DR|AF|MEMO|SPEC|P|B)-[0-9A-Z]+(?:-[0-9A-Za-z]+)*\b")

STOPWORDS = frozenset(
    """a an and are as at be been by can did do does for from had has have how in is it its many may
    much must of on or that the their there these this to under was were what when where which who
    why will with within would you your each any all per than then into about""".split()
)


def tokens(text: str) -> list[str]:
    """Lowercase word tokens. Identifiers such as sop-008 stay whole and are also split."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if "-" in tok or "." in tok:
            out.extend(p for p in re.split(r"[-.]", tok) if p)
    return out


def raw_content_tokens(text: str) -> list[str]:
    """Content words before stemming, for engines that stem for themselves."""
    return [t for t in tokens(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in raw_content_tokens(text)]


def stem(tok: str) -> str:
    """A deliberately light stemmer: plural and common verb endings only."""
    if tok.isdigit() or "-" in tok:
        return tok
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            tok = tok[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    # quarantine / quarantined / quarantining all reduce to the same stem
    return tok[:-1] if tok.endswith("e") and len(tok) > 4 else tok


def sentences(text: str) -> list[str]:
    parts: list[str] = []
    for block in re.split(r"\n{2,}|\n(?=[-|\d])", text.strip()):
        block = block.strip()
        if block:
            parts.extend(s.strip() for s in _SENTENCE.split(block) if s.strip())
    return parts


def numbers(text: str) -> list[str]:
    """Numeric literals, normalised (no thousands separators, no trailing .0)."""
    out = []
    for raw in _NUMBER.findall(strip_record_ids(text)):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.append(n)
    return out


def record_ids(text: str) -> list[str]:
    return _RECORD_ID.findall(text)


def strip_record_ids(text: str) -> str:
    return _RECORD_ID.sub(" ", text)


def slug(text: str) -> str:
    return re.sub(r"^-|-$", "", re.sub(r"[^a-z0-9]+", "-", text.lower()))
