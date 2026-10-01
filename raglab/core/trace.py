"""Tamper-evident, redacted traces.

Every stage writes an event. Each event gets a receipt whose hash covers the
event and the previous receipt's hash, so editing, removing or reordering any
event breaks the chain. The format follows the agentic-evidence-viewer bundle
(trace events, receipts, manifest).

Redaction happens here, at write time. An event has a public payload and an
optional admin payload. The public payload is built from `visible_hits`, which
by construction contains only chunks the principal may see, plus counts. The
admin payload is stored separately and is never part of the public bundle.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from .types import Hit, SearchResult

GENESIS = "genesis"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _hash(prev_hash: str, event: dict) -> str:
    return "sha256:" + hashlib.sha256(prev_hash.encode() + b"\n" + _canonical(event)).hexdigest()


def visible_hits(result: SearchResult | list[Hit], limit: int = 20) -> dict[str, Any]:
    """Public description of a search result: refs and scores of visible hits, and a removed count."""
    hits = result.hits if isinstance(result, SearchResult) else result
    removed = result.removed if isinstance(result, SearchResult) else 0
    out: dict[str, Any] = {"hits": [{"ref": h.chunk.id, "score": round(h.score, 4), "source": h.source} for h in hits[:limit]]}
    if removed:
        out["removed_by_access_filter"] = removed
        out["note"] = f"{removed} inaccessible candidate{'s were' if removed != 1 else ' was'} removed after initial ranking."
    return out


class TraceWriter:
    def __init__(self, run: str, pipeline: str, question_id: str | None = None):
        self.bundle_id = f"{run}:{pipeline}:{question_id or uuid.uuid4().hex[:8]}"
        self.pipeline = pipeline
        self.events: list[dict] = []
        self.receipts: list[dict] = []
        self.admin: list[dict] = []
        self._t0 = time.perf_counter()

    def event(self, stage: str, public: dict[str, Any], admin: dict[str, Any] | None = None) -> None:
        seq = len(self.events)
        ev = {
            "event_id": f"{self.bundle_id}#{seq}",
            "event_type": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "sequence": seq,
            "payload": public,
            "metadata": {"ms_since_start": round((time.perf_counter() - self._t0) * 1000, 1)},
        }
        prev = self.receipts[-1]["hash"] if self.receipts else GENESIS
        self.events.append(ev)
        self.receipts.append(
            {"receipt_id": f"r-{seq:04d}", "event_id": ev["event_id"], "hash": _hash(prev, ev), "prev_hash": prev,
             "timestamp": ev["timestamp"], "event_type": stage}
        )
        if admin is not None:
            self.admin.append({"event_id": ev["event_id"], "payload": admin})

    def bundle(self) -> dict[str, Any]:
        """The public bundle. Admin payloads are deliberately not included."""
        return {
            "manifest": {"bundle_id": self.bundle_id, "version": "1.0.0", "created_at": self.events[0]["timestamp"] if self.events else "",
                         "event_count": len(self.events), "metadata": {"pipeline": self.pipeline}},
            "events": self.events,
            "receipts": self.receipts,
        }

    def public_text(self) -> str:
        return json.dumps(self.bundle(), ensure_ascii=False)


def verify_chain(bundle: dict[str, Any]) -> tuple[bool, str]:
    """Recompute every receipt hash from the events. Stricter than a linkage-only check."""
    events, receipts = bundle["events"], bundle["receipts"]
    if len(events) != len(receipts) or len(events) != bundle["manifest"]["event_count"]:
        return False, "event, receipt and manifest counts differ"
    prev = GENESIS
    for i, (ev, rc) in enumerate(zip(events, receipts)):
        if rc["prev_hash"] != prev:
            return False, f"chain broken at receipt {i}"
        if rc["event_id"] != ev["event_id"] or rc["hash"] != _hash(prev, ev):
            return False, f"receipt {i} does not match its event"
        prev = rc["hash"]
    return True, f"{len(receipts)} receipts verified"
