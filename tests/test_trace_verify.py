import copy
import json

from conftest import QUALITY

from raglab.core.trace import TraceWriter, verify_chain, visible_hits
from raglab.core.types import Chunk, Hit, SearchResult
from raglab.core.verify import LexicalVerifier
from raglab.pipelines.base import run_pipeline


def _hit(text, ref="D#s"):
    return Hit(Chunk(id=ref, document_id="D", ref=ref, text=text, groups=frozenset({"quality"})), 1.0, "test")


def test_chain_verifies_and_detects_tampering():
    t = TraceWriter("r", "p", "Q")
    for i in range(4):
        t.event("stage", {"i": i})
    bundle = t.bundle()
    assert verify_chain(bundle)[0]
    edited = copy.deepcopy(bundle)
    edited["events"][1]["payload"]["i"] = 99
    assert not verify_chain(edited)[0]
    removed = copy.deepcopy(bundle)
    del removed["events"][2], removed["receipts"][2]
    removed["manifest"]["event_count"] = 3
    assert not verify_chain(removed)[0]
    swapped = copy.deepcopy(bundle)
    swapped["events"][1], swapped["events"][2] = swapped["events"][2], swapped["events"][1]
    assert not verify_chain(swapped)[0]


def test_admin_payload_is_not_in_the_public_bundle():
    t = TraceWriter("r", "p", "Q")
    t.event("error", {"type": "ValueError"}, admin={"message": "secret internal detail"})
    assert "secret internal detail" not in t.public_text()
    assert t.admin[0]["payload"]["message"] == "secret internal detail"


def test_removed_candidates_are_a_count_only():
    out = visible_hits(SearchResult([_hit("visible text", "SOP-001#scope")], removed=3))
    assert out["removed_by_access_filter"] == 3
    assert json.dumps(out).count("SOP-001#scope") == 1 and "MEMO" not in json.dumps(out)


def test_naive_trace_shows_count_not_content(lab):
    t = TraceWriter("r", "naive", "x")
    run_pipeline(lab.pipelines["naive"], "Commercial memo price list restricted to the Commercial group", QUALITY, t)
    text = t.public_text()
    assert "removed_by_access_filter" in text
    events = json.dumps([e["payload"] for e in t.events if e["event_type"] != "question"])
    assert "MEMO-" not in events


def test_verifier_accepts_supported_and_rejects_invented():
    hits = [_hit("Maximum bulk hold time before filling: 72 hours. Hold temperature: 2 to 8 °C.")]
    v = LexicalVerifier()
    assert v.verify("Maximum bulk hold time before filling is 72 hours. [1]", hits).supported
    assert not v.verify("Maximum bulk hold time before filling is 96 hours. [1]", hits).supported  # invented figure
    assert not v.verify("Maximum bulk hold time before filling is 72 hours.", hits).supported  # no citation
    assert not v.verify("The bulk hold limit is set by SOP-999. [1]", hits).supported  # invented record id
    assert not v.verify("Maximum bulk hold time is 72 hours. [4]", hits).supported  # cites a source that does not exist
