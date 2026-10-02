"""The live service: uploads, isolation between visitors, and the token."""

import base64

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from raglab.server import create_app  # noqa: E402
from raglab.service import RagService  # noqa: E402

TOKEN = "test-token"
H = {"Authorization": f"Bearer {TOKEN}"}
WS_A, WS_B = "visitor-aaaaaaaa", "visitor-bbbbbbbb"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(RagService("corpus"), token=TOKEN))


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def test_token_is_required(client):
    assert client.get("/describe").status_code == 401
    assert client.get("/describe", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/ask", json={"question": "x", "group": "quality"}).status_code == 401
    assert client.get("/healthz").status_code == 200


def test_service_with_no_token_configured_refuses_everything():
    c = TestClient(create_app(RagService("corpus"), token=""))
    assert c.get("/describe", headers={"Authorization": "Bearer "}).status_code == 401


def test_describe_lists_pipelines_and_sample(client):
    d = client.get("/describe", headers=H).json()
    assert [p["id"] for p in d["pipelines"]][0] == "hybrid" and d["sample"]["documents"] == 60
    assert d["sample"]["groups"] == ["commercial", "quality"]


def test_sample_question_answers_with_a_readable_source(client):
    r = client.post("/ask", headers=H, json={"question": "What does SOP-013 give as the calibration interval for balances and pipettes?",
                                              "source": "sample", "group": "commercial", "pipelines": ["hybrid", "naive"]}).json()
    by = {x["pipeline"]: x for x in r["results"]}
    assert by["hybrid"]["verdict"] == "answered" and "6 months" in by["hybrid"]["text"]
    assert by["hybrid"]["sources"][0]["title"].startswith("SOP-013") and "6 months" in by["hybrid"]["sources"][0]["text"]
    assert by["hybrid"]["steps"]


def test_sample_respects_groups_and_never_returns_restricted_text(client):
    r = client.post("/ask", headers=H, json={"question": "What is the maximum discount allowed on distributor contracts?",
                                              "source": "sample", "group": "quality", "pipelines": ["hybrid", "naive", "graph", "multimodal"]})
    assert "18 percent" not in r.text and "MEMO-" not in r.text
    assert all(x["verdict"] == "access_denied" for x in r.json()["results"])


def test_upload_then_ask(client):
    doc = "# Returns policy\n\nItems may be returned within 45 days of delivery.\n\n# Shipping\n\nOrders ship within 2 business days."
    d = client.post(f"/workspaces/{WS_A}/files", headers=H, json={"name": "policy.md", "content_base64": _b64(doc)})
    assert d.status_code == 200 and d.json()["upload"]["files"][0]["name"] == "policy.md"
    r = client.post("/ask", headers=H, json={"question": "How many days do customers have to return items?", "source": "upload",
                                              "workspace": WS_A, "pipelines": ["hybrid"]}).json()
    assert r["results"][0]["verdict"] == "answered" and "45 days" in r["results"][0]["text"]
    assert r["results"][0]["sources"][0]["title"] == "policy.md"


def test_one_visitor_cannot_query_another_visitors_files(client):
    client.post(f"/workspaces/{WS_B}/files", headers=H, json={"name": "other.md", "content_base64": _b64("The launch code word is heliotrope.")})
    r = client.post("/ask", headers=H, json={"question": "What is the launch code word?", "source": "upload", "workspace": WS_A, "pipelines": ["hybrid", "naive"]})
    assert "heliotrope" not in r.text
    assert client.post("/ask", headers=H, json={"question": "x?", "source": "upload", "workspace": "visitor-zzzzzzzz", "pipelines": ["hybrid"]}).status_code == 410


def test_upload_limits_and_types(client):
    assert client.post(f"/workspaces/{WS_A}/files", headers=H, json={"name": "a.exe", "content_base64": _b64("x")}).status_code == 415
    assert client.post(f"/workspaces/{WS_A}/files", headers=H, json={"name": "big.txt", "content_base64": _b64("x" * 2_000_001)}).status_code == 413
    assert client.post(f"/workspaces/{WS_A}/files", headers=H, json={"name": "policy.md", "content_base64": _b64("again")}).status_code == 409
    assert client.post("/workspaces/../etc/files", headers=H, json={"name": "a.txt", "content_base64": _b64("x")}).status_code in (400, 404)
    assert client.post(f"/workspaces/{WS_A}/files", headers=H, json={"name": "a.txt", "content_base64": "not base64!!"}).status_code == 400


def test_csv_becomes_a_table_and_clear_removes_everything(client):
    d = client.post(f"/workspaces/{WS_A}/files", headers=H, json={"name": "orders.csv", "content_base64": _b64("id,amount,region\n1,10,east\n2,30,west\n")}).json()
    assert any(f["kind"] == "data" and "2 rows" in f["detail"] for f in d["upload"]["files"])
    assert client.delete(f"/workspaces/{WS_A}", headers=H).json()["upload"] is None
    assert client.post("/ask", headers=H, json={"question": "x?", "source": "upload", "workspace": WS_A, "pipelines": ["hybrid"]}).status_code == 410
