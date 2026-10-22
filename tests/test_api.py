"""Tests for the HTTP API."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from conversaflow.api.app import create_app
from conversaflow.config import Settings
from conversaflow.runtime.bootstrap import build_runtime


@pytest.fixture
def client(tmp_path, trained_model):
    settings = Settings.from_env(
        {
            "CONVERSAFLOW_STORE": "memory",
            "CONVERSAFLOW_MODEL_PATH": str(tmp_path / "m.json"),
        }
    )
    rt = build_runtime(settings, model=trained_model)
    app = create_app(settings, runtime=rt)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, settings


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200


def test_unauthenticated(client):
    c, _ = client
    r = c.get("/v1/flows")
    assert r.status_code == 401


def test_turn_flow(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    r = c.post("/v1/sessions", json={}, headers=h)
    assert r.status_code == 201
    sid = r.json()["session_id"]
    r = c.post("/v1/turns", json={"session_id": sid, "text": "hello"}, headers=h)
    assert r.status_code == 200
    assert "ConversaFlow" in r.json()["text"]
    # session-scoped turn
    r = c.post(f"/v1/sessions/{sid}/turns", json={"text": "thanks"}, headers=h)
    assert r.status_code == 200
    # mismatched session id
    r = c.post(f"/v1/sessions/{sid}/turns", json={"session_id": "other", "text": "hi"}, headers=h)
    assert r.status_code == 422


def test_session_get_delete(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    sid = c.post("/v1/sessions", json={}, headers=h).json()["session_id"]
    r = c.get(f"/v1/sessions/{sid}", headers=h)
    assert r.status_code == 200
    r = c.delete(f"/v1/sessions/{sid}", headers=h)
    assert r.status_code == 204


def test_knowledge_ingest_list_delete(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}  # admin
    payload = {"documents": [{"id": "d1", "title": "t", "text": "hello world"}]}
    r = c.post("/v1/knowledge", json=payload, headers=h)
    assert r.status_code == 201
    r = c.get("/v1/knowledge", headers=h)
    assert r.status_code == 200 and "d1" in r.json()["document_ids"]
    r = c.delete("/v1/knowledge/d1", headers=h)
    assert r.status_code == 200
    r = c.delete("/v1/knowledge/d1", headers=h)
    assert r.status_code == 404


def test_knowledge_forbidden(client):
    c, _ = client
    h = {"X-API-Key": "readonly-key"}  # viewer
    payload = {"documents": [{"id": "d1", "title": "t", "text": "hello"}]}
    r = c.post("/v1/knowledge", json=payload, headers=h)
    assert r.status_code == 403


def test_flows_and_agents(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    r = c.get("/v1/flows", headers=h)
    assert r.status_code == 200 and isinstance(r.json(), list)
    r = c.get("/v1/flows/order_status", headers=h)
    assert r.status_code == 200
    r = c.get("/v1/flows/nope", headers=h)
    assert r.status_code == 404
    r = c.get("/v1/agents", headers=h)
    assert r.status_code == 200
    r = c.post("/v1/agents/route", json={"intent": "request_refund", "confidence": 0.9}, headers=h)
    assert r.status_code == 200
    r = c.post("/v1/agents/route", json={"intent": "nope_intent", "confidence": 0.9}, headers=h)
    assert r.status_code == 404


def test_analyze_and_plan(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    r = c.post("/v1/analyze", json={"text": "hello"}, headers=h)
    assert r.status_code == 200
    r = c.post("/v1/plan", json={"text": "book an appointment"}, headers=h)
    assert r.status_code == 200


def test_barge_in_endpoint(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    sid = c.post("/v1/sessions", json={}, headers=h).json()["session_id"]
    r = c.post(f"/v1/sessions/{sid}/barge-in", json={}, headers=h)
    assert r.status_code == 200
    r = c.post(f"/v1/sessions/{sid}/barge-in", headers=h)
    assert r.status_code == 200


def test_metrics_endpoint(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    r = c.get("/metrics", headers=h)
    assert r.status_code == 200


def test_v1_health(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    r = c.get("/v1/health", headers=h)
    assert r.status_code == 200


def test_rate_limit(client):
    c, _ = client
    # create a limiter with limit 1 via app state
    app = c.app
    from conversaflow.api.ratelimit import RateLimiter

    app.state.limiter = RateLimiter(1)
    h = {"X-API-Key": "dev-key-123"}
    r1 = c.get("/v1/health", headers=h)
    r2 = c.get("/v1/health", headers=h)
    assert r1.status_code == 200
    assert r2.status_code == 429


def test_websocket(client):
    c, _ = client
    with c.websocket_connect("/v1/stream?api_key=dev-key-123") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
        ws.send_json({"type": "session.create", "locale": "en-US"})
        msg = ws.receive_json()
        assert msg["type"] == "session.created"
        sid = msg["session_id"]
        ws.send_json({"type": "turn", "session_id": sid, "text": "hello"})
        msg = ws.receive_json()
        assert msg["type"] == "turn"
        ws.send_json({"type": "barge_in", "session_id": sid})
        assert ws.receive_json()["type"] == "barge_in"
        ws.send_json({"type": "session.get", "session_id": sid})
        assert ws.receive_json()["type"] == "session"
        ws.send_json({"type": "session.delete", "session_id": sid})
        assert ws.receive_json()["type"] == "session.deleted"
        ws.send_json({"type": "foo"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json("not json")
        assert ws.receive_json()["type"] == "error"


def test_websocket_bad_auth(client):
    c, _ = client
    with pytest.raises(WebSocketDisconnect), c.websocket_connect("/v1/stream?api_key=bad"):
        pass


def test_error_envelope(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    r = c.post("/v1/turns", json={"text": ""}, headers=h)
    assert r.status_code == 422
    assert "error" in r.json()


def test_validation_error_handler_422_details(client):
    c, _ = client
    h = {"X-API-Key": "dev-key-123"}
    r = c.post("/v1/sessions", json={"locale": "en-xx"}, headers=h)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_websocket_viewer_cannot_mutate(client):
    c, _ = client
    with c.websocket_connect("/v1/stream?api_key=readonly-key") as ws:
        assert ws.receive_json()["type"] == "ready"
        for kind in ("session.create", "turn", "barge_in", "session.delete"):
            ws.send_json({"type": kind})
            assert ws.receive_json()["error"]["code"] == "forbidden"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
