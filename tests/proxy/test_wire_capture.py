"""Forensic wire capture — the in/out payload dump that makes prefix-break
attribution measurable offline.

The properties under test: pairing (one seq links the incoming request to the
forwarded payload), redaction (secret-bearing headers never reach disk),
normalized serialization (a diff of two captures reflects content, not
dict-order noise), and the proxy's standing contract — a broken capture
degrades to no capture, never to a failed request.
"""

from __future__ import annotations

import json

import httpx
import pytest

from contextmanager.proxy.diagnostics import WireCapture


def _body() -> dict:
    return {
        "model": "context-governor",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
    }


async def _post(app, body):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=body,
                                 headers={"Authorization": "Bearer s3cret"})


# ------------------------------------------------------------------ unit level

def test_disabled_by_default_is_a_no_op(tmp_path):
    cap = WireCapture(None)
    assert cap.enabled is False
    assert cap.record_in({}, _body()) is None
    cap.record_out(None, _body())
    assert list(tmp_path.iterdir()) == []


def test_in_out_pairing_shares_one_seq(tmp_path):
    cap = WireCapture(str(tmp_path))
    seq = cap.record_in({"x-test": "1"}, _body())
    assert seq == 1
    cap.record_out(seq, {"payload": True})

    in_rec = json.loads((tmp_path / "req-0001-in.json").read_text(encoding="utf-8"))
    out_rec = json.loads((tmp_path / "req-0001-out.json").read_text(encoding="utf-8"))
    assert in_rec["body"]["messages"][1]["content"] == "hello"
    assert in_rec["headers"]["x-test"] == "1"
    assert out_rec["payload"]["payload"] is True


def test_secret_headers_are_redacted(tmp_path):
    cap = WireCapture(str(tmp_path))
    cap.record_in({"authorization": "Bearer abc", "x-api-key": "k",
                   "user-agent": "ua"}, _body())
    rec = json.loads((tmp_path / "req-0001-in.json").read_text(encoding="utf-8"))
    assert rec["headers"]["authorization"] == "***"
    assert rec["headers"]["x-api-key"] == "***"
    assert rec["headers"]["user-agent"] == "ua"


def test_unserializable_content_degrades_to_repr_not_failure(tmp_path):
    # default=repr makes the dump total: a body carrying a non-JSON object is
    # captured with the object repr'd, never dropped and never raised.
    cap = WireCapture(str(tmp_path))
    assert cap.record_in({}, {"messages": [object()]}) is not None
    rec = json.loads((tmp_path / "req-0001-in.json").read_text(encoding="utf-8"))
    assert "<object object" in rec["body"]["messages"][0]


def test_max_requests_bounds_the_dump(tmp_path):
    cap = WireCapture(str(tmp_path), max_requests=2)
    assert cap.record_in({}, _body()) is not None
    assert cap.record_in({}, _body()) is not None
    assert cap.record_in({}, _body()) is None
    assert len(list(tmp_path.iterdir())) == 2


# ------------------------------------------------------------------- app level

@pytest.mark.anyio
async def test_capture_survives_the_full_request_path(make_app, tmp_path):
    cap_dir = tmp_path / "cap"
    app, upstream, _store, _rw = make_app(
        response={"id": "c1", "choices": [{"index": 0, "message":
                  {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
        wire_capture_dir=str(cap_dir),
    )

    resp = await _post(app, _body())
    assert resp.status_code == 200

    in_rec = json.loads((cap_dir / "req-0001-in.json").read_text(encoding="utf-8"))
    out_rec = json.loads((cap_dir / "req-0001-out.json").read_text(encoding="utf-8"))
    # The outgoing capture is what the upstream actually received...
    assert out_rec["payload"]["messages"] == upstream.last_payload["messages"]
    # ...and the authorization header reached disk only redacted.
    assert in_rec["headers"]["authorization"] == "***"


@pytest.mark.anyio
async def test_capture_failure_never_breaks_the_request(make_app):
    app, _upstream, _store, _rw = make_app(
        response={"id": "c1", "choices": [{"index": 0, "message":
                  {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})

    class Exploding:
        def record_in(self, headers, body):
            raise RuntimeError("boom")

        def record_out(self, seq, payload):
            raise RuntimeError("boom")

    app.state.capture = Exploding()
    resp = await _post(app, _body())
    assert resp.status_code == 200
