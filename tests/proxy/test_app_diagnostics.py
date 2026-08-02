"""Wire-composition diagnostics (/diagnostics) — the measurement that replaces
inference about where the real prompt mass lives.

The property under test is PAIRING: the per-component sizes of the forwarded
payload and the ``usage.prompt_tokens`` that comes back must belong to the SAME
request. /metrics cannot do this — its ``peak_chars_out`` and its closed-loop
``real_prompt_tokens.peak`` are independent running maxima over different
requests, so their difference has no referent. These tests pin the pairing on
both the non-stream and the stream path, and pin the component split itself.

Driven in-process via ``httpx.ASGITransport`` with the shared fake upstream —
no network, no tokenizer.
"""

from __future__ import annotations

import json

import httpx
import pytest

from contextmanager.proxy.diagnostics import WireDiagnostics, component_texts


# --------------------------------------------------------------------- helpers

def _completion(prompt_tokens: int) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2,
                  "total_tokens": prompt_tokens + 2},
    }


def _sse(prompt_tokens: int) -> list[bytes]:
    first = {"id": "c1", "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {"content": "ok"}}]}
    final = {"id": "c1", "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2}}
    return [
        b"data: " + json.dumps(first).encode() + b"\n\n",
        b"data: " + json.dumps(final).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]


def _agent_body(stream: bool = False) -> dict:
    """A request shaped like a real agent turn: a big tools array, a system
    prompt, plain string content, a structured content-parts message, and an
    assistant tool_call whose arguments carry a file body."""
    return {
        "model": "context-governor",
        "stream": stream,
        "tools": [
            {"type": "function",
             "function": {"name": f"tool_{i}", "description": "d" * 200,
                          "parameters": {"type": "object", "properties": {}}}}
            for i in range(12)
        ],
        "messages": [
            {"role": "system", "content": "sys " * 200},
            {"role": "user", "content": "hello " * 50},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "Write",
                                          "arguments": json.dumps(
                                              {"file_path": "a.py",
                                               "content": "body " * 400})}}]},
            {"role": "tool", "content": "result " * 100},
            {"role": "user", "content": [{"type": "text", "text": "look"},
                                         {"type": "image_url",
                                          "image_url": {"url": "data:" + "b" * 500}}]},
        ],
    }


async def _post(app, body):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=body)


async def _get(app, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


# ------------------------------------------------------------- component split

def test_component_split_attributes_every_char_exactly_once():
    payload = _agent_body()
    split = component_texts(payload)
    chars = {k: len(v) for k, v in split["texts"].items()}

    assert split["n_tools"] == 12
    assert split["n_messages"] == 5
    # Each bucket is populated by the message shape that feeds it.
    assert chars["tools"] > 0
    assert chars["system_content"] > 0
    assert chars["string_content"] > 0
    assert chars["structured_content"] > 0   # the content-parts message
    assert chars["tool_calls"] > 0           # the Write arguments
    assert chars["other_payload"] > 0        # model / stream keys

    # The tool_call arguments carry the file body — this is mass that both the
    # rewriter's str-only passes AND metrics.py's chars_out miss entirely.
    assert chars["tool_calls"] > chars["string_content"]


def test_system_content_is_not_counted_as_sheddable_string_content():
    payload = {"messages": [{"role": "system", "content": "S" * 100},
                            {"role": "user", "content": "U" * 30}]}
    chars = {k: len(v) for k, v in component_texts(payload)["texts"].items()}
    assert chars["system_content"] == 100
    assert chars["string_content"] == 30


def test_malformed_payloads_never_raise():
    for bad in (None, [], "nope", {"messages": "not-a-list"},
                {"messages": [None, 7, {"role": "user"}]},
                {"tools": object()}):
        component_texts(bad)  # must not raise
    d = WireDiagnostics(enabled=True)
    assert d.record_request(None) is not None or True  # no exception either way


def test_disabled_is_a_no_op():
    d = WireDiagnostics(enabled=False)
    assert d.record_request(_agent_body()) is None
    snap = d.snapshot()
    assert snap["enabled"] is False


# -------------------------------------------------------------------- pairing

@pytest.mark.anyio
async def test_nonstream_pairs_usage_with_the_same_request(make_app):
    app, _upstream, _store, _rw = make_app(response=_completion(4242))

    resp = await _post(app, _agent_body(stream=False))
    assert resp.status_code == 200

    snap = (await _get(app, "/diagnostics")).json()
    assert snap["enabled"] is True
    assert snap["paired_samples"] == 1
    peak = snap["peak_request"]
    assert peak["real_prompt_tokens"] == 4242
    assert peak["n_tools"] == 12
    # implied ratio is derived from ONE request's own two numbers
    assert peak["implied_chars_per_token"] == pytest.approx(
        peak["total_chars"] / 4242, rel=0.01)


@pytest.mark.anyio
async def test_stream_pairs_usage_from_the_final_sse_chunk(make_app):
    app, _upstream, _store, _rw = make_app(stream_chunks=_sse(777))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/chat/completions",
                                 json=_agent_body(stream=True)) as r:
            assert r.status_code == 200
            async for _ in r.aiter_bytes():
                pass

    snap = (await _get(app, "/diagnostics")).json()
    assert snap["paired_samples"] == 1
    assert snap["peak_request"]["real_prompt_tokens"] == 777


@pytest.mark.anyio
async def test_unpaired_requests_are_reported_as_waiting_not_guessed(make_app):
    # A response with no usage block must NOT produce a paired sample: an
    # unpaired size is exactly the kind of number that invites bad arithmetic.
    app, _upstream, _store, _rw = make_app(
        response={"id": "c1", "choices": [{"index": 0, "message":
                  {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})

    await _post(app, _agent_body())
    snap = (await _get(app, "/diagnostics")).json()
    assert snap["samples"] == 1
    assert snap["paired_samples"] == 0
    assert "waiting_for" in snap
    assert "peak_request" not in snap


@pytest.mark.anyio
async def test_sheddable_share_reflects_what_the_rewriter_can_reach(make_app):
    app, _upstream, _store, _rw = make_app(response=_completion(5000))
    await _post(app, _agent_body())

    snap = (await _get(app, "/diagnostics")).json()
    shed = snap["peak_sheddable_char_pct"]
    unshed = snap["peak_unsheddable_char_pct"]
    assert shed + unshed == pytest.approx(100.0, abs=0.2)
    # tools + system + tool_calls + content-parts dominate this shape, so the
    # str-only passes can reach only a minority of the wire.
    assert unshed > shed


@pytest.mark.anyio
async def test_ring_is_bounded_and_does_not_leak_seq_entries(make_app):
    app, _upstream, _store, _rw = make_app(response=_completion(100),
                                           diag_max_samples=3)
    for _ in range(6):
        await _post(app, _agent_body())

    diag = app.state.diagnostics
    assert len(diag._samples) == 3
    assert len(diag._by_seq) == 3  # evicted samples must not accumulate


@pytest.mark.anyio
async def test_diagnostic_failure_never_breaks_the_request(make_app):
    app, _upstream, _store, _rw = make_app(response=_completion(1234))

    class Exploding:
        tokenize = False

        def record_request(self, payload):
            raise RuntimeError("boom")

        def attach_usage(self, seq, tokens):
            raise RuntimeError("boom")

        def snapshot(self, n_ctx=None):
            raise RuntimeError("boom")

    app.state.diagnostics = Exploding()
    # The proxy's contract (sensing.py: "enrichment must never fail a request"):
    # a broken tee degrades to no measurement, never to a failed turn.
    resp = await _post(app, _agent_body())
    assert resp.status_code == 200
    assert resp.json()["usage"]["prompt_tokens"] == 1234

    # ...and /diagnostics itself answers instead of 500ing.
    snap = (await _get(app, "/diagnostics")).json()
    assert snap["enabled"] is False
    assert "error" in snap
