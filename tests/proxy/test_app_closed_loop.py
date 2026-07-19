"""Phase 14a app-level tests — the response tee and /metrics closed-loop view.

Driven in-process via ``httpx.ASGITransport`` with a fake upstream (same
harness as test_app_stream.py). Proves:

  - the stream tee is a PURE tee: forwarded bytes stay byte-identical while
    usage/timings from the final SSE chunk land in the ledger;
  - the non-stream branch observes usage the same way;
  - /metrics carries the ``closed_loop`` block (real prompt tokens, reuse
    ratio, breaks_by_cause, ledger summary) — zero server-log forensics;
  - sensing failures NEVER fail a request (the Phase-10 lesson applied to 14a).
"""

from __future__ import annotations

import httpx
import pytest


SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
    (b'data: {"choices":[{"delta":{}}],'
     b'"usage":{"prompt_tokens":1234,"completion_tokens":7},'
     b'"timings":{"prompt_n":34,"prompt_ms":50.0,'
     b'"prompt_per_second":680.0,"predicted_per_second":30.0}}\n\n'),
    b"data: [DONE]\n\n",
]

USAGE_RESPONSE = {
    "id": "x", "object": "chat.completion", "choices": [],
    "usage": {"prompt_tokens": 555, "completion_tokens": 9},
}


async def _post(app, json):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=json)


async def _get(app, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_stream_tee_byte_identical_and_observed(make_app):
    app, upstream, store, rewriter = make_app(stream_chunks=SSE_CHUNKS)
    body = {"messages": [{"role": "user", "content": "hello there"}],
            "stream": True}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            received = b"".join([chunk async for chunk in r.aiter_bytes()])

    # The tee must not alter, reorder, or delay a single byte.
    assert received == b"".join(SSE_CHUNKS)

    m = (await _get(app, "/metrics")).json()
    cl = m["closed_loop"]
    assert cl["real_prompt_tokens"]["last"] == 1234
    assert cl["real_prompt_tokens"]["peak"] == 1234
    assert cl["responses_with_timings"] == 1
    assert cl["real_reuse_ratio"] == round(1 - 34 / 1234, 4)


@pytest.mark.asyncio
async def test_nonstream_usage_observed(make_app):
    app, upstream, store, rewriter = make_app(response=USAGE_RESPONSE)
    r = await _post(app, {"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    assert r.json() == USAGE_RESPONSE           # response forwarded unchanged

    cl = (await _get(app, "/metrics")).json()["closed_loop"]
    assert cl["real_prompt_tokens"]["last"] == 555
    assert cl["responses_observed"] == 1
    assert cl["responses_with_timings"] == 0    # no timings block upstream
    assert cl["breaks_by_cause"] == {"new-conversation": 1}
    assert len(cl["conversations"]) == 1


@pytest.mark.asyncio
async def test_second_appended_turn_counts_no_new_break(make_app):
    app, upstream, store, rewriter = make_app(response=USAGE_RESPONSE)
    t1 = [{"role": "system", "content": "You are helpful"},
          {"role": "user", "content": "start the work"}]
    await _post(app, {"messages": t1})
    t2 = t1 + [{"role": "assistant", "content": "done"},
               {"role": "user", "content": "continue"}]
    await _post(app, {"messages": t2})
    cl = (await _get(app, "/metrics")).json()["closed_loop"]
    # One conversation, one new-conversation break, nothing voluntary.
    assert cl["breaks_by_cause"] == {"new-conversation": 1}
    conv = list(cl["conversations"].values())[0]
    assert conv["turns"] == 2


@pytest.mark.asyncio
async def test_sensing_request_failure_never_fails_request(make_app):
    app, upstream, store, rewriter = make_app(response=USAGE_RESPONSE)

    def boom(*a, **kw):
        raise RuntimeError("sensing exploded")

    app.state.controller.observe_request = boom  # type: ignore[method-assign]
    r = await _post(app, {"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    assert upstream.call_count == 1             # request went through untouched


@pytest.mark.asyncio
async def test_sensing_response_failure_never_fails_request(make_app):
    app, upstream, store, rewriter = make_app(response=USAGE_RESPONSE)

    def boom(*a, **kw):
        raise RuntimeError("tee exploded")

    app.state.controller.observe_response = boom  # type: ignore[method-assign]
    r = await _post(app, {"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    assert r.json() == USAGE_RESPONSE


@pytest.mark.asyncio
async def test_sensing_stream_failure_never_breaks_stream(make_app):
    app, upstream, store, rewriter = make_app(stream_chunks=SSE_CHUNKS)

    def boom(*a, **kw):
        raise RuntimeError("tee exploded")

    app.state.controller.observe_response = boom  # type: ignore[method-assign]
    body = {"messages": [{"role": "user", "content": "hello"}], "stream": True}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            received = b"".join([chunk async for chunk in r.aiter_bytes()])
    assert received == b"".join(SSE_CHUNKS)
