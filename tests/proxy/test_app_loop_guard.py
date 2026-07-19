"""Phase 13 app wiring — the loop-breaker on the proxy request path, driven
in-process via ASGITransport with a fake upstream (no network).

Proves:
  - k consecutive near-identical turns -> the k-th outbound payload carries the
    breaker notice APPENDED AT THE TAIL, and everything before it is
    byte-identical to what would have been sent anyway (append-only: the
    upstream prompt-cache prefix stays stable).
  - below k -> payload untouched; cooldown suppresses; second fire escalates.
  - hard-stop mode -> the proxy answers with a synthetic FINAL response
    (non-stream JSON and SSE) and the upstream is NOT called.
  - loop_guard_enabled=False -> feature fully off.
  - llama-server verbatim-recycling timings accelerate the trigger.
  - /metrics exposes loop_injections / loop_hard_stops.
"""

from __future__ import annotations

import httpx
import pytest

from contextmanager.loop_guard import BREAKER_MARKER


FAKE_RESPONSE = {
    "id": "chatcmpl-x",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
}


def turns(tokens: list) -> list[dict]:
    """The livelock shape: a growing transcript whose every turn appends one
    assistant tool-call + one tool result. Contents stay tiny (< the 10-word
    test threshold) so the rewrite is an identity and payload comparisons are
    byte-exact."""
    msgs = [
        {"role": "system", "content": "agent spec"},
        {"role": "user", "content": "do the task"},
    ]
    for tok in tokens:
        msgs.append({"role": "assistant", "content": f"calling tool {tok}"})
        msgs.append({"role": "tool", "content": f"result {tok}"})
    return msgs


async def _post(app, json):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=json)


async def _get(app, path: str):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def drive(app, tokens_per_request: list) -> list:
    """POST one chat request per growing prefix; return each response."""
    out = []
    for i in range(len(tokens_per_request)):
        r = await _post(app, {"messages": turns(tokens_per_request[: i + 1])})
        out.append(r)
    return out


@pytest.mark.asyncio
async def test_breaker_appended_at_tail_on_kth_repeat(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, loop_repeat_k=3,
    )
    await drive(app, ["a", "a", "a"])
    sent = upstream.last_payload["messages"]
    original = turns(["a", "a", "a"])
    # APPEND-ONLY: everything before the breaker is byte-identical to the
    # (identity) rewrite of the original transcript; exactly one message added.
    assert sent[:-1] == original
    breaker = sent[-1]
    assert breaker["role"] == "user"
    assert breaker["content"].startswith(BREAKER_MARKER)
    assert "repeated the same action 3 times" in breaker["content"]


@pytest.mark.asyncio
async def test_no_injection_below_k(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, loop_repeat_k=3,
    )
    await drive(app, ["a", "a"])
    assert upstream.last_payload["messages"] == turns(["a", "a"])
    assert upstream.call_count == 2


@pytest.mark.asyncio
async def test_distinct_turns_never_injected(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, loop_repeat_k=3,
    )
    await drive(app, ["a", "b", "c", "d"])
    assert upstream.last_payload["messages"] == turns(["a", "b", "c", "d"])


@pytest.mark.asyncio
async def test_cooldown_then_escalated_notice(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, loop_repeat_k=3, loop_cooldown_turns=1,
    )
    payloads = []
    for i in range(5):
        await _post(app, {"messages": turns(["a"] * (i + 1))})
        payloads.append(upstream.last_payload["messages"])
    # t3 fires (notice 1), t4 suppressed by the 1-turn cooldown, t5 escalates.
    assert len(payloads[0]) == len(turns(["a"]))            # untouched
    assert payloads[2][-1]["content"].startswith(BREAKER_MARKER)
    assert payloads[3] == turns(["a"] * 4)                  # cooldown: untouched
    assert "FINAL NOTICE" in payloads[4][-1]["content"]


@pytest.mark.asyncio
async def test_hard_stop_returns_synthetic_final_response(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE,
        loop_repeat_k=2, loop_cooldown_turns=0, loop_hard_stop=True,
    )
    responses = await drive(app, ["a"] * 4)
    calls_before_stop = upstream.call_count
    # t2 = notice, t3 = final notice, t4 = HARD STOP (upstream never called).
    stopped = responses[3].json()
    assert stopped["id"] == "loopguard-hardstop"
    assert stopped["choices"][0]["finish_reason"] == "stop"
    assert BREAKER_MARKER in stopped["choices"][0]["message"]["content"]
    assert upstream.call_count == calls_before_stop == 3


@pytest.mark.asyncio
async def test_hard_stop_streaming_returns_sse(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, stream_chunks=[b"data: [DONE]\n\n"],
        loop_repeat_k=2, loop_cooldown_turns=0, loop_hard_stop=True,
    )
    for i in range(3):
        await _post(app, {"messages": turns(["a"] * (i + 1))})
    r = await _post(app, {"messages": turns(["a"] * 4), "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "loopguard-hardstop" in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body
    assert upstream.stream_count == 0  # the synthetic stream, not the upstream's


@pytest.mark.asyncio
async def test_disabled_guard_never_touches_the_wire(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, loop_guard_enabled=False, loop_repeat_k=2,
    )
    assert app.state.loop_guard is None
    await drive(app, ["a"] * 6)
    assert upstream.last_payload["messages"] == turns(["a"] * 6)


@pytest.mark.asyncio
async def test_timings_corroboration_accelerates(make_app):
    recycled = {**FAKE_RESPONSE, "timings": {"draft_n": 500, "draft_n_accepted": 500}}
    app, upstream, store, rewriter = make_app(
        response=recycled, loop_repeat_k=3, loop_timings_m=1,
    )
    # Turn 1 response reports acceptance 1.0 on a 500-token draft -> effective
    # k drops to 2 -> the SECOND identical turn already carries the breaker.
    await drive(app, ["a", "a"])
    assert upstream.last_payload["messages"][-1]["content"].startswith(BREAKER_MARKER)


@pytest.mark.asyncio
async def test_metrics_expose_loop_counters(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE,
        loop_repeat_k=2, loop_cooldown_turns=0, loop_hard_stop=True,
    )
    await drive(app, ["a"] * 4)  # notice, final notice, then a hard stop
    m = (await _get(app, "/metrics")).json()
    assert m["loop_injections"] == 2
    assert m["loop_hard_stops"] == 1
