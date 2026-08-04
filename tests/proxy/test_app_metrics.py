"""/metrics observability tests (Phase 5 §3) — drive the app in-process via
ASGITransport with a fake upstream; assert the proxy-side counter reflects the
prompt transform (and never touches the upstream).
"""

from __future__ import annotations

import httpx
import pytest

BULKY = " ".join(f"word{i}" for i in range(50))  # 50 words >= threshold(10)
FAKE_RESPONSE = {"id": "x", "object": "chat.completion", "choices": []}


async def _post(app, json):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=json)


async def _get(app, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_metrics_counts_handleization_and_wire_shrink(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert m["requests"] == 1
    assert m["messages_in"] == 1
    assert m["messages_handle_ized"] >= 1
    # The wire shrank: the bulky blob became a short stub.
    assert m["chars_out"] < m["chars_in"]
    assert m["chars_saved"] == m["chars_in"] - m["chars_out"]
    assert m["chars_saved"] > 0


@pytest.mark.asyncio
async def test_metrics_small_message_no_savings(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": "hello world"}]})
    m = (await _get(app, "/metrics")).json()
    assert m["requests"] == 1
    assert m["messages_handle_ized"] == 0
    assert m["messages_rehydrated"] == 0
    assert m["chars_saved"] == 0


@pytest.mark.asyncio
async def test_metrics_accumulates_across_requests(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert m["requests"] == 2
    assert m["messages_in"] == 2


@pytest.mark.asyncio
async def test_metrics_does_not_touch_upstream(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _get(app, "/metrics")
    assert upstream.call_count == 0
    assert upstream.stream_count == 0


@pytest.mark.asyncio
async def test_metrics_reports_token_estimates_and_summary(make_app):
    # Stage 8.0: /metrics surfaces an approximate token view + a human-readable summary.
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert m["tokens_in_est"] == m["chars_in"] // 4
    assert m["tokens_out_est"] == m["chars_out"] // 4
    assert m["tokens_saved_est"] == m["chars_saved"] // 4
    # RENAMED 2026-08-02: this was peak_prompt_tokens_est, which read as the peak
    # PROMPT size and is not -- every char here is message content, measured by
    # /diagnostics at 27-35% of the wire, with the tools array and chat template
    # invisible. It under-reported 9,558 against a real 37,864.
    assert m["peak_message_tokens_est"] == m["chars_out"] // 4  # one request -> peak == it
    assert "peak_prompt_tokens_est" not in m, "the misleading name must not come back"
    assert isinstance(m["pct_saved"], (int, float))
    assert "message content" in m["est_scope"]
    assert isinstance(m["summary"], str) and "tokens" in m["summary"]
    # The summary must SAY the savings are over message content, so ~80% cannot
    # be read as "80% of the prompt".
    assert "message content" in m["summary"]


@pytest.mark.asyncio
async def test_summary_prefers_the_observed_prompt_peak(make_app):
    """When sensing has seen a real prompt size, the summary reports THAT rather
    than an estimate over message content."""
    app, upstream, store, rewriter = make_app(
        response={**FAKE_RESPONSE, "usage": {"prompt_tokens": 4321, "completion_tokens": 1}})
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert m.get("peak_prompt_tokens") == 4321
    assert "peak prompt" in m["summary"]
    assert "unobserved" not in m["summary"]


@pytest.mark.asyncio
async def test_metrics_summary_empty_before_any_request(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    m = (await _get(app, "/metrics")).json()
    assert m["summary"] == "no requests yet"
    assert m["tokens_saved_est"] == 0


@pytest.mark.asyncio
async def test_metrics_includes_retrieval_block(make_app):
    # Phase 7 Stage 1: /metrics merges the shared store's retrieval-path counters.
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    # One bulky message -> one note paged out -> corpus_size == 1. Since Phase 10
    # the proxy ITSELF searches (Pass-4 auto-recall probes the store each request),
    # so the retrieval counters are now exercised by the hot path.
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert "retrieval" in m
    r = m["retrieval"]
    assert r["corpus_size"] == 1
    assert r["search_calls"] >= 1          # Pass-4 auto-recall probes the store
    assert "recall_hit_rate" in r
    assert "avg_search_ms" in r
    assert "slices_recalled" in m          # proxy-level recall counter (Phase 10)
    # The CHAT path may lazily probe /props when n_ctx is unknown (_ensure_n_ctx:
    # losing the startup race with llama-server used to disable windowing for the
    # life of the process). What this pins is narrower and unchanged — /metrics
    # ITSELF never touches upstream.
    before = list(upstream.get_paths)
    (await _get(app, "/metrics")).json()
    assert upstream.get_paths == before


@pytest.mark.asyncio
async def test_metrics_exposes_upstream_context_sizing(make_app):
    """Every setpoint derives from the upstream's n_ctx, and n_ctx being
    unresolved does not merely soften the threshold — it disables windowing
    outright. That used to be discoverable only by reading code; `resolved` and
    `windowing_enabled` say it in one field each."""
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    uc = m["upstream_context"]
    assert set(uc) >= {
        "n_ctx", "resolved", "source", "adoptions",
        "handle_threshold_tokens", "windowing_enabled",
    }
    assert uc["resolved"] is bool(uc["n_ctx"])
    # Whatever the fake upstream advertises, the reported threshold must be the
    # one the rewriter is ACTUALLY using — not the static config default.
    assert uc["handle_threshold_tokens"] == rewriter.config.handle_threshold_tokens


@pytest.mark.asyncio
async def test_bracket_agrees_with_the_reported_window(make_app):
    """The lifespan probe used to set app.state.n_ctx directly and bypass the
    resolver, so /metrics reported a window beside a bracket reading
    "unresolved" -- the belief and the evidence behind it disagreeing on the
    surface whose whole job is making the belief auditable."""
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    uc = (await _get(app, "/metrics")).json()["upstream_context"]
    bracket = uc.get("bracket")
    assert bracket is not None
    if uc["n_ctx"]:
        assert bracket["resolved"] is True, "window reported but bracket unresolved"
        assert bracket["window"] == uc["n_ctx"]
