"""Non-streaming app tests (spec §5, §7) — drive the FastAPI app IN-PROCESS via
``httpx.ASGITransport`` with a FAKE upstream injected (NO real llama-server /
network).

Async runner: pytest-asyncio (strict mode — each async test is marked
``@pytest.mark.asyncio``).

Proves:
  - POST /v1/chat/completions (no stream) with a bulky message -> the fake
    upstream RECEIVES a handle-ized payload (stub, not the blob); the proxy
    returns the fake upstream's JSON verbatim.
  - missing / non-list ``messages`` -> HTTP 400, error.type == invalid_request_error.
  - upstream raises UpstreamError -> HTTP 502, error.type == upstream_error.
  - GET /healthz -> 200 {"status":"ok"} and the fake upstream is NOT called.
"""

from __future__ import annotations

import pytest
import httpx

from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.proxy.upstream import UpstreamError


BULKY = " ".join(f"word{i}" for i in range(50))  # 50 words >= threshold(10)
FAKE_RESPONSE = {
    "id": "chatcmpl-x",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
}


async def _post(app, path: str, *, json=None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=json)


async def _get(app, path: str):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_bulky_message_handleized_before_forwarding(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    body = {"model": "test", "messages": [{"role": "user", "content": BULKY}]}
    r = await _post(app, "/v1/chat/completions", json=body)
    assert r.status_code == 200
    # The proxy returns the fake upstream's JSON verbatim (no response-side
    # rewriting in Phase 3).
    assert r.json() == FAKE_RESPONSE
    assert r.headers.get("content-type", "").startswith("application/json")
    # The fake upstream was called exactly once.
    assert upstream.call_count == 1
    payload = upstream.last_payload
    assert payload is not None
    # The bulky message was handle-ized: a stub, not the original blob.
    sent_messages = payload["messages"]
    assert len(sent_messages) == 1
    sent_content = sent_messages[0]["content"]
    assert PromptRewriter.is_stub(sent_content)
    assert BULKY not in sent_content
    # Non-messages fields are forwarded.
    assert payload["model"] == "test"


@pytest.mark.asyncio
async def test_small_message_forwarded_verbatim(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    small = "hello world"
    r = await _post(
        app,
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": small}]},
    )
    assert r.status_code == 200
    assert r.json() == FAKE_RESPONSE
    # Small message was NOT handle-ized.
    assert upstream.last_payload["messages"][0]["content"] == small


@pytest.mark.asyncio
async def test_missing_messages_returns_400(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    r = await _post(app, "/v1/chat/completions", json={"model": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    # Upstream was not called.
    assert upstream.call_count == 0


@pytest.mark.asyncio
async def test_messages_not_a_list_returns_400(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    r = await _post(
        app,
        "/v1/chat/completions",
        json={"messages": "not-a-list"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert upstream.call_count == 0


@pytest.mark.asyncio
async def test_upstream_error_returns_502(make_app):
    app, upstream, store, rewriter = make_app(error=UpstreamError("boom"))
    r = await _post(
        app,
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error"


@pytest.mark.asyncio
async def test_upstream_error_with_status_returns_that_status(make_app):
    """Spec §5 step 5: return the upstream status if it was an HTTP status error."""
    app, upstream, store, rewriter = make_app(error=UpstreamError("up", status_code=500))
    r = await _post(
        app,
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "upstream_error"


@pytest.mark.asyncio
async def test_props_passthrough_both_paths(make_app):
    """Clients on a /v1 base URL probe /v1/props; the proxy serves both /props and
    /v1/props by passing through to the upstream /props (fixes the live 404)."""
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    for path in ("/props", "/v1/props"):
        r = await _get(app, path)
        assert r.status_code == 200
    assert upstream.get_paths == ["/props", "/props"]


def test_apply_model_alias_shapes():
    """The model alias renames the upstream model (OpenAI + Ollama shapes), inherits other
    fields, and is a no-op when disabled or the payload is unrecognized."""
    from contextmanager.proxy.app import _apply_model_alias

    # OpenAI shape: {"data":[{"id":...}]}
    out = _apply_model_alias(
        {"object": "list", "data": [{"id": "real.gguf", "object": "model", "x": 1}]},
        "context-governor",
    )
    assert out["data"] == [{"id": "context-governor", "object": "model", "x": 1}]

    # Ollama shape: {"models":[{"name":...,"model":...}]}
    out = _apply_model_alias(
        {"models": [{"name": "real.gguf", "model": "real.gguf", "y": 2}]},
        "context-governor",
    )
    assert out["models"] == [{"name": "context-governor", "model": "context-governor", "y": 2}]

    # combined shape (llama-server returns BOTH keys) -> alias both
    out = _apply_model_alias(
        {"object": "list",
         "data": [{"id": "real.gguf", "object": "model"}],
         "models": [{"name": "real.gguf", "model": "real.gguf"}]},
        "context-governor",
    )
    assert out["data"][0]["id"] == "context-governor"
    assert out["models"][0]["name"] == "context-governor"
    assert out["models"][0]["model"] == "context-governor"

    # disabled (None / "") and unrecognized payloads pass through unchanged
    d = {"data": [{"id": "real"}]}
    assert _apply_model_alias(d, None) == d
    assert _apply_model_alias(d, "") == d
    assert _apply_model_alias({"foo": "bar"}, "context-governor") == {"foo": "bar"}


def test_resolve_handle_threshold_anchors_on_n_ctx():
    """The handle threshold anchors to the real window when n_ctx is known + ratio>0;
    falls back to the fixed value otherwise; honors the floor for tiny windows."""
    from contextmanager.proxy.app import resolve_handle_threshold
    from contextmanager.proxy.config import ProxyConfig

    base = ProxyConfig(upstream_base_url="http://x", handle_threshold_tokens=2000,
                       handle_threshold_ratio=0.04)
    assert resolve_handle_threshold(base, 75776) == int(75776 * 0.04)   # anchored
    assert resolve_handle_threshold(base, None) == 2000                 # n_ctx unknown -> fixed
    assert resolve_handle_threshold(base, 1000) == 256                  # tiny window -> floor
    off = ProxyConfig(upstream_base_url="http://x", handle_threshold_tokens=2000,
                      handle_threshold_ratio=0.0)
    assert resolve_handle_threshold(off, 75776) == 2000                 # anchoring disabled


def test_inject_context_length_both_shapes():
    from contextmanager.proxy.app import _inject_context_length
    out = _inject_context_length({"data": [{"id": "m"}], "models": [{"name": "m"}]}, 75776)
    assert out["data"][0]["context_length"] == 75776
    assert out["models"][0]["context_length"] == 75776
    d = {"data": [{"id": "m"}]}
    assert _inject_context_length(d, None) == d            # unknown n_ctx -> unchanged


@pytest.mark.asyncio
async def test_probe_n_ctx_reads_props():
    from contextmanager.proxy.app import _probe_n_ctx

    class _Up:
        def __init__(self, resp):
            self.resp = resp
        async def passthrough_get(self, path):
            if self.resp is None:
                raise RuntimeError("server down")
            return self.resp

    assert await _probe_n_ctx(_Up({"default_generation_settings": {"n_ctx": 75776}})) == 75776
    assert await _probe_n_ctx(_Up({"n_ctx": 4096})) == 4096       # top-level fallback
    assert await _probe_n_ctx(_Up({"nope": 1})) is None           # unexpected shape
    assert await _probe_n_ctx(_Up(None)) is None                  # server unreachable


@pytest.mark.asyncio
async def test_ollama_discovery_transparent_when_upstream_answers(make_app):
    """If the upstream DOES serve the Ollama path (real Ollama / multi-backend), the proxy
    returns it verbatim and never overrides it with the model list."""
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)  # no notfound
    r = await _get(app, "/api/tags")
    assert r.status_code == 200
    # forwarded to the ORIGINAL path; no fallback to /v1/models
    assert upstream.get_paths == ["/api/tags"]


@pytest.mark.asyncio
async def test_ollama_discovery_falls_back_to_models_on_404(make_app):
    """When the upstream 404s the Ollama path (llama-server has no /api/*), the proxy
    falls back to the upstream model list -> 200 (clean logs)."""
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, notfound={"/api/tags", "/api/v1/models", "/api/v1/tags"}
    )
    for path in ("/api/tags", "/api/v1/models", "/api/v1/tags"):
        r = await _get(app, path)
        assert r.status_code == 200
    # each tried the original path first, then fell back to /v1/models
    assert upstream.get_paths == [
        "/api/tags", "/v1/models",
        "/api/v1/models", "/v1/models",
        "/api/v1/tags", "/v1/models",
    ]


@pytest.mark.asyncio
async def test_healthz_ok_and_upstream_not_called(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    r = await _get(app, "/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    # /healthz MUST NOT touch the upstream.
    assert upstream.call_count == 0
    assert upstream.stream_count == 0
    assert upstream.get_paths == []


# ---------------------------------------------------------------------------
# GET /v1/models/{model_id} — synthesized retrieve-model reply
# ---------------------------------------------------------------------------

_MODELS_LIST = {
    "object": "list",
    "data": [{"id": "real.gguf", "object": "model", "owned_by": "llamacpp"}],
}


@pytest.mark.asyncio
async def test_retrieve_model_synthesized_from_the_list(make_app):
    """hermes-agent probes /v1/models/<alias> (OpenAI retrieve-model); llama-server
    has no such route, so the proxy synthesizes it from the SAME list /v1/models
    serves — aliased id, real context_length injected."""
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, get_responses={"/v1/models": _MODELS_LIST}
    )
    app.state.n_ctx = 75776
    r = await _get(app, "/v1/models/context-governor")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "context-governor"          # alias presented, not the gguf
    assert body["object"] == "model"
    assert body["owned_by"] == "llamacpp"            # other fields inherited
    assert body["context_length"] == 75776           # true window injected
    assert upstream.get_paths == ["/v1/models"]


@pytest.mark.asyncio
async def test_retrieve_model_unknown_id_is_an_openai_404(make_app):
    app, upstream, store, rewriter = make_app(
        response=FAKE_RESPONSE, get_responses={"/v1/models": _MODELS_LIST}
    )
    r = await _get(app, "/v1/models/no-such-model")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"
    assert "no-such-model" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Trailing assistant-run merge (llama-server 400s >=2 trailing assistants)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trailing_assistant_run_merged_before_forwarding(make_app):
    """The 2026-07-20 live 400: a harness wire ending [.., assistant, assistant]
    is rejected by llama-server's template preflight. The proxy merges the
    trailing run into ONE assistant message; the prefix is untouched."""
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    msgs = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "partial answer"},
        {"role": "assistant", "content": "continue prefill"},
    ]
    r = await _post(app, "/v1/chat/completions", json={"messages": msgs})
    assert r.status_code == 200
    sent = upstream.last_payload["messages"]
    assert len(sent) == 2                               # run collapsed
    assert sent[0] == msgs[0]                           # prefix byte-identical
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] == "partial answer\ncontinue prefill"
    r = await _get(app, "/metrics")
    assert r.json()["assistant_tail_merges"] == 1


@pytest.mark.asyncio
async def test_trailing_run_with_tool_calls_left_untouched(make_app):
    """tool_calls payloads cannot be merged coherently — the wire passes as-is
    (upstream's verdict stands) rather than being corrupted."""
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    msgs = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "function": {"name": "read"}}]},
        {"role": "assistant", "content": "retry prefill"},
    ]
    r = await _post(app, "/v1/chat/completions", json={"messages": msgs})
    assert r.status_code == 200
    assert upstream.last_payload["messages"] == msgs
    r = await _get(app, "/metrics")
    assert r.json()["assistant_tail_merges"] == 0


@pytest.mark.asyncio
async def test_single_trailing_assistant_is_not_a_run(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    msgs = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "one assistant prefill is legal"},
    ]
    r = await _post(app, "/v1/chat/completions", json={"messages": msgs})
    assert r.status_code == 200
    assert upstream.last_payload["messages"] == msgs


@pytest.mark.asyncio
async def test_trailing_run_content_parts_concatenated(make_app):
    """Mixed str + content-parts tails merge into one parts list (multimodal
    wires stay structured; nothing is dropped)."""
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    parts = [{"type": "text", "text": "see this"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    msgs = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "partial answer"},
        {"role": "assistant", "content": parts},
    ]
    r = await _post(app, "/v1/chat/completions", json={"messages": msgs})
    assert r.status_code == 200
    sent = upstream.last_payload["messages"]
    assert len(sent) == 2
    merged = sent[1]["content"]
    assert isinstance(merged, list)
    assert {"type": "text", "text": "partial answer"} in merged
    assert parts[1] in merged                           # image part survives
