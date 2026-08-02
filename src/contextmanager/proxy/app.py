from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..durable import DurableStore
from ..loop_guard import LoopGuard, LoopGuardConfig, LoopGuardDecision, hard_stop_text
from ..tokenizer import LlamaServerTokenCounter
from ..types import TokenCounter
from .config import ProxyConfig
from .diagnostics import WireCapture, WireDiagnostics
from .metrics import StatsCollector
from .rewriter import PromptRewriter, RewriteResult, normalize_volatile_stamps
from .sensing import GovernorController, StreamTee, extract_usage_timings
from .upstream import UpstreamClient, UpstreamError


# Floor for the n_ctx-anchored handle-ization threshold (don't stub trivially small msgs).
_MIN_HANDLE_THRESHOLD = 256


def resolve_handle_threshold(config: ProxyConfig, n_ctx: Optional[int]) -> int:
    """Effective per-message handle-ization threshold. When ``handle_threshold_ratio``
    > 0 AND the upstream's true context size ``n_ctx`` is known, anchor it to the real
    window (``ratio * n_ctx``, floored). Otherwise fall back to the fixed
    ``handle_threshold_tokens``. llama-server is the source of truth for context size."""
    if config.handle_threshold_ratio > 0.0 and n_ctx:
        return max(_MIN_HANDLE_THRESHOLD, int(n_ctx * config.handle_threshold_ratio))
    return config.handle_threshold_tokens


async def _probe_n_ctx(upstream: UpstreamClient) -> Optional[int]:
    """Best-effort read of llama-server's true context size from /props. None on any
    failure (server down at startup, unexpected shape) -> caller falls back."""
    try:
        data = await upstream.passthrough_get("/props")
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return int(data["default_generation_settings"]["n_ctx"])
    except (KeyError, TypeError, ValueError):
        try:
            return int(data["n_ctx"])  # some builds expose it at the top level
        except (KeyError, TypeError, ValueError):
            return None


def _inject_context_length(data: dict, n_ctx: Optional[int]) -> dict:
    """Advertise the upstream's true context size on each model entry (OpenAI `data`
    and Ollama `models` shapes), so clients read the real window from /v1/models."""
    if not n_ctx or not isinstance(data, dict):
        return data
    out = dict(data)
    for key in ("data", "models"):
        items = out.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            out[key] = [{**it, "context_length": n_ctx} for it in items]
    return out


def _apply_model_alias(data: dict, alias: Optional[str]) -> dict:
    """Present the upstream model list under ``alias`` (e.g. "context-governor"),
    inheriting every other field from the real loaded model. Handles both the OpenAI
    shape (`{"data":[{"id":…}]}`) and the Ollama shape (`{"models":[{"name":…}]}`).
    No-op when alias is falsy or the payload has no recognizable model list.
    """
    if not alias or not isinstance(data, dict):
        return data
    out = dict(data)
    # llama-server can return BOTH keys (OpenAI `data` + Ollama `models`); alias each.
    items = out.get("data")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        out["data"] = [{**items[0], "id": alias}]
    items = out.get("models")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        out["models"] = [{**items[0], "name": alias, "model": alias}]
    return out


def _sum_content_chars(messages: list) -> int:
    """Total chars of string-valued message contents (free; no tokenization)."""
    total = 0
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str):
                total += len(c)
    return total


def _prompt_tokens_of(observed: Optional[dict]) -> Optional[int]:
    """``usage.prompt_tokens`` out of an extract_usage_timings() result — the
    upstream's own count of what it actually tokenized. None when absent."""
    if not isinstance(observed, dict):
        return None
    usage = observed.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("prompt_tokens")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_loop_guard(config: ProxyConfig) -> Optional[LoopGuard]:
    """Construct the Phase 13 loop-breaker from the proxy config; None = disabled."""
    if not config.loop_guard_enabled:
        return None
    return LoopGuard(LoopGuardConfig(
        enabled=True,
        repeat_k=config.loop_repeat_k,
        timings_m=config.loop_timings_m,
        accept_threshold=config.loop_accept_threshold,
        draft_n_min=config.loop_draft_n_min,
        cooldown_turns=config.loop_cooldown_turns,
        hard_stop=config.loop_hard_stop,
    ))


def _loop_hard_stop_response(decision: LoopGuardDecision, *, stream: bool):
    """Synthetic FINAL response ending a run the loop guard hard-stopped.

    The upstream is never called; the client sees a normal completion whose
    assistant content explains the stop (finish_reason "stop" ends the
    harness's auto-continue loop the natural way).
    """
    text = hard_stop_text(decision.streak)
    created = int(time.time())
    if not stream:
        return JSONResponse({
            "id": "loopguard-hardstop",
            "object": "chat.completion",
            "created": created,
            "model": "context-governor",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }, media_type="application/json")

    def _chunk(delta: dict, finish: Optional[str]) -> bytes:
        return b"data: " + json.dumps({
            "id": "loopguard-hardstop",
            "object": "chat.completion.chunk",
            "created": created,
            "model": "context-governor",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }).encode("utf-8") + b"\n\n"

    async def _body():
        yield _chunk({"role": "assistant", "content": text}, None)
        yield _chunk({}, "stop")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_body(), media_type="text/event-stream")


def _upstream_error_response(exc: UpstreamError) -> JSONResponse:
    """Map an UpstreamError onto the outbound 502-ish JSON error shape.

    Status is the upstream HTTP status when it is a sane int in 400..599,
    otherwise 502 (bad gateway).
    """
    status = exc.status_code
    if not (isinstance(status, int) and 400 <= status <= 599):
        status = 502
    return JSONResponse(
        {"error": {"message": str(exc), "type": "upstream_error"}},
        status_code=status,
        media_type="application/json",
    )


def _invalid_request(message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error"}},
        status_code=400,
        media_type="application/json",
    )


def _merge_trailing_assistant_run(messages: list) -> tuple:
    """Collapse a TRAILING run of >=2 assistant messages into one.

    llama-server's template preflight rejects such wires outright
    (``Cannot have 2 or more assistant messages at the end of the list`` —
    observed live 2026-07-20 when a harness's continue/retry stitching left
    [..., assistant, assistant] at the tail; the run died on a 400). The proxy
    never produces assistant messages itself, so this is boundary repair of
    the HARNESS's wire: merge text contents (or concatenate content-parts
    lists) into a single assistant message. Tail-only, deterministic, nothing
    dropped — and the prefix up to the run stays byte-stable for the
    upstream's prompt cache. Runs carrying ``tool_calls`` are left untouched
    (their payloads cannot be merged coherently — upstream's verdict stands).

    Returns ``(messages, merged_run_length)``; ``merged_run_length`` is 0 when
    no merge happened.
    """
    if len(messages) < 2:
        return messages, 0
    i = len(messages)
    while (i > 0 and isinstance(messages[i - 1], dict)
           and messages[i - 1].get("role") == "assistant"):
        i -= 1
    run = len(messages) - i
    if run < 2:
        return messages, 0
    tail = messages[i:]
    if any(m.get("tool_calls") for m in tail):
        return messages, 0
    texts: list[str] = []
    parts: list = []
    structured = False
    for m in tail:
        c = m.get("content")
        if isinstance(c, str):
            if c:
                texts.append(c)
                parts.append({"type": "text", "text": c})
        elif isinstance(c, list):
            structured = True
            parts.extend(p for p in c if isinstance(p, dict))
        # content None -> nothing to carry over
    merged = {**tail[-1], "content": parts if structured else "\n".join(texts)}
    return [*messages[:i], merged], run


def create_app(
    config: ProxyConfig,
    *,
    upstream: Optional[UpstreamClient] = None,
    store: Optional[DurableStore] = None,
    counter: Optional[TokenCounter] = None,
) -> FastAPI:
    """Build the FastAPI proxy app.

    The optional ``upstream``/``store``/``counter`` are testability hooks: when
    provided they are used as-is; when omitted, the real Phase 1/2 objects are
    constructed on startup. Tests may also overwrite ``app.state.upstream`` /
    ``app.state.store`` / ``app.state.counter`` / ``app.state.rewriter`` AFTER
    construction and BEFORE issuing requests (the lifespan will not clobber
    instances it did not create itself).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build only what was NOT injected. Idempotent: if a value is already
        # present on app.state (injected by the caller), do not replace it.
        if app.state.store is None:
            app.state.store = DurableStore(
                config.store_root,
                hotness_half_life=config.hotness_half_life_seconds,
            )
            app.state._owns_store = True
        else:
            app.state._owns_store = False

        if app.state.counter is None:
            app.state.counter = LlamaServerTokenCounter(
                config.upstream_base_url, api_key=config.upstream_api_key
            )
            app.state._owns_counter = True
        else:
            app.state._owns_counter = False

        if app.state.upstream is None:
            app.state.upstream = UpstreamClient(config)
            app.state._owns_upstream = True
        else:
            app.state._owns_upstream = False

        # Anchor on llama-server's TRUE context size (the source of truth, not the
        # CLI). Best-effort + short-timed: if the server isn't reachable at startup,
        # fall back to the fixed threshold. Cached for /v1/models propagation.
        try:
            app.state.n_ctx = await asyncio.wait_for(
                _probe_n_ctx(app.state.upstream), timeout=5.0
            )
        except Exception:
            app.state.n_ctx = None

        resolved = config
        effective_threshold = resolve_handle_threshold(config, app.state.n_ctx)
        if effective_threshold != config.handle_threshold_tokens:
            resolved = dataclasses.replace(
                config, handle_threshold_tokens=effective_threshold
            )

        # The rewriter is always (re)built here so it wires to the resolved
        # counter/store. Tests that inject a custom rewriter should set it
        # AFTER startup; the request handlers read app.state.rewriter live.
        app.state.rewriter = PromptRewriter(
            resolved, app.state.counter, app.state.store, n_ctx=app.state.n_ctx
        )
        app.state.config = resolved

        # Phase 14c: restore learned harness profiles (native-compaction
        # ceilings) from the contextstore so a proxy restart keeps its
        # calibration. Best-effort — sensing never blocks startup.
        try:
            app.state.controller.load_profiles(app.state.store.state)
        except Exception:
            pass

        try:
            yield
        finally:
            # Shutdown: close only what we own. Injected instances are the
            # caller's responsibility.
            if getattr(app.state, "_owns_upstream", False) and app.state.upstream is not None:
                await app.state.upstream.aclose()
            if getattr(app.state, "_owns_counter", False) and app.state.counter is not None:
                close_counter = getattr(app.state.counter, "close", None)
                if callable(close_counter):
                    close_counter()
            if getattr(app.state, "_owns_store", False) and app.state.store is not None:
                app.state.store.close()

    app = FastAPI(
        title="contextmanager-proxy",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Pre-seed app.state with injected instances so the lifespan knows what to
    # build vs. reuse. Tests may also overwrite these before issuing requests.
    app.state.config = config
    app.state.upstream = upstream
    app.state.store = store
    app.state.counter = counter
    app.state.rewriter = None  # type: ignore[assignment]
    # In-memory observability (Phase 5 §3). Set here (not only in the lifespan)
    # so it exists in injected-test mode too, where ASGITransport skips lifespan.
    app.state.stats = StatsCollector()
    # Wire-composition tee: per-component sizes of the FORWARDED payload paired
    # with the real usage.prompt_tokens of that same request. /metrics reports
    # peak_chars_out and real_prompt_tokens.peak as independent maxima over
    # DIFFERENT requests, so their difference is meaningless; this is the
    # measured answer to "where does the prompt mass actually live".
    app.state.diagnostics = WireDiagnostics(
        enabled=config.diag_enabled,
        max_samples=config.diag_max_samples,
        tokenize=config.diag_tokenize,
    )
    # Forensic wire capture (in + out payloads per request, diffable offline).
    # Gated on config; disabled = a None check per request.
    app.state.capture = WireCapture(config.wire_capture_dir)
    # Phase 14 closed loop: per-conversation sensing ledger, request-diff
    # classifier, break attribution, learned harness ceilings. Created here so
    # it exists in injected-test mode too; profiles load in the lifespan.
    app.state.controller = GovernorController(
        max_conversations=config.max_conversations
    )
    # Serializes the off-event-loop rewrite (run via asyncio.to_thread) so the store's
    # sqlite connections (opened check_same_thread=False) are never used concurrently.
    app.state.rewrite_lock = asyncio.Lock()
    # Upstream true context size (filled at startup by the lifespan probe); None
    # in injected-test mode (lifespan skipped) -> no context propagation/anchoring.
    app.state.n_ctx = None
    # Phase 13 loop-breaker. Built here (not in the lifespan) so it exists in
    # injected-test mode too; pure-python state, no I/O, None when disabled.
    app.state.loop_guard = build_loop_guard(config)

    # ----------------------------------------------------------- /v1/chat/completions

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception as e:
            return _invalid_request(f"invalid JSON body: {e}")

        if not isinstance(body, dict):
            return _invalid_request("request body must be a JSON object")

        messages = body.get("messages")
        if not isinstance(messages, list):
            return _invalid_request("'messages' must be a list")

        # Forensic tee: the request AS RECEIVED, before any pass touches it.
        # Best-effort — a capture failure must never fail a request.
        capture = getattr(app.state, "capture", None)
        capture_seq = None
        if capture is not None:
            try:
                capture_seq = capture.record_in(request.headers, body)
            except Exception:
                capture_seq = None

        # Pass -1: blank per-request telemetry stamps (Claude Code's billing-header
        # nonce) BEFORE anything reads the wire. Order matters and is load-bearing:
        # record_in above keeps the RAW request as forensic truth, while sensing,
        # the loop guard and the rewrite below all see ONE normalized wire — so the
        # conversation key is derived from exactly the bytes that get forwarded and
        # the two can never drift apart. See rewriter.normalize_volatile_stamps.
        try:
            messages = normalize_volatile_stamps(messages)
        except Exception:
            pass  # fail open: a normalization bug must never fail a request

        # Phase 13: observe the ORIGINAL transcript (pre-rewrite — independent of
        # stubbing/store state) and decide whether this turn needs a breaker.
        guard: Optional[LoopGuard] = app.state.loop_guard
        decision: Optional[LoopGuardDecision] = (
            guard.observe_request(messages) if guard is not None else None
        )
        if decision is not None and decision.hard_stop:
            # End the run with a synthetic final response; upstream never called.
            app.state.stats.record_loop_hard_stop()
            return _loop_hard_stop_response(decision, stream=body.get("stream") is True)

        rewriter: PromptRewriter = app.state.rewriter
        controller: Optional[GovernorController] = app.state.controller
        cfg: ProxyConfig = app.state.config

        # Phase 14a sensing (request side): classify this turn's wire against
        # the conversation's last one and derive the closed-loop inputs.
        # Sensing is ENRICHMENT: any failure degrades to open-loop legacy
        # behavior — it must never fail the request.
        obs = None
        pressure: Optional[int] = None
        high_water: Optional[int] = None
        if controller is not None:
            try:
                # Hand sensing the rewriter's settings so the pressure estimate
                # models the wire that will actually be SENT (post-Pass-1), not
                # the raw incoming one. Without this a bulky tool result is
                # counted at chars/4 even though it leaves as a stub — measured
                # at 37,666 estimated vs 21,259 real, which tripped windowing
                # and broke the prefix for nothing.
                obs = controller.observe_request(
                    messages,
                    handle_threshold_tokens=resolve_handle_threshold(
                        cfg, app.state.n_ctx),
                    stub_tokens_est=PromptRewriter.stub_tokens_estimate(
                        cfg.stub_preview_chars),
                )
                # Windowing setpoints only when windowing is enabled at all —
                # a learned ceiling must never resurrect a feature the user
                # disabled with context_budget_ratio = 0.
                if cfg.context_budget_ratio > 0.0:
                    pressure = obs.pressure_tokens
                    high_water = controller.effective_high_water(
                        app.state.n_ctx, cfg.context_budget_ratio,
                        cfg.ceiling_safety, obs.harness_fp,
                    )
            except Exception:
                obs, pressure, high_water = None, None, None

        # Run the (sync, CPU+IO-bound) rewrite OFF the event loop so a heavy message never
        # freezes the proxy — /healthz, /metrics and other requests stay responsive. The
        # lock serializes store access so the worker thread is the only one touching the
        # sqlite connections at a time.
        async with app.state.rewrite_lock:
            result: RewriteResult = await asyncio.to_thread(
                rewriter.rewrite_outgoing, messages,
                prefix_broken=(obs.prefix_broken if obs is not None else False),
                pressure_tokens=pressure,
                high_water_tokens=high_water,
            )
        out_messages = result.messages
        if decision is not None and decision.breaker_text is not None:
            # APPEND-ONLY, at the very tail: earlier history must stay byte-stable
            # (the upstream's hybrid-SSM prompt cache needs f_keep = 1.0; a
            # mid-history edit forces a 30-60 s full re-prefill). Role "user":
            # strict templates reject mid-wire "system" (same live lesson as the
            # rewriter's synthetic messages).
            out_messages = [*out_messages,
                            {"role": "user", "content": decision.breaker_text}]
        # Boundary repair AFTER all passes (incl. the breaker tail — a trailing
        # breaker user message already ends any assistant run): llama-server
        # 400s wires ending in >=2 assistant messages. Tail-only merge; like
        # the breaker, deliberately NOT folded into note_sent's sent_sig below
        # (transient tail surgery, else next turn reads a phantom tail-edit
        # own-mutation).
        out_messages, tail_merged = _merge_trailing_assistant_run(out_messages)
        payload = {**body, "messages": out_messages}

        # The exact payload about to be forwarded, paired with the incoming
        # capture by seq — diffing in-vs-in and out-vs-out across turns is what
        # attributes a prefix break to the harness or to our own rewrite.
        if capture is not None and capture_seq is not None:
            try:
                capture.record_out(capture_seq, payload)
            except Exception:
                pass

        # Phase 14a: record the REWRITER's wire for break attribution (a prefix
        # break the incoming wire did not already carry is an own-mutation, the
        # voluntary kind 14b drives to zero). Deliberately result.messages, NOT
        # out_messages: the Phase-13 breaker tail is a separately-metered
        # (loop_injections), append-only, next-turn-transient injection — folding
        # it into sent_sig would charge every injection as a phantom tail-edit
        # own-mutation one turn later.
        if controller is not None and obs is not None:
            try:
                controller.note_sent(obs.key, result.messages, observation=obs)
            except Exception:
                pass

        # Record prompt-transform stats right after the rewrite (before
        # forwarding), so measurement reflects what the proxy did to the prompt
        # regardless of the upstream outcome. Char counts are free (no tokenize).
        app.state.stats.record(
            messages_in=len(messages),
            messages_handle_ized=len(result.handle_ized_ids),
            messages_rehydrated=len(result.rehydrated_handles),
            slices_recalled=len(result.recalled_handles),
            windowing_triggered=result.windowing_triggered,
            windowing_emergency=result.windowing_emergency,
            loop_injected=decision is not None and decision.breaker_text is not None,
            assistant_tail_merged=tail_merged > 0,
            chars_in=_sum_content_chars(messages),
            chars_out=_sum_content_chars(out_messages),
        )

        # Wire-composition tee. Measures the FINAL forwarded payload (after every
        # rewrite pass, the breaker tail and the boundary repair), so what it
        # reports is literally what llama-server tokenizes. Returns a seq the
        # response path pairs usage.prompt_tokens against — same request, which
        # is the whole point. Best-effort: a diagnostic must never fail a request.
        diag = getattr(app.state, "diagnostics", None)
        diag_seq = None
        try:
            if diag is not None:
                diag_seq = diag.record_request(payload)
                if diag.tokenize and diag_seq is not None:
                    # 6 upstream /tokenize round-trips — off the event loop, and
                    # under no lock (the counter is independent of the store's
                    # sqlite connections).
                    await asyncio.to_thread(diag.tokenize_request, diag_seq,
                                            payload, app.state.counter)
        except Exception:
            diag_seq = None

        upstream_client: UpstreamClient = app.state.upstream

        if body.get("stream") is True:
            # §9.2 (H1): prime the generator's first chunk inside try/except so
            # an UpstreamError raised before any bytes are emitted maps to a
            # proper JSON error response with the correct status BEFORE the
            # 200/event-stream headers are committed. A mid-stream error
            # after the first chunk cannot change the already-sent status;
            # that is acceptable.
            agen = upstream_client.chat_completion_stream(payload)
            t0 = time.perf_counter()
            first: Optional[bytes]
            try:
                first = await agen.__anext__()
            except StopAsyncIteration:
                first = None
            except UpstreamError as e:
                return _upstream_error_response(e)
            # Proxy-measured TTFT: the prefill-work fallback when the upstream
            # sends no `timings` block (14a).
            ttft_ms = ((time.perf_counter() - t0) * 1000.0
                       if first is not None else None)

            # Phase 14a response tee on the stream branch: accumulate a bounded
            # byte tail of the ALREADY-FORWARDED chunks and parse usage/timings
            # from the final SSE data chunk after the stream completes. Pure
            # tee: forwarded bytes are byte-identical, parsing is best-effort.
            tee = StreamTee()
            tee.feed(first)

            async def _body():
                # The stream is forwarded VERBATIM; the guard only peeks at each
                # chunk for the opportunistic timings signal (cheap substring
                # check first, best-effort parse).
                if first is not None:
                    if guard is not None:
                        guard.observe_stream_chunk(first)
                    yield first
                async for chunk in agen:
                    if guard is not None:
                        guard.observe_stream_chunk(chunk)
                    tee.feed(chunk)
                    yield chunk
                # Parsed once: the diagnostic tee and the controller read the
                # same observation, so they can never disagree about this turn.
                observed = tee.result()
                if diag is not None:
                    try:
                        diag.attach_usage(diag_seq, _prompt_tokens_of(observed))
                    except Exception:
                        pass
                if controller is not None and obs is not None:
                    try:
                        controller.observe_response(obs.key, observed,
                                                    ttft_ms=ttft_ms)
                        if app.state.store is not None:
                            controller.maybe_persist(app.state.store.state)
                    except Exception:
                        pass

            return StreamingResponse(_body(), media_type="text/event-stream")

        try:
            data = await upstream_client.chat_completion(payload)
        except UpstreamError as e:
            return _upstream_error_response(e)
        if guard is not None:
            guard.observe_response(data)
        # Phase 14a response tee (non-stream): the body is already a parsed
        # dict; fold usage/timings into the ledger. Best-effort, response
        # forwarded unchanged either way.
        observed = extract_usage_timings(data)
        if diag is not None:
            try:
                diag.attach_usage(diag_seq, _prompt_tokens_of(observed))
            except Exception:
                pass
        if controller is not None and obs is not None:
            try:
                controller.observe_response(obs.key, observed)
                if app.state.store is not None:
                    controller.maybe_persist(app.state.store.state)
            except Exception:
                pass
        return JSONResponse(data, media_type="application/json")

    # ----------------------------------------------------------- /v1/models + /props

    @app.get("/v1/models")
    async def get_models():
        upstream_client: UpstreamClient = app.state.upstream
        try:
            data = await upstream_client.passthrough_get("/v1/models")
        except UpstreamError as e:
            return _upstream_error_response(e)
        data = _apply_model_alias(data, app.state.config.model_alias)
        data = _inject_context_length(data, app.state.n_ctx)
        return JSONResponse(data, media_type="application/json")

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str):
        # OpenAI retrieve-model shape: ONE model object. llama-server has no
        # such route, so harnesses that probe it (hermes-agent probing
        # /v1/models/context-governor, live 2026-07-20) ate a bare FastAPI 404.
        # Synthesize the reply from the upstream's model list — the SAME data
        # /v1/models serves (alias + real context_length applied) — narrowed to
        # the probed id.
        upstream_client: UpstreamClient = app.state.upstream
        try:
            data = await upstream_client.passthrough_get("/v1/models")
        except UpstreamError as e:
            return _upstream_error_response(e)
        data = _apply_model_alias(data, app.state.config.model_alias)
        data = _inject_context_length(data, app.state.n_ctx)
        items = data.get("data") if isinstance(data, dict) else None
        if not (isinstance(items, list) and items and isinstance(items[0], dict)):
            items = data.get("models") if isinstance(data, dict) else None
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and model_id in (
                        it.get("id"), it.get("name"), it.get("model")):
                    return JSONResponse(dict(it), media_type="application/json")
        return JSONResponse(
            {"error": {
                "message": f"The model '{model_id}' does not exist",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }},
            status_code=404,
            media_type="application/json",
        )

    @app.get("/props")
    @app.get("/v1/props")
    async def get_props():
        # Clients using a /v1 base URL probe /v1/props (llama-server serves /props at
        # root); accept both and pass through to the upstream's /props.
        upstream_client: UpstreamClient = app.state.upstream
        try:
            data = await upstream_client.passthrough_get("/props")
        except UpstreamError as e:
            return _upstream_error_response(e)
        return JSONResponse(data, media_type="application/json")

    # ------------------------------------------------- Ollama-style model discovery
    # Some CLIs auto-detect the backend by probing Ollama's native model-list paths
    # (`/api/tags`, `/api/v1/models`, `/api/v1/tags`). Behaviour, by design:
    #   1. FORWARD the probe to the upstream at its ORIGINAL path first. If the
    #      upstream actually implements it (a real Ollama or multi-backend server),
    #      its answer is returned verbatim — we never override a working `/api/*`.
    #   2. ONLY when the upstream returns 404 (llama-server has no `/api/*` routes)
    #      fall back to the upstream's model list (`/v1/models`) so the probe is
    #      still answered (200) and the client stops logging 404s.
    # DISCOVERY ONLY — chat still goes through `/v1/chat/completions`; we do NOT
    # emulate Ollama's `/api/chat`, so the proxy never falsely claims to be a full
    # Ollama chat backend.
    @app.get("/api/tags")
    @app.get("/api/v1/tags")
    @app.get("/api/v1/models")
    async def ollama_discovery(request: Request):
        upstream_client: UpstreamClient = app.state.upstream
        alias = app.state.config.model_alias
        original_path = request.url.path
        try:
            data = await upstream_client.passthrough_get(original_path)
            return JSONResponse(
                _inject_context_length(_apply_model_alias(data, alias), app.state.n_ctx),
                media_type="application/json",
            )
        except UpstreamError as e:
            if e.status_code == 404:
                # Upstream doesn't serve this Ollama path -> answer from its model list.
                try:
                    data = await upstream_client.passthrough_get("/v1/models")
                    return JSONResponse(
                _inject_context_length(_apply_model_alias(data, alias), app.state.n_ctx),
                media_type="application/json",
            )
                except UpstreamError as fallback_error:
                    return _upstream_error_response(fallback_error)
            # Any non-404 upstream error is surfaced transparently.
            return _upstream_error_response(e)

    # ----------------------------------------------------------- /healthz

    @app.get("/healthz")
    async def healthz():
        # Does NOT touch the upstream.
        return JSONResponse({"status": "ok"}, media_type="application/json")

    # ----------------------------------------------------------- /metrics

    @app.get("/metrics")
    async def metrics():
        # Cumulative prompt-transform stats (Phase 5 §3) + retrieval-path counters
        # from the shared store (Phase 7 Stage 1). Does NOT touch upstream.
        snap = app.state.stats.snapshot()
        store = app.state.store
        if store is not None:
            stats_fn = getattr(store, "stats", None)
            if callable(stats_fn):
                try:
                    snap["retrieval"] = stats_fn()
                except Exception:
                    pass
        # Phase 14a: the closed-loop view — real prompt sizes, reuse ratios,
        # break attribution, learned ceilings, per-conversation ledger. The
        # goal: a session like the 2026-07-19 shakedown needs ZERO server-log
        # forensics.
        ctrl = app.state.controller
        if ctrl is not None:
            try:
                snap["closed_loop"] = ctrl.snapshot()
            except Exception:
                pass
        return JSONResponse(snap, media_type="application/json")

    # ------------------------------------------------------------- /diagnostics

    @app.get("/diagnostics")
    async def diagnostics():
        # Wire composition per component, PAIRED per request with the real
        # usage.prompt_tokens. Unlike /metrics (independent running maxima),
        # every figure here comes from a single forwarded request, so the
        # component split is arithmetic rather than inference. Does NOT touch
        # upstream.
        try:
            snap = app.state.diagnostics.snapshot(app.state.n_ctx)
        except Exception as e:
            snap = {"enabled": False, "error": f"{type(e).__name__}: {e}"}
        return JSONResponse(snap, media_type="application/json")

    return app
