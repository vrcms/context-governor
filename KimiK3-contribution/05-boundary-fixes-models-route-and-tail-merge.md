# 05 — Boundary fixes: `/v1/models/{id}` pseudo-reply + trailing assistant-run merge (2026-07-20)

Follow-up to the regression package (01–04): two live failures observed on the
first session running the fixed proxy, both at the proxy's outer boundary
rather than in the governor core.

## 1. Symptoms

From the live proxy + llama.cpp logs and the hermes-agent session screenshot:

```
INFO:  127.0.0.1:20732 - "GET /v1/models/context-governor HTTP/1.1" 404 Not Found
```

```
INFO:  127.0.0.1:48878 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
srv  operator (): got exception: {"error":{"code":400,"message":
  "Cannot have 2 or more assistant messages at the end of the list.",
   "type":"invalid_request_error"}}
```

## 2. Analysis

### 2.1 The models-route 404s

hermes-agent probes the OpenAI **retrieve-model** endpoint
`GET /v1/models/{model}` during discovery. llama-server has no such route, and
neither did the proxy — so every probe hit FastAPI's bare
`{"detail": "Not Found"}` 404. Not fatal, but noisy and potentially treated by
the harness as "model missing".

### 2.2 The 400 on trailing assistant runs

The message text is llama-server's chat-template **preflight validation**: a
wire may end with at most ONE assistant message (a generation prefill); two or
more are rejected outright. Key findings:

- The proxy **never creates assistant messages** — all its injections (stubs,
  rehydrated notes, recall block, loop-guard breaker) are role `user`, and no
  pass reorders messages. The malformed tail was produced by the **harness**
  (its continue/retry stitching left `[..., assistant, assistant]` at the
  tail after a long multimodal tool call).
- Passing the 400 through kills the agent run — exactly what a governance
  proxy should prevent. Boundary repair is the proxy's job.
- The merge must be **tail-only and deterministic** (prefix cache untouched),
  must run **after** all passes including the loop-guard breaker (a trailing
  breaker user-message already ends any assistant run — no merge needed then),
  and must be **excluded from `note_sent`'s signature** (like the breaker:
  transient tail surgery, otherwise the next turn would be misread as an
  own-mutation tail-edit).
- Runs carrying `tool_calls` cannot be merged coherently → left untouched
  (upstream's verdict stands) rather than corrupted.

### 2.3 Incidental observation (not a defect)

The same llama log showed an erasure cascade at task 20395 ("against 19384",
24.8K re-prefill): the cached prompt was **57,694** tokens and the new one
**24,799** — the harness compacted *for real* (shrink to 43%). That is a
legitimate break the governor rides, not the self-inflicted sickness; and
because it passes the new ≤0.7 shrink check, the hardened learner treats it as
a **genuine** ceiling sample — windowing should pre-empt the next flood at
~0.8 × that size.

## 3. Changes

| File | Change |
|---|---|
| `src/contextmanager/proxy/app.py` | New route `GET /v1/models/{model_id}`: synthesizes the OpenAI retrieve-model reply from the same upstream `/v1/models` list the list-route serves (alias applied, real `context_length` injected, other fields inherited); unknown ids → OpenAI-shaped 404 (`model_not_found`). New helper `_merge_trailing_assistant_run`: collapses a trailing run of ≥2 assistant messages into one (text joined with `\n`; content-parts concatenated so images survive; `tool_calls` runs passed through), wired into `chat_completions` after the breaker append. |
| `src/contextmanager/proxy/metrics.py` | New counter `assistant_tail_merges` (dataclass field, `record()` param, snapshot key). |
| `tests/proxy/conftest.py` | `FakeUpstream`/`make_app` gain `get_responses` (per-path canned passthrough payloads) so tests can serve a realistic `/v1/models` list. |
| `tests/proxy/test_app_nonstream.py` | 6 new tests (below). |

## 4. Tests

- `test_retrieve_model_synthesized_from_the_list` — alias presented, fields
  inherited, `context_length: 75776` injected, upstream asked only for
  `/v1/models`.
- `test_retrieve_model_unknown_id_is_an_openai_404` — `model_not_found` code
  with the probed id in the message.
- `test_trailing_assistant_run_merged_before_forwarding` — run collapsed to
  one assistant (`"partial answer\ncontinue prefill"`), prefix byte-identical,
  `/metrics` counter = 1.
- `test_trailing_run_with_tool_calls_left_untouched` — forwarded unchanged,
  counter = 0.
- `test_single_trailing_assistant_is_not_a_run` — unchanged.
- `test_trailing_run_content_parts_concatenated` — str + parts → one parts
  list; the image part survives.

## 5. Result

```
.venv\Scripts\python.exe -m pytest tests/ -q
491 passed, 1 warning in 19.11s
```

(485 → 491: six new tests, no regressions; the one warning is the
pre-existing Starlette/httpx deprecation.)

Expected live behavior after restart: the `GET /v1/models/context-governor`
probes return 200 with the aliased model object; harness wires ending in
duplicate assistant messages are merged instead of 400ing (each repair visible
as `assistant_tail_merges` in `/metrics` — a non-zero count identifies the
harness as the producer).
