# Full prompt re-processing on every turn: llama.cpp hybrid-SSM + an OpenAI-compat proxy

> ## STATUS: FIXED (2026-07-28) — commits `a4290b9`, `2367da1`
>
> **Root cause:** Claude Code stamps a per-request nonce (`cch=<hex>`) into the
> FIRST content-part of its system prompt. It churned `conversation_key` AND was
> forwarded verbatim, so identity and wire were both unstable. Fix:
> `rewriter.normalize_volatile_stamps()` blanks the value at ingress — before
> sensing, and before forwarding.
>
> **Measured result:** `f_keep = 1.000` on 10 of 11 post-warmup turns; prompt eval
> per turn fell from 25,000-28,000 tokens (35-45 s) to 27-1,387 tokens (0.4-4 s);
> one conversation reached `turns = 16` (previously every conversation was stuck
> at `turns = 1`). Details in `progress.md`.
>
> **Two conclusions in the text below were disproved and are now WRONG:**
> - *"HTTP headers as an identity source"* (open question 1) — dead end. LiteLLM
>   strips everything except `user-agent: litellm/1.93.0`.
> - *"a key-only fix"* — insufficient. The nonce must be normalized on the
>   forwarded WIRE too, not just in the identity hash.
>
> **Remaining work has moved to `problem-2.md`.**

*Written 2026-07-28. Self-contained problem statement for handoff.*

## Stack

- **Model**: Kwaipilot KAT-Coder-V2.5-Dev, GGUF `MTP-UD-Q5_K_XL` (25.5 GiB). Arch `qwen35moe`:
  40 trunk layers + 1 MTP head (`nextn_predict_layers=1`), MoE 256 experts / 8 active,
  **hybrid SSM + attention**.
- **Server**: llama-server (fork `llama-cpp-turboquant`).
  `--ctx-size 53248 --cache-type-k q8_0 --cache-type-v turbo2 --flash-attn on
  --n-cpu-moe 35 --parallel 1 --ubatch-size 2048 --cache-ram 2048
  --spec-type draft-mtp,ngram-mod`
- **Hardware**: RTX 3070 Ti Laptop (8 GB), 31.7 GB RAM, Windows 11. ~397 MiB VRAM free at
  load, ~273 MiB after generation (verified healthy — VRAM is *not* the issue here).
- **Middleware**: "context-governor" — a Python/FastAPI OpenAI-compatible proxy on `:8900`
  forwarding to llama-server `:8080`. It rewrites history (windowing to a low-water mark,
  handle-izing large messages, injecting a sticky recall block from a durable store).
- **Clients**: Claude Code CLI 2.1.119, opencode, hermes, pi — all through the same proxy.

## Symptom

llama-server logs on nearly every turn:

```
slot get_availabl: selected slot by LCP similarity, sim_best = 0.613, f_keep = 0.658
slot update_slots: Checking checkpoint with [25408, 25408] against 16752...
slot update_slots: forcing full prompt re-processing due to lack of cache data
                   (likely due to SWA or hybrid/recurrent memory)
slot update_slots: erased invalidated context checkpoint (n_tokens = 22287, size = 106.768 MiB)
```

A 25-28k-token prompt is re-prefilled from position 0 every turn at ~650-700 t/s →
**35-45 s of dead time per turn**, before any token is emitted.

## Evidence — the divergence point is FIXED

`f_keep` = LCP ÷ previously-cached length. Verified arithmetically across two turns:

```
16752 / 25472 = 0.6577   → logged f_keep = 0.658
16752 / 27158 = 0.6168   → logged f_keep = 0.617
```

So **LCP = 16,752 exactly**, in both turns, despite different cached lengths. Not drift —
a fixed boundary.

It sits ~10,300 tokens *before* the prompt end, which rules out think-block stripping
(that produces divergence 1-2k before the end).

Checkpoints don't help: llama.cpp creates them only at the prefill tail (106-118 MiB each),
always *after* 16,752, so all get invalidated. `-ctxcp` / `-cms` control count and minimum
spacing only — **not placement**.

## Root cause found

The proxy's conversation identity function (`src/contextmanager/proxy/sensing.py`):

```python
def conversation_key(messages: list) -> str:
    first  = message_hash(messages[0])            # full system prompt
    second = message_hash(first_non_system_msg)
    return "conv-" + first + second
```

Claude Code's system prompt's **first content part** is:

```json
{"text":"x-anthropic-billing-header: cc_version=2.1.119.af2; cc_entrypoint=cli; cch=d2005;","type":"text"}
```

`cch` changes on **every request** (`d2005` → `006c8` → `27c6b` observed within one session);
`cc_version` flips its build suffix (`af2` / `c72`). Captured from live traffic.

Consequence — three consecutive turns of one chat:

```
conv-307a2ea5454ae5a9|91ce6d01e8daf020   turns: 1
conv-fc36684c5d5e52e6|91ce6d01e8daf020   turns: 1
conv-ba1bbf3ec605ef44|91ce6d01e8daf020   turns: 1
```

Left half (system prompt) varies; right half (first user message) is stable. Proxy
`/metrics` confirms: `breaks_by_cause: {"new-conversation": N}`, `real_reuse_ratio: 0.0`,
every conversation stuck at `turns: 1`.

Every request is filed as a **new conversation**, which disables everything keyed
per-conversation — sticky recall, the windowing ledger, the hysteresis latch, the learned
compaction ceiling. The proxy rebuilds the prompt from scratch each turn instead of
extending it.

## Why this is fatal here specifically

The model is **hybrid SSM + attention**. Recurrent state cannot be rolled back to an
arbitrary position, so cache reuse requires the new prompt to be a **byte-exact prefix
extension** (`f_keep` exactly 1.0). One byte different at position N forces recomputation
from 0. There is no partial reuse, and no amount of batching/chunking avoids it.

## Fixes attempted and why each failed

1. **Hash only the first 2048 chars of `messages[0]`** — failed. The nonce sits at
   char ~10, inside any window.
2. **Regex-strip `cc_version|cc_entrypoint|cch` before hashing the full message** —
   verified against captured traffic (3 main-chat requests collapsed to 1 key, title-gen
   side-call stayed separate), but **Claude Code still re-processed**. The capture was
   truncated at 2048 chars, so there is likely **another per-request volatile deeper in
   the prompt** that was never observed.

Both were reverted; the codebase is back to its original state (501 tests passing).

## Constraints any solution must satisfy

All measured, not assumed:

- **Claude Code**: per-request nonce at char ~10 of `messages[0]` → must not re-key.
- **opencode / hermes**: stamp `Today's date: Wed Jul 15 2026` into the system prompt →
  must not re-key at midnight during a long session.
- **opencode, two different projects**: their system prompts are **byte-identical for the
  first 8,742 chars**, differing only at `Working directory:`. So truncating the hash
  window collapses two concurrent sessions onto one key. Must still discriminate.
- **Main chat vs. title-gen side-call**: share the identical system prompt, differ only in
  the first user message. Must stay separate.

## Open questions for a solution

1. **Is content hashing the wrong identity source entirely?** The proxy sees raw HTTP
   headers. The OpenAI `/v1/chat/completions` schema has no conversation-ID field, but
   clients may send a stable per-session header. If any harness provides one,
   header-based identity is strictly more robust than hashing mutable content.
   **This was never investigated and is the most promising lead.**
2. If content hashing must be used, how to normalize *unknown* per-request nonces
   generically rather than by a growing regex allowlist? (E.g. diff consecutive requests
   of the same suspected conversation and treat unstable spans as volatile —
   self-calibrating rather than hardcoded.)
3. Even with a stable identity, does the proxy's own rewriting produce a byte-exact prefix
   extension? Its metrics distinguish `own-mutation` from `harness-edit` breaks; this was
   never measured with a working key, because the key never survived a second turn.
4. **Separate, unresolved issue**: the server's prompt cache is over its *token* budget,
   not its MiB budget —
   `cache state: 1 prompts, 556.162 MiB (limits: 2048.000 MiB, 53248 tokens, 104520 est)`.
   `104520 est` exceeds the `53248` token limit, so only one prompt is cached. With
   `--parallel 1`, an interleaved 333-token title-gen side-call evicts the 28k
   main-conversation state, causing a full re-prefill on the next main turn independently
   of the key problem.

## Relevant files

- `src/contextmanager/proxy/sensing.py` — `conversation_key`, `harness_fingerprint`,
  `canonical_content`, `message_hash`, break attribution (`breaks_by_cause`).
- `src/contextmanager/proxy/rewriter.py` — windowing (`_window_out`, `protect_first_n = 2`),
  sticky recall block, hysteresis latch (`_window_latched`).
- `src/contextmanager/proxy/config.py` — defaults: `context_budget_ratio = 0.50`,
  `context_target_ratio = 0.35`, `ceiling_safety = 0.8`, `auto_recall_k = 3`.
- `GET /metrics` → `closed_loop.breaks_by_cause`, `real_reuse_ratio`, per-conversation
  `turns`. This is the diagnostic that identifies the fault in one command.
