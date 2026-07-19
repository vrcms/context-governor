# Phase 14 — Closed-loop governor ("dancing with the harness")

Status: **PLANNED — awaiting approval** (spec drafted 2026-07-19, branch `claude/closed-loop-governor`)

Numbering note: assumes the two pending phases land first — Phase 12 = sticky auto-recall
(uncommitted on main), Phase 13 = loop guard (worktree `admiring-mccarthy-c77255`, renumber
from its local "Phase 12" heading at merge). This phase depends on 12's code (it reworks the
sticky-recall refresh policy); it does not depend on 13.

## Why (diagnosis, 2026-07-19 Hermes shakedown)

The governor runs **open-loop**: it rewrites the wire and never observes consequences.
Live cross-check of proxy `/metrics` against the llama-server log (25 requests, hybrid-SSM
Qwen3.6-35B-A3B) showed:

- The rewriter is **string-only** (~19 `isinstance(content, str)` guards + `_sum_content_chars`),
  so it saw ~47K est tokens while llama-server tokenized ~420K real prompt tokens over the
  same 25 requests (`peak_prompt_tokens_est: 6039` vs real 19–28K per request). Invisible:
  the `tools` array (never in `messages`), content-parts arrays (images+), `tool_calls`
  payloads. The ~19,176-token byte-stable head was mostly tools/template — unmanaged.
- Windowing could therefore never trigger (`windowing_triggers: 0`; high water 0.50×100096
  est-tokens is unreachable), so the **harness's own compaction fired first** at ~28K real
  tokens → task 7220: transcript rewritten above token 1956, full 21K re-prefill (20.4 s).
  The exact flood the proxy exists to prevent.
- Phase-12 sticky recall worked between refreshes (88–600-token prompt evals on 20K+
  contexts), but each **growth-refresh relocated the ~1500-token block mid-wire**: the
  llama-server divergence points 19177 → 20592 → 21298 are the block's successive resting
  places, 3–8K tokens re-processed each. `search_calls: 14/25` — rebuilt on >half the turns,
  aggravated by `_recall_frozen` being one global slot across ≥3 interleaved conversations
  and by the freeze being skipped for non-str anchors.

Root principle learned: **in this system, changing policy is itself a cost** — any voluntary
byte change breaks the prefix and causes the re-prefill it tries to optimize. Static
thresholds can't price that; a closed loop with measured costs can.

## Goal

Zero **uncontrolled** re-prefills (not zero re-prefills — cold starts, image turns, and
*purchased* breaks are legitimate). The governor observes continuously, acts rarely, and
when it must break the prefix, breaks **everything at once**. All setpoints become measured
or learned; nothing needs re-tuning per model, llama.cpp build, or harness.

## Non-goals (this phase)

- Rewriting/stubbing content-parts arrays and slimming the `tools` array (that is content
  *visibility for rewriting* — its own phase; measurement visibility arrives here for free
  via `usage`).
- Anything server-side (image encode caching, checkpoint policy) — llama.cpp's concern.
- Runtime resizing of llama-server resources (impossible; launcher-at-startup concern).

## 14a — Sensing (zero behavior change)

- [ ] Response tee: parse `usage.prompt_tokens` / `usage.completion_tokens` and llama-server's
      `timings` block (`prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_per_second`)
      from non-stream JSON bodies AND the final SSE chunk on the stream branch. Must be a
      pure tee: forwarded bytes byte-identical, parse failures swallowed (the Phase-10
      lesson: enrichment must never fail the request). Fallback when `timings` absent:
      proxy-measured TTFT as a prefill-work proxy.
- [ ] Conversation identity: fingerprint = `stable_id` of the first message (role + canonical
      content serialization, str OR structured). One shared helper; adopted by the ledger now
      and by `_recall_frozen`/`_windowed` keying in 14b.
- [ ] Per-conversation ledger (LRU-bounded dict, in-memory like `_windowed`): rolling-hash
      prefix signature of the last wire sent, last real `prompt_tokens`, last `prompt_n`,
      reuse ratio history, per-turn growth.
- [ ] Request-diff classifier (runs BEFORE forwarding, on message-boundary hashes — never
      full-text diffs): {new-conversation | pure-append | tail-edit | mid-wire-edit(pos) |
      head-rewrite(pos)} × cause {harness-edit | own-mutation | multimodal | unknown}.
- [ ] `/metrics` additions: `real_prompt_tokens` (last/peak), `real_reuse_ratio`,
      `breaks_by_cause` counters, `native_compaction_observed` + ceiling estimate,
      per-conversation ledger summary.
- [ ] Tests: response-shape fixtures (with/without `timings`, stream + non-stream),
      classifier on synthetic wire pairs (incl. the task-7220 head-rewrite signature),
      ledger eviction, byte-identical passthrough on the stream branch.

Acceptance: a session like the 2026-07-19 shakedown requires **zero server-log forensics** —
`/metrics` attributes every break and reports reuse per turn. (This is also the data source
for the roadmap's live A/B/C table.)

## 14b — Break-riding (the mutation queue)

- [ ] All voluntary wire mutations enqueue instead of executing: recall refresh, windowing
      frontier advance, new setpoints taking effect.
- [ ] Flush policy — flush ALL pending mutations only when:
      (a) the classifier says this turn's prefix is already broken (harness edit, new
          conversation, multimodal turn), or
      (b) a hard bound is hit: recall staleness (`recall_refresh_tokens` demoted to
          `recall_max_stale_tokens`, a bound not a cadence) or real pressure nearing the
          ceiling (14c) — one deliberate break, everything batched into it.
- [ ] `_recall_frozen` keyed per conversation (ledger key); anchor matching extended to
      non-str content via the canonical serialization from 14a.
- [ ] Invariants (pinned by tests): on a no-flush turn the wire is a byte-exact prefix
      extension; idempotency (`rewrite(rewrite(x)) == rewrite(x)`) stays green; the rewriter
      remains a pure function of (messages, setpoints, frozen state) — the controller may
      move setpoints only at flush epochs, so behavior within an epoch is deterministic.

Acceptance: steady-state synthetic session → voluntary breaks = 0; every refresh coincides
with an already-broken turn in `breaks_by_cause`.

## 14c — Self-calibrating pressure

- [ ] Windowing pressure driven by REAL `usage.prompt_tokens` (per ledger), not chars/4 est.
- [ ] Learned ceiling: on an observed head-rewrite (native-compaction signature), record the
      real prompt size that preceded it; effective high water =
      `min(context_budget_ratio × n_ctx, ceiling_safety × learned_ceiling)`
      (`ceiling_safety` default 0.8). Until first observation: ratio×n_ctx (today's behavior).
- [ ] Persist the learned harness profile (ceiling, shape stats) in the contextstore so a
      proxy restart keeps calibration; keyed by harness fingerprint (system-prompt head hash).
- [ ] Config: `ceiling_safety` (+ validation/env/flag/TOML, docs in wiki/configuration.md +
      governor.example.toml).

Acceptance: re-run of the 2026-07-19 harness scenario → no task-7220-class flood; the
governor windows first (controlled release ≤ ~5K re-processed vs the observed 21K).

## 14d — Economic layer (stretch; may defer to its own phase)

- [ ] Cost model from measured `timings` (prefill t/s on THIS box/model): each queued
      mutation carries estimated break cost (tokens re-processed × ms/token) vs benefit
      (tokens saved, staleness relieved); flush decisions and window sizing become explicit
      cost/benefit. Thresholds degrade into self-calibrating priors; Phase-11/12 hysteresis
      becomes the *emergent* optimal policy rather than a hand-coded rule.

## Risks / seatbelts

- Stream tee must never alter or delay bytes: tee on the already-forwarded chunk, parse best-effort.
- Controller chasing noise: only *attributed* breaks feed decisions (multimodal turns are
  expected-breaks, never signals to re-optimize against).
- Classifier cost on 100+-message wires: message-boundary rolling hashes, O(n) hashes per
  request, no quadratic diffs, no tokenizer calls.
- Ledger unbounded growth: LRU cap (config, default ~32 conversations).

## Merge-order prerequisites (decide at approval)

1. Commit Phase 12 sticky recall on main (complete, 380 tests green) — this branch rebases
   onto it before implementation starts (14b modifies that code).
2. Loop-guard worktree is independent; land before or after, but its `plan.md`/`config.py`/
   launcher/wiki hunks conflict textually with 12 — merge one, rebase the other, renumber
   its heading to Phase 13.

## Verification gate (before "done")

Full test suite green (380 + new); live shakedown against llama-server + Hermes harness with
a one-time cross-check of `/metrics` reuse numbers against the server log — calibrating the
sensor itself, the last time log forensics should ever be needed.
