# 02 — Solution design: the durable fix package

Approved package (implemented in the order listed): **(c)** ceiling
hardening → **(b)** missed-low-water latch → **(a)** shed-gated flush →
**anchor ordinal pinning**. Each fix is small, independently testable, and
addresses a distinct link of the causal chain; (a)–(c) map to the prior
session's proposal, hardened per the epicrisis findings (§4).

## Fix (c) — Ceiling learning hardening (`src/contextmanager/proxy/sensing.py`)

Root-cause fix. Three cooperating mechanisms:

### c1. Shrink check before learning

A head-rewrite only feeds ceiling learning when the wire **shrank** to ≤ 70%
of its previous message count (`_HEAD_REWRITE_SHRINK = 0.7`).

- Real compaction = bulk replaced by a summary ⇒ drastic shrink.
- An early-message *edit* (refreshed session-state block) diverges in the
  first quarter but keeps the length — now excluded from learning while the
  classifier's `head-rewrite` verdict (used for break attribution) is
  unchanged.
- Applies to both the same-key and cross-key paths.

### c2. Deferred + retractable cross-key confirmation

Same-turn evidence cannot separate a 3-message side-call from a freshly
compacted main chat (both share only the system head; both are much smaller).
The sound discriminators are temporal:

- A cross-key head-rewrite match no longer learns immediately. It **banks a
  pending sample** (`_pending_ceiling[new_key] = {sample, old_key, fp, at}`,
  LRU-bounded, in-memory only — never persisted unconfirmed).
- **Confirmation**: the pending sample is promoted (min-keep learning) only
  when the *new* key takes a subsequent pure-append turn (≥ its 2nd request) —
  a compacted main conversation continues; a one-turn title/summary call never
  does.
- **Retraction**: if the *old* key ever reappears with a pure append, the
  pending sample is cancelled — side-calls coexist with their parent; a true
  compaction kills the old key.
- The wire-shape reclassification (`KIND_HEAD_REWRITE`) is retained for
  attribution; only learning is deferred. `native_compaction_observed` now
  counts **confirmed** compactions.

### c3. Raise-on-contradiction (self-invalidation)

In `observe_response`: when an established conversation (`turns >= 2`) is
observed with `prompt_tokens` **above the learned ceiling** for its harness
fingerprint, the ceiling is **raised** to the observed size.

- A prompt sailing past the ceiling uncompacted falsifies it — this is the
  invariant that was missing (peak 53,242 vs. ceiling 37,804).
- Comparison is against the **ceiling**, not the enforced
  `ceiling_safety × ceiling` high water (a prompt between the two is
  legitimately consistent).
- Raising (not deleting) is self-healing in **both directions**: a later true
  head-rewrite re-lowers the ceiling via min-sample; absent any compaction the
  ceiling rises until `min()` binds at the ratio-based high water.
- `turns >= 2` guard: a fresh conversation's first prompt (e.g. a subagent
  seeded with bulk context) says nothing about steady-state compaction.
- Profiles persist as before; raised values and a per-profile `raised` counter
  round-trip through `state.json`. `/metrics` gains
  `pending_ceiling_samples` and per-profile `raised`.

## Fix (b) — Missed-low-water latch (`src/contextmanager/proxy/rewriter.py`)

Strengthened form of the proposed "futile-trigger latch". Per-conversation
state `_window_latched: conv_key -> pressure at the latching trigger`.

- **Engage**: a Pass-3 trigger that fails to shed down to the low water
  (real-pressure path: `shed < target_shed`; legacy path: post-loop
  `total > low_water`) latches, recording the pre-shed pressure.
- **Suppress**: while latched, a subsequent trigger is skipped unless
  `pressure >= latched_at + (high_water − low_water)` — re-arming only after
  hysteresis-gap growth, mirroring the two-water design ("trigger rarely, cut
  deep") for the regime where cutting deep is impossible.
- **Clear**: pressure falling back to/below the high water pops the latch (the
  crisis is over; the next crossing is tried fresh).
- **Only with a real gap**: `gap <= 0` (collapsed waters, e.g.
  `context_target_ratio = 0`) never latches — legacy per-turn behavior
  preserved.
- Applied symmetrically to the real-pressure and legacy open-loop paths;
  LRU-bounded like `_windowed`.

Rationale: the log proves the wire reaches 53–55K — *above the true ratio
high water* — and is mostly unsheddable. Without the latch, that regime keeps
trickle-breaking every turn even with a perfectly learned ceiling. The latch
converts it into "hold the prefix byte-stable and admit windowing can't
help"; the harness's real compaction (if any) then becomes the only flood,
which the governor rides as an ordinary `prefix_broken` flush.

## Fix (a) — Shed-gated recall flush (`src/contextmanager/proxy/rewriter.py`)

- New internal predicate `windowing_shed`: Pass 3 paged at least one
  previously-verbatim message this call (real path: `shed > 0`; legacy path:
  `total < total_before`). Pass 3a's sticky re-stubbing is byte-identical to
  the previously *sent* wire, so `shed > 0` is an **exact** predicate for
  "windowing broke the prefix this turn".
- Flush rule becomes `flush = prefix_broken or windowing_shed` (was:
  `... or windowing_triggered`).
- `windowing_triggered` keeps its meaning ("crossed the high water") for
  `/metrics` — the diagnostic signal that caught this incident is preserved.
- Net effect: a bare crossing that shed nothing no longer rebuilds the recall
  block — ending the self-fulfilling break.

## Fix (d) — Anchor ordinal pinning (`src/contextmanager/proxy/rewriter.py`)

Found during review (not in the prior proposal).

- `_recall_frozen[conv_key]` now stores `(anchor_mid, anchor_index, block)`,
  where `anchor_index` is the anchor's index in the **block-free** wire at
  freeze time (pure appends never move it).
- `_find_recall_anchor` prefers the occurrence **at the frozen index**;
  falls back to first-match only when the index no longer matches (mid-wire
  insertions/edits — a turn whose prefix the harness already broke anyway).
- Match predicate factored into `_anchor_matches` (identity via
  `stable_id_any`, or stub primary-handle).
- Eliminates the backward relocation of the frozen block on duplicated anchor
  content — a voluntary break per flush epoch in the healthy regime.

## Alternatives considered and rejected

- **Delete the profile on contradiction** instead of raising: loses
  information, thrashes persistence, and re-learns nothing; raising keeps
  min-sample semantics intact and self-heals both ways.
- **Stricter same-turn cross-key heuristics** (min divergence position,
  length guards): all either kill true positives (compaction *replaces* the
  first user message, so requiring shared prefix beyond the head misses it) or
  fail to separate side-calls from compacted chats. Temporal confirmation is
  the sound discriminator.
- **Latch only on `shed = 0`** (the proposal as written): misses the trickle
  regime (`shed > 0` but `≪ target`) and the >50K unsheddable regime —
  replaced by the missed-low-water criterion.
- **Suppress the trigger from metrics when latched**: rejected implicitly —
  `windowing_triggered` semantics were deliberately left unchanged (latched
  turns simply read as "no trigger"), since the counters are the diagnostic
  surface that caught this incident. No new metric fields beyond
  `pending_ceiling_samples` / `raised`.
- **Position-compared flush** (flush only if the frozen block sits *after*
  the first new stub, else defer): noted as a possible future refinement —
  even `shed > 0` flushes can extend a break backward when the block sits
  earlier than the new stubs. Skipped: in practice the block anchors near the
  recent tail (later than the frontier), and riding an existing break beats a
  solo staleness break later.
- **New `windowing_latched` / `windowing_shed` metric counters**: skipped to
  keep blast radius small; latched behavior is inferable from existing
  `closed_loop` metrics (triggers ≈ 0 while real prompt stays high).

## Residual risks / explicitly out of scope

- **Invisible mass / content-visibility**: windowing still cannot see or shed
  the tools array, content-parts, or template overhead. A wire whose real bulk
  is mostly invisible can exceed even the ratio high water while Pass 3 sheds
  ~nothing (observed: 53–55K). The latch keeps this regime prefix-stable, but
  actually *bounding* such wires needs content-visibility work (separate
  project).
- **Slot eviction by side-conversations** (24 full restarts in the log): a
  server-side single-slot phenomenon, untouched by this package.
- **Multi-turn subagent conversations** sharing the system head can still
  confirm a cross-key sample (deferred confirmation passes); retraction and
  the contradiction check bound the damage, and raise-on-contradiction
  self-heals.
- **`ceiling_safety = 0` operator mitigation** (uncommitted edit in
  `governor.example.toml`) was left in place; with hardened calibration it can
  be kept for a conservative session or returned to 0.8 — a false ceiling now
  self-corrects live.
