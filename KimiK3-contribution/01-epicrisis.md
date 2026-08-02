# 01 — Epicrisis: the post-merger full-reprocess regression

## 1. Presenting complaint

After the merger of Phases 12/13/14 on `main` (merge record `037043d`), the
"forcing full re-processing" sickness returned: the llama.cpp server began
re-processing 10–20K tokens of prompt on nearly every agent turn, with
chronically low prefix-cache reuse — the same symptom class the Phase-12
sticky-recall work had eliminated. Evidence bundle: the llama.cpp server log
(`llamacpp_forcing_full_reprocess.txt`, ~635 KB) and the live proxy `/metrics`
snapshot captured by the previous session.

## 2. Prior diagnosis (Fable-5), as received

The previous session concluded, and this review **confirmed in every link**:

1. **A false learned ceiling.** `/metrics` showed `native_compaction_observed:
   4` and a learned native-compaction ceiling of **37,804** tokens, persisted
   in `integration/contextstore/state.json` (surviving restarts). The same
   metrics showed a real prompt **peak of 53,242** and a then-current prompt of
   **40,935** with no compaction — so the harness demonstrably does not compact
   at 37.8K. The four "observations" sampled the main conversation's running
   size, not a real compaction threshold.
2. **Poisoned setpoint.** Effective high water = `min(0.50 × 100096,
   0.8 × 37804)` = **30,243** instead of 50,048. The conversation's real
   pressure (40K+) sat permanently above the trigger.
3. **Windowing triggers every turn but cannot shed.** `windowing_triggers: 62`
   of 101 requests. Most of the wire is unsheddable by Pass 3 (the ~19.2K
   byte-stable tools/template head, protected tail, already-minimal stubs,
   non-string content), so pressure never dropped and the trigger re-fired
   forever.
4. **Each trigger flushed the frozen recall block — the actual byte change.**
   14b's rule "a windowing trigger = the prefix breaks anyway, refresh rides
   free" is only true when windowing actually pages something. When it pages
   nothing, the flush *creates* the break it was supposed to ride — a
   self-fulfilling prophecy. Proxy attribution corroborated: `own-mutation:
   60` ≈ 62 triggers.

## 3. Independent verification (this review)

Every claim was re-checked against source and the raw log rather than taken on
trust.

### 3.1 Code verification

| Claim | Where | Verdict |
|---|---|---|
| 14c pressure = real usage (`last_prompt_tokens + completion + growth_est`) | `proxy/sensing.py` `observe_request` | Confirmed |
| `effective_high_water = min(budget_ratio·n_ctx, ceiling_safety·learned)`; defaults 0.50 / 0.8 → `min(50048, 30243) = 30243` | `proxy/sensing.py`, `proxy/config.py` | Confirmed |
| Trigger fires on `pressure > high_water` **before** the shed loop; `shed` may stay 0 | `proxy/rewriter.py` Pass 3b (old line ~647) | Confirmed |
| Flush keyed to the bare trigger: `flush = prefix_broken or windowing_triggered` | `proxy/rewriter.py` (old line 716) | Confirmed |
| Cross-key match: any ≥3-message wire sharing the system head + diverging in the candidate's first quarter classifies as `head-rewrite` and samples the candidate's `last_prompt_tokens` | `proxy/sensing.py` `observe_request` | Confirmed — a side-call (title/summary/subagent) sharing the system head is structurally indistinguishable from a compacted main chat *by same-turn evidence* |
| No contradiction check: `_peak_prompt_tokens` tracked for metrics only; profiles never invalidated | `proxy/sensing.py` | Confirmed |
| Profiles persist via `maybe_persist` → `StateStore`; reloaded at startup | `proxy/sensing.py`, `state_store.py`, `proxy/app.py` lifespan | Confirmed |
| Pre-14 health was accidental: chars/4 estimate was blind to ~88% of the wire, so windowing never triggered | `proxy/rewriter.py` legacy gate; `tasks/plan.md` ("47K est vs 420K real") | Confirmed |

Config defaults at the time: `context_budget_ratio=0.50`,
`context_target_ratio=0.35`, `protect_first_n=2`, `protect_last_n=6`,
`ceiling_safety=0.8`, `recall_max_stale_tokens=4000`, `max_conversations=32`.

### 3.2 Log forensics (llama.cpp slot log)

Divergence positions ("Checking checkpoint … against N" — N = LCP/keep point):

- **24× at position 1** — full restarts: slot evicted by interleaved
  side-conversations (a pre-existing single-slot phenomenon, not the
  regression).
- **16× tightly clustered at 19175–19288**, marching ~20–90 tokens apart —
  immediately after the ~19.2K byte-stable head. With `protect_first_n=2`, the
  first pageable message sits exactly there: this is the **windowing frontier
  advancing ~one thin message per turn** (the "trickle": each turn, the message
  that ages past `protect_last_n` gets paged; pressure never drops, so the
  trigger fires again next turn).
- **A marching 24K–35K cluster** (24407; 29243; 29450; 30379; 31261/31285;
  33.4–34.7K…) — the **recall block being rebuilt at the new tail** each flush
  turn; each turn's divergence lands at the previous turn's block position.

`f_keep` distribution (124 samples): 33× 0.0 (full restarts), the rest spread
0.35–0.95 — chronic partial reuse, never the healthy ≈1.0. Checkpoint-erasure
cascades illustrate the cost, e.g. task 24492 erased checkpoints at 20679,
21068, 21393, 21702, 29886, 37285 in one turn (f_keep 0.512 ⇒ ~19K tokens
re-processed), and task 30842 re-erased the *same* five positions (37446,
37812, 39885, 40201, 47643) that task 28435 had already erased — repeated
invalidation at identical positions, the signature of own-mutation churn.

Sent-prompt sizes march 37,921 (×6) → 38,687 (×9) → 39,778 (×9) → 40,851 (×9)
→ 41,789 (×13), with peaks at **53,329 and 55,602 (×4)**. Two implications:

- The wire sailed **past the false 37,804 ceiling uncompacted** — the decisive
  proof that the learned ceiling was false (the contradiction the hardened
  learner now acts on).
- The wire also exceeded the **true** ratio high water (50,048) without
  shrinking — Pass 3 *cannot* shed this wire's bulk (invisible template/tools
  mass, protected tail, minimal stubs). Any fix relying on "correct the
  ceiling" alone leaves this >50K regime breaking every turn; this is why the
  latch (fix b) is load-bearing, not optional.

### 3.3 Root cause (confirmed)

> Three individually sound mechanisms composed into the old sickness:
> **14c** made windowing pressure real (previously the estimate was blind and
> windowing never fired — accidentally healthy); the **false learned ceiling**
> (a cross-key false positive with no contradiction check, persisted across
> restarts) made the high water unreachable; and **14b** converted every
> trigger into a recall-block rebuild regardless of whether the trigger
> changed a single byte. Triggers fired every turn, shed ~nothing, and the
> flush supplied the prefix break itself — 10–20K tokens re-processed per
> turn.

### 3.4 Additional defects found during this review (beyond the prior session)

1. **Anchor first-match relocation** (`_find_recall_anchor`): the frozen
   recall anchor was matched by content, first-match-wins. When the anchor
   message's content is duplicated earlier in the wire (repeated boilerplate
   tail messages are common in agent harnesses), the block was re-inserted at
   the *earlier* occurrence on sticky turns — a **backward** relocation, i.e.
   a voluntary prefix break with zero benefit, once per flush epoch even in an
   otherwise healthy regime.
2. **Same-key false-positive path**: the first-quarter-divergence heuristic
   also fired on *edits* to an early message (e.g. a harness refreshing a
   session-state block) — no shrink requirement. The `/metrics` split
   (`harness-edit: 3` vs `native_compaction_observed: 4`) suggests at least one
   of the four samples likely came from such a non-compaction path.
3. **Operational papercut**: `integration/contextstore/state.json` had been
   emptied to **0 bytes** (not deleted, not `{}`). `StateStore.load()` raises
   `StoreError` on it; swallowed on the chat path by try/except, but
   `durable.py`'s eviction `render()` and the MCP service call it unwrapped,
   and profile persistence retried-and-failed silently every response.
4. **Mitigation wiring caveat**: the operator mitigation
   (`ceiling_safety = 0`) had been applied to `governor.**example**.toml`,
   while `integration/run-governor.ps1`'s TOML auto-wiring is commented out —
   it only takes effect if the proxy is launched with `--config` that file,
   `--ceiling-safety 0`, or `CM_CEILING_SAFETY=0`.

## 4. Assessment of the prior session's proposed durable fixes

| Proposed fix | Verdict | Caveat found |
|---|---|---|
| (a) Flush epoch only when windowing shed bytes | **Fit, exact, minimal blast radius** — `shed > 0` ⟺ "Pass 3 changed the sent wire" | **Not sufficient alone**: in the observed regime most turns shed >0 (the trickle), so flushes and breaks would have continued |
| (b) Futile-trigger latch | **Fit in principle, under-specified** — latching only `shed = 0` misses the trickle regime and the >50K unsheddable regime the log proves is reachable | Strengthened to: latch any trigger that **misses low water**, re-arm on hysteresis-gap growth |
| (c) Ceiling self-invalidation + stricter cross-key match | **Fit and load-bearing** — the only fix removing the root cause | Same-turn heuristics cannot separate a side-call from a compacted chat; the sound discriminators are temporal (deferred confirmation + retraction); same-key path needed the shrink check too |
