# 04 — Results

## 1. Test-suite outcome

```
.venv\Scripts\python.exe -m pytest tests/ -q
485 passed, 1 warning in 22.45s
```

- Baseline before changes: **477** tests (merge-record level), with exactly
  one expected failure introduced by design (the old
  `test_cross_key_compaction_detected`, rewritten for deferred learning).
- After changes: **485** tests (8 new, 1 rewritten), all green. The one
  warning is a pre-existing Starlette/httpx deprecation notice, unrelated.

## 2. Files changed

| File | Change |
|---|---|
| `src/contextmanager/proxy/sensing.py` | Fix (c): `_HEAD_REWRITE_SHRINK` gate; `_pending_ceiling` bank with deferred confirmation + retraction; `_learn_ceiling` helper; raise-on-contradiction in `observe_response` (`turns >= 2`); `/metrics` gains `pending_ceiling_samples` and per-profile `raised`; module docstring updated. |
| `src/contextmanager/proxy/rewriter.py` | Fix (b): `_window_latched` per-conversation latch (engage on missed low water, suppress until hysteresis-gap growth, clear below high water, gap > 0 only), both pressure paths. Fix (a): `windowing_shed` predicate; `flush = prefix_broken or windowing_shed`. Fix (d): `_recall_frozen` stores `(anchor_mid, anchor_index, block)`; `_find_recall_anchor` exact-index preference with first-match fallback; `_anchor_matches` factored. Comments updated where semantics changed. |
| `tests/proxy/test_sensing.py` | 1 test rewritten (deferred cross-key), 3 new (retraction, shrink gate, contradiction raise/re-learn). |
| `tests/proxy/test_break_riding.py` | 5 new (duplicated anchor; futile no-flush; shedding still flushes; latch suppress/re-arm; latch clear). |
| `integration/contextstore/state.json` | Operational fix: was **0 bytes** (every `load()` raised `StoreError` — swallowed on the chat path, but hit unwrapped by eviction `render()` and the MCP service, and silently failing profile persistence). Now contains `{}` — the mitigation (no `harness_profiles`) is preserved with valid JSON. |

Not touched: `integration/governor.example.toml` (the operator's uncommitted
`ceiling_safety = 0` mitigation was left as-is), no git commit (per workflow —
commits only on explicit request).

## 3. Behavior deltas (before → after)

| Scenario | Before | After |
|---|---|---|
| Wire permanently above high water, unsheddable | Trigger + recall rebuild **every turn** ⇒ 10–20K re-prefill/turn | One honest trigger; latch holds the wire byte-stable until hysteresis-gap growth; no flush without shed |
| Bare high-water crossing, `shed = 0` | Recall block flushed (self-created break) | Block stays frozen; wire byte-extends |
| Trigger pages real messages (`shed > 0`) | Recall flush rides the break | Unchanged (rides the break) |
| Side-call sharing the system head | Instant false ceiling sample, persisted forever | Banked, then retracted when the parent conversation continues (or never confirmed for 1-turn calls) |
| Early-message edit (no shrink) | Learned as a compaction ceiling | Attribution unchanged; feeds no learning |
| Real prompt sails past the learned ceiling | Nothing (ceiling poisoned permanently) | Ceiling raised to the observed size; later true compaction re-lowers via min-sample |
| Duplicated anchor content | Frozen block relocated backward once per flush epoch | Block pinned at its exact frozen index |

## 4. Operational health criteria (validate in production)

After restarting the proxy on these changes, a healthy session should show:

- `windowing_triggers ≪ requests` (the wiki's standing criterion), and
  crucially triggers **with** visible wire shrink, not bare crossings.
- `closed_loop.breaks_by_cause`: `own-mutation ≈ 0`; the 2026-07-19 session
  had `own-mutation: 60`.
- `closed_loop.real_reuse_ratio ≈ 1.0` between genuine harness edits;
  `native_compaction_observed` counting only confirmed compactions;
  `pending_ceiling_samples` draining to 0; `learned_ceilings[].raised` > 0
  indicating the contradiction check is actively self-correcting.
- llama.cpp side: `f_keep ≈ 1.0` on main-conversation turns; no checkpoint
  erasure cascades except on real harness edits/compactions.

## 5. Known residuals (unchanged by this package)

- **Invisible mass**: Pass 3 still cannot see/shed the tools array,
  content-parts, or template overhead; wires whose bulk is invisible can grow
  past even the ratio high water (observed 53–55K). The latch keeps that
  regime prefix-stable; actually bounding it needs content-visibility work.
- **Slot eviction** by interleaved side-conversations (24 full restarts in the
  log): server-side single-slot behavior, out of scope.
- **Multi-turn subagents** sharing the system head can still confirm a
  cross-key sample; retraction + raise-on-contradiction bound the blast
  radius, and `raised` in `/metrics` makes any such event visible.
