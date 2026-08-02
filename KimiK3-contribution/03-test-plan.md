# 03 — Test plan

Harness: existing pytest suite (`tests/`), rewriter-level tests using
`FakeCounter` (token = word) + a real `DurableStore(tmp_path)`; sensing tests
are pure in-memory except the `StateStore` round-trip. No network, no FastAPI.

## 1. Existing-suite impact analysis (before writing new tests)

| Existing test | Concern | Outcome |
|---|---|---|
| `test_recall.py::test_windowing_trigger_refreshes` | Pins "trigger ⇒ recall flush" | Stays green: its trigger pages real fillers (`shed > 0`), so the shed-gated flush still fires |
| `test_rewriter.py` hysteresis tests (`test_trigger_cuts_deep_toward_low_water`, `test_prefix_byte_stable_between_triggers`, `test_retrigger_advances_but_keeps_old_frontier_bytes`, `test_idempotent_under_hysteresis`, `test_target_zero_restores_legacy_per_turn_nibble`) | Latch could suppress legitimate re-triggers | Stays green: those triggers **reach** low water, so the latch never engages; `target=0` collapses the gap, which disables latching by design |
| `test_break_riding.py::test_windowed_stubs_stay_sticky_across_pressure_turns` | First trigger misses low water (56 < 90 shed) ⇒ latches at 150 | Stays green: turn 2 pressure 90 ≤ high water clears the latch; the no-trigger assertion holds |
| `test_break_riding.py::test_prefix_broken_flushes_below_the_bound`, steady-state and per-conversation tests | Flush policy changed | Unaffected: they exercise `prefix_broken` and sticky paths, not bare crossings |
| `test_sensing.py::test_cross_key_compaction_detected` | Expected **immediate** cross-key learning | **Rewritten** for deferred semantics (see below) — the only pre-existing test requiring an update |
| `test_sensing.py::test_minimum_sample_wins`, `test_effective_high_water`, `test_persistence_round_trip`, `test_fresh_two_message_chat_is_not_compaction` | Learning gates tightened | Unaffected: all drive the same-key path with genuine shrink (12 → 4 messages) |
| App-level closed-loop tests (`test_app_closed_loop.py`) | Sensing/controller changes | Green — enrichment never fails requests |

## 2. New / rewritten tests

### `tests/proxy/test_sensing.py` — `TestCeilingLearning`

- **`test_cross_key_compaction_detected`** *(rewritten)*: the cross-key match
  still reclassifies the turn (`head-rewrite` / `harness-edit`) but banks
  instead of learning (`native_compaction_observed == 0`,
  `learned_ceilings == {}`, `pending_ceiling_samples == 1`); a later
  pure-append turn of the new key confirms ⇒ ceiling 20000 learned, pending
  drained.
- **`test_cross_key_retracted_when_old_key_stays_alive`** *(new)*: the
  2026-07-19 false-ceiling scenario — a side-call sharing the system head
  banks a sample; the parent conversation reappearing with a pure append
  retracts it; nothing is ever learned.
- **`test_early_edit_without_shrink_learns_nothing`** *(new)*: same-key
  divergence in the first quarter *without* shrink still classifies as
  `head-rewrite` (attribution unchanged) but feeds no learning.
- **`test_contradiction_raises_ceiling`** *(new)*: same-key compaction learns
  20000 (high water 16000); the conversation then sails to 25000 uncompacted
  (at `turns >= 2`) ⇒ ceiling raised to 25000, `raised == 1`, high water
  20000; a later true compaction (wire grown first so the divergence lands in
  the first quarter again) re-pins via min-sample at 24000 — self-healing in
  both directions.

### `tests/proxy/test_break_riding.py`

- **`TestSteadyState::test_duplicated_anchor_stays_put`** *(new)*: the freeze
  anchor's content also appears earlier on the wire; the exact-index pin keeps
  the block at its last-sent position (byte-exact prefix extension, frozen
  block reused). Fails under first-match-wins — proves the anchor fix.
- **`TestFutileCrossing::test_futile_trigger_neither_flushes_nor_breaks`**
  *(new)*: the incident's self-own — a crossing that can shed nothing (middle
  messages too small for their stubs) reports `windowing_triggered` but does
  **not** flush the frozen recall block and leaves the wire a byte-exact
  prefix extension.
- **`TestFutileCrossing::test_shedding_trigger_still_flushes`** *(new)*: the
  flush keeps riding *real* breaks — a real-pressure trigger that pages
  fillers (`shed > 0`) refreshes the block, mirroring the legacy-path pin.
- **`TestFutileCrossing::test_missed_low_water_latches_until_gap_growth`**
  *(new)*: after a futile trigger latches at 150, pressure 170 (< 150 + gap
  40) is suppressed (no trigger, byte-stable wire); pressure 195 (≥ 150 + 40)
  re-arms — and a re-armed but still-futile trigger still doesn't flush.
- **`TestFutileCrossing::test_latch_clears_when_pressure_falls_below_high_water`**
  *(new)*: pressure 90 (≤ high water) clears the latch, so 170 (< latched +
  gap) triggers fresh afterward.

## 3. Coverage of the acceptance criteria

- 14b acceptance preserved: steady-state sessions produce zero voluntary
  breaks (existing `test_growing_session_is_pure_prefix_extension` + the new
  futile-crossing pins).
- Flush epochs only on real breaks: harness-edit (existing), shed > 0 (new),
  staleness/anchor loss (existing) — and **not** on bare crossings (new).
- Ceiling learning: shrink-gated, temporally confirmed cross-key, retractable,
  self-invalidating on contradiction, persistence round-trip intact.
