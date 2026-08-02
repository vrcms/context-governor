# KimiK3 Contribution — 2026-07-19 Full-Reprocess Regression: Analysis, Fixes, Tests, Results

This folder documents the review of the previous session's (Claude Fable-5)
incident analysis, the independent verification of that diagnosis against code
and logs, the design and implementation of the durable fix package, and the
test/results record. Scope: the "full re-processing sickness" that returned
after the Phase 12/13/14 merger (commit `037043d`).

## Contents

| File | Contents |
|---|---|
| [01-epicrisis.md](01-epicrisis.md) | Incident epicrisis: symptoms, forensic evidence (llama.cpp slot log, proxy `/metrics`), verification of the Fable-5 diagnosis link by link, root cause, and two additional defects found during review. |
| [02-solution-design.md](02-solution-design.md) | Planning: the four-part fix package as designed and implemented, design rationale per fix, alternatives considered and rejected, residual risks / out-of-scope items. |
| [03-test-plan.md](03-test-plan.md) | Test plan: existing-suite impact analysis, new and rewritten tests with the behavior each one pins. |
| [04-results.md](04-results.md) | Results: full-suite outcome, files changed, operational actions taken, and the operational health criteria for validating the fix in production. |
| [05-boundary-fixes-models-route-and-tail-merge.md](05-boundary-fixes-models-route-and-tail-merge.md) | Follow-up (2026-07-20): synthesized `GET /v1/models/{id}` retrieve-model reply for hermes-agent's probes, and the trailing assistant-run merge repairing the harness wires llama-server 400s ("Cannot have 2 or more assistant messages at the end of the list"). |

## TL;DR

The regression was self-inflicted by three individually sound Phase-14
mechanisms composing badly: a **false learned compaction ceiling** (37,804)
made the windowing high water unreachable (30,243 vs. a 40K+ real wire), so
windowing **triggered every turn but could shed (almost) nothing**, and 14b's
flush policy treated every trigger as a free recall-block rebuild — a
self-fulfilling prefix break on nearly every turn.

Fix package (all implemented, full suite green at 485 tests):

- **(c) Ceiling hardening** (`proxy/sensing.py`) — root cause: shrink check,
  deferred + retractable cross-key confirmation, raise-on-contradiction.
- **(b) Missed-low-water latch** (`proxy/rewriter.py`) — a trigger that can't
  reach low water latches; re-arms only on hysteresis-gap growth.
- **(a) Shed-gated flush** (`proxy/rewriter.py`) — the recall flush rides only
  real byte changes (`windowing_shed`), not bare high-water crossings.
- **Anchor ordinal pinning** (`proxy/rewriter.py`) — frozen recall block keeps
  its exact position when the anchor's content is duplicated.

No git commit was created (per project workflow, commits only on explicit
request).

**Addendum (document 05, suite at 491 tests):** the first live session on the
fixed proxy surfaced two boundary issues — hermes-agent's
`GET /v1/models/context-governor` probes 404ing (llama-server has no
retrieve-model route; the proxy now synthesizes one) and llama-server 400ing
harness wires ending in ≥2 assistant messages (now merged tail-only at the
boundary, counted as `assistant_tail_merges`).
