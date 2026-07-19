"""Phase 13 LoopGuard suite — fingerprinting + the trigger/cooldown/escalation
state machine, mirroring the property-test style of test_compactor.py.

The invariants pinned here:
  1. The guard NEVER fires without ``repeat_k`` consecutive near-identical turns.
  2. It ALWAYS fires at exactly the k-th consecutive repeat (cooldown permitting).
  3. Cooldown suppresses re-injection for exactly ``cooldown_turns`` turns.
  4. Escalation: 1st fire = notice, 2nd = final notice, then capped — unless
     ``hard_stop`` is on, in which case the 3rd fire says "end the run".
  5. A changed fingerprint (loop broken) fully re-arms the guard.
  6. The timings signal is corroboration only: it accelerates the content
     trigger by ONE turn and is never required.
  7. ``observe_request`` never mutates the transcript it observes.

The property test drives the guard against an independently-written reference
model over random turn sequences (hypothesis), the same way
test_post_compaction_below_low_water pins the compactor's no-re-fire theorem.
"""

from __future__ import annotations

import copy
import json

import pytest
from hypothesis import given, settings, strategies as st

from contextmanager.loop_guard import (
    BREAKER_MARKER,
    LoopGuard,
    LoopGuardConfig,
    breaker_notice,
    hard_stop_text,
    normalize_for_fingerprint,
    turn_fingerprint,
)


# ---------------------------------------------------------------------------
# Transcript builders. A "turn" is what an agent CLI appends between two chat
# requests: one assistant tool-call message + its tool result. The transcript
# GROWS every turn (the CLI resends the whole history), so each observation has
# a fresh anchor — exactly the live livelock shape.
# ---------------------------------------------------------------------------


def transcript(tokens: list) -> list[dict]:
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "do the task"},
    ]
    for tok in tokens:
        msgs.append({"role": "assistant", "content": f"calling tool {tok}"})
        msgs.append({"role": "tool", "content": f"result {tok}"})
    return msgs


def feed(guard: LoopGuard, tokens: list) -> list:
    """Observe the growing transcript turn by turn; return the decisions."""
    return [guard.observe_request(transcript(tokens[: i + 1]))
            for i in range(len(tokens))]


def make_guard(**over) -> LoopGuard:
    return LoopGuard(LoopGuardConfig(**over))


# ===========================================================================
# Fingerprinting
# ===========================================================================


def test_fingerprint_whitespace_normalized() -> None:
    a = [{"role": "assistant", "content": "read   file\n\n a.txt"}]
    b = [{"role": "assistant", "content": "read file a.txt"}]
    assert turn_fingerprint(a)[0] == turn_fingerprint(b)[0]


def test_fingerprint_strips_volatile_substrings() -> None:
    a = [{"role": "assistant", "content":
          "done at 2026-07-19T10:32:01Z id=6f9619ff-8b86-d011-b42d-00c04fc964ff"}]
    b = [{"role": "assistant", "content":
          "done at 2026-07-19T11:05:44Z id=00000000-1111-2222-3333-444444444444"}]
    assert turn_fingerprint(a)[0] == turn_fingerprint(b)[0]
    # And normalize itself replaces the volatile parts.
    assert "2026-07-19" not in normalize_for_fingerprint("at 2026-07-19T10:32:01Z")


def test_fingerprint_strips_tool_call_ids() -> None:
    def turn(call_id: str, tc_id: str) -> list[dict]:
        return [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function",
                 "function": {"name": "read", "arguments": '{"path": "a.txt"}'}},
            ]},
            {"role": "tool", "tool_call_id": tc_id, "content": "the file body"},
        ]
    # Fresh ids every turn (what llama-server/CLIs actually do), same action.
    assert (turn_fingerprint(turn("call_x1", "call_x1"))[0]
            == turn_fingerprint(turn("call_z9", "call_z9"))[0])


def test_fingerprint_differs_for_different_action() -> None:
    a = [{"role": "assistant", "content": "read a.txt"}]
    b = [{"role": "assistant", "content": "read b.txt"}]
    assert turn_fingerprint(a)[0] != turn_fingerprint(b)[0]


def test_fingerprint_covers_only_trailing_segment() -> None:
    # Same last turn behind different histories -> same fingerprint (the anchor
    # differs, which is how the guard tells "new repeat" from "retry").
    t1 = transcript(["x", "a"])
    t2 = transcript(["y", "z", "a"])
    fp1, anchor1 = turn_fingerprint(t1)
    fp2, anchor2 = turn_fingerprint(t2)
    assert fp1 == fp2
    assert anchor1 != anchor2


def test_fingerprint_none_without_assistant() -> None:
    assert turn_fingerprint([{"role": "user", "content": "hi"}]) is None
    assert turn_fingerprint([]) is None


# ===========================================================================
# Trigger / cooldown / escalation state machine
# ===========================================================================


def test_never_fires_below_k_and_fires_exactly_at_k() -> None:
    guard = make_guard(repeat_k=3)
    decisions = feed(guard, ["a", "a", "a"])
    assert decisions[0] is None and decisions[1] is None
    d = decisions[2]
    assert d is not None and d.level == 1 and d.streak == 3 and not d.hard_stop
    # The notice is the spec'd text, marked with the [[cm:...]] family marker.
    assert d.breaker_text.startswith(BREAKER_MARKER)
    assert "repeated the same action 3 times" in d.breaker_text
    assert "state that explicitly, summarize what was completed, and stop" in d.breaker_text


def test_distinct_turns_never_fire() -> None:
    guard = make_guard(repeat_k=2)
    assert all(d is None for d in feed(guard, ["a", "b", "c", "d", "e"]))


def test_retry_of_same_request_is_not_a_repeat() -> None:
    guard = make_guard(repeat_k=2)
    msgs = transcript(["a"])
    assert guard.observe_request(msgs) is None
    # Same request re-delivered (identical anchor + fingerprint): NOT turn #2.
    assert guard.observe_request(copy.deepcopy(msgs)) is None
    # A REAL second identical turn does fire (k=2).
    assert guard.observe_request(transcript(["a", "a"])) is not None


def test_cooldown_suppresses_exactly_c_turns_then_escalates() -> None:
    guard = make_guard(repeat_k=3, cooldown_turns=2)
    decisions = feed(guard, ["a"] * 7)
    # t3 fires (level 1); t4, t5 suppressed by cooldown=2; t6 fires level 2.
    assert [d.level if d else None for d in decisions] == [
        None, None, 1, None, None, 2, None,
    ]
    assert "FINAL NOTICE" in decisions[5].breaker_text


def test_escalation_caps_at_final_notice_without_hard_stop() -> None:
    guard = make_guard(repeat_k=2, cooldown_turns=0, hard_stop=False)
    decisions = feed(guard, ["a"] * 6)
    # cooldown=0: fires every turn from t2 on; level 1 once, then 2 forever.
    assert [d.level if d else None for d in decisions] == [None, 1, 2, 2, 2, 2]
    assert all(not d.hard_stop for d in decisions if d)


def test_hard_stop_after_two_ignored_notices() -> None:
    guard = make_guard(repeat_k=2, cooldown_turns=0, hard_stop=True)
    decisions = feed(guard, ["a"] * 5)
    assert [d.level if d else None for d in decisions] == [None, 1, 2, 3, 3]
    stop = decisions[3]
    assert stop.hard_stop is True and stop.breaker_text is None
    assert BREAKER_MARKER in hard_stop_text(stop.streak)


def test_changed_fingerprint_fully_rearms() -> None:
    guard = make_guard(repeat_k=2, cooldown_turns=5)
    tokens = ["a", "a"]          # fires at t2 (level 1, cooldown armed)
    decisions = feed(guard, tokens)
    assert decisions[-1].level == 1
    # The loop breaks: one different turn resets streak, cooldown, escalation.
    tokens += ["b"]
    assert guard.observe_request(transcript(tokens)) is None
    # A fresh cycle must again take k turns and starts back at level 1.
    tokens += ["c"]
    assert guard.observe_request(transcript(tokens)) is None
    tokens += ["c"]
    d = guard.observe_request(transcript(tokens))
    assert d is not None and d.level == 1 and d.streak == 2


def test_observe_request_never_mutates_messages() -> None:
    guard = make_guard(repeat_k=2)
    msgs = transcript(["a", "a"])
    snapshot = copy.deepcopy(msgs)
    guard.observe_request(msgs)
    assert msgs == snapshot


def test_disabled_guard_is_inert() -> None:
    guard = make_guard(enabled=False, repeat_k=2)
    assert all(d is None for d in feed(guard, ["a"] * 10))
    guard.observe_response({"timings": {"draft_n": 1000, "draft_n_accepted": 1000}})
    assert guard._timings_streak == 0


# ===========================================================================
# Timings signal (opportunistic corroboration)
# ===========================================================================


RECYCLED = {"timings": {"draft_n": 500, "draft_n_accepted": 500}}


def test_timings_accelerate_by_one_turn() -> None:
    # WITHOUT corroboration, k=3 means the 2nd identical turn does not fire...
    plain = make_guard(repeat_k=3, timings_m=1)
    assert feed(plain, ["a", "a"])[-1] is None
    # ...WITH a verbatim-recycling response observed after turn 1, the effective
    # k drops to 2 and the 2nd identical turn fires.
    guard = make_guard(repeat_k=3, timings_m=1)
    assert guard.observe_request(transcript(["a"])) is None
    guard.observe_response(RECYCLED)
    d = guard.observe_request(transcript(["a", "a"]))
    assert d is not None and d.streak == 2 and d.level == 1


def test_timings_never_required() -> None:
    # No responses observed at all: the guard still fires at k on content alone.
    guard = make_guard(repeat_k=3, timings_m=1)
    decisions = feed(guard, ["a", "a", "a"])
    assert decisions[-1] is not None


def test_timings_below_draft_n_min_or_absent_reset_streak() -> None:
    guard = make_guard(draft_n_min=200, timings_m=1)
    guard.observe_response(RECYCLED)
    assert guard._timings_streak == 1
    guard.observe_response({"timings": {"draft_n": 50, "draft_n_accepted": 50}})
    assert guard._timings_streak == 0  # tiny drafts are not a loop signal
    guard.observe_response(RECYCLED)
    guard.observe_response({"id": "x"})  # no timings at all
    assert guard._timings_streak == 0


def test_timings_low_acceptance_resets_streak() -> None:
    guard = make_guard(accept_threshold=0.99)
    guard.observe_response(RECYCLED)
    guard.observe_response({"timings": {"draft_n": 500, "draft_n_accepted": 300}})
    assert guard._timings_streak == 0


def test_stream_chunk_parsing_is_opportunistic() -> None:
    guard = make_guard()
    payload = {"choices": [], "timings": {"draft_n": 500, "draft_n_accepted": 500}}
    guard.observe_stream_chunk(b"data: " + json.dumps(payload).encode() + b"\n\n")
    assert guard._timings_streak == 1
    # Garbage / partial chunks are silently skipped, never raise.
    guard.observe_stream_chunk(b'data: {"timings": {"draft_n": 5')
    guard.observe_stream_chunk(b"data: [DONE]\n\n")
    guard.observe_stream_chunk(b"")
    assert guard._timings_streak == 1  # unchanged (nothing parseable observed)


# ===========================================================================
# Config validation
# ===========================================================================


@pytest.mark.parametrize("bad", [
    dict(repeat_k=1),
    dict(repeat_k=0),
    dict(timings_m=0),
    dict(accept_threshold=0.0),
    dict(accept_threshold=1.5),
    dict(draft_n_min=-1),
    dict(cooldown_turns=-1),
])
def test_config_validation_rejects(bad: dict) -> None:
    with pytest.raises(ValueError):
        LoopGuardConfig(**bad)


def test_breaker_notice_levels() -> None:
    assert "Do not repeat it again" in breaker_notice(1, 3)
    assert "FINAL NOTICE" in breaker_notice(2, 9)
    for level in (1, 2):
        assert breaker_notice(level, 4).startswith(BREAKER_MARKER)


# ===========================================================================
# THE property test — guard vs. an independent reference model (hypothesis)
# ===========================================================================


@given(
    seq=st.lists(st.sampled_from("ab"), min_size=0, max_size=40),
    k=st.integers(min_value=2, max_value=4),
    cooldown=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=200, deadline=None)
def test_property_fires_iff_k_consecutive_and_cooldown_clear(
    seq: list, k: int, cooldown: int
) -> None:
    """For ANY turn sequence: a decision is produced iff the trailing run of
    identical turns is >= k AND the cooldown window is clear. This subsumes
    'never fires without k consecutive repeats' and 'always fires at k'."""
    guard = make_guard(repeat_k=k, cooldown_turns=cooldown, hard_stop=False)

    run = 0
    last = None
    cd = 0
    for i, tok in enumerate(seq):
        decision = guard.observe_request(transcript(seq[: i + 1]))
        # Reference model (independent of the implementation's internals).
        if tok == last:
            run += 1
        else:
            run, cd = 1, 0
        last = tok
        fired = False
        if run >= k:
            if cd > 0:
                cd -= 1
            else:
                fired = True
                cd = cooldown
        assert (decision is not None) == fired, (
            f"turn {i}: run={run} cd={cd} expected fired={fired}"
        )
        if decision is not None:
            assert decision.streak == run
            assert decision.breaker_text is not None  # hard_stop off -> always text
