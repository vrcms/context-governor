"""14c pressure estimate — modelling the SENT wire, not the incoming one.

Guards the 2026-07-28 fix for Problem A. The estimate used to be a flat
``(chars - prev_chars) / 4`` over the INCOMING wire, but Pass 1 pages out every
message at or above the handle threshold and sends a ~126-token stub instead. A
60 KB tool read therefore contributed ~15,000 phantom tokens of pressure.

Measured on the live stack, on the turn that tripped it:

    est_pressure 37,666   vs   real prompt 21,259

Pressure crossed the high water, Pass 3 windowed, and windowing sheds from
``protect_first_n`` forward — breaking the prefix and forcing a full re-prefill
for nothing. Live signature: ``windowing_triggers: 2`` matching
``own-mutation: 2``, with the 14b latch suppressing three further crossings.
"""

from __future__ import annotations

from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.proxy.sensing import (
    KIND_APPEND,
    KIND_MID_EDIT,
    GovernorController,
    _growth_estimate,
)

THRESHOLD = 1064          # 0.02 * 53248, the live handle threshold
STUB_EST = PromptRewriter.stub_tokens_estimate(200)   # default stub_preview_chars


def _chars(messages: list) -> int:
    return sum(len(m["content"]) for m in messages if isinstance(m, dict))


class TestStubTokensEstimate:
    def test_sized_from_the_real_stub_format(self):
        # Rendered through make_stub, so it cannot drift if the format changes.
        assert 80 <= STUB_EST <= 200

    def test_scales_with_preview_chars(self):
        assert (PromptRewriter.stub_tokens_estimate(400)
                > PromptRewriter.stub_tokens_estimate(100))

    def test_never_zero(self):
        assert PromptRewriter.stub_tokens_estimate(0) >= 1


class TestGrowthEstimate:
    def test_bulky_append_costs_a_stub_not_its_own_size(self):
        # THE regression: the live turn-11 case, a 60 KB read appended.
        prev = [{"role": "user", "content": "hi"}]
        msgs = prev + [{"role": "assistant", "content": "ok"},
                       {"role": "tool", "content": "x" * 60000}]
        args = (msgs, KIND_APPEND, len(prev), _chars(msgs), _chars(prev))

        legacy = _growth_estimate(*args, None, None)
        fixed = _growth_estimate(*args, THRESHOLD, STUB_EST)

        assert legacy > 14000, "fixture must reproduce the phantom mass"
        assert fixed < STUB_EST + 50, f"bulky content still over-counted: {fixed}"
        assert legacy - fixed > 14000

    def test_small_appends_still_count_in_full(self):
        # Content below the threshold reaches the model verbatim, so it must be
        # counted at chars/4 exactly as before — the fix must not under-report.
        prev = [{"role": "user", "content": "hi"}]
        msgs = prev + [{"role": "user", "content": "y" * 400}]
        args = (msgs, KIND_APPEND, len(prev), _chars(msgs), _chars(prev))
        assert _growth_estimate(*args, THRESHOLD, STUB_EST) == 100

    def test_tool_calls_counted_in_full(self):
        # Pass 1 stubs `content`; `tool_calls` payloads pass through untouched,
        # so they must never be discounted.
        prev = [{"role": "user", "content": "hi"}]
        big = [{"id": "1", "function": {"name": "read", "arguments": "z" * 4000}}]
        msgs = prev + [{"role": "assistant", "content": "", "tool_calls": big}]
        args = (msgs, KIND_APPEND, len(prev), _chars(msgs), _chars(prev))
        assert _growth_estimate(*args, THRESHOLD, STUB_EST) > 900

    def test_non_append_falls_back_to_aggregate(self):
        prev = [{"role": "user", "content": "hi"}]
        msgs = [{"role": "user", "content": "x" * 60000}]
        args = (msgs, KIND_MID_EDIT, 0, _chars(msgs), _chars(prev))
        assert (_growth_estimate(*args, THRESHOLD, STUB_EST)
                == _growth_estimate(*args, None, None))

    def test_unknown_settings_preserve_legacy_behaviour(self):
        prev = [{"role": "user", "content": "hi"}]
        msgs = prev + [{"role": "tool", "content": "x" * 60000}]
        args = (msgs, KIND_APPEND, len(prev), _chars(msgs), _chars(prev))
        assert (_growth_estimate(*args, None, STUB_EST)
                == _growth_estimate(*args, THRESHOLD, None)
                == _growth_estimate(*args, None, None))

    def test_never_negative(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _growth_estimate(msgs, KIND_APPEND, 0, 2, 99999,
                                THRESHOLD, STUB_EST) >= 0


class TestThroughTheController:
    """End-to-end: pressure must stay near the real prompt across a bulky turn."""

    def _turn(self, ctrl, messages, *, prompt_tokens, completion_tokens):
        obs = ctrl.observe_request(
            messages, handle_threshold_tokens=THRESHOLD, stub_tokens_est=STUB_EST)
        ctrl.note_sent(obs.key, messages, observation=obs)
        ctrl.observe_response(obs.key, {"usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens}})
        return obs

    def test_bulky_read_does_not_inflate_pressure(self):
        ctrl = GovernorController()
        t1 = [{"role": "system", "content": "sys"},
              {"role": "user", "content": "start"}]
        self._turn(ctrl, t1, prompt_tokens=20000, completion_tokens=300)

        # Same conversation, appending a 60 KB tool result.
        t2 = t1 + [{"role": "assistant", "content": "ok"},
                   {"role": "tool", "content": "x" * 60000}]
        obs = self._turn(ctrl, t2, prompt_tokens=20200, completion_tokens=100)

        # Real prompt came back at 20,200. Pressure must land near it, NOT
        # ~15,000 above it (which is what crossed the high water live).
        assert obs.pressure_tokens is not None
        assert obs.pressure_tokens < 21000, (
            f"pressure {obs.pressure_tokens} still inflated by pre-stub chars")

    def test_completion_is_not_double_counted(self):
        # The completion arrives inside the appended region, so counting
        # last_completion_tokens separately was a systematic overshoot.
        ctrl = GovernorController()
        t1 = [{"role": "user", "content": "start"}]
        self._turn(ctrl, t1, prompt_tokens=10000, completion_tokens=4000)

        t2 = t1 + [{"role": "assistant", "content": "z" * 400}]
        obs = self._turn(ctrl, t2, prompt_tokens=10100, completion_tokens=10)

        # Appended assistant message is 400 chars ~ 100 tokens. Adding the
        # previous 4,000-token completion on top would land near 14,100.
        assert obs.pressure_tokens is not None
        assert obs.pressure_tokens < 10500, (
            f"pressure {obs.pressure_tokens} looks like a double-count")
