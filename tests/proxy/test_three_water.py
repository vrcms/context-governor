"""Third windowing tier: context_emergency_ratio (the HIGH water).

Guards the 2026-07-28 addition. Pass 3's hysteresis latch exists so a shed
attempt that cannot reach the low water (unsheddable mass) doesn't keep
breaking the prefix for nothing -- it holds the wire steady until pressure
grows by the hysteresis gap. A live session showed pressure climb to 96% of
n_ctx while latched, because the gap-based re-arm didn't fire fast enough.

context_emergency_ratio adds a HIGH water, above the existing MID water
(context_budget_ratio): crossing it OVERRIDES the latch and forces a shed
attempt every request regardless of the hysteresis gap. Disabled by default
(0.0) -- every test here that sets it to 0 (or omits it) must behave IDENTICAL
to before this feature existed.
"""

from __future__ import annotations

import pytest

from conftest import FakeCounter
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.durable import DurableStore


def _cfg(tmp_path, **over):
    kw = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        upstream_api_key=None,
        listen_host="127.0.0.1",
        listen_port=8900,
        handle_threshold_tokens=50,
        stub_preview_chars=10,
        rehydrate_budget_tokens=0,
        context_budget_ratio=0.5,     # MID water @ n_ctx=1000 -> 500
        context_target_ratio=0.2,    # LOW water -> 200 (gap = 300)
        protect_first_n=1,
        protect_last_n=1,
        request_timeout=30.0,
    )
    kw.update(over)
    return ProxyConfig(**kw)


UNSHEDDABLE = {"role": "user", "content": [{"type": "text", "text": "unsheddable"}]}
MESSAGES = [
    {"role": "system", "content": "sys"},   # protected (protect_first_n=1)
    UNSHEDDABLE,                            # non-str content -> _window_out always returns 0
    {"role": "user", "content": "tail"},    # protected (protect_last_n=1)
]


def _rw(tmp_path, emergency_ratio):
    store = DurableStore(str(tmp_path / "store2"))
    cfg = _cfg(tmp_path, context_emergency_ratio=emergency_ratio)
    return PromptRewriter(cfg, FakeCounter(), store, n_ctx=1000)


class TestConfigValidation:
    def test_zero_is_valid_and_disables(self, tmp_path):
        _cfg(tmp_path, context_emergency_ratio=0.0)  # must not raise

    def test_must_be_in_0_1(self, tmp_path):
        with pytest.raises(ValueError, match="context_emergency_ratio"):
            _cfg(tmp_path, context_emergency_ratio=1.5)
        with pytest.raises(ValueError, match="context_emergency_ratio"):
            _cfg(tmp_path, context_emergency_ratio=-0.1)

    def test_must_exceed_mid_water(self, tmp_path):
        # budget_ratio (MID) is 0.5 here; emergency must be strictly above it.
        with pytest.raises(ValueError, match="context_emergency_ratio"):
            _cfg(tmp_path, context_emergency_ratio=0.5)
        with pytest.raises(ValueError, match="context_emergency_ratio"):
            _cfg(tmp_path, context_emergency_ratio=0.3)

    def test_above_mid_water_is_fine(self, tmp_path):
        _cfg(tmp_path, context_emergency_ratio=0.51)  # must not raise


class TestDisabledIsIdenticalToBefore:
    """The whole point of "make it reversible": emergency_ratio=0 (the
    default) must reproduce exactly what Pass 3 did before this feature."""

    def test_latch_holds_indefinitely_when_disabled(self, tmp_path):
        rw = _rw(tmp_path, emergency_ratio=0.0)
        r1 = rw.rewrite_outgoing(MESSAGES, pressure_tokens=550, high_water_tokens=None)
        assert r1.windowing_triggered and not r1.windowing_emergency

        # Pressure climbs well past what would be the emergency line in the
        # ENABLED test below (750) -- but the tier is off, so still latched.
        r2 = rw.rewrite_outgoing(MESSAGES, pressure_tokens=750, high_water_tokens=None)
        assert not r2.windowing_triggered
        assert not r2.windowing_emergency

    def test_windowing_emergency_always_false_when_disabled(self, tmp_path):
        rw = _rw(tmp_path, emergency_ratio=0.0)
        for pressure in (550, 600, 750, 900, 950):
            r = rw.rewrite_outgoing(MESSAGES, pressure_tokens=pressure, high_water_tokens=None)
            assert r.windowing_emergency is False


class TestEmergencyOverride:
    """The mechanism itself: crossing HIGH water forces a retry the latch
    would otherwise have suppressed."""

    def test_first_trigger_latches_normally(self, tmp_path):
        rw = _rw(tmp_path, emergency_ratio=0.7)  # HIGH water -> 700
        r1 = rw.rewrite_outgoing(MESSAGES, pressure_tokens=550, high_water_tokens=None)
        assert r1.windowing_triggered is True
        assert r1.windowing_emergency is False  # a PLAIN trigger, not an override

    def test_stays_latched_below_high_water(self, tmp_path):
        rw = _rw(tmp_path, emergency_ratio=0.7)
        rw.rewrite_outgoing(MESSAGES, pressure_tokens=550, high_water_tokens=None)
        # 600 < high_water(700) and < latched_at(550) + gap(300) = 850 -> suppressed
        r2 = rw.rewrite_outgoing(MESSAGES, pressure_tokens=600, high_water_tokens=None)
        assert r2.windowing_triggered is False
        assert r2.windowing_emergency is False

    def test_crossing_high_water_overrides_the_latch(self, tmp_path):
        rw = _rw(tmp_path, emergency_ratio=0.7)  # HIGH water -> 700
        rw.rewrite_outgoing(MESSAGES, pressure_tokens=550, high_water_tokens=None)  # latch at 550
        rw.rewrite_outgoing(MESSAGES, pressure_tokens=600, high_water_tokens=None)  # still latched
        # 750 >= emergency_water(700) but STILL < latched_at(550)+gap(300)=850 --
        # this is exactly the case a two-tier system cannot handle.
        r3 = rw.rewrite_outgoing(MESSAGES, pressure_tokens=750, high_water_tokens=None)
        assert r3.windowing_triggered is True
        assert r3.windowing_emergency is True

    def test_below_mid_water_rearms_regardless_of_emergency_tier(self, tmp_path):
        rw = _rw(tmp_path, emergency_ratio=0.7)
        rw.rewrite_outgoing(MESSAGES, pressure_tokens=550, high_water_tokens=None)
        r2 = rw.rewrite_outgoing(MESSAGES, pressure_tokens=400, high_water_tokens=None)
        assert r2.windowing_triggered is False  # under MID water: nothing to do
        assert r2.windowing_emergency is False

    def test_when_shedding_succeeds_no_latch_no_emergency_needed(self, tmp_path):
        # A conversation with real sheddable content never needs the override.
        # NOTE: each message must stay BELOW handle_threshold_tokens(50) or
        # Pass 1 stubs it before Pass 3 ever runs, leaving nothing for Pass 3
        # to shed (a single huge message is Pass 1's job, not Pass 3's) --
        # many SMALL messages are what Pass 3 windowing actually exists for.
        store = DurableStore(str(tmp_path / "store3"))
        cfg = _cfg(tmp_path, context_emergency_ratio=0.7)
        rw = PromptRewriter(cfg, FakeCounter(), store, n_ctx=1000)
        middle = [{"role": "user", "content": "word " * 40} for _ in range(20)]
        sheddable = ([{"role": "system", "content": "sys"}] + middle
                     + [{"role": "user", "content": "tail"}])
        r1 = rw.rewrite_outgoing(sheddable, pressure_tokens=550, high_water_tokens=None)
        assert r1.windowing_triggered is True
        assert r1.windowing_emergency is False
        assert rw._window_latched == {}  # confirms the premise: shed succeeded
        r2 = rw.rewrite_outgoing(r1.messages, pressure_tokens=750, high_water_tokens=None)
        # Shedding worked last time (latch cleared), so 750 is a fresh, PLAIN
        # trigger against the (now-stubbed) sticky frontier -- not an override.
        assert r2.windowing_emergency is False


class TestLegacyPathParity:
    """The chars/4 open-loop fallback (pressure_tokens=None) gets the same
    override, mirrored for consistency -- and must not regress when off."""

    def test_legacy_path_runs_with_field_present(self, tmp_path):
        rw = _rw(tmp_path, emergency_ratio=0.0)
        r = rw.rewrite_outgoing(MESSAGES, pressure_tokens=None, high_water_tokens=None)
        assert r.windowing_emergency is False  # never raises, always defined
