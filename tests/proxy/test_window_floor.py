"""Minimum-shrink floor for windowing (2026-08-02).

``_window_out``'s only guard was ``if stub_tokens >= orig_tokens: return 0`` —
a BREAK-EVEN test. Break-even is the wrong threshold for an operation whose
costs the test cannot see:

  - paging a message that was already sent verbatim BREAKS THE UPSTREAM PREFIX
    (a full re-prefill of the whole prompt), and
  - it hides the content from the model.

At break-even those costs are pure loss. Measured on a live hermes run: 27
messages under 100 tokens were paged for a net 39 tokens each after the
~25-token stub, five of them assistant turns — the model's own prior reasoning.
Against the same wire the MEDIAN windowed message was 230 tokens vs a 25-token
stub (9.2x), so a 4.0 floor sits far below production reality: it rejects 27 of
74 paged messages but only 5.5% of the token yield.
"""

from __future__ import annotations

from conftest import FakeCounter
from contextmanager.durable import DurableStore
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter

import pytest


def _rw(tmp_path, n_ctx=None, **over):
    kw = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        handle_threshold_tokens=100000,   # keep Pass 1 out of the way entirely
        stub_preview_chars=0,
        rehydrate_budget_tokens=0,
        auto_recall_k=0,
        request_timeout=30.0,
    )
    kw.update(over)
    cfg = ProxyConfig(**kw)
    store = DurableStore(cfg.store_root)
    return cfg, PromptRewriter(cfg, FakeCounter(), store, n_ctx=n_ctx)


def _wire(mid_words: int, n_mid: int = 6):
    """head(2) + n_mid middles of mid_words tokens + protected tail(6)."""
    head = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    mids = [{"role": "user", "content": " ".join(f"m{i}w{j}" for j in range(mid_words))}
            for i in range(n_mid)]
    tail = [{"role": "user", "content": f"tail {w}"} for w in "abcdef"]
    return head + mids + tail


def _stubs(msgs):
    return [m for m in msgs
            if isinstance(m.get("content"), str) and PromptRewriter.is_stub(m["content"])]


class TestFloorRejectsMarginalTrades:
    def test_marginal_message_is_not_paged(self, tmp_path):
        """20 tokens against an ~8-token stub = 2.5x. Below the 4.0 floor."""
        _, rw = _rw(tmp_path, n_ctx=200, context_budget_ratio=0.5,
                    context_target_ratio=0.3)
        r = rw.rewrite_outgoing(_wire(20), pressure_tokens=195)
        assert r.windowing_triggered is True, "the trigger itself must still fire"
        assert _stubs(r.messages) == [], (
            "a 2.5x 'shrink' was paged — break-even accepted a trade that costs "
            "a prefix break and hides content to save a handful of tokens"
        )

    def test_genuine_shrink_is_still_paged(self, tmp_path):
        """80 tokens against an ~8-token stub = 10x. Comfortably worth it."""
        _, rw = _rw(tmp_path, n_ctx=200, context_budget_ratio=0.5,
                    context_target_ratio=0.3)
        r = rw.rewrite_outgoing(_wire(80), pressure_tokens=195)
        assert r.windowing_triggered is True
        assert _stubs(r.messages), "a 10x shrink was refused — the floor is too aggressive"

    def test_ratio_zero_restores_break_even(self, tmp_path):
        """0 must reproduce the historical behaviour exactly, so the change is
        opt-out for anyone depending on it."""
        _, rw = _rw(tmp_path, n_ctx=200, context_budget_ratio=0.5,
                    context_target_ratio=0.3, window_min_shrink_ratio=0.0)
        r = rw.rewrite_outgoing(_wire(20), pressure_tokens=195)
        assert _stubs(r.messages), "ratio=0 should page the marginal message as before"

    def test_negative_ratio_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="window_min_shrink_ratio"):
            ProxyConfig(upstream_base_url="http://x", window_min_shrink_ratio=-1.0)


class TestFloorDoesNotDisableWindowing:
    """The mechanics suites disable the ratio to keep their arithmetic legible,
    so this is the one place windowing runs end-to-end at the PRODUCTION
    default — the interaction that would otherwise go uncovered."""

    def test_windowing_still_sheds_at_default_ratio(self, tmp_path):
        _, rw = _rw(tmp_path, n_ctx=200, context_budget_ratio=0.5,
                    context_target_ratio=0.3)
        assert rw.config.window_min_shrink_ratio == 4.0, "not testing the default"
        r = rw.rewrite_outgoing(_wire(80), pressure_tokens=195)
        assert r.windowing_triggered is True
        assert len(_stubs(r.messages)) >= 1

    def test_frontier_stays_sticky_at_default_ratio(self, tmp_path):
        """A message paged under the floor must still re-stub byte-identically
        next turn, or the floor has broken monotonic handle-ization."""
        _, rw = _rw(tmp_path, n_ctx=200, context_budget_ratio=0.5,
                    context_target_ratio=0.3)
        w = _wire(80)
        r1 = rw.rewrite_outgoing(w, pressure_tokens=195)
        assert _stubs(r1.messages)
        r2 = rw.rewrite_outgoing(w + [{"role": "user", "content": "next"}],
                                 pressure_tokens=90)
        assert r2.windowing_triggered is False
        assert r2.messages[: len(r1.messages)] == r1.messages
