"""The sticky-recall staleness bound, anchored to the real window.

`recall_max_stale_tokens = 4000` was the last absolute token constant among
setpoints that are otherwise all fractions of n_ctx, and an absolute here is
three different policies on three servers:

    4000 tokens  ==   2% of a 200K window   -> rebuild almost every turn
                      6% of  65K            -> ~12 rebuilds on the measured run
                     50% of   8K            -> effectively never rebuild

Only the middle case was ever observed, and there 6 of 13 prefix breaks on the
opencode run were the recall block moving off-epoch. These tests pin the derived
bound at both extremes so the behaviour cannot silently become
machine-specific again.
"""

from __future__ import annotations

import pytest

from conftest import FakeCounter
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.durable import DurableStore, Message


def _rw(tmp_path, n_ctx=None, **over):
    kw = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        listen_host="127.0.0.1",
        listen_port=8900,
        handle_threshold_tokens=1310,
        stub_preview_chars=200,
        rehydrate_budget_tokens=4000,
        request_timeout=30.0,
    )
    kw.update(over)
    r = PromptRewriter(ProxyConfig(**kw), FakeCounter(),
                       DurableStore(str(tmp_path / "s")))
    if n_ctx:
        r.update_context_size(n_ctx, 1310)
    return r


class TestAnchoredToTheWindow:
    def test_bound_scales_with_n_ctx(self, tmp_path):
        assert _rw(tmp_path, n_ctx=65536)._recall_stale_bound() == 16384
        assert _rw(tmp_path, n_ctx=200000)._recall_stale_bound() == 50000
        assert _rw(tmp_path, n_ctx=8192)._recall_stale_bound() == 2048

    def test_the_new_default_is_looser_than_the_old_constant(self, tmp_path):
        """At the window this was measured on, the whole point is FEWER forced
        rebuilds — the old 4000 produced ~12 on a conversation reaching 46-49K."""
        assert _rw(tmp_path, n_ctx=65536)._recall_stale_bound() > 4000

    def test_bound_is_above_the_windowing_band(self, tmp_path):
        """0.25 sits above budget - target (0.50 - 0.35 = 0.15), so the block
        refreshes at most once per windowing cycle and tends to land ON a flush
        epoch, where the prefix is already broken and the refresh is free."""
        r = _rw(tmp_path, n_ctx=65536)
        band = r.config.context_budget_ratio - r.config.context_target_ratio
        assert r.config.recall_max_stale_ratio > band


class TestEscapeHatches:
    def test_ratio_zero_falls_back_to_the_fixed_value(self, tmp_path):
        r = _rw(tmp_path, n_ctx=65536, recall_max_stale_ratio=0.0)
        assert r._recall_stale_bound() == 4000

    def test_unknown_n_ctx_falls_back_to_the_fixed_value(self, tmp_path):
        """Server down at startup must not silently disable sticky recall."""
        assert _rw(tmp_path)._recall_stale_bound() == 4000

    def test_zero_tokens_still_means_legacy_per_turn_recompute(self, tmp_path):
        """`recall_max_stale_tokens = 0` is the operator explicitly selecting
        per-turn recompute. A ratio must not re-enable stickiness they turned
        off — the ratio is an anchoring policy, not an override."""
        r = _rw(tmp_path, n_ctx=65536, recall_max_stale_tokens=0,
                recall_max_stale_ratio=0.25)
        assert r._recall_stale_bound() == 0


class TestTheBoundIsActuallyWired:
    """The tests above exercise the helper in isolation, which proves the
    arithmetic and nothing about whether `rewrite_outgoing` consults it. This
    one drives the real sticky-recall path: the FIXED bound is set absurdly low
    (50) and the DERIVED bound high (0.25 * 65536), so growth that lands between
    them distinguishes the two — refresh means the fixed value is still in
    force, reuse means the derived one is."""

    def test_growth_past_the_fixed_bound_but_under_the_derived_one_reuses(
        self, tmp_path
    ):
        cfg = ProxyConfig(
            upstream_base_url="http://upstream.test",
            store_root=str(tmp_path / "store"),
            handle_threshold_tokens=10,
            stub_preview_chars=10,
            rehydrate_budget_tokens=4000,
            request_timeout=30.0,
            recall_max_stale_tokens=50,      # would refresh
            recall_max_stale_ratio=0.25,     # 16,384 — would not
        )
        store = DurableStore(cfg.store_root)
        rw = PromptRewriter(cfg, FakeCounter(), store)
        rw.update_context_size(65536, 1310)
        assert rw._recall_stale_bound() == 16384

        handle = store.page_out(Message(
            role="user", id="old-1",
            content="unicorn banana smoothie recipe with quantum sprinkles",
        ))
        t1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "unicorn banana smoothie recipe quantum?"},
        ]
        r1 = rw.rewrite_outgoing(t1)
        assert r1.recalled_handles == [handle]

        # ~280 chars == ~70 est-tokens of post-anchor growth: over the fixed 50,
        # far under the derived 16,384.
        big = " ".join(f"smoothiepreparationworklogentry{i:02d}" for i in range(9))
        t2 = t1 + [
            {"role": "assistant", "content": big},
            {"role": "user", "content": "unicorn smoothie quantum sprinkles next?"},
        ]
        r2 = rw.rewrite_outgoing(t2)

        assert r2.recalled_handles == [], "block was rebuilt: the fixed bound is still in force"
        recall_1 = [m for m in r1.messages if PromptRewriter.is_recall(m.get("content", ""))]
        recall_2 = [m for m in r2.messages if PromptRewriter.is_recall(m.get("content", ""))]
        assert recall_2 == recall_1
        # The whole previous wire is a byte-exact prefix of this one — the
        # property the staleness bound exists to protect.
        assert r2.messages[: len(r1.messages)] == r1.messages


class TestValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_ratio_out_of_range_is_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError, match="recall_max_stale_ratio"):
            _rw(tmp_path, recall_max_stale_ratio=bad)
