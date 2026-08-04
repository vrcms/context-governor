"""Dynamic upstream context size (2026-08-02).

n_ctx is probed from llama-server /props and EVERY setpoint derives from it —
handle threshold, high/low/emergency water. It used to be sampled once, at
startup, with a 5 s timeout, and never again. Two silent failure modes:

  - llama-server not up yet when the proxy started -> n_ctx None for the life of
    the process. That does not merely fall back to the static threshold, it
    disables windowing ENTIRELY (high_water is None), with nothing saying so.
  - somebody restarts llama-server with a different -c -> the proxy keeps sizing
    itself to a window that no longer exists. If n_ctx SHRANK, its high water now
    sits above the real limit.

The trap when fixing this: the handle threshold is DERIVED from n_ctx, so naively
adopting a new one re-cuts every stub decision on every live conversation at
once — a prefix break in both directions, and upward it is un-handle-ization,
the regression this codebase spent the day eliminating. Hence _pinned_threshold.
"""

from __future__ import annotations

from conftest import FakeCounter
from contextmanager.durable import DurableStore
from contextmanager.proxy.app import resolve_handle_threshold
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.proxy.sensing import conversation_key


def _rw(tmp_path, n_ctx=None, **over):
    kw = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        handle_threshold_tokens=50,
        stub_preview_chars=10,
        rehydrate_budget_tokens=0,
        auto_recall_k=0,
        request_timeout=30.0,
    )
    kw.update(over)
    cfg = ProxyConfig(**kw)
    store = DurableStore(cfg.store_root)
    return cfg, store, PromptRewriter(cfg, FakeCounter(), store, n_ctx=n_ctx)


MID = " ".join(f"w{i}" for i in range(80))      # 80 FakeCounter tokens


class TestScalesWithTheBox:
    def test_threshold_and_water_track_n_ctx(self):
        cfg = ProxyConfig(upstream_base_url="http://x")
        small = resolve_handle_threshold(cfg, 65536)
        big = resolve_handle_threshold(cfg, 262144)
        # Proportional up to integer truncation: 65536*0.02 -> 1310.72 -> 1310,
        # 262144*0.02 -> 5242.88 -> 5242, so 4*small is 5240, not 5242.
        assert abs(big - 4 * small) <= 4, "threshold does not track the real window"
        assert int(65536 * cfg.context_budget_ratio) < int(262144 * cfg.context_budget_ratio)

    def test_unknown_n_ctx_falls_back_to_static(self):
        cfg = ProxyConfig(upstream_base_url="http://x")
        assert resolve_handle_threshold(cfg, None) == cfg.handle_threshold_tokens


class TestUpdatePreservesStickyState:
    def test_sticky_state_survives_a_context_size_change(self, tmp_path):
        """A rebuild would drop every frontier and frozen block — i.e. break the
        prefix of every live conversation to adopt a setpoint."""
        cfg, store, rw = _rw(tmp_path, n_ctx=65536)
        wire = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "start"},
            {"role": "user", "content": MID},
        ]
        rw.rewrite_outgoing(wire)
        key = conversation_key(wire)
        rw._windowed[key] = {"msg-x": 123}
        rw._recall_frozen[key] = ("anchor", 1, "block")

        assert rw.update_context_size(262144, 5242) is True
        assert rw._n_ctx == 262144
        assert rw.config.handle_threshold_tokens == 5242
        assert rw._windowed[key] == {"msg-x": 123}, "windowing frontier was dropped"
        assert rw._recall_frozen[key] == ("anchor", 1, "block"), "recall freeze dropped"

    def test_no_change_reports_false(self, tmp_path):
        cfg, store, rw = _rw(tmp_path, n_ctx=65536)
        assert rw.update_context_size(65536, cfg.handle_threshold_tokens) is False


class TestThresholdIsPinnedPerConversation:
    def test_live_conversation_keeps_its_original_threshold(self, tmp_path):
        """Threshold UP must not un-handle-ize a message already sent as a stub."""
        cfg, store, rw = _rw(tmp_path, n_ctx=65536, handle_threshold_tokens=50)
        wire = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "start"},
            {"role": "user", "content": MID},          # 80 tokens >= 50 -> stub
        ]
        first = rw.rewrite_outgoing(wire)
        assert PromptRewriter.is_stub(first.messages[2]["content"])

        # llama-server restarted much larger: threshold 50 -> 500, which would put
        # this 80-token message BELOW the line.
        rw.update_context_size(262144, 500)

        wire2 = wire + [{"role": "user", "content": "next"}]
        second = rw.rewrite_outgoing(wire2)
        assert PromptRewriter.is_stub(second.messages[2]["content"]), (
            "an already-stubbed message came back VERBATIM after an n_ctx change "
            "— un-handle-ization, and a prefix break on every live conversation"
        )
        assert second.messages[: len(first.messages)] == first.messages

    def test_new_conversation_adopts_the_new_threshold(self, tmp_path):
        """Pinning must not freeze the setpoint forever — only for wires already
        in flight. A conversation starting after the change uses the new value."""
        cfg, store, rw = _rw(tmp_path, n_ctx=65536, handle_threshold_tokens=50)
        rw.rewrite_outgoing([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old conversation"},
            {"role": "user", "content": MID},
        ])
        rw.update_context_size(262144, 500)          # 80-token msgs now below line
        fresh = rw.rewrite_outgoing([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "brand new conversation"},
            {"role": "user", "content": MID},
        ])
        assert not PromptRewriter.is_stub(fresh.messages[2]["content"]), (
            "new conversation did not pick up the raised threshold"
        )
