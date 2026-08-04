"""Phase 14b break-riding + 14c pressure tests — voluntary breaks go to zero.

Rewriter-level invariants:

  - STEADY STATE: a growing conversation below every bound produces a wire that
    is a byte-exact prefix extension of the previous turn's, every turn —
    voluntary breaks = 0 (the 14b acceptance criterion).
  - FLUSH EPOCH: ``prefix_broken=True`` (the classifier says the harness
    already broke this turn's prefix) refreshes the recall block below the
    staleness bound — the refresh rides the break instead of causing one.
  - Sticky state (``_recall_frozen``, ``_windowed``) is keyed PER CONVERSATION:
    interleaved conversations no longer thrash one global slot.
  - Non-str (content-parts) anchors freeze too.
  - 14c: windowing pressure comes from REAL usage-derived tokens when provided
    (the chars/4 estimate was blind to ~88% of the wire), and the effective
    high water can be overridden by the learned-ceiling setpoint.

Same harness as test_recall.py: FakeCounter (token = word count) + a real
``DurableStore(tmp_path)``. No network, no FastAPI.
"""

from __future__ import annotations

from conftest import FakeCounter
from contextmanager.durable import DurableStore
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.proxy.sensing import conversation_key
from contextmanager.types import Message


def _config(tmp_path, **over) -> ProxyConfig:
    kw = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        handle_threshold_tokens=10,
        # These suites drive windowing with deliberately TINY messages so the
        # n_ctx=200 arithmetic stays legible (20 tokens vs an 8-token stub =
        # 2.5x shrink). window_min_shrink_ratio defaults to 4.0, which
        # correctly rejects trades that marginal — so it is disabled here to
        # keep these tests about windowing MECHANICS. The floor itself is
        # covered in test_window_floor.py, including an end-to-end shed at the
        # production default.
        window_min_shrink_ratio=0.0,
        stub_preview_chars=10,
        rehydrate_budget_tokens=4000,
        request_timeout=30.0,
    )
    kw.update(over)
    return ProxyConfig(**kw)


def _rewriter(tmp_path, n_ctx=None, **over):
    cfg = _config(tmp_path, **over)
    store = DurableStore(cfg.store_root)
    counter = FakeCounter()
    return cfg, store, counter, PromptRewriter(cfg, counter, store, n_ctx=n_ctx)


def _seed(store: DurableStore, mid: str, content: str) -> str:
    return store.page_out(Message(role="user", content=content, id=mid))


def _recall_messages(messages: list) -> list:
    return [m for m in messages
            if PromptRewriter.is_recall(m.get("content", ""))]


# ---------------------------------------------------------------------------
# 14b — steady state: zero voluntary breaks
# ---------------------------------------------------------------------------


class TestSteadyState:
    def test_growing_session_is_pure_prefix_extension(self, tmp_path):
        """The 14b acceptance: a steady-state synthetic session produces ZERO
        voluntary breaks — every turn's wire byte-extends the previous one."""
        cfg, store, counter, rw = _rewriter(tmp_path)
        h = _seed(store, "old-1", "unicorn banana smoothie recipe with quantum sprinkles")
        t = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "unicorn banana smoothie recipe quantum?"},
        ]
        r_prev = rw.rewrite_outgoing(t)
        assert r_prev.recalled_handles == [h]  # the one legitimate build
        for i in range(5):
            t = t + [
                {"role": "assistant", "content": f"step {i} done"},
                {"role": "user", "content": f"continue with step {i + 1}"},
            ]
            r = rw.rewrite_outgoing(t)
            assert r.recalled_handles == []            # no fresh search
            assert r.windowing_triggered is False
            assert r.messages[: len(r_prev.messages)] == r_prev.messages
            r_prev = r

    def test_prefix_broken_flushes_below_the_bound(self, tmp_path):
        """A harness-broken turn is a FLUSH EPOCH: the pending refresh rides it
        even though growth is far below recall_max_stale_tokens."""
        cfg, store, counter, rw = _rewriter(tmp_path)
        h = _seed(store, "old-1", "unicorn banana smoothie recipe with quantum sprinkles")
        t1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "unicorn banana smoothie recipe quantum?"},
        ]
        r1 = rw.rewrite_outgoing(t1)
        assert r1.recalled_handles == [h]
        t2 = t1 + [
            {"role": "assistant", "content": "here it is"},
            {"role": "user", "content": "unicorn smoothie quantum next?"},
        ]
        r2 = rw.rewrite_outgoing(t2, prefix_broken=True)
        # Fresh build, re-anchored at the new tail — on the harness's dime.
        assert r2.recalled_handles == [h]
        blocks = _recall_messages(r2.messages)
        assert len(blocks) == 1
        assert r2.messages[-2] is blocks[0]
        # And the new freeze holds again on the next quiet turn.
        t3 = t2 + [{"role": "assistant", "content": "ok"},
                   {"role": "user", "content": "thanks"}]
        r3 = rw.rewrite_outgoing(t3)
        assert r3.recalled_handles == []
        assert r3.messages[: len(r2.messages)] == r2.messages

    def test_duplicated_anchor_stays_put(self, tmp_path):
        """The freeze anchors by content, and the anchor message's content also
        appears EARLIER on the wire (a repeated boilerplate turn). The
        exact-index pin keeps the block at its last-sent position — first-match
        used to relocate it BACKWARD, a voluntary prefix break per flush."""
        cfg, store, counter, rw = _rewriter(tmp_path)
        h = _seed(store, "old-1", "unicorn banana smoothie recipe with quantum sprinkles")
        dup = "unicorn smoothie quantum repeat me"
        t1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": dup},
            {"role": "assistant", "content": "mid turn answer"},
            {"role": "user", "content": dup},   # the anchor: content duplicated at index 1
        ]
        r1 = rw.rewrite_outgoing(t1)
        assert r1.recalled_handles == [h]
        t2 = t1 + [
            {"role": "assistant", "content": "next answer"},
            {"role": "user", "content": "fresh question"},
        ]
        r2 = rw.rewrite_outgoing(t2)
        assert r2.recalled_handles == []                       # frozen block reused
        assert r2.messages[: len(r1.messages)] == r1.messages  # ...at the SAME position


# ---------------------------------------------------------------------------
# 14b — per-conversation sticky state
# ---------------------------------------------------------------------------


class TestPerConversationState:
    def test_interleaved_conversations_do_not_thrash(self, tmp_path):
        """The 2026-07-19 bug: one global frozen slot across >=3 interleaved
        conversations meant every switch rebuilt the block (search_calls
        14/25). Keyed per conversation, A's freeze survives B's turns."""
        cfg, store, counter, rw = _rewriter(tmp_path)
        ha = _seed(store, "old-a", "unicorn banana smoothie recipe with quantum sprinkles")
        hb = _seed(store, "old-b", "gearbox lighthouse assembly torque manual pages")
        a1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "unicorn banana smoothie recipe quantum?"},
        ]
        b1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "gearbox lighthouse assembly torque?"},
        ]
        ra1 = rw.rewrite_outgoing(a1)
        assert ra1.recalled_handles == [ha]
        rb1 = rw.rewrite_outgoing(b1)         # B's build must not evict A's
        assert rb1.recalled_handles == [hb]
        a2 = a1 + [
            {"role": "assistant", "content": "smoothie ready"},
            {"role": "user", "content": "add the quantum sprinkles"},
        ]
        ra2 = rw.rewrite_outgoing(a2)
        assert ra2.recalled_handles == []      # A's frozen block was reused
        assert ra2.messages[: len(ra1.messages)] == ra1.messages
        b2 = b1 + [
            {"role": "assistant", "content": "torque spec found"},
            {"role": "user", "content": "assemble the gearbox"},
        ]
        rb2 = rw.rewrite_outgoing(b2)
        assert rb2.recalled_handles == []      # and so was B's
        assert rb2.messages[: len(rb1.messages)] == rb1.messages

    def test_lru_cap_evicts_oldest_conversation(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, max_conversations=1)
        ha = _seed(store, "old-a", "unicorn banana smoothie recipe with quantum sprinkles")
        hb = _seed(store, "old-b", "gearbox lighthouse assembly torque manual pages")
        a1 = [{"role": "system", "content": "You are helpful"},
              {"role": "user", "content": "unicorn banana smoothie recipe quantum?"}]
        b1 = [{"role": "system", "content": "You are helpful"},
              {"role": "user", "content": "gearbox lighthouse assembly torque?"}]
        assert rw.rewrite_outgoing(a1).recalled_handles == [ha]
        assert rw.rewrite_outgoing(b1).recalled_handles == [hb]  # evicts A (cap 1)
        a2 = a1 + [{"role": "assistant", "content": "ready"},
                   {"role": "user", "content": "sprinkles please"}]
        ra2 = rw.rewrite_outgoing(a2)
        assert ra2.recalled_handles == [ha]    # A's freeze was evicted -> rebuilt

    def test_non_str_anchor_freezes(self, tmp_path):
        """Phase 12 skipped the freeze when the final message was content-parts
        — every image turn became a guaranteed per-turn prefix break. 14b
        freezes on the canonical serialization instead."""
        cfg, store, counter, rw = _rewriter(tmp_path)
        h = _seed(store, "old-1", "unicorn banana smoothie recipe with quantum sprinkles")
        parts_tail = {"role": "user", "content": [
            {"type": "text", "text": "and here is the smoothie photo"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}
        t1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "unicorn banana smoothie recipe quantum?"},
            parts_tail,
        ]
        r1 = rw.rewrite_outgoing(t1)
        assert r1.recalled_handles == [h]
        t2 = t1 + [
            {"role": "assistant", "content": "nice photo"},
            {"role": "user", "content": "now the quantum sprinkles"},
        ]
        r2 = rw.rewrite_outgoing(t2)
        # The frozen block came back byte-identically at the parts-anchor.
        assert r2.recalled_handles == []
        assert r2.messages[: len(r1.messages)] == r1.messages


# ---------------------------------------------------------------------------
# 14b — windowing frontier keyed per conversation
# ---------------------------------------------------------------------------


class TestWindowedFrontierPerConversation:
    def test_one_conversations_frontier_does_not_stub_another(self, tmp_path):
        cfg = _config(
            tmp_path,
            recall_max_stale_tokens=10_000,
            handle_threshold_tokens=50,
            context_budget_ratio=0.5,
            context_target_ratio=0.3,
        )
        store = DurableStore(cfg.store_root)
        rw = PromptRewriter(cfg, FakeCounter(), store, n_ctx=200)
        # Conversation A crosses the high water and windows its middle.
        fillers = [
            {"role": "user",
             "content": " ".join(f"fillstuff{i}word{j}" for j in range(15))}
            for i in range(8)
        ]
        tail = [{"role": "user", "content": f"alpha beta gamma step {w}"}
                for w in ("one", "two", "three", "four", "five", "six")]
        a = [{"role": "system", "content": "You are helpful"},
             {"role": "user", "content": "big conversation about filler stuff"}]
        ra = rw.rewrite_outgoing(a + fillers + tail)
        assert ra.windowing_triggered is True
        assert any(PromptRewriter.is_stub(m.get("content", "")) for m in ra.messages)
        # Conversation B reuses the SAME filler contents but is small: A's
        # frontier must not leak into B (the old global dict would stub these).
        b = [{"role": "system", "content": "You are helpful"},
             {"role": "user", "content": "tiny unrelated conversation"}]
        rb = rw.rewrite_outgoing(b + fillers[:2])
        assert rb.windowing_triggered is False
        assert not any(PromptRewriter.is_stub(m.get("content", ""))
                       for m in rb.messages)


# ---------------------------------------------------------------------------
# 14c — real pressure replaces the blind chars/4 estimate
# ---------------------------------------------------------------------------


def _small_wire():
    """12 small messages: est chars are tiny (the string-only view), so only a
    REAL pressure signal can reveal the invisible mass (tools, content-parts,
    template overhead)."""
    head = [{"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "start the work"}]
    middles = [
        {"role": "user",
         "content": " ".join(f"middlefill{i}w{j}" for j in range(20))}
        for i in range(4)
    ]
    tail = [{"role": "user", "content": f"tail step {w}"}
            for w in ("a", "b", "c", "d", "e", "f")]
    return head + middles + tail


class TestRealPressure:
    def test_real_pressure_triggers_when_est_is_blind(self, tmp_path):
        cfg, store, counter, rw = _rewriter(
            tmp_path, n_ctx=200, handle_threshold_tokens=50,
            context_budget_ratio=0.5, context_target_ratio=0.3,
            auto_recall_k=0,
        )
        msgs = _small_wire()
        # est tokens ~ (sum of chars)/4 << 100 == high water; real pressure says
        # the wire (with its invisible mass) is at 150.
        r = rw.rewrite_outgoing(msgs, pressure_tokens=150)
        assert r.windowing_triggered is True
        stubs = [m for m in r.messages if PromptRewriter.is_stub(m.get("content", ""))]
        assert len(stubs) == 4        # every eligible middle was paged

    def test_real_pressure_below_high_never_triggers(self, tmp_path):
        cfg, store, counter, rw = _rewriter(
            tmp_path, n_ctx=200, handle_threshold_tokens=1000,
            context_budget_ratio=0.5, context_target_ratio=0.3,
            auto_recall_k=0,
        )
        # est view WOULD trigger (15 fat messages, word count > 100) but the
        # real measurement says we are fine: real beats estimate.
        msgs = [{"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hello"}] + [
            {"role": "user",
             "content": " ".join(f"bigfill{i}w{j}" for j in range(30))}
            for i in range(15)
        ]
        r = rw.rewrite_outgoing(msgs, pressure_tokens=80)
        assert r.windowing_triggered is False
        assert not any(PromptRewriter.is_stub(m.get("content", ""))
                       for m in r.messages)

    def test_learned_high_water_override(self, tmp_path):
        cfg, store, counter, rw = _rewriter(
            tmp_path, n_ctx=200, handle_threshold_tokens=50,
            context_budget_ratio=0.5, context_target_ratio=0.3,
            auto_recall_k=0,
        )
        msgs = _small_wire()
        # Ratio high water would be 100; the learned-ceiling setpoint says 50.
        r = rw.rewrite_outgoing(msgs, pressure_tokens=60, high_water_tokens=50)
        assert r.windowing_triggered is True

    def test_windowed_stubs_stay_sticky_across_pressure_turns(self, tmp_path):
        cfg, store, counter, rw = _rewriter(
            tmp_path, n_ctx=200, handle_threshold_tokens=50,
            context_budget_ratio=0.5, context_target_ratio=0.3,
            auto_recall_k=0,
        )
        msgs = _small_wire()
        r1 = rw.rewrite_outgoing(msgs, pressure_tokens=150)
        assert r1.windowing_triggered is True
        # Next turn: pressure released (the stubs shrank the real wire), the
        # host CLI resends originals + a new tail. Frontier re-applies without
        # re-triggering and the wire byte-extends.
        msgs2 = msgs + [{"role": "user", "content": "tail step g"}]
        r2 = rw.rewrite_outgoing(msgs2, pressure_tokens=90)
        assert r2.windowing_triggered is False
        assert r2.messages[: len(r1.messages)] == r1.messages


# ---------------------------------------------------------------------------
# 14b hardening — futile crossings: no flush, latch, re-arm (2026-07-19 incident)
# ---------------------------------------------------------------------------


class TestFutileCrossing:
    """n_ctx=200, HIGH=100, LOW=60, hysteresis gap=40 (FakeCounter: token = word)."""

    def _setup(self, tmp_path, **over):
        cfg, store, counter, rw = _rewriter(
            tmp_path, n_ctx=200,
            context_budget_ratio=0.5, context_target_ratio=0.3,
            recall_max_stale_tokens=10_000,
            **over,
        )
        h = _seed(store, "old-1",
                  "unicorn banana smoothie recipe with quantum sprinkles")
        t1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "unicorn banana smoothie recipe quantum?"},
        ]
        r1 = rw.rewrite_outgoing(t1)
        assert r1.recalled_handles == [h]
        # Middle messages too SMALL to page: a 2-word message's minimal stub
        # costs more words than it saves, so Pass 3 sheds nothing.
        tiny = [{"role": "user", "content": "tick tack"} for _ in range(4)]
        tail = [{"role": "user", "content": f"tail step {w}"}
                for w in ("one", "two", "three", "four", "five", "six")]
        t2 = t1 + tiny + tail
        return rw, t1, r1, t2

    def test_futile_trigger_neither_flushes_nor_breaks(self, tmp_path):
        """The 2026-07-19 self-own: a high-water crossing that sheds NOTHING
        broke nothing — flushing the frozen recall block then CREATES the
        prefix break it exists to ride."""
        rw, t1, r1, t2 = self._setup(tmp_path)
        r2 = rw.rewrite_outgoing(t2, pressure_tokens=150)
        assert r2.windowing_triggered is True       # the crossing is real...
        assert r2.recalled_handles == []            # ...but the block was reused
        assert r2.messages[: len(r1.messages)] == r1.messages  # byte-stable prefix

    def test_shedding_trigger_still_flushes(self, tmp_path):
        """The flush keeps riding REAL breaks: a trigger that actually pages
        messages (shed > 0) refreshes the block, as the legacy path pins in
        test_recall.test_windowing_trigger_refreshes."""
        rw, t1, r1, t2 = self._setup(tmp_path, handle_threshold_tokens=50)
        fillers = [
            {"role": "user",
             "content": " ".join(f"fillstuff{i}word{j}" for j in range(15))}
            for i in range(8)
        ]
        tail = [{"role": "user", "content": f"unicorn smoothie quantum step {w}"}
                for w in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")]
        r2 = rw.rewrite_outgoing(t1 + fillers + tail, pressure_tokens=150)
        assert r2.windowing_triggered is True
        assert r2.recalled_handles == ["old-1"]    # shed > 0 -> flush rode the break

    def test_missed_low_water_latches_until_gap_growth(self, tmp_path):
        """Unsheddable mass above the high water: after one honest attempt the
        latch holds the wire byte-stable instead of re-breaking every turn,
        re-arming only when pressure grows by the hysteresis gap (40)."""
        rw, t1, r1, t2 = self._setup(tmp_path)
        r2 = rw.rewrite_outgoing(t2, pressure_tokens=150)   # sheds 0 -> latched @150
        assert r2.windowing_triggered is True
        t3 = t2 + [{"role": "assistant", "content": "ack"},
                   {"role": "user", "content": "tail step seven"}]
        r3 = rw.rewrite_outgoing(t3, pressure_tokens=170)   # < 150 + 40: suppressed
        assert r3.windowing_triggered is False
        assert r3.recalled_handles == []
        assert r3.messages[: len(r2.messages)] == r2.messages
        t4 = t3 + [{"role": "assistant", "content": "ack"},
                   {"role": "user", "content": "tail step eight"}]
        r4 = rw.rewrite_outgoing(t4, pressure_tokens=195)   # >= 150 + 40: re-armed
        assert r4.windowing_triggered is True
        assert r4.recalled_handles == []                    # still futile -> no flush
        assert r4.messages[: len(r3.messages)] == r3.messages

    def test_latch_clears_when_pressure_falls_below_high_water(self, tmp_path):
        rw, t1, r1, t2 = self._setup(tmp_path)
        r2 = rw.rewrite_outgoing(t2, pressure_tokens=150)   # latched @150
        assert r2.windowing_triggered is True
        t3 = t2 + [{"role": "assistant", "content": "ack"},
                   {"role": "user", "content": "tail step seven"}]
        r3 = rw.rewrite_outgoing(t3, pressure_tokens=90)    # <= HIGH: latch cleared
        assert r3.windowing_triggered is False
        t4 = t3 + [{"role": "assistant", "content": "ack"},
                   {"role": "user", "content": "tail step eight"}]
        # 170 < 150 + 40 would have been suppressed had the latch survived.
        r4 = rw.rewrite_outgoing(t4, pressure_tokens=170)
        assert r4.windowing_triggered is True


# ---------------------------------------------------------------------------
# Monotonic handle-ization under interleaving (2026-08-02)
# ---------------------------------------------------------------------------


def _wire_for(tag: str):
    """`_small_wire()` with a distinct first user turn -> distinct conv_key."""
    w = [dict(m) for m in _small_wire()]
    w[1] = {"role": "user", "content": f"start the work on {tag}"}
    return w


class TestFrontierSurvivesInterleaving:
    """THE BUG (2026-08-02): ``_windowed`` — the sticky windowing frontier — was
    read with ``.get()`` in Pass 3a and LRU-touched ONLY on a windowing TRIGGER.
    So the healthy steady state (frontier re-applied byte-identically, no
    trigger) never refreshed recency: a conversation aged toward eviction
    precisely because it was behaving, while one-shot side-calls kept inserting
    fresh keys ahead of it.

    Eviction of the frontier is not a cache miss, it is a REGRESSION: every
    previously-windowed message returns to the wire verbatim, breaking the
    prefix and forcing the next turn to re-window the same messages — an
    own-mutation, and a full re-prefill.

    ``_recall_frozen`` was already touched on its reuse path; only ``_windowed``
    had the hole, so the two desynchronized on exactly the conversations that
    mattered most.
    """

    def _rw(self, tmp_path):
        cfg, store, counter, rw = _rewriter(
            tmp_path, n_ctx=200, handle_threshold_tokens=50,
            context_budget_ratio=0.5, context_target_ratio=0.3,
            auto_recall_k=0, max_conversations=2,
        )
        return rw

    def test_reused_frontier_is_not_evicted_by_newer_conversations(self, tmp_path):
        rw = self._rw(tmp_path)
        a = _wire_for("alpha")
        a_stubbed = rw.rewrite_outgoing(a, pressure_tokens=150)
        assert a_stubbed.windowing_triggered is True

        # A keeps being rewritten, below the trigger — the REUSE path. Pre-fix
        # this did not count as use.
        a2 = a + [{"role": "user", "content": "tail step g"}]
        r = rw.rewrite_outgoing(a2, pressure_tokens=90)
        assert r.windowing_triggered is False
        assert r.messages[: len(a_stubbed.messages)] == a_stubbed.messages

        # Two OTHER conversations each trigger and take a frontier slot, with
        # an A reuse turn interleaved — the real shape of a main conversation
        # alternating with title/summary side-calls.
        rw.rewrite_outgoing(_wire_for("bravo"), pressure_tokens=150)
        rw.rewrite_outgoing(a2, pressure_tokens=90)
        rw.rewrite_outgoing(_wire_for("charlie"), pressure_tokens=150)

        # A returns. Its frontier must still be there: the previously-windowed
        # messages stay stubs and the wire still byte-extends the turn that
        # first stubbed them.
        a3 = a2 + [{"role": "user", "content": "tail step h"}]
        final = rw.rewrite_outgoing(a3, pressure_tokens=90)
        assert final.windowing_triggered is False, (
            "frontier was lost: pressure had to re-trigger windowing"
        )
        assert final.messages[: len(a_stubbed.messages)] == a_stubbed.messages, (
            "previously handle-ized messages returned verbatim — "
            "handle-ization regressed, breaking the prefix"
        )

    def test_genuinely_idle_conversation_still_evicts(self, tmp_path):
        """The cap must still bound memory: touching on READ protects ACTIVE
        conversations, not abandoned ones."""
        rw = self._rw(tmp_path)
        a = _wire_for("alpha")
        rw.rewrite_outgoing(a, pressure_tokens=150)
        rw.rewrite_outgoing(_wire_for("bravo"), pressure_tokens=150)
        rw.rewrite_outgoing(_wire_for("charlie"), pressure_tokens=150)
        assert len(rw._windowed) == 2
        assert conversation_key(a) not in rw._windowed
