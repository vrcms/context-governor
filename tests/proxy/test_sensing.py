"""Phase 14a/14c sensing tests — the governor's afferent path.

Unit tests for the pure helpers (canonical serialization, boundary hashes,
conversation identity, the request-diff classifier, the SSE response tee) and
for ``GovernorController`` (ledger LRU, break attribution, reuse ratios,
native-compaction ceiling learning + persistence, real-usage pressure).

No network, no FastAPI — everything here is deterministic and in-memory except
the StateStore persistence round-trip (tmp_path).
"""

from __future__ import annotations

from contextmanager.proxy.sensing import (
    CAUSE_HARNESS,
    CAUSE_MULTIMODAL,
    CAUSE_NEW,
    CAUSE_OWN,
    CAUSE_UNKNOWN,
    KIND_APPEND,
    KIND_HEAD_REWRITE,
    KIND_MID_EDIT,
    KIND_NEW,
    KIND_TAIL_EDIT,
    GovernorController,
    StreamTee,
    canonical_content,
    classify_wire_diff,
    conversation_key,
    extract_final_sse_json,
    extract_usage_timings,
    harness_fingerprint,
    message_hash,
    wire_signature,
)
from contextmanager.state_store import StateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conv(*contents: str, system: str = "You are helpful") -> list:
    """[system] + alternating user/assistant messages with the given contents."""
    out = [{"role": "system", "content": system}]
    for i, c in enumerate(contents):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": c})
    return out


# ---------------------------------------------------------------------------
# Canonical serialization + hashes
# ---------------------------------------------------------------------------


class TestCanonicalContent:
    def test_str_passthrough(self):
        assert canonical_content("hello world") == "hello world"

    def test_none_is_empty(self):
        assert canonical_content(None) == ""

    def test_structured_deterministic_sorted_keys(self):
        a = [{"type": "text", "text": "x"}]
        b = [{"text": "x", "type": "text"}]  # same object, different key order
        assert canonical_content(a) == canonical_content(b)
        assert canonical_content(a) == '[{"text":"x","type":"text"}]'


class TestMessageHash:
    def test_role_and_content_matter(self):
        base = {"role": "user", "content": "abc"}
        assert message_hash(base) != message_hash({"role": "assistant", "content": "abc"})
        assert message_hash(base) != message_hash({"role": "user", "content": "abd"})
        assert message_hash(base) == message_hash(dict(base))

    def test_tool_calls_participate(self):
        # Assistant tool-call messages often carry content: null — without the
        # payload in the hash every such message would collide.
        a = {"role": "assistant", "content": None}
        b = {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "function": {"name": "read"}}]}
        c = {"role": "assistant", "content": None,
             "tool_calls": [{"id": "2", "function": {"name": "read"}}]}
        assert message_hash(a) != message_hash(b)
        assert message_hash(b) != message_hash(c)

    def test_str_vs_parts_differ(self):
        assert message_hash({"role": "user", "content": "x"}) != message_hash(
            {"role": "user", "content": [{"type": "text", "text": "x"}]}
        )


class TestConversationIdentity:
    def test_stable_under_append(self):
        t1 = _conv("first question")
        t2 = t1 + [{"role": "assistant", "content": "answer"},
                   {"role": "user", "content": "follow-up"}]
        assert conversation_key(t1) == conversation_key(t2)

    def test_same_system_different_first_user_separates(self):
        # Interleaved conversations sharing one harness system prompt (main
        # chat vs. title/summary side-calls) must not collide.
        a = _conv("work on the parser")
        b = _conv("generate a short title")
        assert conversation_key(a) != conversation_key(b)

    def test_empty(self):
        assert conversation_key([]) == "conv-empty"

    def test_harness_fingerprint_tracks_system_head_only(self):
        a = _conv("alpha", system="SYS PROMPT")
        b = _conv("totally different conversation", system="SYS PROMPT")
        c = _conv("alpha", system="OTHER HARNESS")
        assert harness_fingerprint(a) == harness_fingerprint(b)
        assert harness_fingerprint(a) != harness_fingerprint(c)


# ---------------------------------------------------------------------------
# Request-diff classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_no_previous_is_new(self):
        assert classify_wire_diff(None, ["a"]) == (KIND_NEW, 0)
        assert classify_wire_diff([], ["a"]) == (KIND_NEW, 0)

    def test_pure_append_and_equal(self):
        prev = wire_signature(_conv("q1", "a1"))
        new = wire_signature(_conv("q1", "a1", "q2"))
        assert classify_wire_diff(prev, new) == (KIND_APPEND, len(prev))
        assert classify_wire_diff(prev, prev) == (KIND_APPEND, len(prev))

    def test_tail_edit(self):
        prev = wire_signature(_conv("q1", "a1", "q2", "a2", "q3"))
        edited = _conv("q1", "a1", "q2", "a2", "q3-regenerated")
        kind, pos = classify_wire_diff(prev, wire_signature(edited))
        assert kind == KIND_TAIL_EDIT
        assert pos == len(prev) - 1

    def test_mid_wire_edit(self):
        contents = [f"m{i}" for i in range(9)]  # 10 messages with the system head
        prev = wire_signature(_conv(*contents))
        contents[4] = "m4-edited"  # index 5 on the wire: 4*5 >= 10, 5 < 8
        kind, pos = classify_wire_diff(prev, wire_signature(_conv(*contents)))
        assert kind == KIND_MID_EDIT
        assert pos == 5

    def test_head_rewrite_task_7220_signature(self):
        # The native-compaction flood: the system head survives, everything
        # above an early message is rewritten (transcript rewritten above token
        # 1956 -> full 21K re-prefill).
        prev = wire_signature(_conv(*[f"m{i}" for i in range(20)]))  # 21 msgs
        compacted = _conv("compacted summary of prior work", "recent tail")
        kind, pos = classify_wire_diff(prev, wire_signature(compacted))
        assert kind == KIND_HEAD_REWRITE
        assert pos == 1

    def test_truncation_with_matching_prefix_is_tail_edit(self):
        prev = wire_signature(_conv(*[f"m{i}" for i in range(9)]))
        shorter = prev[:-1]
        kind, pos = classify_wire_diff(prev, shorter)
        assert kind == KIND_TAIL_EDIT
        assert pos == len(shorter)


# ---------------------------------------------------------------------------
# Response tee (SSE + non-stream shapes)
# ---------------------------------------------------------------------------


_FINAL_CHUNK = (
    b'data: {"choices":[{"delta":{}}],'
    b'"usage":{"prompt_tokens":1234,"completion_tokens":7},'
    b'"timings":{"prompt_n":34,"prompt_ms":50.0,'
    b'"prompt_per_second":680.0,"predicted_per_second":30.0}}\n\n'
)


class TestResponseTee:
    def test_extract_usage_timings_nonstream(self):
        body = {"id": "x", "usage": {"prompt_tokens": 10}, "timings": {"prompt_n": 3}}
        out = extract_usage_timings(body)
        assert out == {"usage": {"prompt_tokens": 10}, "timings": {"prompt_n": 3}}
        assert extract_usage_timings({"id": "x"}) is None
        assert extract_usage_timings("not a dict") is None

    def test_usage_without_timings(self):
        out = extract_usage_timings({"usage": {"prompt_tokens": 5}})
        assert out == {"usage": {"prompt_tokens": 5}}

    def test_final_sse_chunk_parsed(self):
        chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                  _FINAL_CHUNK, b"data: [DONE]\n\n"]
        tee = StreamTee()
        for c in chunks:
            tee.feed(c)
        out = tee.result()
        assert out is not None
        assert out["usage"]["prompt_tokens"] == 1234
        assert out["timings"]["prompt_n"] == 34

    def test_chunk_boundary_split_mid_json(self):
        # The tail buffer is contiguous, so an event split across feeds parses.
        tee = StreamTee()
        tee.feed(_FINAL_CHUNK[:40])
        tee.feed(_FINAL_CHUNK[40:])
        tee.feed(b"data: [DONE]\n\n")
        out = tee.result()
        assert out is not None and out["usage"]["prompt_tokens"] == 1234

    def test_done_only_or_garbage_is_none(self):
        tee = StreamTee()
        tee.feed(b"data: [DONE]\n\n")
        assert tee.result() is None
        assert extract_final_sse_json(b"data: {broken json}\n\n") is None
        assert extract_final_sse_json(b"") is None

    def test_feed_never_raises(self):
        tee = StreamTee()
        tee.feed(None)          # type: ignore[arg-type]
        tee.feed("not bytes")   # type: ignore[arg-type]
        assert tee.result() is None


# ---------------------------------------------------------------------------
# GovernorController — ledger, attribution, calibration
# ---------------------------------------------------------------------------


class TestLedger:
    def test_lru_eviction(self):
        ctrl = GovernorController(max_conversations=2)
        k1 = ctrl.observe_request(_conv("conversation one")).key
        k2 = ctrl.observe_request(_conv("conversation two")).key
        k3 = ctrl.observe_request(_conv("conversation three")).key
        convs = ctrl.snapshot()["conversations"]
        assert k1 not in convs
        assert k2 in convs and k3 in convs


class TestBreakAttribution:
    def test_new_conversation(self):
        ctrl = GovernorController()
        t1 = _conv("hello parser work")
        obs = ctrl.observe_request(t1)
        assert obs.kind == KIND_NEW and obs.cause == CAUSE_NEW
        assert obs.prefix_broken is True
        ctrl.note_sent(obs.key, t1, observation=obs)
        assert ctrl.snapshot()["breaks_by_cause"] == {CAUSE_NEW: 1}

    def test_pure_append_is_not_a_break(self):
        ctrl = GovernorController()
        t1 = _conv("hello parser work")
        obs1 = ctrl.observe_request(t1)
        ctrl.note_sent(obs1.key, t1, observation=obs1)
        t2 = t1 + [{"role": "assistant", "content": "done"},
                   {"role": "user", "content": "next step"}]
        obs2 = ctrl.observe_request(t2)
        assert obs2.kind == KIND_APPEND and obs2.prefix_broken is False
        ctrl.note_sent(obs2.key, t2, observation=obs2)
        breaks = ctrl.snapshot()["breaks_by_cause"]
        assert breaks == {CAUSE_NEW: 1}  # nothing beyond the initial send

    def test_own_mutation_attributed(self):
        # Incoming wire is a pure append, but the SENT wire diverged from the
        # last sent wire -> the break is OURS (the voluntary kind 14b kills).
        ctrl = GovernorController()
        t1 = _conv("hello parser work")
        obs1 = ctrl.observe_request(t1)
        ctrl.note_sent(obs1.key, t1, observation=obs1)
        t2 = t1 + [{"role": "user", "content": "next"}]
        obs2 = ctrl.observe_request(t2)
        assert obs2.prefix_broken is False
        mutated = [t1[0], {"role": "user", "content": "[[cm:recall]]moved"}] + t2[1:]
        ctrl.note_sent(obs2.key, mutated, observation=obs2)
        assert ctrl.snapshot()["breaks_by_cause"][CAUSE_OWN] == 1

    def test_harness_edit_attributed(self):
        ctrl = GovernorController()
        contents = [f"m{i}" for i in range(9)]
        t1 = _conv(*contents)
        obs1 = ctrl.observe_request(t1)
        ctrl.note_sent(obs1.key, t1, observation=obs1)
        contents[4] = "m4-edited-by-harness"  # mid-wire, first user msg intact
        t2 = _conv(*contents)
        obs2 = ctrl.observe_request(t2)
        assert obs2.kind == KIND_MID_EDIT and obs2.cause == CAUSE_HARNESS
        assert obs2.prefix_broken is True
        ctrl.note_sent(obs2.key, t2, observation=obs2)
        assert ctrl.snapshot()["breaks_by_cause"][CAUSE_HARNESS] == 1

    def test_multimodal_append_is_expected_break(self):
        ctrl = GovernorController()
        t1 = _conv("look at this diagram")
        obs1 = ctrl.observe_request(t1)
        ctrl.note_sent(obs1.key, t1, observation=obs1)
        t2 = t1 + [{"role": "user", "content": [
            {"type": "text", "text": "here"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        ]}]
        obs2 = ctrl.observe_request(t2)
        assert obs2.kind == KIND_APPEND
        assert obs2.cause == CAUSE_MULTIMODAL and obs2.prefix_broken is True


class TestResponseObservation:
    def test_reuse_ratio_and_peaks(self):
        ctrl = GovernorController()
        t1 = _conv("measure this conversation")
        obs = ctrl.observe_request(t1)
        ctrl.note_sent(obs.key, t1, observation=obs)
        ctrl.observe_response(obs.key, {
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
            "timings": {"prompt_n": 100, "prompt_ms": 500.0},
        })
        snap = ctrl.snapshot()
        assert snap["real_prompt_tokens"] == {"last": 1000, "peak": 1000}
        assert snap["real_reuse_ratio"] == 0.9
        assert snap["responses_observed"] == 1
        assert snap["responses_with_timings"] == 1
        conv = snap["conversations"][obs.key]
        assert conv["last_prompt_tokens"] == 1000 and conv["last_prompt_n"] == 100

    def test_ttft_fallback_without_timings(self):
        ctrl = GovernorController()
        obs = ctrl.observe_request(_conv("no timings here"))
        ctrl.observe_response(obs.key, None, ttft_ms=123.4)
        conv = ctrl.snapshot()["conversations"][obs.key]
        assert conv["last_ttft_ms"] == 123.4

    def test_growth_history(self):
        ctrl = GovernorController()
        t1 = _conv("grow me")
        obs = ctrl.observe_request(t1)
        ctrl.observe_response(obs.key, {"usage": {"prompt_tokens": 1000}})
        t2 = t1 + [{"role": "user", "content": "more"}]
        obs2 = ctrl.observe_request(t2)
        ctrl.observe_response(obs2.key, {"usage": {"prompt_tokens": 1400}})
        conv = ctrl.snapshot()["conversations"][obs.key]
        assert conv["last_growth_tokens"] == 400


class TestPressure:
    def test_no_reals_no_pressure(self):
        ctrl = GovernorController()
        assert ctrl.observe_request(_conv("first ever")).pressure_tokens is None

    def test_pressure_from_reals_plus_growth(self):
        ctrl = GovernorController()
        t1 = _conv("pressure test conversation")
        obs1 = ctrl.observe_request(t1)
        ctrl.observe_response(obs1.key, {
            "usage": {"prompt_tokens": 1000, "completion_tokens": 52},
        })
        grown = "x" * 400  # 100 estimated tokens of growth
        t2 = t1 + [{"role": "assistant", "content": grown}]
        obs2 = ctrl.observe_request(t2)
        assert obs2.pressure_tokens == 1000 + 52 + 100


class TestCeilingLearning:
    def _grow_and_observe(self, ctrl, n_msgs=11, prompt_tokens=20000,
                          system="SYS PROMPT"):
        t1 = _conv(*[f"long turn {i}" for i in range(n_msgs)], system=system)
        obs = ctrl.observe_request(t1)
        ctrl.note_sent(obs.key, t1, observation=obs)
        ctrl.observe_response(obs.key, {"usage": {"prompt_tokens": prompt_tokens}})
        return t1, obs

    def test_same_key_head_rewrite_learns_ceiling(self):
        ctrl = GovernorController()
        t1, obs1 = self._grow_and_observe(ctrl)
        # Compaction that KEEPS the first user message: same conversation key,
        # divergence at index 2 (well inside the first quarter of 12 messages).
        compacted = t1[:2] + [
            {"role": "user", "content": "compacted summary"},
            {"role": "user", "content": "recent tail"},
        ]
        obs2 = ctrl.observe_request(compacted)
        assert obs2.kind == KIND_HEAD_REWRITE and obs2.cause == CAUSE_HARNESS
        snap = ctrl.snapshot()
        assert snap["native_compaction_observed"] == 1
        assert snap["learned_ceilings"][obs1.harness_fp]["ceiling"] == 20000

    def test_cross_key_compaction_detected(self):
        # Compaction that REPLACES the first user message changes the
        # conversation key — the shared system head must still be matched.
        ctrl = GovernorController()
        t1, obs1 = self._grow_and_observe(ctrl)
        compacted = [t1[0],
                     {"role": "user", "content": "summary of everything"},
                     {"role": "user", "content": "recent tail"},
                     {"role": "assistant", "content": "ok"}]
        obs2 = ctrl.observe_request(compacted)
        assert obs2.key != obs1.key
        assert obs2.kind == KIND_HEAD_REWRITE and obs2.cause == CAUSE_HARNESS
        assert ctrl.snapshot()["native_compaction_observed"] == 1
        assert ctrl.snapshot()["learned_ceilings"][obs1.harness_fp]["ceiling"] == 20000

    def test_fresh_two_message_chat_is_not_compaction(self):
        ctrl = GovernorController()
        self._grow_and_observe(ctrl)
        obs = ctrl.observe_request(_conv("brand new question",
                                         system="SYS PROMPT"))
        assert obs.kind == KIND_NEW
        assert ctrl.snapshot()["native_compaction_observed"] == 0

    def test_minimum_sample_wins(self):
        ctrl = GovernorController()
        t1, obs1 = self._grow_and_observe(ctrl, prompt_tokens=30000)
        compacted = t1[:2] + [{"role": "user", "content": "summary one"},
                              {"role": "user", "content": "tail"}]
        ctrl.observe_request(compacted)
        # Second cycle in a fresh conversation, same harness, smaller ceiling.
        t2, _ = self._grow_and_observe(ctrl, prompt_tokens=18000)
        compacted2 = t2[:2] + [{"role": "user", "content": "summary two"},
                               {"role": "user", "content": "tail"}]
        ctrl.observe_request(compacted2)
        assert ctrl.snapshot()["learned_ceilings"][obs1.harness_fp]["ceiling"] == 18000

    def test_effective_high_water(self):
        ctrl = GovernorController()
        t1, obs1 = self._grow_and_observe(ctrl)  # ceiling sample 20000
        compacted = t1[:2] + [{"role": "user", "content": "summary"},
                              {"role": "user", "content": "tail"}]
        ctrl.observe_request(compacted)
        fp = obs1.harness_fp
        # min(ratio*n_ctx, safety*ceiling)
        assert ctrl.effective_high_water(100_000, 0.5, 0.8, fp) == 16000
        # ratio bound wins when tighter
        assert ctrl.effective_high_water(20_000, 0.5, 0.8, fp) == 10000
        # no n_ctx -> learned alone; safety 0 disables the learned ceiling
        assert ctrl.effective_high_water(None, 0.5, 0.8, fp) == 16000
        assert ctrl.effective_high_water(100_000, 0.5, 0.0, fp) == 50000
        # unknown harness -> ratio only
        assert ctrl.effective_high_water(100_000, 0.5, 0.8, "hp-other") == 50000

    def test_persistence_round_trip(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        ctrl = GovernorController()
        t1, obs1 = self._grow_and_observe(ctrl)
        compacted = t1[:2] + [{"role": "user", "content": "summary"},
                              {"role": "user", "content": "tail"}]
        ctrl.observe_request(compacted)
        ctrl.maybe_persist(store)
        # A restarted proxy keeps its calibration.
        reborn = GovernorController()
        reborn.load_profiles(store)
        assert reborn.effective_high_water(100_000, 0.5, 0.8, obs1.harness_fp) == 16000

    def test_maybe_persist_only_when_dirty(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        ctrl = GovernorController()
        ctrl.maybe_persist(store)   # nothing learned -> no write
        assert not (tmp_path / "state.json").exists()
