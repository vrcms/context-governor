"""ContextWindowResolver — the number every other threshold scales from.

The OpenAI standard does not carry a context window: measured 2026-08-02, a chat
completion response has usage and timings and nothing else, and the spec has no
such field in /v1/models, the envelope, or headers. So the window cannot be read
universally — it has to be BRACKETED by evidence that is available everywhere:
the largest prompt that succeeded (floor) and a length that was rejected
(ceiling).
"""

from __future__ import annotations

from contextmanager.proxy.window import (
    ContextWindowResolver,
    looks_like_overflow,
    parse_overflow_limit,
    scan_for_window,
)


class TestModelsScan:
    """No single standard key exists; these are the shapes actually in the wild."""

    def test_llama_cpp_shape(self):
        body = {"data": [{"id": "m", "meta": {"n_ctx": 65536, "n_ctx_train": 262144}}]}
        assert scan_for_window(body) == 65536, "must prefer the LOADED window"

    def test_vllm_shape(self):
        assert scan_for_window({"data": [{"id": "m", "max_model_len": 32768}]}) == 32768

    def test_lmstudio_shape(self):
        assert scan_for_window({"data": [{"loaded_context_length": 8192}]}) == 8192

    def test_plain_context_length(self):
        assert scan_for_window({"data": [{"context_length": 16384}]}) == 16384

    def test_openai_shape_carries_nothing(self):
        """The official spec has no context field — this must not invent one."""
        body = {"object": "list", "data": [
            {"id": "gpt-x", "object": "model", "created": 1, "owned_by": "openai"}]}
        assert scan_for_window(body) is None

    def test_implausible_values_are_ignored(self):
        assert scan_for_window({"data": [{"n_ctx": 8}]}) is None

    def test_never_raises_on_junk(self):
        for junk in (None, 3, "x", [], {}, {"data": None}, {"a": {"b": [{"c": 1}]}}):
            assert scan_for_window(junk) is None or isinstance(scan_for_window(junk), int)


class TestOverflowClassification:
    def test_recognises_vendor_phrasings(self):
        assert looks_like_overflow("This model's maximum context length is 8192 tokens")
        assert looks_like_overflow("the request exceeds the available context size")

    def test_unrelated_failures_are_not_overflow(self):
        """An unclassifiable failure must never move the belief — otherwise a
        network blip shrinks every threshold in the governor."""
        for text in (None, "", "connection reset by peer",
                     "500 Internal Server Error", "model not found"):
            assert looks_like_overflow(text) is False

    def test_extracts_the_declared_limit(self):
        assert parse_overflow_limit(
            "This model's maximum context length is 8192 tokens, however you "
            "requested 9000") == 8192

    def test_no_limit_when_none_stated(self):
        assert parse_overflow_limit("the request exceeds the available context size") is None


class TestPrecedence:
    def test_unresolved_when_nothing_is_known(self):
        r = ContextWindowResolver()
        assert r.window is None and r.source == "unresolved"

    def test_config_outranks_discovery(self):
        r = ContextWindowResolver(configured=4096)
        r.offer(65536, "v1/models")
        assert r.window == 4096 and r.source == "config"

    def test_discovery_used_when_no_config(self):
        r = ContextWindowResolver()
        r.offer(65536, "v1/models")
        assert r.window == 65536 and r.source == "v1/models"

    def test_floor_only_as_last_resort(self):
        """A proven-to-fit prompt is a safe fallback, but it is NOT discovery —
        it cannot grow past what the governor itself lets through."""
        r = ContextWindowResolver()
        r.observe_success(20000)
        assert r.window == 20000 and r.source == "observed-floor"


class TestEvidenceOverridesClaims:
    def test_a_success_larger_than_the_belief_raises_it(self):
        r = ContextWindowResolver()
        r.offer(8192, "v1/models")
        assert r.observe_success(20000) is True, "should report the contradiction"
        assert r.window == 20000
        assert "floor-override" in r.source
        assert r.snapshot()["contradictions"] == 1

    def test_an_overflow_clamps_a_too_large_belief(self):
        r = ContextWindowResolver()
        r.offer(200000, "v1/models")
        r.observe_overflow("maximum context length is 65536 tokens")
        assert r.window == 65536
        assert "ceiling-clamp" in r.source

    def test_unclassifiable_failure_leaves_the_belief_alone(self):
        r = ContextWindowResolver()
        r.offer(65536, "v1/models")
        assert r.observe_overflow("connection reset by peer") is False
        assert r.window == 65536
        assert r.snapshot()["observed_ceiling"] is None


class TestStaleEvidence:
    """The upstream can change under us — a restart with a different -c. Then
    old evidence describes a server that no longer exists, and a bracket built
    from both is inconsistent (floor above ceiling), which silently let the
    floor-override win and sized the governor above a limit it had been told."""

    def test_a_success_above_the_ceiling_retires_the_ceiling(self):
        r = ContextWindowResolver()
        r.observe_overflow("maximum context length is 8192 tokens")
        assert r.snapshot()["observed_ceiling"] == 8192
        r.observe_success(40000)                       # server grew
        assert r.snapshot()["observed_ceiling"] is None
        assert r.window == 40000

    def test_an_overflow_below_the_floor_retires_the_floor(self):
        r = ContextWindowResolver()
        r.observe_success(40000)
        r.observe_overflow("maximum context length is 8192 tokens")   # server shrank
        snap = r.snapshot()
        assert snap["observed_floor"] is None
        assert snap["observed_ceiling"] == 8192
        assert r.window == 8192, "must not stay sized above a limit just reported"

    def test_bracket_is_never_inconsistent(self):
        r = ContextWindowResolver()
        for step in range(6):
            r.observe_success(10000 * (step + 1))
            r.observe_overflow("maximum context length is %d tokens" % (5000 * (step + 1)))
            snap = r.snapshot()
            lo, hi = snap["observed_floor"], snap["observed_ceiling"]
            if lo and hi:
                assert lo <= hi, f"floor {lo} above ceiling {hi}"


class TestSafety:
    def test_never_raises_on_garbage_input(self):
        r = ContextWindowResolver()
        for bad in (None, "", "abc", -1, 0, 3.7, [], {}):
            r.observe_success(bad)
            r.offer(bad, "junk")
            r.observe_overflow(bad if isinstance(bad, str) else None)
        assert isinstance(r.snapshot(), dict)

    def test_snapshot_exposes_the_whole_bracket(self):
        r = ContextWindowResolver()
        r.offer(65536, "v1/models")
        r.observe_success(30000)
        snap = r.snapshot()
        assert set(snap) >= {"window", "source", "resolved", "configured", "declared",
                             "declared_source", "observed_floor", "observed_ceiling",
                             "contradictions", "overflows"}
