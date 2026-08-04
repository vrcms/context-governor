"""The tool_call ARGUMENT threshold: its own setpoint, anchored and floored.

Measured 2026-08-03 across three wire captures. tool_call arguments are 43% of
opencode's peak prompt and are invisible to Pass 3 windowing, so they accumulate
for the life of a conversation and are never shed. The per-message threshold
(2% of n_ctx) reached almost none of that mass — it is mid-tail, not a few
giants: p50 94 chars, p90 1,630, and 0 of 333 verbatim values were at or above
the message threshold on the sent wire.

Two bounds, both derived rather than tuned:

  ratio * n_ctx        self-sizes to the server's real window, like every other
                       setpoint in the file
  min_shrink * stub    break-even. A stub costs ~125 tokens to render at the
                       default preview, so stubbing anything smaller makes the
                       wire BIGGER. Without the floor, 0.004 * 8192 would set
                       the threshold to 32 tokens and every fire would lose.

Both directions are asserted here, because a threshold that is only correct at
one context size is exactly the machine-specific tuning this project keeps
finding and removing.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakeCounter
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.durable import DurableStore

N_CTX = 65536
MSG_THRESHOLD = 1310          # 0.02 * 65536, the live value on the measured runs


def _cfg(tmp_path, **over):
    kw = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        listen_host="127.0.0.1",
        listen_port=8900,
        handle_threshold_tokens=MSG_THRESHOLD,
        stub_preview_chars=200,        # production default; the floor tracks it
        rehydrate_budget_tokens=4000,
        request_timeout=30.0,
        auto_recall_k=0,               # keep Pass 4 out of these assertions
        handleize_toolcall_args=True,  # OFF in production; see TestDefaultIsOff
    )
    kw.update(over)
    return ProxyConfig(**kw)


@pytest.fixture
def rw(tmp_path):
    store = DurableStore(str(tmp_path / "store2"))
    r = PromptRewriter(_cfg(tmp_path), FakeCounter(), store)
    r.update_context_size(N_CTX, MSG_THRESHOLD)
    return r


def _words(n):
    """n whitespace-separated words == n tokens under FakeCounter."""
    return " ".join(f"word{i:05d}" for i in range(n))


def _msg(arguments: dict, call_id="call_1", name="bash"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}},
    ]}


def _args_of(out_msg, i=0):
    return json.loads(out_msg["tool_calls"][i]["function"]["arguments"])


class TestReachesTheMidTail:
    def test_argument_below_the_message_threshold_is_stubbed(self, rw):
        """The whole point. 400 tokens is far below the 1310 message threshold,
        so before this change nothing here was ever stubbed."""
        assert 400 < MSG_THRESHOLD
        out = rw.rewrite_outgoing([_msg({"command": _words(400)})]).messages
        assert "[[cm:stored" in _args_of(out[0])["command"]

    def test_protocol_critical_fields_survive(self, rw):
        """`id` and `function.name` are how the model and the server pair a call
        with its result. Only `arguments` may ever be rewritten."""
        out = rw.rewrite_outgoing([_msg({"command": _words(400)})]).messages
        call = out[0]["tool_calls"][0]
        assert call["id"] == "call_1"
        assert call["function"]["name"] == "bash"
        assert call["type"] == "function"

    def test_other_argument_keys_are_untouched(self, rw):
        out = rw.rewrite_outgoing(
            [_msg({"command": _words(400), "cwd": "/srv/app", "timeout": 30})]
        ).messages
        args = _args_of(out[0])
        assert args["cwd"] == "/srv/app"
        assert args["timeout"] == 30


class TestTheFloorHolds:
    def test_value_smaller_than_its_own_stub_is_left_verbatim(self, rw):
        """Break-even is the wrong bar (bug #8). A 150-token value replaced by a
        ~125-token stub nets 25 tokens and costs a hidden payload; below the
        floor it would be an outright loss."""
        payload = _words(150)
        out = rw.rewrite_outgoing([_msg({"command": payload})]).messages
        assert _args_of(out[0])["command"] == payload

    def test_floor_dominates_on_a_small_context(self, tmp_path):
        """0.004 * 8192 == 32 tokens, well under the ~125-token stub. The floor
        must win, or every fire on a small server bloats the wire."""
        r = PromptRewriter(_cfg(tmp_path), FakeCounter(),
                           DurableStore(str(tmp_path / "s")))
        r.update_context_size(8192, 164)
        stub_tokens = r.stub_tokens_estimate(200)
        assert r._toolcall_threshold() == int(2.0 * stub_tokens)
        assert r._toolcall_threshold() > int(0.004 * 8192)

    def test_ratio_dominates_on_a_large_context(self, tmp_path):
        """The mirror case: on a 200K window the floor is irrelevant and the
        threshold must scale up, or the governor over-compresses a server with
        room to spare."""
        r = PromptRewriter(_cfg(tmp_path), FakeCounter(),
                           DurableStore(str(tmp_path / "s")))
        r.update_context_size(200000, 4000)
        assert r._toolcall_threshold() == int(0.004 * 200000)
        assert r._toolcall_threshold() > int(2.0 * r.stub_tokens_estimate(200))

    def test_unknown_context_falls_back_to_the_floor(self, tmp_path):
        """n_ctx unknown (server down at startup) must not disable the pass —
        the floor is still a valid, safe setpoint on its own."""
        r = PromptRewriter(_cfg(tmp_path), FakeCounter(),
                           DurableStore(str(tmp_path / "s")))
        assert r._toolcall_threshold() == int(2.0 * r.stub_tokens_estimate(200))


class TestPinnedPerConversation:
    def test_a_rising_n_ctx_does_not_un_stub_a_live_conversation(self, rw):
        """The regression this codebase spent 2026-08-02 eliminating. A larger
        window raises the threshold; applying that to a conversation already in
        flight would send arguments back VERBATIM that have already gone out as
        stubs — an own-mutation on every affected message at once."""
        msgs = [_msg({"command": _words(400)})]
        first = rw.rewrite_outgoing(msgs).messages
        assert "[[cm:stored" in _args_of(first[0])["command"]

        # Server restarts with a much larger -c: the new setpoint (800) is above
        # this argument's 400 tokens.
        rw.update_context_size(200000, 4000)
        assert rw._toolcall_threshold() == 800

        second = rw.rewrite_outgoing(msgs).messages
        assert "[[cm:stored" in _args_of(second[0])["command"], (
            "pinning failed: the argument came back verbatim"
        )

    def test_a_new_conversation_gets_the_new_setpoint(self, rw):
        """Pinning must not freeze the setpoint globally — a conversation that
        STARTS after the change is the one moment adopting it is free."""
        rw.rewrite_outgoing([_msg({"command": _words(400)})])
        rw.update_context_size(200000, 4000)
        out = rw.rewrite_outgoing(
            [{"role": "user", "content": "a different conversation"},
             _msg({"command": _words(400)}, call_id="call_2")]
        ).messages
        assert "[[cm:stored" not in _args_of(out[1])["command"]


class TestDefaultIsOff:
    """The 2026-08-03 finding, pinned as a test.

    A stub in `content` is context the model reads. A stub in
    `tool_calls[].function.arguments` sits in the model's OWN PRIOR OUTPUT — the
    slot it pattern-matches when producing the next tool call. On a live opencode
    run it copied the markers verbatim, CM's real handles included, into new bash
    commands, and the shell received them as literal input:

        /bin/bash: line 1: [[cm:stored: command not found

    Every token counter scored that as a win while the task produced wrong
    output. Evidence: `_runs/wire-ABORTED-toolcall-imitation-111732`.
    """

    def test_the_shipped_default_is_off(self, tmp_path):
        cfg = _cfg(tmp_path)
        object.__setattr__(cfg, "handleize_toolcall_args",
                           ProxyConfig.handleize_toolcall_args)
        assert cfg.handleize_toolcall_args is False

    def test_arguments_are_untouched_at_the_default(self, tmp_path):
        """Not merely 'not stubbed' — the tool_calls object must come through
        byte-identical, since re-serializing an unchanged dict can reorder keys
        and break the prefix for no benefit."""
        cfg = _cfg(tmp_path, handleize_toolcall_args=False)
        r = PromptRewriter(cfg, FakeCounter(), DurableStore(str(tmp_path / "s")))
        r.update_context_size(N_CTX, MSG_THRESHOLD)
        msg = _msg({"command": _words(400)})
        original = json.loads(json.dumps(msg["tool_calls"]))
        out = r.rewrite_outgoing([msg]).messages
        assert out[0]["tool_calls"] == original
        assert "[[cm:stored" not in _args_of(out[0])["command"]

    def test_nothing_is_paged_out_at_the_default(self, tmp_path):
        """The store must not fill with argument notes either — a handle that
        exists is a handle recall can surface."""
        cfg = _cfg(tmp_path, handleize_toolcall_args=False)
        store = DurableStore(str(tmp_path / "s"))
        r = PromptRewriter(cfg, FakeCounter(), store)
        r.update_context_size(N_CTX, MSG_THRESHOLD)
        before = store.corpus_size()
        r.rewrite_outgoing([_msg({"command": _words(400)})])
        assert store.corpus_size() == before


class TestValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_ratio_out_of_range_is_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError, match="toolcall_threshold_ratio"):
            _cfg(tmp_path, toolcall_threshold_ratio=bad)

    def test_negative_shrink_ratio_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="toolcall_min_shrink_ratio"):
            _cfg(tmp_path, toolcall_min_shrink_ratio=-1.0)

    def test_both_disabled_still_yields_a_usable_threshold(self, tmp_path):
        """Belt and braces: zeroing both knobs must not produce a threshold of
        0, which would stub every argument down to a stub twice its size."""
        r = PromptRewriter(
            _cfg(tmp_path, toolcall_threshold_ratio=0.0,
                 toolcall_min_shrink_ratio=0.0),
            FakeCounter(), DurableStore(str(tmp_path / "s")),
        )
        r.update_context_size(N_CTX, MSG_THRESHOLD)
        assert r._toolcall_threshold() >= 1


class TestIdempotence:
    def test_rewriting_our_own_output_is_a_no_op(self, rw):
        """rewrite(rewrite(x)) == rewrite(x). A stub is short enough to fall
        back under the threshold, so the second pass must not re-stub it."""
        once = rw.rewrite_outgoing([_msg({"command": _words(400)})]).messages
        twice = rw.rewrite_outgoing(once).messages
        assert _args_of(twice[0]) == _args_of(once[0])
