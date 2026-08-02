"""Pass 1 extension: handle-ization of large tool_calls arguments.

Guards the 2026-07-28 fix for the second half of the "own-mutation windowing"
incident (see problem-2.md). A live wire capture showed `tool_calls`
accounting for 116,736 of 222,177 total wire chars (53%) on one request — an
agentic turn dominated by a large write_file/edit_file/shell argument — while
Pass 1 handle-ization and Pass 3 windowing both only ever looked at `content`.
That mass was structurally unsheddable: windowing fired 4 times on a live
session and pressure still climbed to 96% of n_ctx, because there was nothing
left it was allowed to touch.

The fix edits STRING VALUES inside the already-JSON-decoded `arguments`
object and re-encodes the same shape back to a JSON string — never replaces
`arguments` with a non-JSON stub, because llama-server's chat template parses
`arguments` into a mapping before rendering (`tool_call.arguments|items`
requires a dict; verified against the live template file).
"""

from __future__ import annotations

import json

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
        stub_preview_chars=20,
        rehydrate_budget_tokens=4000,
        request_timeout=30.0,
    )
    kw.update(over)
    return ProxyConfig(**kw)


@pytest.fixture
def rw(tmp_path):
    store = DurableStore(str(tmp_path / "store2"))
    return PromptRewriter(_cfg(tmp_path), FakeCounter(), store)


def _big(n=500):
    return " ".join(f"word{i}" for i in range(n))


def _tool_call_msg(arguments: dict, call_id="call_1", name="write_file"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}},
    ]}


class TestStructuralSafety:
    def test_arguments_stays_valid_json_object(self, rw):
        msg = _tool_call_msg({"path": "foo.py", "content": _big()})
        out = rw.rewrite_outgoing([msg]).messages
        args = out[0]["tool_calls"][0]["function"]["arguments"]
        parsed = json.loads(args)  # must not raise
        assert isinstance(parsed, dict)

    def test_small_keys_survive_untouched(self, rw):
        msg = _tool_call_msg({"path": "foo.py", "content": _big(), "mode": "w"})
        out = rw.rewrite_outgoing([msg]).messages
        parsed = json.loads(out[0]["tool_calls"][0]["function"]["arguments"])
        assert parsed["path"] == "foo.py"
        assert parsed["mode"] == "w"

    def test_large_value_gets_stubbed(self, rw):
        big = _big()
        msg = _tool_call_msg({"path": "foo.py", "content": big})
        out = rw.rewrite_outgoing([msg]).messages
        parsed = json.loads(out[0]["tool_calls"][0]["function"]["arguments"])
        assert "[[cm:stored handle=" in parsed["content"]
        assert len(parsed["content"]) < len(big)

    def test_full_content_retrievable_from_store(self, rw):
        big = _big()
        msg = _tool_call_msg({"content": big})
        rw.rewrite_outgoing([msg])
        stored = rw.store.get(rw.stable_id("toolcall-arg:content", big))
        assert stored == big


class TestFailOpen:
    """Any shape mismatch must leave tool_calls completely untouched, never raise."""

    def test_tool_calls_not_a_list(self, rw):
        msg = {"role": "assistant", "content": None, "tool_calls": "not-a-list"}
        out = rw.rewrite_outgoing([msg]).messages
        assert out[0]["tool_calls"] == "not-a-list"

    def test_arguments_not_json(self, rw):
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "not json {{{"}},
        ]}
        out = rw.rewrite_outgoing([msg]).messages
        assert out[0]["tool_calls"][0]["function"]["arguments"] == "not json {{{"

    def test_arguments_decodes_to_non_dict(self, rw):
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": json.dumps([1, 2, 3])}},
        ]}
        out = rw.rewrite_outgoing([msg]).messages
        assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == [1, 2, 3]

    def test_missing_function_key(self, rw):
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function"},
        ]}
        out = rw.rewrite_outgoing([msg]).messages
        assert out[0]["tool_calls"] == [{"id": "c1", "type": "function"}]

    def test_never_raises_on_junk_entries(self, rw):
        msg = {"role": "assistant", "content": None,
               "tool_calls": [None, 42, "x", {"function": "not-a-dict"}]}
        rw.rewrite_outgoing([msg])  # must not raise


class TestNoUnnecessaryRewrite:
    """Untouched tool_calls must forward byte-identically, not re-serialized."""

    def test_small_arguments_are_the_same_object(self, rw):
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": json.dumps({"path": "x"})}},
        ]}
        out = rw.rewrite_outgoing([msg]).messages
        assert out[0]["tool_calls"] is msg["tool_calls"]

    def test_input_message_never_mutated(self, rw):
        big = _big()
        original_args = json.dumps({"content": big})
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "write_file", "arguments": original_args}},
        ]}
        rw.rewrite_outgoing([msg])
        assert msg["tool_calls"][0]["function"]["arguments"] == original_args


class TestDeterminism:
    def test_two_independent_runs_produce_identical_bytes(self, rw):
        msg = _tool_call_msg({"content": _big()})
        a = rw.rewrite_outgoing([dict(msg)]).messages
        b = rw.rewrite_outgoing([dict(msg)]).messages
        assert (a[0]["tool_calls"][0]["function"]["arguments"]
                == b[0]["tool_calls"][0]["function"]["arguments"])

    def test_idempotent_on_already_transformed_output(self, rw):
        msg = _tool_call_msg({"content": _big()})
        once = rw.rewrite_outgoing([msg]).messages
        twice = rw.rewrite_outgoing(once).messages
        assert (once[0]["tool_calls"][0]["function"]["arguments"]
                == twice[0]["tool_calls"][0]["function"]["arguments"])


class TestMultipleToolCallsAndArgs:
    def test_only_the_large_value_among_several_is_stubbed(self, rw):
        msg = _tool_call_msg({"a": "tiny", "b": _big(), "c": "also tiny"})
        out = rw.rewrite_outgoing([msg]).messages
        parsed = json.loads(out[0]["tool_calls"][0]["function"]["arguments"])
        assert parsed["a"] == "tiny"
        assert parsed["c"] == "also tiny"
        assert "[[cm:stored handle=" in parsed["b"]

    def test_multiple_tool_calls_each_handled_independently(self, rw):
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "write_file",
                          "arguments": json.dumps({"content": _big()})}},
            {"id": "c2", "type": "function",
             "function": {"name": "read_file",
                          "arguments": json.dumps({"path": "x"})}},
        ]}
        out = rw.rewrite_outgoing([msg]).messages
        args1 = json.loads(out[0]["tool_calls"][0]["function"]["arguments"])
        args2 = json.loads(out[0]["tool_calls"][1]["function"]["arguments"])
        assert "[[cm:stored handle=" in args1["content"]
        assert args2 == {"path": "x"}

    def test_non_string_values_left_alone(self, rw):
        msg = _tool_call_msg({"count": 42, "enabled": True, "content": _big()})
        out = rw.rewrite_outgoing([msg]).messages
        parsed = json.loads(out[0]["tool_calls"][0]["function"]["arguments"])
        assert parsed["count"] == 42
        assert parsed["enabled"] is True
