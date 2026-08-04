"""Content-parts handle-ization (2026-08-02) — the 66% of the wire we could not see.

Pass 1 handle-izes only ``isinstance(content, str)``; ``_window_out`` skips
anything that is not a str. A harness that sends tool results as OpenAI
content-parts is therefore structurally invisible to the governor. Measured on a
live opencode run: structured_content 66.3% of the wire vs string_content 12.9%,
with 88/191 messages already handle-ized and the prompt still climbing to 47 K.

These tests pin the things that break something if this is done wrong:
  - OFF by default: a parts wire is passed through byte-identically.
  - Shape preserved: a list stays a list, parts keep their keys.
  - Image parts NEVER stubbed (silent multimodal destruction).
  - Idempotent: rewrite(rewrite(x)) == rewrite(x) — else Pass 1 fails to
    recognise its own output and re-stubs it every turn, which is the
    non-monotonic prefix break this whole effort exists to remove.
  - Lossless: the original text is retrievable from the store.
"""

from __future__ import annotations

from conftest import FakeCounter
from contextmanager.durable import DurableStore
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter


def _rw(tmp_path, **over):
    kw = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        handle_threshold_tokens=10,
        stub_preview_chars=10,
        rehydrate_budget_tokens=0,
        auto_recall_k=0,
        request_timeout=30.0,
    )
    kw.update(over)
    cfg = ProxyConfig(**kw)
    store = DurableStore(cfg.store_root)
    return cfg, store, PromptRewriter(cfg, FakeCounter(), store)


BIG = " ".join(f"chunk{i}" for i in range(60))     # 60 words -> 60 FakeCounter tokens
SMALL = "tiny result"


def _parts_wire(extra_parts=None):
    parts = [{"type": "text", "text": BIG}]
    if extra_parts:
        parts = parts + list(extra_parts)
    return [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "do the thing"},
        {"role": "tool", "content": parts},
    ]


class TestDefaultOff:
    def test_parts_untouched_when_flag_off(self, tmp_path):
        _, _, rw = _rw(tmp_path)                      # handleize_content_parts defaults False
        wire = _parts_wire()
        out = rw.rewrite_outgoing(wire)
        assert out.messages[2]["content"] == wire[2]["content"]
        assert out.handle_ized_ids == []


class TestShapeAndSafety:
    def test_oversized_text_part_is_stubbed_in_place(self, tmp_path):
        _, _, rw = _rw(tmp_path, handleize_content_parts=True)
        out = rw.rewrite_outgoing(_parts_wire())
        content = out.messages[2]["content"]
        assert isinstance(content, list), "shape changed — parts array was flattened"
        assert content[0]["type"] == "text", "part lost its keys"
        assert PromptRewriter.is_stub(content[0]["text"])
        assert BIG not in content[0]["text"]
        assert len(out.handle_ized_ids) == 1

    def test_image_parts_are_never_touched(self, tmp_path):
        _, _, rw = _rw(tmp_path, handleize_content_parts=True)
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}}
        out = rw.rewrite_outgoing(_parts_wire([image]))
        content = out.messages[2]["content"]
        assert content[1] == image, "an image part was rewritten"
        assert content[1] is image, "image part was needlessly copied"

    def test_small_text_parts_pass_through_by_identity(self, tmp_path):
        _, _, rw = _rw(tmp_path, handleize_content_parts=True)
        wire = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": [{"type": "text", "text": SMALL}]},
        ]
        out = rw.rewrite_outgoing(wire)
        assert out.messages[2]["content"] is wire[2]["content"], (
            "unchanged parts message was copied — wire is no longer byte-stable"
        )
        assert out.handle_ized_ids == []

    def test_non_text_unknown_part_types_pass_through(self, tmp_path):
        _, _, rw = _rw(tmp_path, handleize_content_parts=True)
        weird = {"type": "input_audio", "input_audio": {"data": "Z" * 5000}}
        out = rw.rewrite_outgoing(_parts_wire([weird]))
        assert out.messages[2]["content"][1] is weird


class TestIdempotenceAndLossless:
    def test_rewrite_of_rewrite_is_identical(self, tmp_path):
        _, _, rw = _rw(tmp_path, handleize_content_parts=True)
        first = rw.rewrite_outgoing(_parts_wire())
        second = rw.rewrite_outgoing(first.messages)
        assert second.messages == first.messages, (
            "not idempotent — Pass 1 failed to recognise its own stub and would "
            "re-cut the prefix every turn"
        )
        assert second.handle_ized_ids == []

    def test_original_text_is_retrievable(self, tmp_path):
        _, store, rw = _rw(tmp_path, handleize_content_parts=True)
        out = rw.rewrite_outgoing(_parts_wire())
        handles = PromptRewriter.parse_handles(out.messages[2]["content"][0]["text"])
        assert handles, "stub carries no handle — content is unreachable"
        assert store.get(handles[0]) == BIG

    def test_input_wire_is_never_mutated(self, tmp_path):
        _, _, rw = _rw(tmp_path, handleize_content_parts=True)
        wire = _parts_wire()
        snapshot = [{k: (list(v) if isinstance(v, list) else v) for k, v in m.items()}
                    for m in wire]
        rw.rewrite_outgoing(wire)
        assert wire == snapshot, "caller's messages were mutated in place"


class TestMetricsAccounting:
    """chars_in/chars_out counted string-valued content only, so a parts-shaped
    wire scored ~0 going in while the string-shaped recall block scored on the
    way out: a live request that shrank 34 KB to 1,786 prompt tokens reported
    pct_saved = -11975%. The savings ratio has to mean something on both shapes."""

    def test_parts_content_is_counted(self):
        from contextmanager.proxy.app import _sum_content_chars
        big = "x" * 5000
        as_string = [{"role": "tool", "content": big}]
        as_parts = [{"role": "tool", "content": [{"type": "text", "text": big}]}]
        assert _sum_content_chars(as_string) >= 5000
        assert _sum_content_chars(as_parts) >= 5000, (
            "content-parts scored zero — the same blind spot that hid the "
            "governor from opencode, now in the metrics"
        )

    def test_stubbing_parts_shows_as_a_saving(self, tmp_path):
        from contextmanager.proxy.app import _sum_content_chars
        _, _, rw = _rw(tmp_path, handleize_content_parts=True)
        wire = _parts_wire()
        out = rw.rewrite_outgoing(wire)
        assert _sum_content_chars(out.messages) < _sum_content_chars(wire)
