"""Pass -1 volatile-stamp normalization.

Regression cover for the 2026-07-28 incident: Claude Code 2.1.x stamps a
per-request nonce into the FIRST content-part of its system prompt
(``x-anthropic-billing-header: ... cch=<hex>;``). Unnormalized it broke the
pipeline twice over — it churned ``conversation_key`` (so every turn was filed
as a new conversation, which flushed the frozen recall block and rebuilt it at
the tail) AND it was forwarded verbatim, so the hybrid-SSM prompt cache could
never find a byte-exact prefix and re-prefilled 25-28k tokens every turn.

The shapes below are taken from a live wire capture, not invented.
"""

from __future__ import annotations

from contextmanager.proxy.rewriter import normalize_volatile_stamps
from contextmanager.proxy.sensing import conversation_key

# Verbatim from the capture (part lengths 81 / 57 / 25815 in the real wire).
BILLING = ("x-anthropic-billing-header: cc_version=2.1.119.{ver}; "
           "cc_entrypoint=cli; cch={nonce};")
IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def cc_system(nonce: str = "9b25f", ver: str = "af2") -> dict:
    """Claude Code's real shape: system content as three text parts."""
    return {"role": "system", "content": [
        {"type": "text", "text": BILLING.format(ver=ver, nonce=nonce)},
        {"type": "text", "text": IDENTITY},
        {"type": "text", "text": "\nYou are an interactive agent that helps."},
    ]}


class TestNormalizerBehaviour:
    def test_blanks_nonce_and_version_keeps_entrypoint(self):
        out = normalize_volatile_stamps([cc_system()])
        text = out[0]["content"][0]["text"]
        assert text == ("x-anthropic-billing-header: cc_version=; "
                        "cc_entrypoint=cli; cch=;")
        # cc_entrypoint keeps its VALUE on purpose: it separates cli/sdk/vscode
        # sessions and is a legitimate identity discriminator.
        assert "cc_entrypoint=cli" in text

    def test_other_parts_untouched(self):
        out = normalize_volatile_stamps([cc_system()])
        assert out[0]["content"][1]["text"] == IDENTITY
        assert out[0]["content"][2]["text"] == "\nYou are an interactive agent that helps."

    def test_string_content_shape_also_handled(self):
        msg = {"role": "system", "content": BILLING.format(ver="af2", nonce="cdb12") + IDENTITY}
        out = normalize_volatile_stamps([msg])
        assert "cch=;" in out[0]["content"]
        assert out[0]["content"].endswith(IDENTITY)

    def test_non_system_messages_untouched(self):
        # A user pasting telemetry-looking text must never be rewritten.
        msg = {"role": "user", "content": BILLING.format(ver="af2", nonce="dead1")}
        out = normalize_volatile_stamps([msg])
        assert out[0]["content"] == msg["content"]

    def test_system_without_marker_untouched(self):
        msg = {"role": "system", "content": "You are helpful. cch=abc123; stays."}
        out = normalize_volatile_stamps([msg])
        assert out[0]["content"] == msg["content"]

    def test_occurrence_beyond_leading_window_untouched(self):
        # Replacement is bounded to the leading window, so real prompt content
        # far from the header can never be clipped.
        tail = "later in the prompt: cch=realcontent;"
        msg = {"role": "system",
               "content": BILLING.format(ver="af2", nonce="9b25f") + ("x" * 600) + tail}
        out = normalize_volatile_stamps([msg])
        assert out[0]["content"].endswith(tail)

    def test_pure_and_idempotent(self):
        original = [cc_system()]
        snapshot = original[0]["content"][0]["text"]
        once = normalize_volatile_stamps(original)
        twice = normalize_volatile_stamps(once)
        assert original[0]["content"][0]["text"] == snapshot  # input not mutated
        assert twice[0]["content"][0]["text"] == once[0]["content"][0]["text"]

    def test_returns_input_object_when_nothing_changed(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert normalize_volatile_stamps(msgs) is msgs

    def test_never_raises_on_junk(self):
        for junk in ("not a list", None, [None], [{"role": "system"}],
                     [{"role": "system", "content": 42}],
                     [{"role": "system", "content": [None, "x", {"no": "text"}]}]):
            normalize_volatile_stamps(junk)


class TestIdentityStability:
    """The money test: the nonce must stop churning the conversation key."""

    def test_key_stable_across_nonce_and_version_churn(self):
        # Three consecutive turns of ONE chat, with the nonce values actually
        # observed in the capture and a build-suffix flip on the third.
        t1 = [cc_system("9b25f", "af2"), {"role": "user", "content": "hello"}]
        t2 = [cc_system("cdb12", "af2"), {"role": "user", "content": "hello"},
              {"role": "assistant", "content": "hi"},
              {"role": "user", "content": "follow-up"}]
        t3 = [cc_system("88fab", "c72"), {"role": "user", "content": "hello"},
              {"role": "assistant", "content": "hi"},
              {"role": "user", "content": "follow-up"},
              {"role": "assistant", "content": "ok"},
              {"role": "user", "content": "more"}]

        keys = {conversation_key(normalize_volatile_stamps(t))
                for t in (t1, t2, t3)}
        assert len(keys) == 1, keys

    def test_unnormalized_would_still_churn(self):
        # Pins WHY the pass is required — without it the key moves every turn.
        t1 = [cc_system("9b25f"), {"role": "user", "content": "hello"}]
        t2 = [cc_system("cdb12"), {"role": "user", "content": "hello"}]
        assert conversation_key(t1) != conversation_key(t2)

    def test_side_call_still_separated(self):
        # Main chat vs. title/summary side-call share the system prompt and must
        # NOT collapse onto one key.
        main = normalize_volatile_stamps(
            [cc_system("9b25f"), {"role": "user", "content": "work on the parser"}])
        side = normalize_volatile_stamps(
            [cc_system("cdb12"), {"role": "user", "content": "Generate a concise title"}])
        assert conversation_key(main) != conversation_key(side)

    def test_different_entrypoints_stay_separate(self):
        # cc_entrypoint is preserved, so a CLI session and a VSCode session with
        # the same opening message remain distinct conversations.
        cli = normalize_volatile_stamps(
            [cc_system("9b25f"), {"role": "user", "content": "hello"}])
        vscode_sys = cc_system("cdb12")
        vscode_sys["content"][0]["text"] = vscode_sys["content"][0]["text"].replace(
            "cc_entrypoint=cli", "cc_entrypoint=vscode")
        vscode = normalize_volatile_stamps(
            [vscode_sys, {"role": "user", "content": "hello"}])
        assert conversation_key(cli) != conversation_key(vscode)


class TestWireStability:
    """Identity alone is not enough — the FORWARDED bytes must be stable too."""

    def test_normalized_system_bytes_identical_across_turns(self):
        a = normalize_volatile_stamps([cc_system("9b25f", "af2")])
        b = normalize_volatile_stamps([cc_system("88fab", "c72")])
        assert a[0]["content"] == b[0]["content"]
