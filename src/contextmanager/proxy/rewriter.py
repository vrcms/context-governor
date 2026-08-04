"""PromptRewriter — the Phase 3 core: a pure, deterministic transform over an
OpenAI-compatible `messages` list.

Normative per Phase 3 spec §3 (3.1–3.5). Free of any FastAPI/httpx/network
imports: it depends only on the stdlib (`hashlib`, `re`, `dataclasses`) and on
the contextmanager `TokenCounter`/`Message`/`DurableStore` types.

The headline invariant (§3.5) is PREFIX STABILITY / IDEMPOTENCY: a second pass
of `rewrite_outgoing` over its own output is a no-op for handle-ization (stubs
stay byte-identical), and handle-ization of message i depends ONLY on message i
(never on its neighbors). Both hold by construction here because:
  - `stable_id` is a pure function of (role, content);
  - the handle-ization decision (`counter.count_text(content) >= threshold`) is
    per-message;
  - already-stub content is detected and left untouched.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Optional

from ..types import Message, TokenCounter
from ..durable import DurableStore
from .recall import extract_query, select_diverse, similarity_ratio
from .recall import _Indel  # re-exported: tests assert the accelerator is wired
from .sensing import canonical_content, conversation_key


def _similarity(base: str, content: str) -> float:
    """Normalized similarity in [0,1] used to pick a diff-encoding base.

    Thin wrapper over `recall.similarity_ratio`, which is the single shared
    implementation (it lives there because `rewriter` imports `recall`, so the
    reverse would be circular). Full rationale and measurements are in that
    function's docstring; the diff-encoding specifics are below.

    WHY THIS EXISTS (measured 2026-07-28 on real store notes, ~19-20 KB each —
    exactly the size that reaches this code path under `diff_max_chars`):

        pair                difflib      rapidfuzz     speedup
        19898 x 19703        9.26 s       0.0109 s        851x
        19898 x 19449        8.41 s       0.0106 s        793x
        19703 x 19449       17.22 s       0.0091 s       1893x
                                          overall         981x

    `difflib.SequenceMatcher` is O(n*m) in pure Python, and this runs
    SYNCHRONOUSLY on the request path. At `diff_lookback = 6` that is up to ~60 s
    of CPU per eligible message — which matched, almost exactly, the 56.6 s of
    governor CPU measured during a single 97 s "dead wait" where llama-server sat
    idle and the client was blocked. Two thirds of session wall-clock was going
    here.

    METRICS AGREE WHERE IT MATTERS. Indel normalized similarity and
    Ratcliff-Obershelp differ on unrelated documents (0.05 vs 0.33), but there
    both are far below any sane threshold and the decision is identical. On the
    case this feature exists for — a file re-read after an edit — they agree to
    3-4 decimals:

        edits:      0       1       3      10      50     200
        difflib   1.0000  0.9996  0.9968  0.9895  0.9408  0.7587
        rapidfuzz 1.0000  0.9996  0.9969  0.9896  0.9432  0.7692

    So `diff_min_similarity` keeps its meaning and needed no recalibration.
    `tests/proxy/test_diff_similarity.py` pins that agreement so a future
    rapidfuzz release cannot silently move the decision boundary.

    NOTE: `difflib.unified_diff` (used to RENDER the delta) is deliberately left
    alone — it matches on lines, not characters, so its n is ~3 orders of
    magnitude smaller and it was never part of the cost.
    """
    return similarity_ratio(base, content)


# Regex matching the opening line of a stub (spec §3.1). The handle is captured;
# role and tokens are matched but not captured.
_HANDLE_RE = re.compile(
    r"\[\[cm:stored handle=(?P<handle>\S+) role=\S+ tokens=\d+\]\]"
)

# The literal prefix that identifies a stub (spec §3.3 is_stub).
_STUB_PREFIX = "[[cm:stored handle="

# Regex matching the opening marker line of a synthetic rehydrated message
# (spec §9.1): `[[cm:rehydrated handle=<H>]]`. The handle is captured. Used by
# `is_rehydrated` (prefix check) and `_parse_rehydrated_handle` (full parse).
_REHYDRATED_RE = re.compile(r"\[\[cm:rehydrated handle=(?P<handle>\S+)\]\]")
_REHYDRATED_PREFIX = "[[cm:rehydrated handle="

# Diff-stub: a delta-compressed handle-ization. A bulky message that is a near-
# duplicate of an already-stored note is replaced by the BASE handle + a unified
# diff instead of a head/tail preview — lossless (full content still paged out
# under `handle`), and tiny + informative for iterative content (file re-reads,
# repeated state dumps). Marker:
#   [[cm:diff handle=<new> base=<base> role=<r> tokens=<n>]]\n<diff>\n[[/cm:diff]]
_DIFF_PREFIX = "[[cm:diff handle="
# Extracts the PRIMARY handle (the full content) from either a stored- or diff-stub.
_PRIMARY_HANDLE_RE = re.compile(r"\[\[cm:(?:stored|diff) handle=(?P<handle>\S+)")

# Auto-recall (Phase 10): ONE synthetic system message per request carrying store
# slices relevant to the LIVE TAIL (anticipatory demand paging — the model cannot
# page-fault on memory it cannot see). Marker:
#   [[cm:recall]]\n[[cm:recalled handle=<h>]]\n<slice>\n…\n[[/cm:recall]]
# Recall blocks are STRIPPED on entry and recomputed fresh each call, so at most one
# exists on the wire at any time and rewrite(rewrite(x)) cannot grow. `[[cm:recalled`
# deliberately does NOT match `_HANDLE_RE` (`[[cm:stored`), so Pass 2 never re-expands
# a recalled slice.
_RECALL_PREFIX = "[[cm:recall]]"

# Handles already VISIBLE on the wire (stored-/diff-stub headers, diff `base=`
# references, rehydrated markers). These are not recall candidates: auto-recall
# targets OFF-wire memory only (content the host CLI compacted away, evicted notes,
# MCP-saved state, prior sessions). Handles are filesystem-safe slugs, hence the
# explicit character class.
_ONWIRE_HANDLE_RE = re.compile(
    r"\[\[cm:(?:stored|diff|rehydrated) handle=(?P<h>[A-Za-z0-9._-]+)"
)
_DIFF_BASE_RE = re.compile(
    r"\[\[cm:diff handle=[A-Za-z0-9._-]+ base=(?P<h>[A-Za-z0-9._-]+)"
)

# ------------------------------------------------- Pass -1: volatile stamps
# Some harnesses stamp PER-REQUEST telemetry into the system prompt. Claude Code
# 2.1.x (through LiteLLM) sends messages[0].content as content-parts whose FIRST
# part is nothing but a billing header — measured 2026-07-28 from a live wire
# capture:
#     part0 (81 chars): "x-anthropic-billing-header: cc_version=2.1.119.af2;
#                        cc_entrypoint=cli; cch=9b25f;"
#     part1 (57 chars): "You are Claude Code, Anthropic's official CLI for Claude."
#     part2 (25815)   : the actual system prompt
# `cch` changes on EVERY request (9b25f / cdb12 / 88fab observed in one session)
# and `cc_version` flips its build suffix (af2 / c72) within a single session.
#
# WHY THIS MATTERS TWICE. Two independent failures both trace to this nonce:
#   1. IDENTITY — conversation_key() hashes messages[0], so the key churned every
#      turn. Every request was filed as a new conversation, which forced
#      prefix_broken=True, which made the rewriter FLUSH its frozen recall block
#      and rebuild it at the tail each turn (the ~16752-token divergence point in
#      the server log). All per-conversation state was dead: sticky recall, the
#      windowing ledger, hysteresis, the learned ceiling.
#   2. WIRE — even with a perfect key, the nonce is still FORWARDED. On a hybrid
#      SSM/recurrent model, cache reuse needs a byte-exact prefix, so a differing
#      char 94 forces a full re-prefill from token ~24 regardless of identity.
# An earlier attempt normalized only for hashing (case 1) and still saw full
# re-processing, because case 2 was untouched. Normalizing the WIRE at ingress
# fixes both at once: identity is derived from the same normalized bytes that
# are forwarded, so the two can never drift apart.
#
# SCOPE IS DELIBERATELY NARROW: system messages only, only when the marker is
# present in the leading window, only the two measured-volatile VALUES.
# `cc_entrypoint` keeps its value on purpose — it distinguishes cli / sdk /
# vscode sessions and is a legitimate identity discriminator. Unknown patterns
# are NOT stripped; blind normalization would eat real content (dates, versions
# the model legitimately needs).
_BILLING_HEADER_MARKER = "x-anthropic-billing-header:"
_BILLING_VOLATILE_RE = re.compile(r"\b(cc_version|cch)=[^;\s]*")
# The header sits at chars 0-81. Bounding replacement to a leading window means a
# `cch=` appearing later in real prompt content can never be touched.
_VOLATILE_WINDOW = 512


def _normalize_volatile_text(text) -> Optional[str]:
    """Return normalized text, or None when nothing changed / not applicable."""
    if not isinstance(text, str):
        return None
    head = text[:_VOLATILE_WINDOW]
    if _BILLING_HEADER_MARKER not in head:
        return None
    new_head = _BILLING_VOLATILE_RE.sub(r"\1=", head)
    if new_head == head:
        return None
    return new_head + text[_VOLATILE_WINDOW:]


def _normalize_volatile_content(content):
    """Normalize str content or an OpenAI content-parts list. None = unchanged."""
    if isinstance(content, str):
        return _normalize_volatile_text(content)
    if isinstance(content, list):
        new_parts = None
        for i, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            new_text = _normalize_volatile_text(part.get("text"))
            if new_text is None:
                continue
            if new_parts is None:
                new_parts = list(content)
            new_part = dict(part)
            new_part["text"] = new_text
            new_parts[i] = new_part
        return new_parts
    return None


def normalize_volatile_stamps(messages: list) -> list:
    """Blank per-request telemetry values in system messages.

    PURE: returns a new list only when something changed, sharing every untouched
    message object; the input is never mutated. Idempotent — running it twice is
    identical to running it once. Never raises: any unexpected shape is returned
    unchanged, which fails OPEN (the old churn resumes and is visible in
    /metrics as breaks_by_cause={"new-conversation": N}) rather than dropping a
    request.

    MUST be called at ingress, BEFORE sensing — conversation_key() runs in
    controller.observe_request() and hashes whatever it is handed.
    """
    if not isinstance(messages, list):
        return messages
    out = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        new_content = _normalize_volatile_content(msg.get("content"))
        if new_content is None:
            continue
        if out is None:
            out = list(messages)
        new_msg = dict(msg)
        new_msg["content"] = new_content
        out[i] = new_msg
    return messages if out is None else out

# Rough chars-per-token used ONLY to estimate the size of content too large to send to
# the tokenizer (the stub's `tokens=` field is informational; the decision to handle-ize
# such content is already certain from its char length).
_EST_CHARS_PER_TOKEN = 4


@dataclass
class RewriteResult:
    """Result of `PromptRewriter.rewrite_outgoing`.

    Attributes:
        messages: the rewritten OpenAI messages to send upstream.
        handle_ized_ids: ids of messages that were replaced by a stub this call.
        rehydrated_handles: handles whose content was paged back in this call.
        recalled_handles: handles auto-recalled into a FRESH Pass-4 block this
            call. Empty on sticky turns that re-injected the frozen block
            (nothing new was searched or recalled — that is the point).
        windowing_triggered: True iff Pass 3 crossed the high water this call and
            advanced the stub frontier (should be RARE — that is the point of the
            hysteresis: triggers << requests means the KV prefix is being reused).
        windowing_emergency: True iff this trigger only happened because the
            HIGH-water override (context_emergency_ratio) forced it — i.e. the
            hysteresis latch WOULD have suppressed it otherwise. Always False
            when the emergency tier is disabled (the default). Should be RARER
            than windowing_triggered; a nonzero rate under normal operation
            means something is chronically unsheddable and worth investigating
            directly, not just papering over with more overrides.
    """

    messages: list[dict]
    handle_ized_ids: list[str]
    rehydrated_handles: list[str]
    recalled_handles: list[str] = field(default_factory=list)
    windowing_triggered: bool = False
    windowing_emergency: bool = False


class PromptRewriter:
    """Pure, deterministic rewriter of an OpenAI messages list.

    Constructed with a `ProxyConfig`, a `TokenCounter`, and a `DurableStore`.
    `rewrite_outgoing` produces the rewritten messages plus bookkeeping lists.
    All the static helpers (`stable_id`, `make_stub`, `parse_handles`,
    `is_stub`) are deterministic and free of instance state.
    """

    def __init__(self, config: "ProxyConfig", counter: TokenCounter,
                 store: DurableStore, n_ctx: Optional[int] = None) -> None:
        # Import locally to avoid a circular import at module load time
        # (proxy/__init__. imports both config and rewriter).
        from .config import ProxyConfig
        if not isinstance(config, ProxyConfig):  # defensive, cheap
            raise TypeError("config must be a ProxyConfig")
        self.config = config
        self.counter = counter
        self.store = store
        # The upstream's true context size (from /props), if known. Enables the
        # total-budget windowing pass. None -> windowing disabled.
        self._n_ctx = n_ctx
        # STICKY windowing frontier (Phase 11), keyed PER CONVERSATION (Phase
        # 14b): conv_key -> {stable_id: original token count} of every message
        # Pass 3 has windowed out for that conversation. Re-applied each turn
        # (the CLI resends originals) so previously-windowed messages become
        # byte-identical stubs again WITHOUT re-triggering — this is what keeps
        # the wire prefix stable between high-water crossings, and thereby the
        # upstream KV cache warm. In-memory by design: its useful lifetime
        # equals the upstream's prefix-cache lifetime (a proxy restart just
        # re-derives it once). LRU-bounded by config.max_conversations.
        self._windowed: "OrderedDict[str, dict]" = OrderedDict()
        # MISSED-LOW-WATER latch (Phase 14b hardening): conv_key -> pressure at
        # the last windowing trigger that could NOT shed down to the low water
        # (unsheddable mass: invisible template/tools tokens, protected tail,
        # already-minimal stubs — paging has nothing left to give). While
        # latched, re-triggers are SUPPRESSED until pressure grows by the
        # hysteresis gap, converting futile per-turn prefix breaks into a
        # byte-stable wire the upstream cache can extend. Cleared when pressure
        # falls back below the high water (the crisis is over; a fresh trigger
        # may well be sheddable). Never engaged when the hysteresis gap is
        # collapsed (target 0 = legacy per-turn line).
        self._window_latched: "OrderedDict[str, int]" = OrderedDict()
        # STICKY recall block (Phase 12), keyed PER CONVERSATION (Phase 14b —
        # one global slot thrashed across the ≥3 interleaved conversations of
        # the 2026-07-19 shakedown): conv_key -> (anchor_mid, block_text) of
        # the last Pass-4 block built, where anchor_mid identifies the message
        # the block was inserted BEFORE. Re-injected byte-identically at the
        # same anchor each turn; recomputed (in one jump, at the new tail) only
        # at a FLUSH EPOCH: the prefix is already broken this turn (harness
        # edit / new conversation / multimodal / a Pass-3 windowing trigger —
        # the refresh rides the break for free) or a hard bound is hit (growth
        # past recall_max_stale_tokens, anchor loss).
        self._recall_frozen: "OrderedDict[str, tuple]" = OrderedDict()
        # PINNED handle-ization threshold (2026-08-02), conv_key -> tokens. The
        # threshold is derived from the upstream's n_ctx, and n_ctx can CHANGE
        # under us (llama-server restarted with a different -c). Re-cutting a
        # live conversation with a new threshold breaks its prefix in EITHER
        # direction — see _pinned_threshold — so it is frozen at first sight.
        self._threshold: "OrderedDict[str, int]" = OrderedDict()
        # PINNED tool_call ARGUMENT threshold (2026-08-03), conv_key -> tokens.
        # Pinned for exactly the same reason as _threshold: it is derived from
        # n_ctx, and lowering it under a live conversation turns arguments
        # already sent verbatim into stubs — an own-mutation on every affected
        # message at once. Kept as its own cache rather than folded into
        # _threshold so the two setpoints can move independently.
        self._tc_threshold: "OrderedDict[str, int]" = OrderedDict()

    # ------------------------------------------------------------------ ids
    @staticmethod
    def stable_id(role: str, content: str) -> str:
        """Deterministic per-message id: same (role, content) -> same id.

        `id = "msg-" + sha1(role + "\\x00" + content).hexdigest()[:16]`.
        The NUL byte separates role from content so that pairs like
        ("ab", "c") and ("a", "bc") do not collide.
        """
        digest = hashlib.sha1((role + "\x00" + content).encode("utf-8")).hexdigest()
        return "msg-" + digest[:16]

    @staticmethod
    def stable_id_any(role: str, content) -> str:
        """`stable_id` extended to ANY content shape (Phase 14b) via the shared
        canonical serialization. Byte-compatible with `stable_id` for strings —
        so ids frozen before this method existed still match — and defined for
        content-parts lists (the non-str anchors the Phase-12 freeze used to
        skip, breaking the prefix on every image turn)."""
        return PromptRewriter.stable_id(role, canonical_content(content))

    def _lru_touch(self, cache: "OrderedDict", key: str) -> None:
        """Mark `key` most-recent in a per-conversation LRU cache and evict the
        oldest entries beyond config.max_conversations."""
        if key in cache:
            cache.move_to_end(key)
        cap = self.config.max_conversations
        while len(cache) > cap:
            cache.popitem(last=False)

    def _touch_conversation(self, conv_key: str) -> None:
        """Mark this conversation most-recent in EVERY per-conversation cache.

        THE BUG THIS FIXES (measured 2026-08-02). All three sticky caches are
        READ with ``.get()`` — Pass 3a's windowing frontier, the missed-low-water
        latch, the frozen recall block — but were only ever LRU-touched on the
        paths that WRITE them (a windowing trigger, a recall rebuild). So the
        HEALTHY steady state — frontier re-applied byte-identically, recall block
        reused frozen, no trigger — never refreshed recency: **a conversation
        aged toward eviction precisely because it was behaving.** Meanwhile every
        one-shot side-call (title/summary generation) inserts a fresh key and
        pushes it further back. The 2026-08-02 session ran 28 distinct
        conversations against ``max_conversations = 32``.

        Losing ``_windowed`` is not a cache miss, it is a REGRESSION. Every
        previously-windowed message returns to the wire verbatim, so:

            Pass 3a re-applies nothing -> wire jumps by the full original mass
            -> pressure crosses the high water -> Pass 3b re-windows the SAME
            messages -> prefix breaks at protect_first_n -> full re-prefill

        which is recorded as an ``own-mutation`` and costs a ~25 s re-read of a
        30 K prompt. The next turn can evict it again, which is what a run of
        breaks at one fixed position looks like from the server side.

        Touching on READ is what makes handle-ization **monotonic under
        interleaving**: a conversation being actively rewritten can no longer
        have its frontier evicted by conversations that are merely newer. Within
        a live key the frontier was already monotonic (``_window_out`` only ever
        adds to it); eviction was the sole path by which a message that had been
        handle-ized could become un-handle-ized.
        """
        for cache in (self._windowed, self._window_latched,
                      self._recall_frozen, self._threshold, self._tc_threshold):
            if conv_key in cache:
                self._lru_touch(cache, conv_key)

    # ---------------------------------------------------------------- stubs
    @staticmethod
    def make_stub(handle: str, role: str, tokens: int, content: str,
                  preview_chars: int) -> str:
        """Render the deterministic stub text (spec §3.1).

        Format (when content length > 2*preview_chars):
            [[cm:stored handle=<handle> role=<role> tokens=<n>]]
            <preview-head>
            …(truncated <m> chars)…
            <preview-tail>
            [[/cm:stored]]

        When content length <= 2*preview_chars, the truncated-line and the
        tail are omitted (just the head, which is the whole content):
            [[cm:stored handle=<handle> role=<role> tokens=<n>]]
            <content>
            [[/cm:stored]]

        `<m>` is the number of characters omitted between head and tail.
        """
        header = f"[[cm:stored handle={handle} role={role} tokens={tokens}]]"
        footer = "[[/cm:stored]]"
        n = len(content)
        if n <= 2 * preview_chars:
            # Omit the truncated-line and the tail; the head IS the whole content.
            return f"{header}\n{content}\n{footer}"
        head = content[:preview_chars]
        # `content[len(content)-preview_chars:]` is the last `preview_chars`
        # chars; this is correct even when preview_chars == 0 (yields "").
        tail = content[n - preview_chars:]
        omitted = n - 2 * preview_chars
        return (
            f"{header}\n{head}\n"
            f"…(truncated {omitted} chars)…\n"
            f"{tail}\n{footer}"
        )

    @staticmethod
    def stub_tokens_estimate(preview_chars: int) -> int:
        """Approximate tokens a HANDLE-IZED message costs on the sent wire.

        Sized by rendering a representative stub through `make_stub` rather than
        hard-coding a constant, so it cannot drift if the stub format changes.
        Used by the 14c pressure estimate (`sensing.observe_request`): a message
        at or above the handle threshold does NOT reach the upstream at its own
        size — the rewriter pages it out and sends this instead. Estimating it at
        `chars / 4` is what made a 60 KB tool result look like ~15,000 tokens of
        pressure when its real contribution to the wire was ~50.
        """
        sample = PromptRewriter.make_stub(
            handle="msg-0123456789abcdef", role="tool", tokens=999999,
            content="x" * (2 * preview_chars + 1), preview_chars=preview_chars,
        )
        return max(1, len(sample) // _EST_CHARS_PER_TOKEN)

    @staticmethod
    def parse_handles(text: str) -> list[str]:
        """Return all handles referenced by `[[cm:stored handle=…]]` markers in
        `text`, in order of first occurrence, deduplicated."""
        seen: set[str] = set()
        out: list[str] = []
        for m in _HANDLE_RE.finditer(text):
            h = m.group("handle")
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out

    @staticmethod
    def is_stub(content: str) -> bool:
        """True iff `content` (a string) begins, after optional leading
        whitespace, with `[[cm:stored handle=` — i.e. the message IS a stub we
        produced. Non-string input returns False (defensive; callers should
        ensure `content` is a str before relying on the str-typed contract)."""
        if not isinstance(content, str):
            return False
        return content.lstrip().startswith(_STUB_PREFIX)

    @staticmethod
    def is_rehydrated(content: str) -> bool:
        """True iff `content` is a str whose `lstrip()` starts with
        `[[cm:rehydrated handle=` — i.e. the message is one of the synthetic
        rehydrated messages produced by Pass 2 of `rewrite_outgoing`.

        Rehydrated messages are ALREADY-REWRITTEN output: they must never be
        re-handle-ized (Pass 1 leaves them untouched) and must never be
        re-rehydrated (Pass 2 skips handles already present as rehydrated
        markers). This is the §9.1 idempotency fix.
        """
        if not isinstance(content, str):
            return False
        return content.lstrip().startswith(_REHYDRATED_PREFIX)

    @staticmethod
    def is_recall(content: str) -> bool:
        """True iff `content` is a str whose `lstrip()` starts with `[[cm:recall]]`
        — i.e. the message is the Pass-4 auto-recall block. Recall blocks are
        already-rewritten output that must be STRIPPED on entry and recomputed
        (locality shifts every turn), never re-handle-ized or accumulated."""
        if not isinstance(content, str):
            return False
        return content.lstrip().startswith(_RECALL_PREFIX)

    @staticmethod
    def _parse_rehydrated_handle(content: str) -> Optional[str]:
        """Extract the handle from a `[[cm:rehydrated handle=H]]` marker at the
        start of `content` (after optional leading whitespace). Returns the
        handle string, or None if `content` does not start with such a marker.
        """
        if not isinstance(content, str):
            return None
        m = _REHYDRATED_RE.match(content.lstrip())
        return m.group("handle") if m is not None else None

    # ---------------------------------------------------------- diff-encoding
    @staticmethod
    def is_diff_stub(content: str) -> bool:
        """True iff `content` is a diff-stub we produced (already-rewritten output;
        passed through unchanged on re-entry, like a normal stub)."""
        if not isinstance(content, str):
            return False
        return content.lstrip().startswith(_DIFF_PREFIX)

    @staticmethod
    def make_diff_stub(handle: str, base_handle: str, role: str, tokens: int,
                       diff_text: str) -> str:
        """Render a diff-stub. The FULL content is still paged out under `handle`
        (lossless, rehydratable); the wire carries only a unified diff against
        `base_handle`'s stored content."""
        header = (
            f"[[cm:diff handle={handle} base={base_handle} "
            f"role={role} tokens={tokens}]]"
        )
        return f"{header}\n{diff_text}\n[[/cm:diff]]"

    @staticmethod
    def _primary_handle(content: str) -> Optional[str]:
        """Handle of the FULL content behind a stored- or diff-stub, else None."""
        if not isinstance(content, str):
            return None
        m = _PRIMARY_HANDLE_RE.match(content.lstrip())
        return m.group("handle") if m is not None else None

    @staticmethod
    def _unified_diff(base: str, new: str) -> str:
        """Deterministic, lossless line diff base->new (file-header lines dropped
        for compactness)."""
        lines = difflib.unified_diff(base.splitlines(), new.splitlines(), lineterm="", n=1)
        return "\n".join(ln for ln in lines if not ln.startswith(("--- ", "+++ ")))

    def _maybe_diff_stub(self, handle: str, role: str, tokens: int, content: str,
                         recent_stubs: list) -> Optional[str]:
        """Return a diff-stub if a recent same-role stored note is similar enough
        AND the diff comes out smaller than a normal stub; else None (caller falls
        back to ``make_stub``). Deterministic: depends only on prior messages, so it
        preserves the per-message prefix-stability invariant under append."""
        if self.config.diff_min_similarity <= 0.0:
            return None
        # SIZE GUARD (critical): difflib.SequenceMatcher is O(n*m) and pathological on
        # large, repetitive content (log files), and it runs synchronously on the request
        # path — without this cap a single bulky read can freeze the proxy for minutes.
        # Above the cap, fall back to a normal stub (still lossless, just no delta).
        cap = self.config.diff_max_chars
        if cap and len(content) > cap:
            return None
        # Most-recent `diff_lookback` same-role stubs, newest first.
        candidates = [h for (h, r) in recent_stubs if r == role][-self.config.diff_lookback:]
        best_handle: Optional[str] = None
        best_base: Optional[str] = None
        best_ratio = -1.0
        for h in reversed(candidates):
            if h == handle:
                continue
            try:
                base = self.store.get(h)
            except Exception:
                continue
            if cap and len(base) > cap:
                continue  # same O(n*m) guard for an oversized base note
            # See _similarity(): rapidfuzz Indel when available (~981x faster on
            # real 20 KB store notes), exact difflib fallback otherwise. Both
            # agree to 3-4 decimals on near-duplicates, which is the only regime
            # where diff_min_similarity actually decides anything.
            ratio = _similarity(base, content)
            if ratio > best_ratio:
                best_ratio, best_handle, best_base = ratio, h, base
        if best_handle is None or best_base is None or best_ratio < self.config.diff_min_similarity:
            return None
        diff_stub = self.make_diff_stub(
            handle, best_handle, role, tokens, self._unified_diff(best_base, content)
        )
        normal = self.make_stub(handle, role, tokens, content, self.config.stub_preview_chars)
        return diff_stub if len(diff_stub) < len(normal) else None

    # ------------------------------------------------- dynamic context size
    def update_context_size(self, n_ctx: Optional[int],
                            handle_threshold_tokens: Optional[int]) -> bool:
        """Adopt a new upstream context size WITHOUT losing sticky state.

        ``n_ctx`` is probed from llama-server ``/props`` and every setpoint is
        derived from it (handle threshold, high/low/emergency water). It used to
        be sampled ONCE at startup and never again, so:

          - llama-server down when the proxy started -> n_ctx None forever ->
            the threshold silently falls back to the static default AND windowing
            is disabled entirely (``high_water`` None), with nothing saying so;
          - somebody restarts llama-server with a different ``-c`` -> the proxy
            keeps sizing itself to a window that no longer exists. If n_ctx
            SHRANK, the high water now sits above the real limit.

        Rebuilding the rewriter to adopt a new size is not an option: it would
        drop every windowing frontier and frozen recall block, breaking the
        prefix of every live conversation at once. So the size is updated in
        place and all per-conversation state is preserved.

        Returns True iff anything actually changed.
        """
        changed = False
        if n_ctx and int(n_ctx) != (self._n_ctx or 0):
            self._n_ctx = int(n_ctx)
            changed = True
        if (handle_threshold_tokens
                and int(handle_threshold_tokens)
                != self.config.handle_threshold_tokens):
            self.config = _dc_replace(
                self.config, handle_threshold_tokens=int(handle_threshold_tokens)
            )
            changed = True
        return changed

    def _pinned_threshold(self, conv_key: str) -> int:
        """The handle-ization threshold this conversation was FIRST seen with.

        Applying a NEW threshold to a conversation already in flight re-cuts
        every stub decision already made, in whichever direction it moved:

            threshold DOWN -> messages sent verbatim become stubs
            threshold UP   -> messages sent as stubs come back VERBATIM

        The second is un-handle-ization — the regression this codebase spent
        2026-08-02 eliminating — and both are a prefix break on every live
        conversation simultaneously, i.e. exactly the storm a naive n_ctx
        refresh would cause on every llama-server restart.

        So the threshold is pinned per conversation at first sight and never
        moves. A new n_ctx applies to conversations that START after it, which
        is the only moment at which changing it is free.
        """
        t = self._threshold.get(conv_key)
        if t is None:
            t = self.config.handle_threshold_tokens
            self._threshold[conv_key] = t
            self._lru_touch(self._threshold, conv_key)
        return t

    def _toolcall_threshold(self) -> int:
        """Current setpoint for stubbing a tool_call ARGUMENT value.

            max( toolcall_min_shrink_ratio * stub_tokens , ratio * n_ctx )

        Both terms are derived; neither is a tuned constant.

        The RATIO term is the same anchoring every other setpoint uses, so the
        governor self-sizes to whatever window the server actually runs. It is
        much lower than the per-message ratio because this mass behaves
        differently: tool_call arguments are invisible to Pass 3 windowing, so
        they accumulate for the life of the conversation and are never shed.
        Measured 2026-08-03 on the opencode capture, they reached 43% of the
        peak prompt while the message threshold fired on none of them.

        The FLOOR term is not a preference — it is the break-even of the
        operation. A stub costs ~137 tokens to render, so stubbing anything
        smaller makes the wire BIGGER. Without it, 0.004 * 8192 would set the
        threshold at 33 tokens and every fire would be a net loss. Expressed
        against `stub_tokens_estimate` so it tracks the stub format rather than
        drifting from it, the same discipline as `window_min_shrink_ratio`.

        With the defaults the floor dominates below ~68K of context and the
        ratio takes over above it, which is the intended behaviour at both ends.
        """
        stub_tokens = self.stub_tokens_estimate(self.config.stub_preview_chars)
        floor = int(self.config.toolcall_min_shrink_ratio * stub_tokens)
        anchored = 0
        ratio = self.config.toolcall_threshold_ratio
        if ratio > 0 and self._n_ctx:
            anchored = int(ratio * self._n_ctx)
        return max(1, floor, anchored)

    def _recall_stale_bound(self) -> int:
        """Growth (in estimated tokens) the frozen recall block tolerates before
        it is rebuilt, anchored to the real window.

        A FIXED bound is three different policies on three servers: 4000 tokens
        is 2% of a 200K window (rebuild almost every turn), 6% of 65K, and 50%
        of 8K (never rebuild). Only the middle case was ever measured, and there
        it forced ~12 rebuilds on a conversation growing to 46-49K — 6 of which
        landed off-epoch and cost a prefix break. Every other setpoint in this
        file is a fraction of n_ctx; this one was the exception.

        NOT pinned per conversation, unlike the two handle-ization thresholds.
        Those decide whether a message is a stub, so moving one re-cuts bytes
        already on the wire. This one only decides WHEN to refresh a block, and
        a refresh is a normal, already-handled event — it re-freezes at the new
        tail rather than rewriting history.

        `recall_max_stale_tokens = 0` means the operator selected legacy
        per-turn recompute; a ratio must not silently re-enable stickiness they
        turned off, so that answer is returned unchanged.
        """
        fixed = self.config.recall_max_stale_tokens
        if fixed <= 0:
            return 0
        ratio = self.config.recall_max_stale_ratio
        if ratio > 0 and self._n_ctx:
            return max(1, int(ratio * self._n_ctx))
        return fixed

    def _pinned_toolcall_threshold(self, conv_key: str) -> int:
        """`_toolcall_threshold` frozen at the conversation's first sight.

        Same hazard as `_pinned_threshold`: n_ctx can move under a live
        conversation, and a LOWER tool_call threshold would stub arguments that
        have already gone out verbatim — a simultaneous own-mutation on every
        message carrying one. Pinning confines a new setpoint to conversations
        that start after it.
        """
        t = self._tc_threshold.get(conv_key)
        if t is None:
            t = self._toolcall_threshold()
            self._tc_threshold[conv_key] = t
            self._lru_touch(self._tc_threshold, conv_key)
        return t

    def _handleize_content_parts(self, parts, role: str,
                                 handle_ized_ids: list,
                                 threshold: Optional[int] = None) -> Optional[list]:
        """Stub oversized TEXT parts inside an OpenAI content-parts list.

        THE GAP THIS CLOSES (measured 2026-08-02). Pass 1 handle-izes only
        ``isinstance(content, str)`` and ``_window_out`` skips anything that is
        not a str, so a harness sending tool results as content-parts is
        structurally invisible to the governor. One opencode run measured:

            structured_content  66.3%    <- no code path
            tools               17.6%    <- harness-owned, resent every turn
            string_content      12.9%    <- the only thing we could reach

        88 of 191 messages were handle-ized and the wire still climbed to 47 K,
        because two thirds of it was in a shape we passed through untouched. The
        windowing latch was not misbehaving — it correctly saw unsheddable mass
        and stopped breaking the prefix for no gain.

        SHAPE IS PRESERVED. The list stays a list and every part keeps its keys;
        only an oversized ``text`` value is replaced by the ordinary stub string.
        Flattening a parts array to a plain-string stub would be smaller, but
        harnesses validate tool-result shapes, and a reshaped message is a
        different message — a prefix break by construction, every turn.

        NON-TEXT PARTS ARE NEVER TOUCHED. ``image_url`` (and anything whose
        ``type`` is not ``"text"``) passes through by identity: stubbing an image
        part would silently destroy multimodal input, which no token saving
        justifies.

        Deliberately NOT delta-compressed (``_maybe_diff_stub``): that base
        depends on the order of earlier stubs, and one new variable at a time is
        how this stays debuggable.

        PURE: returns a NEW list only when something changed, sharing every
        untouched part object; None when nothing did, so the caller keeps the
        original message and the wire stays byte-identical.
        IDEMPOTENT: a part whose text is already one of our markers is skipped,
        so ``rewrite(rewrite(x)) == rewrite(x)``.
        """
        if not isinstance(parts, list):
            return None
        if threshold is None:
            threshold = self.config.handle_threshold_tokens
        out: list = []
        changed = False
        for part in parts:
            if not isinstance(part, dict) or part.get("type") != "text":
                out.append(part)          # image_url & friends: identity
                continue
            text = part.get("text")
            if not isinstance(text, str):
                out.append(part)
                continue
            if (self.is_stub(text) or self.is_diff_stub(text)
                    or self.is_rehydrated(text)):
                out.append(part)          # already ours -> byte-identical
                continue
            tokens = self._count_for_handleization(text, threshold)
            if tokens is None or tokens < threshold:
                out.append(part)
                continue
            mid = self.stable_id(role, text)
            handle = self.store.page_out(   # idempotent: same id -> same handle
                Message(role=role, content=text, id=mid)
            )
            new_part = dict(part)
            new_part["text"] = self.make_stub(
                handle, role, tokens, text, self.config.stub_preview_chars
            )
            out.append(new_part)
            handle_ized_ids.append(mid)
            changed = True
        return out if changed else None

    def _count_for_handleization(self, content: str,
                                 threshold: Optional[int] = None) -> Optional[int]:
        """Token count for the handle-ization decision, AVOIDING a /tokenize round-trip on
        the easy cases:
          - tokens <= chars ALWAYS, so content shorter than the threshold (in chars) can
            never reach the threshold in tokens -> return None (not bulky), no tokenize;
          - content larger than ``tokenize_max_chars`` is bulky-by-size and too big to POST
            to /tokenize (slow + a DoS risk) -> return a char-based ESTIMATE (clamped to at
            least the threshold so it is always handle-ized), no tokenize;
          - otherwise one exact count.
        Returning None means "below threshold, leave it alone".
        """
        n = len(content)
        if threshold is None:
            threshold = self.config.handle_threshold_tokens
        if n < threshold:
            return None
        cap = self.config.tokenize_max_chars
        if cap and n > cap:
            return max(threshold, n // _EST_CHARS_PER_TOKEN)
        return self.counter.count_text(content)

    def _handleize_tool_calls(self, tool_calls, handle_ized_ids: list,
                              threshold: int) -> Optional[list]:
        """Pass 1 extension: page out large STRING values found inside
        ``tool_calls[].function.arguments``.

        Returns a NEW list only when something was actually stubbed; ``None``
        means "leave tool_calls exactly as received" — re-serializing an
        UNCHANGED dict through ``json.dumps`` can still reorder/reformat bytes
        the client sent, which would break prefix stability for no reason, so
        the untouched original object is always what gets forwarded when there
        is nothing to compress.

        WHY THIS EXISTS (measured 2026-07-28, live wire capture): on one real
        request, ``tool_calls`` accounted for 116,736 of 222,177 total wire
        chars (53%) — an agentic turn dominated by a large write_file / diff /
        shell-output ARGUMENT — while Pass 1 handle-ization only ever looked at
        ``content``, and Pass 3 windowing (below) only ever SHEDS ``content``.
        That mass was invisible and unsheddable to the whole rewriter: pressure
        climbed to 96% of n_ctx even after windowing fired 4 times, because
        there was nothing left it was ALLOWED to touch.

        `threshold` is the PINNED per-conversation tool_call setpoint, NOT
        `handle_threshold_tokens`. It used to be the latter, which was wrong
        twice over. Sizing: measured 2026-08-03, the per-message threshold
        caught the handful of giants (10 stubs on the opencode capture — the
        ledger's claim that this never fires was itself wrong) while missing a
        mid-tail worth ~15,400 net tokens. Pinning: it read the LIVE config
        value, so an n_ctx change mid-conversation re-cut arguments already sent
        verbatim — the hazard `_pinned_threshold` exists to prevent, which this
        path was silently exempt from.

        SAFETY (this is why it edits values in place rather than replacing the
        whole field): the OpenAI wire format for ``function.arguments`` is a
        JSON-encoded STRING, and llama-server's chat template parses it into a
        mapping before rendering (`tool_call.arguments|items` requires a dict —
        verified against the live template). Replacing ``arguments`` with an
        arbitrary non-JSON stub string would make that parse fail — a hard
        request error, not just a quality regression. So this only ever edits
        STRING VALUES inside the already-decoded object and re-encodes the
        SAME object shape back to a JSON string. Any shape that does not match
        exactly — not a list, a non-dict entry, a missing/malformed
        ``function``, ``arguments`` that is not a string, or that does not
        decode to a JSON object — is left completely untouched (fail open),
        the same posture as `normalize_volatile_stamps`.

        Deterministic and idempotent by construction: `_count_for_handleization`
        and `make_stub` are pure functions of content, `store.page_out` is
        idempotent (same id -> same handle), and a value that is ALREADY a
        stub is short enough to fall back under the threshold on the next
        pass, so re-running this on already-transformed tool_calls is a no-op.

        KNOWN GAP (not fixed here, low severity): stub markers embedded inside
        an argument value are not scanned by Pass 2 auto-rehydration or the
        Pass 4 on-wire-handle filter (`_ONWIRE_HANDLE_RE`), both of which only
        ever look at top-level `content`. Worst case is recall re-surfacing
        content that is already present here in stub form — a mild budget
        inefficiency, not a correctness or prefix-stability issue.
        """
        if not isinstance(tool_calls, list):
            return None
        out: Optional[list] = None
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if not isinstance(args, str):
                continue
            try:
                parsed = json.loads(args)
            except (TypeError, ValueError):
                continue
            if not isinstance(parsed, dict):
                continue
            new_parsed: Optional[dict] = None
            for key, value in parsed.items():
                if not isinstance(value, str):
                    continue
                tokens = self._count_for_handleization(value, threshold)
                if tokens is None or tokens < threshold:
                    continue
                # Namespaced by arg name so a large "content" value and a large
                # "diff" value never collide even if (improbably) byte-identical.
                role = f"toolcall-arg:{key}"
                mid = self.stable_id(role, value)
                handle = self.store.page_out(  # idempotent: same id -> same handle
                    Message(role=role, content=value, id=mid)
                )
                stub = self.make_stub(
                    handle, role, tokens, value, self.config.stub_preview_chars
                )
                if new_parsed is None:
                    new_parsed = dict(parsed)
                new_parsed[key] = stub
                handle_ized_ids.append(mid)
            if new_parsed is None:
                continue
            # Compact separators (matches sensing.canonical_content's convention):
            # every byte here is wire cost we are trying to reduce.
            new_args = json.dumps(new_parsed, ensure_ascii=False,
                                  separators=(",", ":"))
            if out is None:
                out = list(tool_calls)
            new_tc = dict(tc)
            new_fn = dict(fn)
            new_fn["arguments"] = new_args
            new_tc["function"] = new_fn
            out[i] = new_tc
        return out

    def _append_with_tool_calls(self, rewritten: list, out_msg, msg,
                                handle_ized_ids: list, tc_threshold: int) -> None:
        """Append `out_msg` (Pass 1's role/content resolution for `msg`) to
        `rewritten`, first transforming any large tool_calls argument strings.
        `out_msg` keeps its ORIGINAL tool_calls untouched when nothing needed
        stubbing (see `_handleize_tool_calls`); non-dict `out_msg`/`msg` (the
        defensive non-dict passthrough case) are left alone entirely.

        Gated OFF by default since 2026-08-03 (`handleize_toolcall_args`): a stub
        placed in the model's own prior output becomes a template it imitates,
        and the markers end up in the next shell command. See the config docstring
        for the measurement."""
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if (tool_calls is not None and isinstance(out_msg, dict)
                and self.config.handleize_toolcall_args):
            new_tc = self._handleize_tool_calls(
                tool_calls, handle_ized_ids, tc_threshold
            )
            if new_tc is not None:
                out_msg["tool_calls"] = new_tc
        rewritten.append(out_msg)

    # -------------------------------------------------------------- rewrite
    def rewrite_outgoing(self, messages: list[dict], *,
                         prefix_broken: bool = False,
                         pressure_tokens: Optional[int] = None,
                         high_water_tokens: Optional[int] = None) -> RewriteResult:
        """Rewrite an OpenAI `messages` array (spec §3.4), deterministically.

        Phase 14 closed-loop inputs (all optional; omitted = open-loop legacy):
          - ``prefix_broken``: the request-diff classifier says this turn's
            prefix is ALREADY broken (harness edit, new conversation,
            multimodal turn) — a free flush epoch: pending voluntary mutations
            (the stale recall block) ride the break instead of causing one.
          - ``pressure_tokens``: REAL context pressure for this conversation
            (from measured usage.prompt_tokens + estimated growth), replacing
            the string-only chars/4 estimate that was blind to ~88% of the wire
            (tools array, content-parts, tool_calls, template overhead).
          - ``high_water_tokens``: the controller's effective windowing high
            water (``min(ratio*n_ctx, ceiling_safety*learned_ceiling)``),
            moved only at flush epochs.
        The rewriter itself stays a PURE function of (messages, these
        setpoints, frozen state) — same inputs, same bytes out.

        For each message, in order:
          1. If already a stub (`is_stub(content)`) -> leave as-is.
          2. Else if `content` is a str whose token count is >=
             `config.handle_threshold_tokens` -> page it out via the
             `DurableStore` (id = `stable_id`; `Message(role, content, id)`) and
             replace `content` with `make_stub(...)`; record the id in
             `handle_ized_ids`.
          3. Else pass through unchanged. If `content` is not a plain string
             (e.g. the OpenAI content-parts list) it is passed through
             untouched (never handle-ized).

        After the handle-ization pass, auto-rehydrate: scan the rewritten
        messages for explicit `[[cm:stored handle=H]]` references in NON-stub
        message content; for each referenced handle present in the store and
        not already expanded this call, append a synthetic message
        `{"role": "user", "content": "[[cm:rehydrated handle=H]]\\n<full>" }`
        (role "user" — strict chat templates reject mid-conversation "system")
        immediately AFTER the referencing message, subject to a running token
        budget of `config.rehydrate_budget_tokens` (truncate the last synthetic
        message via `counter.truncate_to_tokens` to fit; never exceed the
        budget). Unknown handles are skipped silently. Stubs themselves do NOT
        trigger auto-rehydration — only explicit references in non-stub
        content do.
        """
        handle_ized_ids: list[str] = []
        rewritten: list[dict] = []
        # (handle, role) of stored-/diff-stubs seen so far, for diff-base lookup.
        # Built in order, so a message's diff base depends only on EARLIER messages
        # (preserves the §3.5 per-message prefix-stability invariant under append).
        recent_stubs: list[tuple[str, str]] = []

        # ---- Pass 0: strip any previous auto-recall block (Phase 10) ----
        # Recall is recomputed fresh each call (the conversation's locality shifts
        # every turn), so a block from a prior pass over this list is dropped BEFORE
        # anything else runs. This is what makes Pass 4 non-accumulating: at most one
        # recall block ever exists, so rewrite(rewrite(x)) cannot grow.
        messages = [
            m for m in messages
            if not (isinstance(m, dict) and self.is_recall(m.get("content")))
        ]

        # Conversation identity (Phase 14b): all sticky state (windowing
        # frontier, frozen recall block) is keyed per conversation, so
        # interleaved conversations (main chat + title/summary side-calls)
        # can no longer thrash one global slot. Computed AFTER the recall
        # strip so a rewrite of our own output maps to the same key.
        conv_key = conversation_key(messages)
        # Refresh recency for ALL of this conversation's sticky state BEFORE any
        # pass reads it. Reuse must count as use, or the steady state evicts
        # itself and un-handle-izes messages it already paged out (see
        # _touch_conversation).
        self._touch_conversation(conv_key)
        # Threshold PINNED to this conversation: n_ctx (and therefore the
        # threshold) can move under a live conversation, and re-cutting its stub
        # decisions mid-flight breaks the prefix in either direction.
        threshold = self._pinned_threshold(conv_key)
        # tool_call ARGUMENTS get their own, much lower setpoint — pinned for the
        # same reason, and until 2026-08-03 not pinned at all.
        tc_threshold = self._pinned_toolcall_threshold(conv_key)

        # ---- Pass 1: handle-ization (per-message deterministic) ----
        for msg in messages:
            # Preserve the message as a shallow copy so we never mutate input.
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content") if isinstance(msg, dict) else None

            if isinstance(content, str) and (self.is_stub(content) or self.is_diff_stub(content)):
                # Already a stub (stored or diff): leave byte-identical, but remember
                # its full-content handle as a candidate diff base for later messages.
                h = self._primary_handle(content)
                if h is not None:
                    recent_stubs.append((h, role))
                self._append_with_tool_calls(rewritten, dict(msg), msg,
                                             handle_ized_ids, tc_threshold)
                continue

            if isinstance(content, str) and self.is_rehydrated(content):
                # Already a synthetic rehydrated message from a prior turn:
                # already-rewritten output. Pass through UNCHANGED (never
                # handle-ize, never page out) — §9.1 idempotency fix.
                self._append_with_tool_calls(rewritten, dict(msg), msg,
                                             handle_ized_ids, tc_threshold)
                continue

            if isinstance(content, str):
                # ONE token measurement, and skip the /tokenize round-trip entirely on the
                # easy/dangerous cases (tiny content can't be bulky; huge content is
                # bulky-by-size and unsafe to tokenize). None => below threshold.
                tokens = self._count_for_handleization(content, threshold)
                if tokens is not None and tokens >= threshold:
                    mid = self.stable_id(role, content)
                    handle = self.store.page_out(  # idempotent: same id -> same handle
                        Message(role=role, content=content, id=mid)
                    )
                    # Delta-compress against a recent near-duplicate of the same role if
                    # one exists and the diff is smaller; else a normal head/tail stub.
                    stub = self._maybe_diff_stub(handle, role, tokens, content, recent_stubs)
                    if stub is None:
                        stub = self.make_stub(
                            handle, role, tokens, content, self.config.stub_preview_chars
                        )
                    self._append_with_tool_calls(
                        rewritten, {"role": role, "content": stub}, msg,
                        handle_ized_ids, tc_threshold
                    )
                    handle_ized_ids.append(mid)
                    recent_stubs.append((handle, role))
                    continue

            # Content-parts (opt-in): stub oversized TEXT parts in place, shape
            # preserved, image parts untouched. Falls through to the plain
            # passthrough below when nothing crossed the threshold, so an
            # unchanged parts message keeps its ORIGINAL object and stays
            # byte-identical on the wire.
            if self.config.handleize_content_parts and isinstance(content, list):
                new_parts = self._handleize_content_parts(
                    content, role, handle_ized_ids, threshold
                )
                if new_parts is not None:
                    out_msg = dict(msg)
                    out_msg["content"] = new_parts
                    self._append_with_tool_calls(
                        rewritten, out_msg, msg, handle_ized_ids, tc_threshold
                    )
                    continue

            # Pass through unchanged (covers non-string content too). This is
            # the common path for a tool-call turn: an assistant message with
            # content=None and tool_calls=[...] never matches any of the
            # isinstance(content, str) branches above.
            out_msg = dict(msg) if isinstance(msg, dict) else msg
            self._append_with_tool_calls(rewritten, out_msg, msg, handle_ized_ids,
                                         tc_threshold)

        # ---- Pass 2: auto-rehydration of explicit references ----
        rehydrated_handles: list[str] = []
        if self.config.rehydrate_budget_tokens > 0:
            budget = self.config.rehydrate_budget_tokens
            used = 0
            out: list[dict] = []

            # §9.1 #3: collect handles already expanded as rehydrated markers
            # in the working message list (from prior turns) so we never append
            # a duplicate synthetic message for a handle that is already
            # present inline. This is the idempotency fix for Pass 2.
            already_rehydrated: set[str] = set()
            for msg in rewritten:
                c = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(c, str) and self.is_rehydrated(c):
                    h = self._parse_rehydrated_handle(c)
                    if h is not None:
                        already_rehydrated.add(h)

            for msg in rewritten:
                out.append(msg)
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, str):
                    continue
                if self.is_stub(content):
                    # Stubs do not auto-rehydrate.
                    continue
                handles = self.parse_handles(content)
                for h in handles:
                    # Skip handles we already rehydrated earlier this call or
                    # that are already present as rehydrated markers.
                    if h in rehydrated_handles or h in already_rehydrated:
                        continue
                    remaining = budget - used
                    # M2: stop the loop only when the budget is exhausted.
                    if remaining <= 0:
                        break
                    # Load the full content from the store. Unknown/missing
                    # handle -> skip silently, no crash.
                    try:
                        full = self.store.get(h)
                    except Exception:
                        continue
                    if full is None:
                        continue
                    # L2: the `[[cm:rehydrated handle=H]]` marker line is NEVER
                    # truncated — only the `<full>` body is. This keeps the
                    # marker intact so the message stays detectable next turn.
                    marker = f"[[cm:rehydrated handle={h}]]\n"
                    marker_tokens = self.counter.count_text(marker)
                    # M2: if the marker alone already exceeds the remaining
                    # budget, this handle does not fit — try the next one
                    # (continue) rather than aborting the whole loop.
                    if marker_tokens >= remaining:
                        continue
                    body_budget = remaining - marker_tokens
                    body_tokens = self.counter.count_text(full)
                    if body_tokens <= body_budget:
                        body = full
                    else:
                        body = self.counter.truncate_to_tokens(full, body_budget)
                    synth = marker + body
                    count = self.counter.count_text(synth)
                    # Defensive: never exceed the budget and never emit empty.
                    if count <= 0 or count > remaining:
                        continue
                    # Role "user", NOT "system" (learned live 2026-07-01): strict
                    # chat templates (Qwen: "System message must be at the
                    # beginning") reject a system message anywhere but index 0.
                    # "user" is the only role every template accepts mid-wire;
                    # the [[cm:...]] marker keeps the content distinguishable.
                    out.append({"role": "user", "content": synth})
                    used += count
                    rehydrated_handles.append(h)
                    already_rehydrated.add(h)
                    if used >= budget:
                        break
            rewritten = out

        # ---- Pass 3: total-budget windowing (LOSSLESS, three-water hysteresis) ----
        # Bound the TOTAL wire below context_budget_ratio * n_ctx (MID water = the
        # trigger AND the ceiling) by paging out the OLDEST non-pinned middle
        # messages (head + recent tail kept verbatim; paged content becomes a
        # retrievable stub -> lossless). Phase 11: because the CLI resends the
        # ORIGINAL transcript every turn, a stateless check would re-trigger and
        # advance the stub frontier EVERY turn — invalidating the upstream's KV
        # prefix each time. So the frontier is STICKY (3a): previously-windowed
        # messages are re-stubbed byte-identically from cached counts, and paging
        # only ADVANCES (3b) when the wire crosses the mid water — then it cuts
        # deep to the LOW water (context_target_ratio) in one bite. Between
        # triggers the wire prefix is byte-stable, so the KV cache is reused and
        # only the tail (+ recall block) is re-processed.
        #
        # THIRD TIER (2026-07-28, opt-in via context_emergency_ratio, 0 = off):
        # the HIGH water. A trigger that cannot reach the low water LATCHES
        # (below) rather than re-breaking the prefix for nothing — but a live
        # session showed pressure climb to 96% of n_ctx while latched, because
        # the hysteresis-gap re-arm didn't fire fast enough against unsheddable
        # mass (large tool_calls arguments — see the 2026-07-28 Pass 1 fix; the
        # HIGH water is a general safety net, not specific to that one cause).
        # Crossing the HIGH water OVERRIDES the latch: a shed is attempted every
        # request regardless, on the reasoning that a failed retry costs no more
        # than the first trigger already did (see `emergency_water` below).
        windowing_triggered = False
        # 14b: did windowing actually CHANGE BYTES this call (page at least one
        # previously-verbatim message)? A high-water crossing that sheds
        # nothing breaks nothing — and therefore has no break for a recall
        # flush to ride (see Pass 4). This is the predicate the flush policy
        # consumes; windowing_triggered stays the crossing signal for metrics.
        windowing_shed = False
        # True iff the HIGH-water override actually FIRED — i.e. the latch
        # WOULD have suppressed this turn's trigger and didn't only because
        # pressure had crossed emergency_water. Distinct from a plain trigger
        # (which needs no override) so /metrics can show whether the third
        # tier is doing anything, not just whether it is configured.
        windowing_emergency = False
        high_water: Optional[int] = high_water_tokens
        if high_water is None and self._n_ctx and self.config.context_budget_ratio > 0.0:
            high_water = int(self._n_ctx * self.config.context_budget_ratio)
        if high_water is not None and high_water > 0:
            budget = self.config.context_budget_ratio
            target = self.config.context_target_ratio
            low_water = (int(self._n_ctx * target)
                         if (self._n_ctx and target > 0.0) else high_water)
            if budget > 0.0 and target > 0.0:
                # Scale the low water with a (possibly learned-lowered) high
                # water so the hysteresis gap survives 14c calibration.
                low_water = min(low_water, int(high_water * target / budget))
            if low_water >= high_water:
                low_water = high_water  # collapsed/disabled gap -> legacy per-turn line
            gap = high_water - low_water
            # HIGH water (third tier): only meaningful when set ABOVE the mid
            # water (config validates this at construction) — None means "no
            # emergency tier", preserving today's behavior exactly.
            emergency_water: Optional[int] = None
            if self._n_ctx and self.config.context_emergency_ratio > 0.0:
                emergency_water = int(self._n_ctx * self.config.context_emergency_ratio)
            tail_start = len(rewritten) - self.config.protect_last_n
            windowed = self._windowed.get(conv_key)

            # Pass 3a: re-apply the sticky frontier (per conversation). Cached
            # counts -> the stub text is byte-identical to the turn it was first
            # windowed; no tokenizer calls (a notes.has stat guards the rare
            # re-page_out after a wipe).
            if windowed and self.config.protect_first_n < tail_start:
                for i in range(self.config.protect_first_n, tail_start):
                    msg = rewritten[i]
                    content = msg.get("content") if isinstance(msg, dict) else None
                    role = msg.get("role", "") if isinstance(msg, dict) else ""
                    if (
                        not isinstance(content, str)
                        or self.is_stub(content)
                        or self.is_diff_stub(content)
                        or self.is_rehydrated(content)
                    ):
                        continue
                    mid = self.stable_id(role, content)
                    tokens = windowed.get(mid)
                    if tokens is None:
                        continue
                    handle = self.store.notes.handle_for(mid)
                    if not self.store.notes.has(handle):
                        self.store.page_out(Message(role=role, content=content, id=mid))
                    rewritten[i] = {
                        "role": role,
                        "content": self.make_stub(handle, role, tokens, content, 0),
                    }
                    handle_ized_ids.append(mid)

            # Pass 3b: trigger at HIGH water, page down to LOW water.
            if pressure_tokens is not None:
                # 14c: REAL pressure (measured usage.prompt_tokens + growth) —
                # sees the tools array, content-parts, and template overhead the
                # chars/4 estimate was blind to. The shed target is expressed in
                # real tokens too: page until the visible savings cover the
                # overshoot (or the middle runs out — the rest is invisible mass
                # only content-visibility work can reach).
                if pressure_tokens <= high_water:
                    # Below the trigger the crisis is over: re-arm the latch so
                    # the next crossing is tried fresh.
                    self._window_latched.pop(conv_key, None)
                elif self.config.protect_first_n < tail_start:
                    latched_at = self._window_latched.get(conv_key)
                    would_latch = (latched_at is not None and gap > 0
                                  and pressure_tokens < latched_at + gap)
                    emergency = (emergency_water is not None
                                and pressure_tokens >= emergency_water)
                    if would_latch and not emergency:
                        # LATCHED: the last trigger could not reach low water
                        # (unsheddable mass) — re-firing now would break the
                        # prefix for nothing. Hold the wire byte-stable until
                        # pressure grows by the hysteresis gap — UNLESS pressure
                        # has crossed the HIGH water, in which case the latch is
                        # overridden below regardless of the gap.
                        pass
                    else:
                        if would_latch and emergency:
                            windowing_emergency = True
                        windowing_triggered = True
                        if windowed is None:
                            windowed = self._windowed.setdefault(conv_key, {})
                        self._lru_touch(self._windowed, conv_key)
                        target_shed = pressure_tokens - low_water
                        shed = 0
                        i = self.config.protect_first_n
                        while shed < target_shed and i < tail_start:
                            shed += self._window_out(rewritten, i, windowed,
                                                     handle_ized_ids, None)
                            i += 1
                        windowing_shed = shed > 0
                        if gap > 0:
                            if shed < target_shed:
                                # Missed low water: unsheddable mass — latch.
                                self._window_latched[conv_key] = pressure_tokens
                                self._lru_touch(self._window_latched, conv_key)
                            else:
                                self._window_latched.pop(conv_key, None)
            else:
                # Legacy open-loop gate: estimate tokens as chars/4 to avoid
                # per-message tokenizer calls on the common under-trigger case;
                # only go precise when plausibly over.
                est_chars = sum(
                    len(m["content"]) for m in rewritten
                    if isinstance(m, dict) and isinstance(m.get("content"), str)
                )
                if est_chars / 4 > high_water and self.config.protect_first_n < tail_start:
                    # Count each message ONCE (count_text may be a network call),
                    # then maintain a running total as messages are paged out.
                    counts = [
                        self.counter.count_text(m["content"])
                        if isinstance(m, dict) and isinstance(m.get("content"), str) else 0
                        for m in rewritten
                    ]
                    total = sum(counts)
                    if total > high_water:
                        latched_at = self._window_latched.get(conv_key)
                        would_latch = (latched_at is not None and gap > 0
                                      and total < latched_at + gap)
                        emergency = (emergency_water is not None
                                    and total >= emergency_water)
                        if would_latch and not emergency:
                            pass  # latched (see the real-pressure branch above)
                        else:
                            if would_latch and emergency:
                                windowing_emergency = True
                            windowing_triggered = True
                            if windowed is None:
                                windowed = self._windowed.setdefault(conv_key, {})
                            self._lru_touch(self._windowed, conv_key)
                            total_before = total
                            i = self.config.protect_first_n
                            while total > low_water and i < tail_start:
                                total -= self._window_out(rewritten, i, windowed,
                                                          handle_ized_ids, counts[i])
                                i += 1
                            windowing_shed = total < total_before
                            if gap > 0:
                                if total > low_water:
                                    self._window_latched[conv_key] = total_before
                                    self._lru_touch(self._window_latched, conv_key)
                                else:
                                    self._window_latched.pop(conv_key, None)
                    else:
                        # Measured under the high water after all: re-arm.
                        self._window_latched.pop(conv_key, None)

        # ---- Pass 4: auto-recall (anticipatory demand paging, Phase 10) ----
        # The live run proved agents do not ask for their memory back
        # (messages_rehydrated: 0) — and a model cannot page-fault on content it
        # cannot see. So the governor recalls FOR it: derive an implicit query from
        # the live tail (locality), search the store, and inject the top slices of
        # OFF-wire memory as one budgeted, clearly-marked user message right
        # before the final message. Recall flows through store.search(), so
        # retrieval metrics and hotness warming come free — recall feeds the
        # working-set signal that drives eviction.
        #
        # STICKY (Phase 12): a fresh block at a tail-tracking insertion point is a
        # guaranteed per-turn prefix break (measured live: divergence at a constant
        # ~recall_budget_tokens before the previous prompt end, every turn — fatal
        # for hybrid-SSM caches that need byte-exact pure extension). So the block
        # is FROZEN once built and re-injected byte-identically before the same
        # anchor message each turn; it is recomputed — in one jump, at the new
        # tail — only on a refresh trigger: enough growth since the freeze, the
        # anchor gone (host CLI compacted), or a Pass-3 windowing pass that
        # actually paged a message (that turn re-prefills anyway, so the
        # refresh rides for free). Sticky turns
        # skip store.search() (no hotness warming) — the refresh cadence warms
        # instead.
        recalled_handles: list[str] = []
        if (self.config.auto_recall_k > 0
                and self.config.recall_budget_tokens > 0 and rewritten):
            stale_bound = self._recall_stale_bound()
            # FLUSH EPOCH (Phase 14b): the prefix is already broken this turn —
            # by the harness (classifier verdict passed in as prefix_broken) or
            # by Pass 3 ACTUALLY PAGING a message (windowing_shed — that turn
            # re-prefills anyway, so the refresh rides the break for free).
            # A bare high-water crossing that shed nothing breaks nothing, so a
            # flush then would CREATE the very break it exists to ride — the
            # self-fulfilling re-prefill loop of 2026-07-19. Otherwise the
            # block stays frozen until the HARD staleness bound (growth past
            # recall_max_stale_tokens, or anchor loss).
            flush = prefix_broken or windowing_shed
            frozen = self._recall_frozen.get(conv_key)
            reused = False
            if stale_bound > 0 and not flush and frozen is not None:
                anchor_mid, anchor_index, block = frozen
                idx = self._find_recall_anchor(rewritten, anchor_mid, anchor_index)
                if idx is not None:
                    # canonical_content: structured (content-parts) growth counts
                    # too — previously invisible, so image tails never aged the
                    # block.
                    grown_chars = sum(
                        len(canonical_content(m.get("content")))
                        for m in rewritten[idx + 1:] if isinstance(m, dict)
                    )
                    if grown_chars / _EST_CHARS_PER_TOKEN < stale_bound:
                        rewritten.insert(idx, {"role": "user", "content": block})
                        self._lru_touch(self._recall_frozen, conv_key)
                        reused = True
            if not reused:
                try:
                    block, recalled_handles = self._build_recall_block(rewritten)
                except Exception:
                    # Recall is ENRICHMENT, never a dependency: a store/tokenizer
                    # hiccup degrades to "no recall this turn" — it must never fail
                    # the request. (Learned live 2026-07-01: a /tokenize parse error
                    # inside the recall builder 500'd every chat completion.)
                    block, recalled_handles = None, []
                # Whatever happens below, this conversation's previous frozen
                # block is dead: its flush/staleness trigger fired (or it never
                # existed).
                self._recall_frozen.pop(conv_key, None)
                if block is not None:
                    insert_at = len(rewritten) - 1 if len(rewritten) >= 2 else len(rewritten)
                    # Role "user", NOT "system": strict templates reject mid-wire
                    # system messages (same live lesson as Pass 2 above).
                    rewritten.insert(insert_at, {"role": "user", "content": block})
                    if stale_bound > 0 and insert_at + 1 < len(rewritten):
                        anchor = rewritten[insert_at + 1]
                        # stable_id_any: non-str anchors (content-parts) freeze
                        # too — the Phase-12 skip made every image turn a
                        # guaranteed prefix break.
                        # insert_at is the anchor's index in a BLOCK-FREE wire
                        # (the block now sits at insert_at, the anchor shifted
                        # to insert_at + 1) — stored so the anchor's OCCURRENCE
                        # can be pinned next turn even when its content is
                        # duplicated earlier on the wire.
                        self._recall_frozen[conv_key] = (
                            self.stable_id_any(anchor.get("role", ""),
                                               anchor.get("content")),
                            insert_at,
                            block,
                        )
                        self._lru_touch(self._recall_frozen, conv_key)

        return RewriteResult(
            messages=rewritten,
            handle_ized_ids=handle_ized_ids,
            rehydrated_handles=rehydrated_handles,
            recalled_handles=recalled_handles,
            windowing_triggered=windowing_triggered,
            windowing_emergency=windowing_emergency,
        )

    # ----------------------------------------------------------- windowing
    def _window_out(self, rewritten: list, i: int, windowed: dict,
                    handle_ized_ids: list, orig_tokens: Optional[int]) -> int:
        """Page message ``i`` out as a minimal (no-preview) stub if eligible;
        returns the tokens SAVED (0 when skipped: non-str content, already a
        stub/rehydrated, or the stub would not be smaller). ``orig_tokens``
        None -> count lazily (the real-pressure path counts only what it pages;
        the legacy path passes its precomputed count)."""
        msg = rewritten[i]
        content = msg.get("content") if isinstance(msg, dict) else None
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        if (
            not isinstance(content, str)
            or self.is_stub(content)
            or self.is_diff_stub(content)
            or self.is_rehydrated(content)
        ):
            return 0
        if orig_tokens is None:
            orig_tokens = self.counter.count_text(content)
        mid = self.stable_id(role, content)
        handle = self.store.notes.handle_for(mid)
        # Minimal stub (no preview) -> the archive marker is tiny.
        minimal = self.make_stub(handle, role, orig_tokens, content, 0)
        stub_tokens = self.counter.count_text(minimal)
        if stub_tokens >= orig_tokens:  # never bloat a tiny old message
            return 0
        # ...and never take a MARGINAL trade either. The break-even test above
        # models ONLY tokens. Paging also breaks the upstream prefix (a full
        # re-prefill of the whole prompt) and hides the content from the model —
        # costs that test cannot see, so at break-even they are pure loss.
        # Measured 2026-08-02: 27 messages under 100 tokens were paged for a net
        # 39 tokens each, five of them assistant turns (the model's own prior
        # reasoning). Require a real shrink, not merely a non-negative one.
        ratio = self.config.window_min_shrink_ratio
        if ratio > 0 and orig_tokens < ratio * stub_tokens:
            return 0
        self.store.page_out(Message(role=role, content=content, id=mid))
        # Frontier remembers the ORIGINAL count so Pass 3a can regenerate this
        # exact stub on every later turn.
        windowed[mid] = orig_tokens
        rewritten[i] = {"role": role, "content": minimal}
        handle_ized_ids.append(mid)
        return orig_tokens - stub_tokens

    # ---------------------------------------------------------- auto-recall
    def _anchor_matches(self, msg, anchor_mid: str, anchor_handle: str) -> bool:
        """True iff `msg` IS the frozen anchor: by identity (`stable_id` equals
        `anchor_mid` — byte-identical content, the common case; non-str
        content-parts match via `stable_id_any`, 14b), or — if Pass 1/3 stubbed
        it since the freeze — by its stub's primary handle."""
        if not isinstance(msg, dict):
            return False
        content = msg.get("content")
        if self.stable_id_any(msg.get("role", ""), content) == anchor_mid:
            return True
        return (isinstance(content, str)
                and (self.is_stub(content) or self.is_diff_stub(content))
                and self._primary_handle(content) == anchor_handle)

    def _find_recall_anchor(self, messages: list[dict], anchor_mid: str,
                            anchor_index: Optional[int] = None) -> Optional[int]:
        """Index of the message the frozen recall block precedes, or None if it
        left the transcript (host-CLI compaction/edit -> refresh).

        The occurrence AT `anchor_index` (its position in the block-free wire
        at freeze time) wins first: under pure append an anchor's index never
        moves, so an exact-index match keeps the block at its last-sent
        position even when the anchor's CONTENT is duplicated earlier on the
        wire — a repeated boilerplate message used to relocate the block
        BACKWARD to the first match, a voluntary prefix break per flush epoch.
        When the frozen index no longer matches (mid-wire insertions/edits
        shifted it — a turn whose prefix the harness already broke), fall back
        to the first match, which is stable under append."""
        anchor_handle = self.store.notes.handle_for(anchor_mid)
        if anchor_index is not None and 0 <= anchor_index < len(messages):
            if self._anchor_matches(messages[anchor_index], anchor_mid, anchor_handle):
                return anchor_index
        for i, msg in enumerate(messages):
            if self._anchor_matches(msg, anchor_mid, anchor_handle):
                return i
        return None

    def _on_wire_handles(self, messages: list[dict]) -> set:
        """Every handle the current wire already carries — as a stored-/diff-stub,
        a diff ``base=`` reference, a rehydrated marker, OR as verbatim content
        (a message that WOULD map to that handle if paged out). Recall must never
        duplicate what the model can already see."""
        on_wire: set = set()
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, str):
                continue
            for m in _ONWIRE_HANDLE_RE.finditer(content):
                on_wire.add(m.group("h"))
            for m in _DIFF_BASE_RE.finditer(content):
                on_wire.add(m.group("h"))
            if not (self.is_stub(content) or self.is_diff_stub(content)
                    or self.is_rehydrated(content)):
                # Verbatim message: its content lives on the wire even if a copy
                # was paged out in some earlier turn — computing its would-be
                # handle is one local sha1, no I/O.
                role = msg.get("role", "")
                on_wire.add(self.store.notes.handle_for(self.stable_id(role, content)))
        return on_wire

    def _build_recall_block(self, messages: list[dict]) -> tuple[Optional[str], list[str]]:
        """Assemble the Pass-4 recall block: implicit tail query -> store search ->
        off-wire filter -> near-duplicate suppression -> budgeted assembly (the
        marker lines are never truncated, only slice bodies — same discipline as
        Pass 2). Returns ``(block, handles)`` or ``(None, [])`` when there is
        nothing worth recalling (empty/trivial query, no hits, everything already
        on the wire, or nothing fits the budget). Deterministic given the message
        list and the store's state."""
        query = extract_query(messages)
        if not query:
            return None, []
        k = self.config.auto_recall_k
        try:
            slices = self.store.search(query, k=k * 3)  # pool for the filters below
        except Exception:
            return None, []
        if not slices:
            return None, []
        on_wire = self._on_wire_handles(messages)
        candidates = [s for s in slices if s.handle not in on_wire]
        if not candidates:
            return None, []
        keep = select_diverse([s.content for s in candidates])
        candidates = [candidates[i] for i in keep][:k]

        header = "[[cm:recall]]"
        footer = "[[/cm:recall]]"
        budget = self.config.recall_budget_tokens
        used = self.counter.count_text(f"{header}\n{footer}")
        parts: list[str] = [header]
        picked: list[str] = []
        for sl in candidates:
            marker = f"[[cm:recalled handle={sl.handle}]]\n"
            marker_tokens = self.counter.count_text(marker)
            remaining = budget - used
            if marker_tokens >= remaining:
                continue
            body_budget = remaining - marker_tokens
            body_tokens = self.counter.count_text(sl.content)
            body = (sl.content if body_tokens <= body_budget
                    else self.counter.truncate_to_tokens(sl.content, body_budget))
            piece = marker + body
            count = self.counter.count_text(piece)
            if count <= 0 or count > remaining:
                continue
            parts.append(piece)
            picked.append(sl.handle)
            used += count
            if used >= budget:
                break
        if not picked:
            return None, []
        parts.append(footer)
        return "\n".join(parts), picked
