"""Sensing + closed-loop control (Phase 14) — the governor's afferent path.

Phase 14a diagnosis: the proxy ran OPEN-LOOP — it rewrote the wire and never
observed consequences. The rewriter is string-only, so ~88% of the real prompt
mass (the ``tools`` array, content-parts, ``tool_calls`` payloads, template
overhead) was invisible to its chars/4 estimates; windowing could never trigger
and the harness's own lossy compaction fired first. This module closes the loop
with three pieces, all of them PURE TEES over data the proxy already carries:

  1. Wire signatures + a request-diff classifier: every message is reduced to a
     boundary hash (role + canonical content + tool_calls — O(n) hashes, no
     full-text diffs, no tokenizer calls); comparing this turn's signature to
     the last one classifies the turn as {new-conversation | pure-append |
     tail-edit | mid-wire-edit | head-rewrite} and attributes the cause.
     A head-rewrite is the native-compaction signature (the task-7220 flood).
  2. A response tee: ``usage.prompt_tokens`` / ``usage.completion_tokens`` and
     llama-server's ``timings`` block are parsed from non-stream JSON bodies and
     from the final SSE data chunk. Forwarded bytes stay byte-identical; parse
     failures are swallowed (the Phase-10 lesson: enrichment must never fail a
     request).
  3. A per-conversation ledger (LRU-bounded, in-memory) + ``GovernorController``:
     real prompt sizes, reuse ratios, per-turn growth, break attribution — and
     the Phase-14c self-calibration: windowing pressure derived from REAL
     ``usage.prompt_tokens`` and a learned native-compaction ceiling (persisted
     in the contextstore's StateStore, keyed by harness fingerprint) so the
     governor windows (controlled, lossless) BEFORE the harness floods.
     Calibration is hardened against false samples: learning requires the wire
     to SHRINK (an early-message edit is not a compaction), cross-key matches
     are confirmed temporally (a continuing new key) and retracted when the old
     key stays alive, and any established conversation observed PAST the
     learned ceiling uncompacted raises it (self-healing in both directions).

Everything here is stdlib-only and side-effect-free with respect to the wire.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

# Rough chars-per-token, consistent with the rewriter's estimate. Estimates are
# only ever used for GROWTH deltas here — absolute sizes come from real usage.
_EST_CHARS_PER_TOKEN = 4

# How many chars of the first message identify the harness (system-prompt head).
_HARNESS_FP_CHARS = 2048

# A head-rewrite only feeds ceiling learning when the wire SHRANK to at most
# this fraction of its previous message count. Native compaction replaces the
# bulk with a summary (drastic shrink); an early-message EDIT (a refreshed
# session-state block) keeps the length and must not pose as a compaction.
_HEAD_REWRITE_SHRINK = 0.7

# Bounded per-conversation history (reuse ratios, growth) — enough for trend
# reading on /metrics without unbounded growth.
_HISTORY_LEN = 32

# Classifier kinds (wire-shape) and causes (attribution).
KIND_NEW = "new-conversation"
KIND_APPEND = "pure-append"
KIND_TAIL_EDIT = "tail-edit"
KIND_MID_EDIT = "mid-wire-edit"
KIND_HEAD_REWRITE = "head-rewrite"

CAUSE_HARNESS = "harness-edit"
CAUSE_OWN = "own-mutation"
CAUSE_MULTIMODAL = "multimodal"
CAUSE_NEW = "new-conversation"
CAUSE_UNKNOWN = "unknown"


# --------------------------------------------------------------- canonical ids

def canonical_content(content) -> str:
    """ONE serialization of message content for hashing/identity, str OR
    structured (the OpenAI content-parts list). Strings pass through unchanged —
    so every id derived through here is byte-compatible with the historical
    str-only ``stable_id`` — and structured content gets a deterministic
    compact-JSON rendering (sorted keys). Never raises."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(content)


def message_hash(msg) -> str:
    """Boundary hash of one message: role + canonical content + tool_calls.
    ``tool_calls`` participates because assistant tool-call messages often carry
    ``content: null`` — without the payload every such message would collide and
    the classifier would go blind exactly where agent wires are densest."""
    if not isinstance(msg, dict):
        blob = repr(msg)
        return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:16]
    role = msg.get("role", "")
    content = canonical_content(msg.get("content"))
    tool_calls = ""
    if msg.get("tool_calls") is not None:
        tool_calls = canonical_content(msg.get("tool_calls"))
    digest = hashlib.sha1(
        (role + "\x00" + content + "\x00" + tool_calls).encode("utf-8", "replace")
    ).hexdigest()
    return digest[:16]


def wire_signature(messages: list) -> list:
    """Rolling per-message boundary hashes of a wire — O(n), no full-text diffs."""
    return [message_hash(m) for m in messages]


def conversation_key(messages: list) -> str:
    """Stable identity of a conversation: the first message's hash combined with
    the first NON-system message's hash. The first message alone (the spec's
    minimum) collides when interleaved conversations share one harness system
    prompt — the main chat vs. title/summary side-calls of the 2026-07-19
    shakedown — so the first user turn is folded in to separate them. Both
    components are stable under append for the conversation's whole life."""
    if not messages:
        return "conv-empty"
    first = message_hash(messages[0])
    second = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") != "system":
            second = message_hash(m)
            break
    return "conv-" + first + second


def harness_fingerprint(messages: list) -> str:
    """Identity of the HARNESS (not the conversation): a hash of the system-prompt
    head. All conversations run by the same harness share it, so the learned
    native-compaction ceiling (14c) transfers across sessions and restarts."""
    if not messages:
        return "hp-empty"
    head = canonical_content(
        messages[0].get("content") if isinstance(messages[0], dict) else messages[0]
    )[:_HARNESS_FP_CHARS]
    return "hp-" + hashlib.sha1(head.encode("utf-8", "replace")).hexdigest()[:16]


def _growth_estimate(messages: list, kind: str, pos: int, chars: int,
                     prev_chars: int, threshold: Optional[int],
                     stub_est: Optional[int]) -> int:
    """Estimated tokens this turn ADDS to the wire that is actually SENT.

    THE BUG THIS FIXES (measured 2026-07-28). The estimate used to be a flat
    ``(chars - prev_chars) / 4`` over the INCOMING wire. But the incoming wire is
    not what reaches llama-server: Pass 1 pages out every message at or above the
    handle threshold and sends a ~50-token stub in its place. On one live turn a
    60 KB file read therefore contributed ~15,000 phantom tokens:

        est_pressure 37,666   vs   real prompt 21,259

    That pushed pressure past the high water, Pass 3 windowed, and windowing
    sheds messages from ``protect_first_n`` forward — breaking the prefix and
    forcing a full re-prefill. Two such spurious triggers were observed live
    (``windowing_triggers: 2`` matching ``own-mutation: 2``); the 14b latch
    suppressed three more.

    THE FIX is to estimate each appended message at what it will COST ON THE
    WIRE: content at or above the threshold becomes a stub, so it counts as
    ``stub_est``, not as its own size. Everything else counts at chars/4 as
    before. ``tool_calls`` payloads are never stubbed, so they always count in
    full.

    Note this is still a cheap PRE-GATE, not a precise count — it deliberately
    makes no tokenizer calls. Residual error is on the order of tens of tokens
    per turn (stub-size approximation, and Pass 2 rehydration, which is bounded
    by ``rehydrate_budget_tokens`` and can add content the incoming wire lacks).
    Verifying against the post-Pass-1 wire before shedding is the precise
    follow-up, deliberately deferred.

    Falls back to the legacy aggregate when the rewriter's settings are unknown,
    or on any non-append turn (tail edits, mid-wire edits, head rewrites), where
    there is no well-defined "appended region" — those turns are rare and usually
    already carry a flush.
    """
    legacy = max(0, (chars - prev_chars) // _EST_CHARS_PER_TOKEN)
    if threshold is None or stub_est is None or kind != KIND_APPEND:
        return legacy
    est = 0
    for m in messages[pos:]:
        if not isinstance(m, dict):
            continue
        body = len(canonical_content(m.get("content"))) // _EST_CHARS_PER_TOKEN
        est += stub_est if body >= threshold else body
        if m.get("tool_calls") is not None:
            est += (len(canonical_content(m.get("tool_calls")))
                    // _EST_CHARS_PER_TOKEN)
    return max(0, est)


# ---------------------------------------------------------------- classifier

def classify_wire_diff(prev: Optional[list], new: list) -> tuple:
    """Classify this turn's wire against the previous one, on boundary hashes.

    Returns ``(kind, pos)`` where pos is the first differing message index:
      - no previous wire                          -> (new-conversation, 0)
      - previous is a prefix of new (or equal)    -> (pure-append, len(prev))
      - divergence in the first QUARTER of prev   -> (head-rewrite, i) — the
        native-compaction signature: the system head survives, everything above
        an early token is rewritten (task 7220: 21K re-prefilled in one turn)
      - divergence within the last 2 messages     -> (tail-edit, i) — a retry /
        regenerated tail
      - anything else                             -> (mid-wire-edit, i)
    """
    if not prev:
        return KIND_NEW, 0
    n_common = 0
    for a, b in zip(prev, new):
        if a != b:
            break
        n_common += 1
    if n_common == len(prev):
        return KIND_APPEND, len(prev)
    i = n_common
    if 4 * i < len(prev):
        return KIND_HEAD_REWRITE, i
    if i >= len(prev) - 2:
        return KIND_TAIL_EDIT, i
    return KIND_MID_EDIT, i


def _has_structured_content(messages: list) -> bool:
    """True iff any message carries non-string content (content-parts — the
    multimodal/expected-break signature)."""
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content")
            if c is not None and not isinstance(c, str):
                return True
    return False


# ------------------------------------------------------------- response tee

def extract_usage_timings(obj) -> Optional[dict]:
    """Pull ``{"usage": .., "timings": ..}`` out of a parsed response object.
    Returns None when neither block is present (or the input is not a dict) —
    the caller treats None as "nothing observed this turn". Never raises."""
    if not isinstance(obj, dict):
        return None
    out: dict = {}
    usage = obj.get("usage")
    if isinstance(usage, dict):
        out["usage"] = usage
    timings = obj.get("timings")
    if isinstance(timings, dict):
        out["timings"] = timings
    return out or None


def extract_final_sse_json(buf: bytes) -> Optional[dict]:
    """Parse the LAST ``data: {...}`` event carrying usage/timings out of an SSE
    byte tail. llama-server puts usage+timings on the final content chunk before
    ``data: [DONE]``. Malformed lines are skipped; never raises."""
    try:
        text = buf.decode("utf-8", errors="replace")
    except Exception:
        return None
    result: Optional[dict] = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        found = extract_usage_timings(obj)
        if found is not None:
            result = found
    return result


class StreamTee:
    """Bounded tail-buffer accumulator over a forwarded SSE stream.

    A PURE tee: ``feed`` copies bytes into a rolling tail (default 64 KB — ample
    for the final usage/timings chunk) and never raises, so the forwarding path
    cannot be altered, delayed, or failed by sensing. ``result`` parses the tail
    once, after the stream completes."""

    def __init__(self, max_tail: int = 65536) -> None:
        self._tail = b""
        self._max = max(1024, int(max_tail))

    def feed(self, chunk) -> None:
        try:
            if isinstance(chunk, (bytes, bytearray)):
                self._tail = (self._tail + bytes(chunk))[-self._max:]
        except Exception:
            pass

    def result(self) -> Optional[dict]:
        return extract_final_sse_json(self._tail)


# ------------------------------------------------------------------- ledger

@dataclass
class ConvEntry:
    """Per-conversation rolling state (in-memory, like the windowing frontier:
    its useful lifetime equals the upstream's prefix-cache lifetime)."""

    incoming_sig: list = field(default_factory=list)   # last INCOMING wire hashes
    sent_sig: list = field(default_factory=list)       # last SENT wire hashes
    incoming_chars: int = 0                            # canonical chars of last incoming wire
    harness_fp: str = ""
    turns: int = 0
    # Real numbers from the response tee (0 = not yet observed).
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_prompt_n: int = 0            # tokens actually (re)processed upstream
    last_prompt_ms: float = 0.0
    last_ttft_ms: float = 0.0         # proxy-measured fallback when timings absent
    reuse_history: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    growth_history: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))


@dataclass
class RequestObservation:
    """What the classifier concluded about ONE incoming request — computed
    BEFORE forwarding, consumed by the rewriter's flush policy (14b) and by
    /metrics attribution."""

    key: str
    harness_fp: str
    kind: str
    pos: int
    cause: str                       # attribution when the prefix is broken
    prefix_broken: bool              # harness edit / new conv / multimodal turn
    pressure_tokens: Optional[int]   # real-usage pressure estimate (14c); None until observed


class GovernorController:
    """The closed loop's state: ledger + break attribution + learned ceilings.

    Thread-safe (one lock around all state). Every public method is designed to
    be called inside a try/except by the app layer — sensing must never fail a
    request — but is also internally defensive."""

    def __init__(self, *, max_conversations: int = 32) -> None:
        self._lock = Lock()
        self._ledger: "OrderedDict[str, ConvEntry]" = OrderedDict()
        self._max = max(1, int(max_conversations))
        self._breaks: Counter = Counter()
        self._native_compactions = 0
        # harness_fp -> {"ceiling": int, "observations": int, "raised": int,
        #                "last_sample": int, "updated_at": float}
        self._profiles: dict = {}
        self._profiles_dirty = False
        # Deferred cross-key ceiling samples: new_conv_key -> {"sample", "old_key",
        # "fp", "at"}. A cross-key head-rewrite is only a SUSPECTED compaction
        # (a side-call sharing the system head has the same wire shape — the
        # false-ceiling incident of 2026-07-19); the sample is promoted to a
        # learned ceiling only when the new key proves to be a CONTINUING
        # conversation (a later pure-append turn), and retracted if the old key
        # turns out to be alive (side-calls coexist with their parent; a true
        # compaction kills it). In-memory only: a pending sample dies with the
        # proxy rather than being persisted unconfirmed.
        self._pending_ceiling: "OrderedDict[str, dict]" = OrderedDict()
        self._last_prompt_tokens = 0
        self._peak_prompt_tokens = 0
        self._last_reuse_ratio: Optional[float] = None
        self._responses_observed = 0
        self._responses_with_timings = 0
        # Own-mutation forensics (2026-08-02): "kind@pos" -> count, plus a
        # bounded ring of the most recent sites. Answers WHICH of our own
        # rewrites broke the prefix and WHERE, so a fix can be aimed instead of
        # guessed from server-side prefill timings.
        self._own_sites: Counter = Counter()
        self._own_recent: deque = deque(maxlen=32)

    # ---------------------------------------------------------- request side
    def _entry(self, key: str) -> ConvEntry:
        entry = self._ledger.get(key)
        if entry is None:
            entry = ConvEntry()
            self._ledger[key] = entry
        self._ledger.move_to_end(key)
        while len(self._ledger) > self._max:
            self._ledger.popitem(last=False)
        return entry

    def observe_request(self, messages: list, *,
                        handle_threshold_tokens: Optional[int] = None,
                        stub_tokens_est: Optional[int] = None,
                        ) -> RequestObservation:
        """Classify the incoming wire against this conversation's last one and
        update the ledger's request-side state. Runs BEFORE forwarding.

        ``handle_threshold_tokens`` / ``stub_tokens_est`` let the 14c pressure
        estimate account for the rewrite that is about to happen — see
        `_growth_estimate`. Both omitted (the default) keeps the legacy aggregate
        estimate, so callers that do not know the rewriter's settings, and the
        existing tests, are unaffected."""
        key = conversation_key(messages)
        sig = wire_signature(messages)
        fp = harness_fingerprint(messages)
        chars = sum(len(canonical_content(m.get("content")))
                    for m in messages if isinstance(m, dict))
        with self._lock:
            entry = self._entry(key)
            prev_sig = entry.incoming_sig
            kind, pos = classify_wire_diff(prev_sig or None, sig)

            # Cross-key compaction detection: native compaction usually REPLACES
            # the first user message (system head + summary + recent tail), which
            # changes the conversation key — the head-rewrite would masquerade as
            # a brand-new conversation and the ceiling would never be learned.
            # Match the shared system head against recent ledger entries; the
            # >=3-message guard filters the genuinely-fresh-chat shape
            # ([system, user]) that would otherwise false-positive.
            # The match only BANKS a deferred sample (see _pending_ceiling): a
            # side-call sharing the system head has the exact same wire shape,
            # so same-turn evidence cannot separate the two — confirmation is
            # temporal, not structural.
            ceiling_source: Optional[ConvEntry] = None
            if kind == KIND_NEW and len(sig) >= 3:
                for other_key in reversed(self._ledger):
                    if other_key == key:
                        continue
                    cand = self._ledger[other_key]
                    if not cand.incoming_sig or cand.last_prompt_tokens <= 0:
                        continue
                    if cand.incoming_sig[0] != sig[0]:
                        continue
                    k2, p2 = classify_wire_diff(cand.incoming_sig, sig)
                    if k2 == KIND_HEAD_REWRITE and self._shrank(sig, cand.incoming_sig):
                        kind, pos = KIND_HEAD_REWRITE, p2
                        self._bank_pending_ceiling(
                            key, sample=cand.last_prompt_tokens,
                            old_key=other_key, fp=cand.harness_fp or fp,
                        )
                        break
            elif kind == KIND_HEAD_REWRITE and entry.last_prompt_tokens > 0:
                # Same-key compaction must SHRINK the wire: an early-message
                # edit (a harness refreshing a session-state block) diverges in
                # the first quarter but keeps the length — it is not a
                # compaction and must not lower the ceiling.
                if self._shrank(sig, prev_sig):
                    ceiling_source = entry

            # Cause attribution. Multimodal content anywhere in/after the diff
            # region marks an expected-break turn (images re-encode; content the
            # rewriter cannot see) — never a signal to re-optimize against.
            diverged_region = messages[pos:] if pos < len(messages) else []
            multimodal = _has_structured_content(diverged_region)
            if kind == KIND_NEW:
                cause = CAUSE_NEW
                prefix_broken = True
            elif kind == KIND_APPEND:
                cause = CAUSE_MULTIMODAL if multimodal else CAUSE_UNKNOWN
                # An appended image message still breaks upstream reuse (the
                # spec treats a multimodal turn as an already-broken prefix).
                prefix_broken = multimodal
            else:
                cause = CAUSE_MULTIMODAL if multimodal else CAUSE_HARNESS
                prefix_broken = True

            # 14c ceiling learning: a head-rewrite by the HARNESS is the native-
            # compaction signature; the real prompt size that preceded it is a
            # sample of the harness's compaction ceiling. Keep the MINIMUM
            # positive sample (conservative: window before the earliest observed
            # flood point). Multimodal-attributed breaks never feed learning.
            if (kind == KIND_HEAD_REWRITE and cause == CAUSE_HARNESS
                    and ceiling_source is not None):
                self._learn_ceiling(ceiling_source.harness_fp or fp,
                                    ceiling_source.last_prompt_tokens)

            # Deferred cross-key samples: CONFIRM when the suspected-compacted
            # conversation proves alive (a pure-append turn at >= its 2nd
            # request); otherwise drop it — a one-turn side-call never
            # confirms. The banking turn itself has entry.turns == 0 and is
            # skipped, so a fresh pending survives to be judged later.
            pend = self._pending_ceiling.get(key)
            if pend is not None and entry.turns >= 1:
                if kind == KIND_APPEND:
                    self._learn_ceiling(pend["fp"], pend["sample"])
                del self._pending_ceiling[key]
            # RETRACTION: the OLD key reappearing with a pure append means it
            # was never compacted — any pending sample reading its head-rewrite
            # as a compaction was a side-call false positive.
            if kind == KIND_APPEND:
                for pk, p in list(self._pending_ceiling.items()):
                    if p["old_key"] == key:
                        del self._pending_ceiling[pk]

            # 14c pressure: this turn's real prompt will be ~ last real prompt
            # + the new growth. Growth is the only estimated term — absolute mass
            # (tools, template, content-parts) is already inside
            # last_prompt_tokens.
            #
            # `last_completion_tokens` is deliberately NOT added: on an append
            # turn the completion is part of the appended region and is already
            # counted there, and on the aggregate fallback it is inside the char
            # delta. Adding it again was a systematic double-count.
            pressure: Optional[int] = None
            if entry.last_prompt_tokens > 0:
                pressure = entry.last_prompt_tokens + _growth_estimate(
                    messages, kind, pos, chars, entry.incoming_chars,
                    handle_threshold_tokens, stub_tokens_est,
                )

            entry.incoming_sig = sig
            entry.incoming_chars = chars
            entry.harness_fp = fp
            entry.turns += 1
        return RequestObservation(
            key=key, harness_fp=fp, kind=kind, pos=pos, cause=cause,
            prefix_broken=prefix_broken, pressure_tokens=pressure,
        )

    def note_sent(self, key: str, sent_messages: list,
                  observation: Optional[RequestObservation] = None) -> None:
        """Record the wire actually FORWARDED and attribute any prefix break.
        A break the incoming wire already carried keeps its incoming cause; a
        break that appeared only after our rewrite is an own-mutation — the
        voluntary kind Phase 14b exists to drive to zero."""
        sig = wire_signature(sent_messages)
        with self._lock:
            entry = self._entry(key)
            kind, pos = classify_wire_diff(entry.sent_sig or None, sig)
            if kind == KIND_NEW:
                self._breaks[CAUSE_NEW] += 1
            elif kind != KIND_APPEND:
                if observation is not None and observation.prefix_broken:
                    self._breaks[observation.cause] += 1
                else:
                    self._breaks[CAUSE_OWN] += 1
                    # WHERE, not just how many. classify_wire_diff already
                    # computes the first divergent message index and it used to
                    # be discarded, so every own-mutation post-mortem had to be
                    # inferred from llama-server logs. A run of breaks at ONE
                    # site is the signature of a specific mutation (a lost
                    # windowing frontier re-cutting at protect_first_n; a recall
                    # block refreshing off-epoch); a scatter is churn. Bounded
                    # ring + histogram, both cheap.
                    site = f"{kind}@{pos}"
                    self._own_sites[site] += 1
                    self._own_recent.append({
                        "conv": key,
                        "kind": kind,
                        "pos": pos,
                        "n_prev": len(entry.sent_sig or ()),
                        "n_sent": len(sig),
                    })
            entry.sent_sig = sig

    # --------------------------------------------------------- response side
    def observe_response(self, key: str, parsed: Optional[dict],
                         ttft_ms: Optional[float] = None) -> None:
        """Fold one response's usage/timings (from the tee) into the ledger.
        ``parsed`` is ``extract_usage_timings``' output; None records only the
        proxy-measured TTFT fallback."""
        with self._lock:
            entry = self._ledger.get(key)
            if entry is None:
                return
            self._ledger.move_to_end(key)
            if ttft_ms is not None:
                entry.last_ttft_ms = float(ttft_ms)
            if not parsed:
                return
            usage = parsed.get("usage") or {}
            timings = parsed.get("timings") or {}
            prompt_tokens = _as_int(usage.get("prompt_tokens"))
            completion_tokens = _as_int(usage.get("completion_tokens"))
            prompt_n = _as_int(timings.get("prompt_n"))
            prompt_ms = _as_float(timings.get("prompt_ms"))
            if prompt_tokens > 0:
                if entry.last_prompt_tokens > 0:
                    entry.growth_history.append(
                        prompt_tokens - entry.last_prompt_tokens
                    )
                entry.last_prompt_tokens = prompt_tokens
                self._last_prompt_tokens = prompt_tokens
                if prompt_tokens > self._peak_prompt_tokens:
                    self._peak_prompt_tokens = prompt_tokens
                # 14c contradiction check: an established conversation sailing
                # PAST the learned ceiling uncompacted falsifies that ceiling
                # (a false-positive head-rewrite sample would otherwise poison
                # the high water forever — the 2026-07-19 incident). RAISE the
                # ceiling to the observed size; a later TRUE head-rewrite
                # re-lowers it via min-sample, so the check is self-healing in
                # both directions. turns >= 2: a fresh conversation's first
                # prompt (e.g. a subagent seeded with bulk context) says
                # nothing about the harness's steady-state compaction point.
                prof = self._profiles.get(entry.harness_fp)
                if (prof is not None and entry.turns >= 2
                        and prompt_tokens > _as_int(prof.get("ceiling"))):
                    prof["ceiling"] = prompt_tokens
                    prof["raised"] = _as_int(prof.get("raised")) + 1
                    prof["updated_at"] = time.time()
                    self._profiles_dirty = True
            if completion_tokens > 0:
                entry.last_completion_tokens = completion_tokens
            self._responses_observed += 1
            if timings:
                self._responses_with_timings += 1
                if prompt_n >= 0 and prompt_tokens > 0:
                    entry.last_prompt_n = prompt_n
                if prompt_ms > 0:
                    entry.last_prompt_ms = prompt_ms
            # PREFIX REUSE, portably (2026-08-02). This whole computation used
            # to sit inside `if timings:` -- and timings is a llama.cpp
            # extension. On vLLM or LM Studio the closed loop recorded NO reuse
            # at all, which silently blinds break attribution, own-mutation
            # forensics and every ratio built on them.
            #
            #   usage.prompt_tokens_details.cached_tokens   OpenAI STANDARD
            #   timings.prompt_n                            llama.cpp only
            #
            # They carry the same information from opposite ends:
            #   reuse = cached_tokens / prompt_tokens
            #   reuse = 1 - prompt_n / prompt_tokens
            # Presence of the field decides, not its value -- a legitimate
            # cached_tokens of 0 is a real observation of zero reuse, and must
            # not fall through to a path that may not exist.
            if prompt_tokens > 0:
                details = usage.get("prompt_tokens_details")
                has_cached = (isinstance(details, dict)
                              and "cached_tokens" in details)
                reuse = None
                if has_cached:
                    cached = _as_int(details.get("cached_tokens"))
                    reuse = cached / prompt_tokens
                    if not timings:
                        # Keep "tokens actually reprocessed" meaningful without
                        # the vendor field it used to come from.
                        entry.last_prompt_n = max(0, prompt_tokens - cached)
                elif timings and prompt_n >= 0:
                    reuse = 1.0 - (prompt_n / prompt_tokens)
                if reuse is not None:
                    reuse = max(0.0, min(1.0, reuse))
                    entry.reuse_history.append(reuse)
                    self._last_reuse_ratio = reuse

    # ------------------------------------------------------- 14c calibration
    @staticmethod
    def _shrank(sig: list, prev_sig: list) -> bool:
        """True iff the wire SHRANK enough to be a plausible compaction (the
        bulk replaced by a summary), not merely an early-message edit."""
        return bool(prev_sig) and len(sig) <= int(_HEAD_REWRITE_SHRINK * len(prev_sig))

    def _bank_pending_ceiling(self, new_key: str, *, sample: int, old_key: str,
                              fp: str) -> None:
        """Park a SUSPECTED cross-key compaction sample for temporal
        confirmation (LRU-bounded like the ledger)."""
        self._pending_ceiling[new_key] = {
            "sample": int(sample), "old_key": old_key, "fp": fp,
            "at": time.time(),
        }
        self._pending_ceiling.move_to_end(new_key)
        while len(self._pending_ceiling) > self._max:
            self._pending_ceiling.popitem(last=False)

    def _learn_ceiling(self, fp_key: str, sample: int) -> None:
        """Min-keep one native-compaction ceiling sample for a harness
        fingerprint (callers hold the lock)."""
        sample = int(sample)
        if sample <= 0:
            return
        prof = self._profiles.get(fp_key)
        if prof is None:
            prof = {"ceiling": sample, "observations": 0, "raised": 0}
        prof["ceiling"] = min(_as_int(prof.get("ceiling")), sample) \
            if _as_int(prof.get("ceiling")) > 0 else sample
        prof["observations"] = _as_int(prof.get("observations")) + 1
        prof["last_sample"] = sample
        prof["updated_at"] = time.time()
        self._profiles[fp_key] = prof
        self._profiles_dirty = True
        self._native_compactions += 1

    def effective_high_water(self, n_ctx: Optional[int], budget_ratio: float,
                             ceiling_safety: float, harness_fp: str) -> Optional[int]:
        """The windowing HIGH water this turn:
        ``min(budget_ratio * n_ctx, ceiling_safety * learned_ceiling)``.
        Falls back to the ratio alone until a ceiling is observed (today's
        behavior), and to the learned ceiling alone when n_ctx is unknown.
        None = windowing stays disabled (nothing known)."""
        base: Optional[int] = None
        if n_ctx and budget_ratio > 0.0:
            base = int(n_ctx * budget_ratio)
        if ceiling_safety <= 0.0:
            return base
        with self._lock:
            prof = self._profiles.get(harness_fp)
        if not prof:
            return base
        learned = int(int(prof.get("ceiling", 0)) * ceiling_safety)
        if learned <= 0:
            return base
        return min(base, learned) if base is not None else learned

    def load_profiles(self, state_store) -> None:
        """Rehydrate learned harness profiles from the contextstore, so a proxy
        restart keeps its calibration. Best-effort."""
        try:
            data = state_store.load().get("harness_profiles")
        except Exception:
            return
        if not isinstance(data, dict):
            return
        with self._lock:
            for fp, prof in data.items():
                if isinstance(fp, str) and isinstance(prof, dict) \
                        and _as_int(prof.get("ceiling")) > 0:
                    self._profiles.setdefault(fp, dict(prof))

    def maybe_persist(self, state_store) -> None:
        """Write profiles back when (and only when) something was learned."""
        with self._lock:
            if not self._profiles_dirty:
                return
            snapshot = {fp: dict(p) for fp, p in self._profiles.items()}
            self._profiles_dirty = False
        try:
            state_store.update({"harness_profiles": snapshot})
        except Exception:
            with self._lock:
                self._profiles_dirty = True

    # -------------------------------------------------------------- /metrics
    def snapshot(self) -> dict:
        """JSON-able closed-loop state for /metrics — the goal: a session like
        the 2026-07-19 shakedown needs ZERO server-log forensics."""
        with self._lock:
            conversations = {}
            for key, e in self._ledger.items():
                reuse = list(e.reuse_history)
                conversations[key] = {
                    "turns": e.turns,
                    "last_prompt_tokens": e.last_prompt_tokens,
                    "last_prompt_n": e.last_prompt_n,
                    "last_reuse_ratio": round(reuse[-1], 4) if reuse else None,
                    "avg_reuse_ratio": (round(sum(reuse) / len(reuse), 4)
                                        if reuse else None),
                    "last_growth_tokens": (e.growth_history[-1]
                                           if e.growth_history else None),
                    "last_ttft_ms": round(e.last_ttft_ms, 1),
                }
            return {
                "real_prompt_tokens": {
                    "last": self._last_prompt_tokens,
                    "peak": self._peak_prompt_tokens,
                },
                "real_reuse_ratio": (round(self._last_reuse_ratio, 4)
                                     if self._last_reuse_ratio is not None else None),
                "breaks_by_cause": dict(self._breaks),
                "own_mutation_sites": dict(self._own_sites),
                "own_mutation_recent": list(self._own_recent),
                "native_compaction_observed": self._native_compactions,
                "pending_ceiling_samples": len(self._pending_ceiling),
                "learned_ceilings": {
                    fp: {"ceiling": _as_int(p.get("ceiling")),
                         "observations": _as_int(p.get("observations")),
                         "raised": _as_int(p.get("raised"))}
                    for fp, p in self._profiles.items()
                },
                "responses_observed": self._responses_observed,
                "responses_with_timings": self._responses_with_timings,
                "conversations": conversations,
            }


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
