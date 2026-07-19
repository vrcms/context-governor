"""LoopGuard — mechanical loop-breaker for degenerate agent cycles (Phase 13).

Grounded in a real livelock (2026-07-19, opencode + Qwen3.6, ~40 wasted turns): a
checklist step the agent could not verify with its tools ("check if the render is
visible") produced >= 9 consecutive turns that were byte-identical (same tool call,
same result, same reply; context growing by exactly +329 tokens/turn), 13 turns of
llama-server draft acceptance = 1.0 (the whole output drafted from existing context
= verbatim recycling), and 530 reads of the same file. The harness auto-continue
prompt never stopped it. Full anatomy:
C:\\tools\\llama-cpp-turboquant\\wiki\\hybrid-ssm-prompt-cache-misses.md.

Detection (two signals, both configurable):
  1. CONTENT (primary, model/server-agnostic): each completed turn — the trailing
     segment of the incoming transcript from the LAST assistant message to the end
     (assistant reply + tool calls + tool results) — is normalized (whitespace
     collapsed, volatile substrings such as timestamps/UUIDs/call-ids stripped) and
     hashed. ``repeat_k`` consecutive equal fingerprints = a degenerate cycle.
  2. SERVER TIMINGS (opportunistic corroboration, never required): llama-server
     responses carry ``timings.draft_n`` / ``timings.draft_n_accepted`` when
     speculative decoding is on. Acceptance >= ``accept_threshold`` with
     ``draft_n > draft_n_min`` for ``timings_m`` consecutive turns indicates
     verbatim recycling and ACCELERATES the content trigger by one turn. With
     spec decoding off the guard works unchanged on signal 1 alone.

Action on trigger: the caller APPENDS a breaker notice at the TAIL of the next
outbound request. Tail-append only — NEVER a mid-history edit: the downstream
hybrid-SSM prompt cache needs a byte-stable prefix (f_keep = 1.0); a mid-history
mutation forces a 30-60 s full re-prefill.

Escalation: first trigger -> notice; if the cycle survives the cooldown, a second
trigger -> one stronger FINAL notice; further triggers repeat the final notice —
unless ``hard_stop`` (OFF by default) is set, in which case the caller is told to
end the run with a synthetic final response.

Like the rewriter's sticky frontier, the state is in-memory and per-proxy-lifetime
(one live session per governor); a restart just re-arms the guard.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Optional


# Marker prefix for everything the guard emits, consistent with the [[cm:...]]
# marker family the governor already puts on the wire.
BREAKER_MARKER = "[[cm:loop-breaker]]"

# Volatile substrings stripped before hashing so "near-identical" turns (same
# action, fresh ids) still collide: ISO timestamps, bare clock times, UUIDs,
# long hex runs (call ids, commit-ish ids), long digit runs (epochs, counters).
_VOLATILE_RES = [
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    re.compile(r"\b[0-9a-fA-F]{8,}\b"),
    re.compile(r"\b\d{10,}\b"),
]
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class LoopGuardConfig:
    """Thresholds for the loop-breaker. All tunable; ``enabled=False`` turns the
    whole feature off (observe/decide become no-ops).

    Attributes:
        enabled: master switch.
        repeat_k: consecutive identical turn fingerprints that trigger the
            breaker (>= 2 — one occurrence is never a "repeat").
        timings_m: consecutive verbatim-recycling timings observations needed
            before the content trigger is accelerated by one turn.
        accept_threshold: draft acceptance ratio (accepted/draft_n) at or above
            which a response counts as verbatim recycling.
        draft_n_min: minimum ``draft_n`` for a timings observation to count
            (tiny drafts accept fully all the time — not a loop signal).
        cooldown_turns: after an injection, suppress re-injection for this many
            subsequent (still-degenerate) turns.
        hard_stop: when the cycle survives TWO injected notices, end the run
            with a synthetic final response instead of injecting again.
            OFF by default.
    """

    enabled: bool = True
    repeat_k: int = 3
    timings_m: int = 3
    accept_threshold: float = 0.99
    draft_n_min: int = 200
    cooldown_turns: int = 3
    hard_stop: bool = False

    def __post_init__(self) -> None:
        if self.repeat_k < 2:
            raise ValueError(f"repeat_k must be >= 2, got {self.repeat_k}")
        if self.timings_m < 1:
            raise ValueError(f"timings_m must be >= 1, got {self.timings_m}")
        if not (0.0 < self.accept_threshold <= 1.0):
            raise ValueError(
                f"accept_threshold must be in (0.0, 1.0], got {self.accept_threshold}"
            )
        if self.draft_n_min < 0:
            raise ValueError(f"draft_n_min must be >= 0, got {self.draft_n_min}")
        if self.cooldown_turns < 0:
            raise ValueError(f"cooldown_turns must be >= 0, got {self.cooldown_turns}")


@dataclass(frozen=True)
class LoopGuardDecision:
    """What the caller must do with the CURRENT request.

    ``breaker_text`` is the notice to APPEND (tail-only!) as one extra message;
    None when ``hard_stop`` is set instead. ``level``: 1 = first notice,
    2 = final notice, 3 = hard stop. ``streak`` = consecutive identical turns
    observed so far (for the notice text and for logging).
    """

    breaker_text: Optional[str]
    hard_stop: bool
    level: int
    streak: int


# --------------------------------------------------------------- fingerprinting

def normalize_for_fingerprint(text: str) -> str:
    """Whitespace-collapse + volatile-substring stripping. Two renderings of the
    same action (fresh call ids, new timestamps) normalize identically."""
    for pat in _VOLATILE_RES:
        text = pat.sub("§", text)
    return _WS_RE.sub(" ", text).strip()


def _serialize_message(msg: dict) -> str:
    """Stable serialization of one message for hashing. Volatile STRUCTURE is
    dropped here (tool-call ``id`` / ``tool_call_id`` — regenerated per turn by
    most servers even when the action is byte-identical); volatile TEXT is
    handled by ``normalize_for_fingerprint`` afterwards."""
    role = msg.get("role", "")
    content = msg.get("content")
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True, default=str)
    parts = [role, content]
    name = msg.get("name")
    if isinstance(name, str):
        parts.append(name)
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        stripped = [
            {k: v for k, v in call.items() if k != "id"} if isinstance(call, dict) else call
            for call in tool_calls
        ]
        parts.append(json.dumps(stripped, sort_keys=True, default=str))
    return "\x1f".join(parts)


def turn_fingerprint(messages: list) -> Optional[tuple[str, tuple[int, int]]]:
    """Fingerprint of the LAST COMPLETED TURN in an incoming transcript: the
    trailing segment from the last assistant message to the end (assistant reply
    + tool calls + tool results / follow-up user msg).

    Returns ``(sha1_hex, anchor)`` where ``anchor = (last_assistant_index,
    len(messages))`` identifies the turn's POSITION — an unchanged anchor with an
    unchanged fingerprint means the same request re-delivered (client retry),
    not a new repeated turn. Returns None when the transcript has no assistant
    message yet (nothing completed to fingerprint).
    """
    last_assistant = None
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_assistant = i
    if last_assistant is None:
        return None
    serialized = "\x1e".join(
        _serialize_message(m) if isinstance(m, dict) else repr(m)
        for m in messages[last_assistant:]
    )
    normalized = normalize_for_fingerprint(serialized)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest, (last_assistant, len(messages))


# ------------------------------------------------------------- breaker notices

def breaker_notice(level: int, streak: int) -> str:
    """The notice text for injection ``level`` (1 = first, 2 = final)."""
    if level <= 1:
        return (
            f"{BREAKER_MARKER} You have repeated the same action {streak} times "
            "with identical results. Do not repeat it again. If the remaining "
            "step requires capabilities you lack (e.g., visual verification of "
            "rendered output), state that explicitly, summarize what was "
            "completed, and stop."
        )
    return (
        f"{BREAKER_MARKER} FINAL NOTICE: you have now repeated the same action "
        f"{streak} times with identical results, and an earlier notice did not "
        "change your behavior. You MUST NOT issue this action again. Choose one, "
        "now: (a) take a genuinely different action, or (b) state explicitly "
        "which capability you lack, summarize what was completed, and stop."
    )


def hard_stop_text(streak: int) -> str:
    """Assistant-content of the synthetic final response that ends the run."""
    return (
        f"{BREAKER_MARKER} Run ended by the context governor: the same action "
        f"was repeated {streak} times with identical results and two breaker "
        "notices did not stop the loop. The remaining step appears to require "
        "capabilities this agent lacks; it should be completed or verified "
        "manually. Review the transcript above for what was completed."
    )


# ---------------------------------------------------------------- the guard

class LoopGuard:
    """Trigger/cooldown/escalation state machine over turn fingerprints.

    Call ``observe_request(messages)`` once per outbound chat request with the
    ORIGINAL (pre-rewrite) transcript; it returns a :class:`LoopGuardDecision`
    when the caller must act, else None. Feed upstream responses back via
    ``observe_response`` (parsed JSON) or ``observe_stream_chunk`` (raw SSE
    bytes) so the timings signal can corroborate.

    Never mutates the transcript it observes. Single-session state, same
    lifetime model as the rewriter's sticky windowing frontier.
    """

    def __init__(self, config: LoopGuardConfig) -> None:
        self.config = config
        self._last_fp: Optional[str] = None
        self._last_anchor: Optional[tuple[int, int]] = None
        self._streak: int = 0
        self._timings_streak: int = 0
        self._cooldown: int = 0
        # Injection level reached for the CURRENT cycle: 0 none, 1 notice,
        # 2 final notice. Resets when the fingerprint changes (loop broken).
        self._level: int = 0

    # -------------------------------------------------------------- observe

    def observe_request(self, messages: list) -> Optional[LoopGuardDecision]:
        """Advance the state machine with the latest transcript; decide."""
        if not self.config.enabled:
            return None
        fingerprinted = turn_fingerprint(messages)
        if fingerprinted is None:
            return None
        fp, anchor = fingerprinted

        if fp == self._last_fp and anchor == self._last_anchor:
            # Same request re-delivered (client retry/timeout) — NOT a new turn.
            return None
        if fp == self._last_fp:
            self._streak += 1
        else:
            # The cycle broke: full re-arm (streak, cooldown, escalation).
            self._streak = 1
            self._cooldown = 0
            self._level = 0
        self._last_fp, self._last_anchor = fp, anchor

        effective_k = self.config.repeat_k
        if self._timings_streak >= self.config.timings_m and effective_k > 2:
            effective_k -= 1  # verbatim-recycling corroboration: fire one turn earlier

        if self._streak < effective_k:
            return None
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if self._level >= 2:
            if self.config.hard_stop:
                return LoopGuardDecision(
                    breaker_text=None, hard_stop=True, level=3, streak=self._streak
                )
            # Escalation is capped at the final notice; repeat it each cooldown.
            self._cooldown = self.config.cooldown_turns
            return LoopGuardDecision(
                breaker_text=breaker_notice(2, self._streak),
                hard_stop=False, level=2, streak=self._streak,
            )

        self._level += 1
        self._cooldown = self.config.cooldown_turns
        return LoopGuardDecision(
            breaker_text=breaker_notice(self._level, self._streak),
            hard_stop=False, level=self._level, streak=self._streak,
        )

    # -------------------------------------------------------------- timings

    def observe_response(self, data: object) -> None:
        """Feed one upstream response (parsed JSON). Reads llama-server's
        ``timings.draft_n`` / ``timings.draft_n_accepted`` when present; any
        response without a qualifying verbatim-recycling signature resets the
        timings streak (conservative — stale corroboration never lingers)."""
        if not self.config.enabled:
            return
        timings = data.get("timings") if isinstance(data, dict) else None
        recycled = False
        if isinstance(timings, dict):
            draft_n = timings.get("draft_n")
            accepted = timings.get("draft_n_accepted")
            if (
                isinstance(draft_n, (int, float)) and isinstance(accepted, (int, float))
                and draft_n > self.config.draft_n_min
                and accepted / draft_n >= self.config.accept_threshold
            ):
                recycled = True
        self._timings_streak = self._timings_streak + 1 if recycled else 0

    def observe_stream_chunk(self, chunk: bytes) -> None:
        """Opportunistic timings scan of a passthrough SSE chunk. llama-server
        puts ``timings`` in the final data event; anything unparseable (event
        split across chunks, non-JSON) is silently skipped — this signal is
        best-effort by design and the stream itself is NEVER altered."""
        if not self.config.enabled or b'"timings"' not in chunk:
            return
        for line in chunk.split(b"\n"):
            if not line.startswith(b"data:") or b'"timings"' not in line:
                continue
            try:
                self.observe_response(json.loads(line[5:].strip()))
            except Exception:
                continue
