"""ContextWindowResolver — one belief about the upstream's usable window.

WHY THIS EXISTS. Every scaling decision in the governor is a fraction of the
upstream context size: the handle-ization threshold, the low/mid/high water
marks, the learned-ceiling setpoint. If that number is wrong, everything
downstream is wrong by the same factor. It used to come from exactly one place —
llama-server's /props, probed once at startup — so a server slow to boot left it
None for the process's life, which does not merely soften the threshold but
disables windowing ENTIRELY.

THE STANDARD DOES NOT CARRY IT. Measured 2026-08-02: an OpenAI chat completion
response carries usage and (on llama.cpp) timings, and no context size at all.
The OpenAI spec has no such field anywhere — not in /v1/models, not in the
completion envelope, not in headers. Every server exposing it does so as a
vendor extension. There is nothing to "handshake" about; the window cannot
simply be read.

IT CAN BE BRACKETED. Two bounds are observable on any provider:

    FLOOR    the largest prompt that SUCCEEDED — proof the window is at least
             that big. Cannot be wrong: an observation, not a claim.
    CEILING  a length the upstream REJECTED — proof the window is smaller.

A NOTE ON "start conservative and learn upward", which does NOT work on its own
and is written down so nobody re-derives it. Once the governor is windowing, IT
decides how large prompts get. A too-low belief windows the wire down, so the
prompts it observes are small, so the floor never rises above the belief that
produced it. The floor is therefore a CONTRADICTION DETECTOR — it catches a
declared value that is too low — not a way to discover the window from nothing.
Discovery must come from a declared source; the floor keeps that source honest.

PRECEDENCE (highest first):
    1. explicit config   — the operator knows something we cannot measure
    2. discovery         — /v1/models scan, then provider endpoints
    3. observed floor    — only when nothing declared a value at all
Whatever wins is CLAMPED into the observed bracket: raised to the floor if it
contradicts proof, lowered to the ceiling if it exceeds proof.
"""

from __future__ import annotations

import re
from threading import Lock
from typing import Any, Optional

# Keys carrying a context window on an OpenAI-shaped listing. No single standard
# exists; these are the conventions actually in the wild.
#   n_ctx                  llama.cpp (under data[].meta)
#   max_model_len          vLLM
#   loaded_context_length  LM Studio
#   context_length         common
_WINDOW_KEYS = (
    "n_ctx",
    "max_model_len",
    "context_length",
    "max_context_length",
    "loaded_context_length",
    "max_position_embeddings",
)

# Below this is not a plausible window; treat as noise rather than adopt a value
# that would make every derived threshold nonsense.
_MIN_PLAUSIBLE = 256

_OVERFLOW_PATTERNS = (
    re.compile(r"maximum context length is (\d+)", re.I),
    re.compile(r"max_model_len[^\d]{0,20}(\d+)", re.I),
    re.compile(r"context (?:window|size|length)[^\d]{0,30}(\d+)", re.I),
)
_OVERFLOW_HINTS = (
    "maximum context length", "context length", "context window",
    "exceeds the available context", "too many tokens", "max_model_len",
    "prompt is too long", "context size",
)


def scan_for_window(obj: Any, depth: int = 0) -> Optional[int]:
    """Deep-scan an arbitrary JSON body for a plausible context window.

    Recursive because the location is not standardised: llama.cpp puts it at
    data[].meta.n_ctx, vLLM at data[].max_model_len. Returns the SMALLEST
    plausible hit — when a body advertises both a trained maximum and a loaded
    window (llama.cpp reports n_ctx beside n_ctx_train), the loaded one is the
    operative limit and taking the larger would over-size everything.
    """
    if depth > 8:
        return None
    found: list = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _WINDOW_KEYS and isinstance(value, (int, float)):
                candidate = int(value)
                if candidate >= _MIN_PLAUSIBLE:
                    found.append(candidate)
            elif isinstance(value, (dict, list)):
                nested = scan_for_window(value, depth + 1)
                if nested:
                    found.append(nested)
    elif isinstance(obj, list):
        for item in obj:
            nested = scan_for_window(item, depth + 1)
            if nested:
                found.append(nested)
    return min(found) if found else None


def looks_like_overflow(text: Optional[str]) -> bool:
    """True iff an upstream error body reads as a context-length rejection.

    Deliberately narrow. An unclassifiable failure must NOT move the belief:
    treating an unrelated 500 as proof of a small window would shrink every
    threshold on the strength of a network blip.
    """
    if not text:
        return False
    low = text.lower()
    return any(h in low for h in _OVERFLOW_HINTS)


def parse_overflow_limit(text: Optional[str]) -> Optional[int]:
    """The declared limit inside an overflow error, when the server states one."""
    if not text:
        return None
    for pat in _OVERFLOW_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                value = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if value >= _MIN_PLAUSIBLE:
                return value
    return None


class ContextWindowResolver:
    """Thread-safe. Never raises — a sizing belief must not be able to fail a
    request."""

    def __init__(self, configured: Optional[int] = None) -> None:
        self._lock = Lock()
        self._configured = int(configured) if configured else None
        self._declared: Optional[int] = None          # from discovery
        self._declared_source: Optional[str] = None
        self._floor = 0                               # largest prompt PROVEN to fit
        self._ceiling: Optional[int] = None           # smallest length PROVEN too big
        self._contradictions = 0
        self._overflows = 0

    # -------------------------------------------------------------- evidence
    def observe_success(self, prompt_tokens: Optional[int]) -> bool:
        """A prompt of this size was accepted. Raises the floor. True iff this
        contradicted the current belief (the belief was too low)."""
        try:
            value = int(prompt_tokens or 0)
        except (TypeError, ValueError):
            return False
        if value <= 0:
            return False
        with self._lock:
            if value > self._floor:
                self._floor = value
            # A prompt larger than a length we previously saw REJECTED means the
            # world changed under us — llama-server restarted with a bigger -c,
            # a different model loaded. The rejection is stale evidence about a
            # server that no longer exists; the success is current. Newest wins.
            if self._ceiling is not None and value > self._ceiling:
                self._ceiling = None
            declared = self._configured or self._declared
            if declared is not None and value > declared:
                self._contradictions += 1
                return True
            return False

    def observe_overflow(self, error_text: Optional[str] = None,
                         attempted_tokens: Optional[int] = None) -> bool:
        """An upstream rejection. Lowers the ceiling ONLY when the body reads as
        a context-length error; an unclassifiable failure leaves it untouched."""
        if not looks_like_overflow(error_text):
            return False
        limit = parse_overflow_limit(error_text)
        with self._lock:
            self._overflows += 1
            candidate = limit
            if candidate is None and attempted_tokens:
                try:
                    candidate = int(attempted_tokens)
                except (TypeError, ValueError):
                    candidate = None
            if candidate and candidate >= _MIN_PLAUSIBLE:
                if self._ceiling is None or candidate < self._ceiling:
                    self._ceiling = candidate
                # Symmetrically: a rejection below a length we previously saw
                # SUCCEED means the window shrank (restart with a smaller -c).
                # The old success is stale proof about a server that is gone.
                # Without this the bracket is inconsistent — floor above ceiling
                # — and _resolve's floor-override silently wins, leaving the
                # governor sized above a limit it has just been told about.
                if self._floor and self._floor > self._ceiling:
                    self._floor = 0
            return True

    def offer(self, value: Optional[int], source: str) -> bool:
        """A value discovered from an endpoint. True iff the declared belief moved."""
        try:
            candidate = int(value or 0)
        except (TypeError, ValueError):
            return False
        if candidate < _MIN_PLAUSIBLE:
            return False
        with self._lock:
            if self._declared == candidate:
                return False
            self._declared = candidate
            self._declared_source = source
            return True

    # ---------------------------------------------------------------- belief
    @property
    def window(self) -> Optional[int]:
        with self._lock:
            return self._resolve()[0]

    @property
    def source(self) -> str:
        with self._lock:
            return self._resolve()[1]

    def _resolve(self) -> tuple:
        if self._configured:
            value, source = self._configured, "config"
        elif self._declared:
            value, source = self._declared, self._declared_source or "discovered"
        elif self._floor:
            # Nothing declared. The floor is PROVEN to fit, so adopting it is
            # conservative and safe — but it cannot grow past what the governor
            # itself lets through, so it is a fallback, not discovery.
            return self._floor, "observed-floor"
        elif self._ceiling:
            # Nothing declared and no surviving floor — but a rejection told us
            # an upper limit. Sizing to it is conservative: every water mark is
            # a fraction of it, so the wire is held well below a length already
            # known to be too long. Better than unresolved, which turns
            # windowing off entirely.
            return self._ceiling, "observed-ceiling"
        else:
            return None, "unresolved"
        if self._floor and value < self._floor:
            return self._floor, source + "+floor-override"
        if self._ceiling and value > self._ceiling:
            return self._ceiling, source + "+ceiling-clamp"
        return value, source

    def snapshot(self) -> dict:
        with self._lock:
            value, source = self._resolve()
            return {
                "window": value,
                "source": source,
                "resolved": value is not None,
                "configured": self._configured,
                "declared": self._declared,
                "declared_source": self._declared_source,
                "observed_floor": self._floor or None,
                "observed_ceiling": self._ceiling,
                "contradictions": self._contradictions,
                "overflows": self._overflows,
            }
