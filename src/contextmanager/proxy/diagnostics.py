"""Wire-composition diagnostics — measure WHERE the real prompt mass lives.

The rewriter's passes are string-content-only, so the proxy's chars/4 estimates
are blind to the ``tools`` array, structured content-parts, ``tool_calls``
payloads and chat-template overhead. sensing.py's Phase-14a docstring puts the
invisible share at "~88%", and 14c works around it by deriving pressure from
real ``usage.prompt_tokens`` — but nothing in this codebase has ever MEASURED
the split per component. Every sizing decision downstream (should we prune the
tools array? stub tool_calls arguments? handle structured content?) is a guess
until it is.

This module is a PURE TEE over the payload the proxy is about to forward, paired
with the ``usage.prompt_tokens`` that comes back for THAT SAME request. It
reports, with numbers instead of inference:

  * chars contributed by each component of the OUTGOING wire
    (tools / system / string content / structured content-parts / tool_calls),
  * the real prompt token count for that exact request,
  * therefore the implied chars-per-token, and — in tokenize mode — the
    UNACCOUNTED residual (chat template + role scaffolding), the term no
    estimate in the codebase models at all.

PAIRING IS THE POINT. /metrics reports ``peak_chars_out`` and
``real_prompt_tokens.peak`` as INDEPENDENT running maxima (metrics.py:87 and
sensing.py's ledger), accumulated over different requests. Subtracting one from
the other does not yield a floor — it yields a number with no referent. Every
sample here carries both figures from a single request, which is the only way
the difference means anything.

Chars mode is free (len() plus one compact serialization of the non-string
parts) and safe to leave on. Tokenize mode issues 6 extra /tokenize calls per
sampled request and is OFF by default.

Never raises, never mutates the payload, never fails a request.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

# Wire components, in prompt order. "other_payload" catches the small
# non-message/non-tools keys (model, temperature, stream, ...) so the accounted
# total is the WHOLE forwarded body, not a convenient subset.
COMPONENTS = (
    "tools",
    "system_content",
    "string_content",
    "structured_content",
    "tool_calls",
    "other_payload",
)

# Components the rewriter's string-only passes can currently reach. Everything
# else is unsheddable mass no amount of windowing can touch.
_SHEDDABLE = ("string_content",)


def _dumps(obj: Any) -> str:
    """Compact deterministic serialization; never raises."""
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(obj)


def component_texts(payload: dict) -> dict:
    """Split the outgoing payload into one concatenated text per component.

    Both the (free) char measurement and the (costly) exact tokenization read
    this, so the two can never disagree about what belongs where.
    """
    texts = {k: [] for k in COMPONENTS}
    n_messages = 0
    n_tools = 0

    if not isinstance(payload, dict):
        return {"texts": {k: "" for k in COMPONENTS},
                "n_messages": 0, "n_tools": 0}

    tools = payload.get("tools")
    if tools is not None:
        texts["tools"].append(_dumps(tools))
        if isinstance(tools, list):
            n_tools = len(tools)

    messages = payload.get("messages")
    if isinstance(messages, list):
        n_messages = len(messages)
        for m in messages:
            if not isinstance(m, dict):
                texts["structured_content"].append(_dumps(m))
                continue
            role = m.get("role", "")
            content = m.get("content")
            if isinstance(content, str):
                # The ONLY bucket the rewriter's passes can page out.
                if role == "system":
                    texts["system_content"].append(content)
                else:
                    texts["string_content"].append(content)
            elif content is not None:
                # OpenAI content-parts list (multimodal / block content).
                texts["structured_content"].append(_dumps(content))
            tool_calls = m.get("tool_calls")
            if tool_calls is not None:
                # Assistant tool-call arguments: a Write/Edit call carries a
                # whole file body here, with content: null. Invisible to every
                # str-only pass AND to metrics.py's chars_out.
                texts["tool_calls"].append(_dumps(tool_calls))

    for key, value in payload.items():
        if key in ("messages", "tools"):
            continue
        texts["other_payload"].append(key)
        texts["other_payload"].append(_dumps(value))

    return {
        "texts": {k: "".join(v) for k, v in texts.items()},
        "n_messages": n_messages,
        "n_tools": n_tools,
    }


@dataclass
class WireSample:
    """One forwarded request, and what came back for it."""
    seq: int
    ts: float
    n_messages: int
    n_tools: int
    chars: dict
    tokens: Optional[dict] = None            # exact per-component (tokenize mode)
    real_prompt_tokens: Optional[int] = None  # usage.prompt_tokens, SAME request

    @property
    def total_chars(self) -> int:
        return sum(self.chars.values())

    def as_row(self) -> dict:
        row = {
            "seq": self.seq,
            "n_messages": self.n_messages,
            "n_tools": self.n_tools,
            "total_chars": self.total_chars,
            "chars": dict(self.chars),
            "real_prompt_tokens": self.real_prompt_tokens,
        }
        if self.real_prompt_tokens:
            row["implied_chars_per_token"] = round(
                self.total_chars / self.real_prompt_tokens, 2)
        if self.tokens:
            row["tokens"] = dict(self.tokens)
            accounted = sum(self.tokens.values())
            row["accounted_tokens"] = accounted
            if self.real_prompt_tokens:
                # Chat template + role scaffolding: real minus everything we
                # can attribute to a component. The term nothing else models.
                row["unaccounted_tokens"] = self.real_prompt_tokens - accounted
        return row


class WireDiagnostics:
    """Thread-safe bounded ring of wire samples. Disabled = every call a no-op."""

    def __init__(self, enabled: bool = True, max_samples: int = 64,
                 tokenize: bool = False) -> None:
        self.enabled = enabled
        self.tokenize = tokenize
        self._samples: "deque[WireSample]" = deque(maxlen=max(1, max_samples))
        self._by_seq: dict = {}
        self._seq = 0
        self._lock = Lock()

    # ------------------------------------------------------------- write path

    def record_request(self, payload: dict) -> Optional[int]:
        """Measure the outgoing payload. Returns a seq to pair usage against,
        or None when disabled / on any failure."""
        if not self.enabled:
            return None
        try:
            split = component_texts(payload)
            chars = {k: len(v) for k, v in split["texts"].items()}
            with self._lock:
                self._seq += 1
                sample = WireSample(
                    seq=self._seq,
                    ts=time.time(),
                    n_messages=split["n_messages"],
                    n_tools=split["n_tools"],
                    chars=chars,
                )
                if len(self._samples) == self._samples.maxlen:
                    evicted = self._samples[0]
                    self._by_seq.pop(evicted.seq, None)
                self._samples.append(sample)
                self._by_seq[sample.seq] = sample
                return sample.seq
        except Exception:
            return None

    def tokenize_request(self, seq: Optional[int], payload: dict, counter) -> None:
        """Exact per-component token counts (6 /tokenize calls). Blocking —
        call from a worker thread. Best-effort; failure leaves chars-only."""
        if seq is None or not self.enabled or not self.tokenize or counter is None:
            return
        try:
            texts = component_texts(payload)["texts"]
            tokens = {}
            for name, text in texts.items():
                tokens[name] = counter.count_text(text) if text else 0
        except Exception:
            return
        with self._lock:
            sample = self._by_seq.get(seq)
            if sample is not None:
                sample.tokens = tokens

    def attach_usage(self, seq: Optional[int], prompt_tokens: Optional[int]) -> None:
        """Pair the real prompt size with the request that produced it."""
        if seq is None or not self.enabled or not prompt_tokens:
            return
        try:
            value = int(prompt_tokens)
        except (TypeError, ValueError):
            return
        if value <= 0:
            return
        with self._lock:
            sample = self._by_seq.get(seq)
            if sample is not None:
                sample.real_prompt_tokens = value

    # -------------------------------------------------------------- read path

    def snapshot(self, n_ctx: Optional[int] = None) -> dict:
        if not self.enabled:
            return {"enabled": False,
                    "note": "set CM_DIAG_ENABLED=1 to measure wire composition"}
        with self._lock:
            samples = list(self._samples)
        paired = [s for s in samples if s.real_prompt_tokens]

        out: dict = {
            "enabled": True,
            "tokenize_mode": self.tokenize,
            "samples": len(samples),
            "paired_samples": len(paired),
            "note": ("chars are exact; per-component TOKENS require "
                     "CM_DIAG_TOKENIZE=1" if not self.tokenize else
                     "per-component tokens exact (concatenated per component)"),
        }
        if not paired:
            out["waiting_for"] = ("a completed request carrying usage.prompt_tokens "
                                  "— paired samples are the only valid basis for a split")
            return out

        peak = max(paired, key=lambda s: s.real_prompt_tokens or 0)
        out["peak_request"] = peak.as_row()

        # Mean component share across paired samples. Shares, not absolutes:
        # averaging absolute sizes across differently-sized requests hides the
        # thing we are trying to see.
        share_totals = {k: 0.0 for k in COMPONENTS}
        for s in paired:
            total = s.total_chars or 1
            for k in COMPONENTS:
                share_totals[k] += s.chars.get(k, 0) / total
        out["mean_char_share_pct"] = {
            k: round(v / len(paired) * 100.0, 1) for k, v in share_totals.items()
        }

        # The decision number: what fraction of the peak request could the
        # rewriter's string-only passes even reach?
        sheddable = sum(peak.chars.get(k, 0) for k in _SHEDDABLE)
        total = peak.total_chars or 1
        out["peak_sheddable_char_pct"] = round(sheddable / total * 100.0, 1)
        out["peak_unsheddable_char_pct"] = round((total - sheddable) / total * 100.0, 1)

        if n_ctx:
            out["n_ctx"] = n_ctx
            out["peak_prompt_pct_of_n_ctx"] = round(
                (peak.real_prompt_tokens or 0) / n_ctx * 100.0, 1)

        out["recent"] = [s.as_row() for s in paired[-8:]]
        return out


# --------------------------------------------------------------- wire capture

# Headers whose values never belong on disk.
_REDACTED_HEADERS = frozenset({
    "authorization", "x-api-key", "api-key", "proxy-authorization", "cookie",
})


class WireCapture:
    """Forensic per-request dump of what ENTERS and what LEAVES the proxy.

    The closed loop's break attribution (own-mutation vs harness-edit) was
    designed in Phase 14 but never measured against a working identity, so the
    question "does the proxy's own rewrite break the upstream's prefix cache,
    or did the incoming wire arrive broken?" has only ever been answered by
    inference. This answers it from the wire itself: each request gets ONE seq
    and up to two files — ``req-<seq>-in.json`` (headers + body as received)
    and ``req-<seq>-out.json`` (the exact payload forwarded upstream) — so
    consecutive turns can be diffed offline, in both directions.

    Serialization is normalized (sorted keys, compact separators) so a diff of
    two captures reflects content, not dict-order noise. Secret-bearing
    headers are redacted. Disabled unless a directory is configured; when
    enabled it still never raises and never fails a request.
    """

    def __init__(self, directory: Optional[str] = None,
                 max_requests: int = 256) -> None:
        self._dir = directory or None
        self._max = max(1, int(max_requests))
        self._seq = 0
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self._dir is not None

    def record_in(self, headers: Any, body: dict) -> Optional[int]:
        """Dump the request as received. Returns the seq to pair the outgoing
        payload against, or None when disabled / over the cap / on failure."""
        if not self.enabled:
            return None
        try:
            hdrs = {}
            if headers is not None:
                for k, v in dict(headers).items():
                    hdrs[str(k).lower()] = ("***" if str(k).lower()
                                            in _REDACTED_HEADERS else v)
            record = {"ts": time.time(), "headers": hdrs, "body": body}
            return self._write("in", record)
        except Exception:
            return None

    def record_out(self, seq: Optional[int], payload: dict) -> None:
        """Dump the exact payload forwarded upstream, paired by seq."""
        if not self.enabled or seq is None:
            return
        try:
            self._write("out", {"ts": time.time(), "payload": payload}, seq=seq)
        except Exception:
            pass

    def _write(self, tag: str, record: dict,
               seq: Optional[int] = None) -> Optional[int]:
        # Serialize BEFORE touching the filesystem so a non-serializable body
        # leaves no partial file behind.
        blob = json.dumps(record, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), default=repr)
        with self._lock:
            if seq is None:
                if self._seq >= self._max:
                    return None
                self._seq += 1
                seq = self._seq
            os.makedirs(self._dir, exist_ok=True)  # type: ignore[arg-type]
            path = os.path.join(self._dir, f"req-{seq:04d}-{tag}.json")  # type: ignore[operator]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(blob)
            return seq
