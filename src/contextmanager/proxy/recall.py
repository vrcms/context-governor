"""Auto-recall (Phase 10) — the implicit-query half of demand paging.

The proxy's Pass 4 uses these pure helpers to turn the LIVE TAIL of a
conversation into a search query (locality-driven: what the agent is working on
right now is the best predictor of which externalized memory it needs), and to
spend the recall token budget on DISTINCT information rather than near-duplicate
slices (agent sessions re-read the same files over and over).

Everything here is deterministic: same messages in, same query out — required so
the rewriter's prefix-stability reasoning stays checkable. Stdlib-only apart from
one optional accelerator (rapidfuzz, in ``similarity_ratio``) which has an exact
stdlib fallback, so a bare install behaves identically — only slower.
Extracted terms are ``\\w``-only, which makes them inherently safe to hand to an
FTS5 MATCH (no operators, no quotes) as well as to the pure-Python retriever.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

# OPTIONAL ACCELERATOR — see similarity_ratio(). Absence is not an error.
try:  # pragma: no cover - trivial import guard
    from rapidfuzz.distance import Indel as _Indel
except ImportError:  # pragma: no cover
    _Indel = None


def similarity_ratio(a: str, b: str) -> float:
    """Normalized similarity in [0,1]. THE shared metric for near-duplicate work.

    Lives here rather than in the rewriter because `rewriter` imports `recall`,
    so the reverse would be circular. Both callers use it identically:
      * `select_diverse` below — near-duplicate suppression in the recall pool.
      * `rewriter._similarity` — picking a base for diff-encoding.

    WHY NOT difflib (measured 2026-07-28, on real store notes):

        size / caller                          difflib    rapidfuzz   speedup
        ~20 KB, rewriter diff-encoding          9.26 s      0.011 s      851x
        ~20 KB, worst-case repetitive        29,840 ms      0.03 ms   893,000x
        4 KB, select_diverse (cap_chars)        301 ms      1.17 ms      258x

    `difflib.SequenceMatcher` is O(n*m) in pure Python and both callers run it
    SYNCHRONOUSLY on the request path. In the rewriter that was ~56.6 s of CPU
    per turn -- about two thirds of session wall clock -- with the client blocked
    and llama-server idle throughout. In `select_diverse` the greedy pairwise
    loop over a 9-candidate pool is up to ~36 comparisons, i.e. ~10.8 s per
    request on top of that.

    rapidfuzz is CPU-only (no CUDA): a C++ bit-parallel Myers implementation that
    packs 64 DP cells into a machine word, plus SIMD. That is where the speedup
    comes from -- an algorithmic change, not hardware.

    CALIBRATION: Indel and Ratcliff-Obershelp diverge on UNRELATED documents
    (0.05 vs 0.33) where both sit far below any threshold and the decision is the
    same. On near-duplicates -- the only regime either caller's threshold decides
    -- they agree to 3-4 decimals (1.0000/1.0000 ... 0.7587/0.7692), so neither
    `diff_min_similarity` nor `max_similarity` needed retuning. Pinned by
    tests/proxy/test_diff_similarity.py.
    """
    if _Indel is not None:
        return _Indel.normalized_similarity(a, b)
    # autojunk=False: the default heuristic marks "popular" characters as junk on
    # strings >= 200 chars -- and every candidate here is bulky by definition --
    # collapsing the ratio for exactly the long near-duplicates this catches.
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

# Words of 3+ word-characters; \w keeps identifiers like handle_threshold intact.
_TERM_RE = re.compile(r"\w{3,}")

# Lines that are governor plumbing, not conversation signal: stub/marker headers,
# footers, and the truncation ellipsis line inside stubs.
_MARKER_LINE_RE = re.compile(r"^\s*(\[\[/?cm:\S*.*\]\]|…\(truncated \d+ chars\)…)\s*$")

# Small, closed English stopword set + chat-plumbing words. Deliberately tiny:
# over-aggressive stopping hurts recall of code-flavored text more than it helps.
_STOPWORDS = frozenset("""
    the and for are but not you all any can had her was one our out day get has
    him his how man new now old see two way who boy did its let put say she too
    use that with have this will your from they know want been good much some
    time very when come here just like long make many more only over such take
    than them well were what your about after again before below between both
    could does doing down during each further having into itself more most other
    same should then there these those through under until while would with
    role user assistant system tool content message messages
""".split())


def _strip_markers(text: str) -> str:
    """Remove governor marker lines (stub headers/footers, truncation lines) so
    query terms come from real content, never from our own plumbing."""
    return "\n".join(
        line for line in text.split("\n") if not _MARKER_LINE_RE.match(line)
    )


def extract_query(messages: list[dict], *, tail_messages: int = 6,
                  max_terms: int = 12) -> str:
    """Derive the implicit recall query from the newest ``tail_messages`` messages.

    Term frequency is weighted by message recency (the newest message counts
    ~2x the oldest of the window) — the conversation's locality, in the
    virtual-memory sense. Returns up to ``max_terms`` distinct terms joined by
    spaces (ties broken alphabetically for determinism), or "" when there is not
    enough signal (fewer than 2 distinct terms).
    """
    tail = messages[-tail_messages:] if tail_messages > 0 else []
    weights: Counter[str] = Counter()
    n = len(tail)
    for i, msg in enumerate(tail):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str) or not content:
            continue
        # Newest message weight 2.0, oldest 1.0, linear in between.
        w = 1.0 + (i / (n - 1)) if n > 1 else 2.0
        for term in _TERM_RE.findall(_strip_markers(content)):
            t = term.lower()
            if t in _STOPWORDS or t.isdigit():
                continue
            weights[t] += w
    if len(weights) < 2:
        return ""
    # Highest weight first; alphabetical tie-break keeps the query deterministic.
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    return " ".join(t for t, _ in ranked[:max_terms])


def select_diverse(contents: list[str], *, max_similarity: float = 0.85,
                   cap_chars: int = 4000) -> list[int]:
    """Greedy near-duplicate suppression: return the INDICES of ``contents`` to
    keep, in order, skipping any item whose similarity to an already-kept item
    is >= ``max_similarity``. Comparisons are capped to the first ``cap_chars``
    characters (difflib is O(n*m); slices can be large). ``max_similarity >= 1``
    or a single candidate disables suppression naturally.
    """
    kept: list[int] = []
    for i, text in enumerate(contents):
        head = text[:cap_chars]
        duplicate = False
        for j in kept:
            # See similarity_ratio(): rapidfuzz when available (258x at this
            # cap_chars), exact difflib fallback otherwise. This greedy loop is
            # up to ~36 comparisons on a 9-candidate pool, which was ~10.8 s per
            # request on difflib.
            if similarity_ratio(
                contents[j][:cap_chars], head,
            ) >= max_similarity:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return kept
