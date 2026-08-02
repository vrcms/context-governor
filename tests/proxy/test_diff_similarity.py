"""Diff-candidate similarity scoring (`rewriter._similarity`).

Guards the 2026-07-28 accelerator swap. `difflib.SequenceMatcher` is O(n*m) in
pure Python and ran synchronously on the request path: measured 5-17 s per
comparison on real ~20 KB store notes, x`diff_lookback` candidates = ~56.6 s of
governor CPU per turn, which was ~66% of session wall clock. rapidfuzz's Indel
normalized similarity is ~981x faster on the same data.

The swap is only safe because the two metrics agree in the regime where
`diff_min_similarity` actually decides anything. These tests pin that agreement
so a future rapidfuzz release cannot silently move the decision boundary.
"""

from __future__ import annotations

import difflib
from functools import lru_cache

import pytest

from contextmanager.proxy.rewriter import _Indel, _similarity


@lru_cache(maxsize=None)
def _difflib_ratio(a: str, b: str) -> float:
    """The exact metric that _similarity replaced.

    Memoized, and BASE below is deliberately SMALL (~2 KB, not the ~20 KB of a
    real store note). difflib is O(n*m), so a realistic fixture made this very
    test file take 108 s — which is precisely the pathology the accelerator
    exists to remove. The agreement property under test is size-independent; the
    large-content measurements live in the class docstring below and in
    `rewriter._similarity`.
    """
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


BASE = "".join(f"line {i}: the quick brown fox jumps over the lazy dog\n"
               for i in range(40))


def _edited(n_edits: int) -> str:
    lines = BASE.splitlines(True)
    step = max(1, len(lines) // n_edits) if n_edits else len(lines) + 1
    for i in range(0, len(lines), step):
        if n_edits <= 0:
            break
        lines[i] = f"MODIFIED LINE {i}\n"
        n_edits -= 1
    return "".join(lines)


class TestContract:
    def test_identical_is_one(self):
        assert _similarity(BASE, BASE) == pytest.approx(1.0)

    def test_disjoint_is_low(self):
        assert _similarity("a" * 500, "b" * 500) < 0.1

    def test_bounded_and_symmetric(self):
        a, b = BASE, _edited(20)
        assert 0.0 <= _similarity(a, b) <= 1.0
        assert _similarity(a, b) == pytest.approx(_similarity(b, a), abs=1e-9)


class TestAgreesWithDifflibOnNearDuplicates:
    """The regime that matters: a file re-read after an edit.

    Measured on a real 19,000-char store note, difflib vs rapidfuzz:
        edits:      0       1       3      10      50     200
        difflib   1.0000  0.9996  0.9968  0.9895  0.9408  0.7587
        rapidfuzz 1.0000  0.9996  0.9969  0.9896  0.9432  0.7692
    """

    @pytest.mark.parametrize("n_edits", [0, 1, 3, 8, 20])
    def test_ratios_track_difflib(self, n_edits):
        edited = _edited(n_edits)
        assert _similarity(BASE, edited) == pytest.approx(
            _difflib_ratio(BASE, edited), abs=0.05
        )

    @pytest.mark.parametrize("n_edits", [0, 1, 3, 8, 20])
    def test_threshold_decision_matches_difflib(self, n_edits):
        # What the rewriter actually branches on. Divergence here would silently
        # change which messages get delta-encoded.
        edited = _edited(n_edits)
        for threshold in (0.5, 0.7, 0.9):
            assert (_similarity(BASE, edited) >= threshold) == (
                _difflib_ratio(BASE, edited) >= threshold
            ), f"disagreement at {n_edits} edits, threshold {threshold}"

    def test_near_duplicate_stays_well_above_default_threshold(self):
        # The autojunk=False lesson: the naive default collapsed a one-line
        # re-read from ~0.999 to ~0.51, losing its delta encoding. Whatever the
        # backend, a lightly-edited file must stay clearly diff-worthy — i.e.
        # far above the default diff_min_similarity of 0.5.
        # Threshold is 0.9, not 0.99, because BASE is only 40 lines: one edit is
        # a 2.5% change here versus 0.25% on a realistic 400-line note (which
        # measured 0.9996). The invariant under test is the margin over 0.5.
        assert _similarity(BASE, _edited(1)) > 0.9
        assert _difflib_ratio(BASE, _edited(1)) > 0.9


@pytest.mark.skipif(_Indel is None, reason="rapidfuzz not installed")
def test_accelerator_is_actually_in_use():
    # Pins that the fast path is wired, not silently falling back to difflib.
    from rapidfuzz.distance import Indel
    assert _similarity(BASE, _edited(10)) == pytest.approx(
        Indel.normalized_similarity(BASE, _edited(10)), abs=1e-12
    )
