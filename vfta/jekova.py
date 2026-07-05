"""JEKOVA candidate detector (Jekova and Krasteva 2004).

The JEKOVA algorithm passes the signal through a 14.6 Hz integer IIR band-pass
and derives three per-second counts (Count1-3) from the filtered output. vftx
exposes those counts as three separate features (``count1`` [13], ``count2``
[14], ``count3`` [15]) for the feature screen, but each of those calls redoes
the band-pass loop, which is the expensive step. This module consolidates them:
one band-pass pass yields all three counts, matching the real-time integer-only
form the paper credits, and is the home for the Phase 4 grid search over the
three count thresholds.

The band-pass and per-second counting live in ``vftx._count_helpers`` (validated
against the reference); this module is the detector-level wrapper so ``vfta`` and
the notebook talk to one JEKOVA object instead of three feature columns.
"""

from __future__ import annotations

import numpy as np

from vftx._count_helpers import _aux_counts
from vftx.types import PreprocessedSignal, SegmentConfig

COUNT_NAMES = ("count1", "count2", "count3")


def counts(pp: PreprocessedSignal, cfg: SegmentConfig) -> tuple[int, int, int]:
    """Return (Count1, Count2, Count3) from a single 14.6 Hz band-pass pass."""
    return _aux_counts(pp, cfg)
pass #def


def decide(c1: int, c2: int, c3: int, thr: tuple[float, float, float]) -> bool:
    """Shockable decision from the three counts and a threshold triple.

    A window is flagged shockable when every count meets its threshold. The
    thresholds are tuned in Phase 4 by a small grid search; this is the fixed
    decision rule the search sweeps.
    """
    t1, t2, t3 = thr
    return bool(c1 >= t1 and c2 >= t2 and c3 >= t3)
pass #def


def count_matrix(pp_windows, cfg: SegmentConfig) -> np.ndarray:
    """Stack (Count1, Count2, Count3) for an iterable of preprocessed windows."""
    return np.array([counts(w, cfg) for w in pp_windows], dtype=float)
pass #def
