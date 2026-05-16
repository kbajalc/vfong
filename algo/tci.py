"""
Feature [1] — TCI: Threshold Crossing Intervals.

Reference: vf_features.pyx:threshold_crossing_intervals()

Algorithm summary:
  Enumerate all threshold-crossing pulse pairs (rise + fall = one pulse) at the
  20 % amplitude threshold.  For each interior 1-second window (skip first and
  last) count pulses that fall within it and record distances to neighbouring
  pulses.  Average TCI metrics across windows.  The boundary seconds are excluded
  to avoid artefacts from truncated pulses.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_tci(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Threshold Crossing Intervals feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration — uses ``cfg.threshold`` and
        ``cfg.signal.sampling_rate``.

    Returns
    -------
    float
        TCI value averaged over interior 1-second windows.
    """
    # --- STUB ---
    return 0.0
