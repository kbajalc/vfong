"""
Feature [12] — MAV: Mean Absolute Value.

Reference: vf_features.pyx:mean_absolute_value()

Algorithm summary:
  Compute the mean of |signal| over a sliding 2-second window, then average
  the per-window MAV values across the segment.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_mav(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Mean Absolute Value feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration — uses ``cfg.energy.mav_window_sec`` and
        ``cfg.signal.sampling_rate``.

    Returns
    -------
    float
        MAV averaged over 2-second sliding windows.
    """
    # --- STUB ---
    return 0.0
