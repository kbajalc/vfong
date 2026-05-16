"""
Feature [0] — TCSC: Threshold Crossing Sample Count.

Reference: vf_features.pyx:threshold_crossing_sample_count()

Algorithm summary:
  Apply a Tukey (tapered cosine) window to overlapping 3-second sub-windows
  (step 1 second).  In each window count samples that cross BOTH the +20 % and
  -20 % thresholds of the peak-to-peak amplitude.  Average the count across all
  windows and normalise by window length.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_tcsc(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Threshold Crossing Sample Count feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (thresholds, window parameters).

    Returns
    -------
    float
        TCSC value.
    """
    # --- STUB ---
    return 0.0
