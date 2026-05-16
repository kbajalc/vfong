"""
Feature [13] — Count1: Samples in 50–100 % of peak amplitude range.

Reference: vf_features.pyx:count_features() — index 0

Algorithm summary:
  Resample signal to 250 Hz if needed.  Count samples whose absolute value
  falls in the upper half of the peak-to-peak amplitude range (above 50 % of
  the maximum absolute value).  Normalise by segment length.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_count1(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Count1 feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (``cfg.count_target_rate``).

    Returns
    -------
    float
        Fraction of samples in the 50–100 % amplitude range.
    """
    # --- STUB ---
    return 0.0
