"""
Feature [14] — Count2: Samples above mean amplitude.

Reference: vf_features.pyx:count_features() — index 1

Algorithm summary:
  Resample signal to 250 Hz if needed.  Count samples whose absolute value
  exceeds the mean absolute value of the segment.  Normalise by segment length.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_count2(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Count2 feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (``cfg.count_target_rate``).

    Returns
    -------
    float
        Fraction of samples above mean absolute amplitude.
    """
    # --- STUB ---
    return 0.0
