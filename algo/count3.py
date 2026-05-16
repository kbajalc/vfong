"""
Feature [15] — Count3: Samples within mean ± mean-deviation band.

Reference: vf_features.pyx:count_features() — index 2

Algorithm summary:
  Resample signal to 250 Hz if needed.  Compute the mean absolute value μ and
  the mean absolute deviation σ.  Count samples whose absolute value falls
  within [μ - σ, μ + σ].  Normalise by segment length.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_count3(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Count3 feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (``cfg.count_target_rate``).

    Returns
    -------
    float
        Fraction of samples within the mean ± mean-deviation band.
    """
    # --- STUB ---
    return 0.0
