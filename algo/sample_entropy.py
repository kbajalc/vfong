"""
Feature [11] — SpEn: Sample Entropy.

Reference: pyeeg/__init__.py:samp_entropy() called from vf_features.pyx

Algorithm summary:
  Resample signal to 250 Hz if needed.  Use the last 5 seconds (1250 samples).
  Compute sample entropy with embedding dimension m=2 and tolerance r=0.2×std.
  Sample entropy measures the unpredictability of the time series.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_sample_entropy(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Sample Entropy feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (``cfg.spen_duration_sec``, ``cfg.spen_m``,
        ``cfg.spen_r``, ``cfg.count_target_rate``).

    Returns
    -------
    float
        Sample entropy value.
    """
    # --- STUB ---
    return 0.0
