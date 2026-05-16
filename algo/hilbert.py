"""
Feature [5] — HILB: Hilbert-Transform Phase Space Reconstruction.

Reference: vf_features.pyx:hilbert_psr()

Algorithm summary:
  Compute the analytic signal via Hilbert transform.  Form the 2-D trajectory
  (real part, imaginary part) and project onto the same 40×40 grid as PSR.
  HILB is the proportion of occupied grid cells.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_hilbert(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Hilbert-transform PSR feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (``cfg.psr_grid_size``).

    Returns
    -------
    float
        Fraction of 40×40 grid cells occupied by the Hilbert phase-space
        trajectory.
    """
    # --- STUB ---
    return 0.0
