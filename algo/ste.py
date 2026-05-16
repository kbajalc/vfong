"""
Feature [2] — STE: Short-Time Energy.

Reference: vf_features.pyx:modified_exponential_algorithm() (global-peak variant)

Algorithm summary:
  Weight each sample by an exponential decay anchored at the global signal peak,
  time constant τ = 3 s.  STE is the sum of squared weighted samples, normalised
  by segment length.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_ste(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Short-Time Energy feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (``cfg.ste_tau_sec``).

    Returns
    -------
    float
        STE value.
    """
    # --- STUB ---
    return 0.0
