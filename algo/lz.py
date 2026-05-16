"""
Feature [10] — LZ: Lempel-Ziv complexity.

Reference: vf_features_native.c:lempel_ziv_complexity()

Algorithm summary:
  Binarise the signal at its mean value (1 if above, 0 if below).  Apply the
  Lempel-Ziv 1976 complexity measure: count the number of distinct substrings
  encountered scanning left-to-right.  Normalise by sequence length.

  Critical detail: initial complexity counter C(n) = 1 (not 0 — the reference
  implementation fixed this bug in commit 0022678).
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_lz(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Lempel-Ziv complexity feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration.

    Returns
    -------
    float
        Normalised LZ complexity value.
    """
    # --- STUB ---
    return 0.0
