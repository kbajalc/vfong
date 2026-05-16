"""
Feature [3] — MEA: Modified Exponential Algorithm.

Reference: vf_features.pyx:modified_exponential_algorithm() (local-peak variant)

Algorithm summary:
  Same exponential weighting as STE but anchored at each local peak rather than
  the global peak, and with a much shorter time constant τ = 0.2 s.  Captures
  the energy envelope around rapid local oscillations.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_mea(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Modified Exponential Algorithm feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration — uses ``cfg.energy.mea_tau_sec`` and
        ``cfg.signal.sampling_rate``.

    Returns
    -------
    float
        MEA value.
    """
    # --- STUB ---
    return 0.0
