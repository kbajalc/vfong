"""
Feature [8] — A2: Energy ratio around peak frequency.

Reference: vf_features.pyx:spec_a2()

Algorithm summary:
  Hamming-windowed FFT.  Locate the dominant spectral peak frequency f_peak.
  A2 is the ratio of energy in a narrow band around f_peak to total energy,
  capturing how concentrated the spectrum is at its dominant frequency.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_spectral_a2(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the A2 energy-ratio feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration — uses ``cfg.signal.sampling_rate``.

    Returns
    -------
    float
        Energy ratio A2.
    """
    # --- STUB ---
    return 0.0
