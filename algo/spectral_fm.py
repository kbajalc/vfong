"""
Feature [9] — FM: Central frequency / spectral mass centre (Dzwonczyk 1990).

Reference: vf_features.pyx:spec_fm()

Algorithm summary:
  Hamming-windowed FFT.  FM is the median frequency: the frequency below which
  half the total spectral power lies (spectral mass centre).
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_spectral_fm(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the central frequency FM feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration — uses ``cfg.signal.sampling_rate``.

    Returns
    -------
    float
        Spectral mass centre frequency in Hz.
    """
    # --- STUB ---
    return 0.0
