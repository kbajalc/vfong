"""
Feature [16] — Amplitude: Peak-to-peak amplitude in millivolts.

Reference: vf_features.pyx:extract_features() — computed on raw_mv before normalisation,
           using scipy.signal.argrelmax / argrelmin for local extrema detection.

Algorithm summary:
  On the raw (un-normalised) mV signal, find all local maxima and local minima
  using a neighbourhood comparison.  Amplitude = (mean of maxima - mean of minima) / 2.
  This is NOT simply np.ptp(); it uses local extrema to be robust to outlier spikes.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_amplitude(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the peak-to-peak amplitude in millivolts.

    Parameters
    ----------
    sig:
        Preprocessed signal — uses ``sig.raw_mv`` (pre-normalisation).
    cfg:
        Extraction configuration.

    Returns
    -------
    float
        Peak-to-peak amplitude in mV.
    """
    # --- STUB ---
    return 0.0
