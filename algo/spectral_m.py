"""
Feature [7] — M: Spectral centroid-like parameter (Barro 1989).

Reference: vf_features.pyx:spec_m()

Algorithm summary:
  Hamming-windowed FFT.  M is a weighted centroid of the power spectrum:
  sum(f_i * |X_i|^2) / sum(|X_i|^2) over the relevant frequency range.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_spectral_m(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the spectral M parameter.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration — uses ``cfg.signal.sampling_rate``.

    Returns
    -------
    float
        Spectral centroid-like parameter M.
    """
    # --- STUB ---
    return 0.0
