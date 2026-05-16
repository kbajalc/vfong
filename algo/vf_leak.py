"""
Feature [6] — VF: VF Leak (Kuo & Dillman 1978).

Reference: vf_features.pyx:vf_leak()

Algorithm summary:
  Apply a Hamming window to the segment and compute the FFT.  VF Leak is the
  ratio of spectral energy in the VF band (4–8 Hz) to total energy.  A high
  ratio indicates a dominant periodic oscillation in the VF frequency range.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_vf_leak(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the VF Leak feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration — uses ``cfg.signal.sampling_rate``.

    Returns
    -------
    float
        Ratio of VF-band energy to total spectral energy.
    """
    # --- STUB ---
    return 0.0
