"""
Features [17–21] — IMF1_LZ … IMF5_LZ: LZ complexity of EMD intrinsic mode functions.

Reference: vf_features.pyx:imf_lempel_ziv() + vf_features_native.c:imf_lempel_ziv_complexity()

Algorithm summary:
  Decompose the processed signal into intrinsic mode functions (IMFs) using
  Empirical Mode Decomposition (EMD) via the bundled PTSA library (ptsa/ptsa/emd.py)
  or a Python replacement (PyEMD).  For each of the five IMFs, binarise at the
  IMF's mean and apply the same LZ complexity measure as feature [10].

  All five IMFs must be computed from a single EMD call — decomposing separately
  would produce different results since EMD is a greedy sifting algorithm.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_imf_lz(
    sig: PreprocessedSignal,
    cfg: SegmentConfig,
) -> tuple[float, float, float, float, float]:
    """Return LZ complexity for each of the five EMD intrinsic mode functions.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (``cfg.sampling_rate``).

    Returns
    -------
    tuple of five floats
        ``(imf1_lz, imf2_lz, imf3_lz, imf4_lz, imf5_lz)``
    """
    # --- STUB ---
    return 0.0, 0.0, 0.0, 0.0, 0.0
