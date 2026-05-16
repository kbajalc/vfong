"""
Feature [4] — PSR: Phase Space Reconstruction.

Reference: vf_features.pyx:phase_space_reconstruction()

Algorithm summary:
  Form a 2-D delay-embedded signal: x(t) vs x(t + delay) where delay = 0.5 s.
  Project the trajectory onto a 40×40 grid spanning [min, max] × [min, max].
  PSR is the proportion of grid cells visited (occupied cells / total cells).
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def compute_psr(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Phase Space Reconstruction feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Extraction configuration (``cfg.psr_delay_sec``, ``cfg.psr_grid_size``).

    Returns
    -------
    float
        Fraction of 40×40 grid cells visited by the phase-space trajectory.
    """
    # --- STUB ---
    return 0.0
