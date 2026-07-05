"""
Feature [4] — PSR: Phase Space Reconstruction.

Reference: vf_features.pyx:phase_space_reconstruction()

Algorithm:
  Form a 2-D delay-embedded trajectory: x(t) vs x(t + delay) where delay = 0.5 s.
  Both axes share the same range [min(samples), max(samples)].
  Project onto a 40×40 occupancy grid; PSR = occupied cells / 1600.
"""

import numpy as np

from vftx.types import PreprocessedSignal, SegmentConfig


def compute_psr(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Phase Space Reconstruction feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.phase_space`` and ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    ps = cfg.phase_space
    n_delay = int(ps.delay_sec * sig.sampling_rate)
    g = ps.grid_size  # 40

    x = samples[:-n_delay]
    y = samples[n_delay:]

    # single axis range for both dimensions (reference code uses global min/max)
    offset = np.min(samples)
    axis_range = np.max(samples) - offset

    grid_x = ((x - offset) * (g - 1) / axis_range).astype(np.int8)
    grid_y = ((y - offset) * (g - 1) / axis_range).astype(np.int8)

    grid = np.zeros((g, g), dtype=np.int8)
    grid[grid_y, grid_x] = 1
    return float(np.sum(grid)) / float(g * g)
pass #def
