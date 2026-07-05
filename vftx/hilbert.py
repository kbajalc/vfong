"""
Feature [5] — HILB: Hilbert-Transform Phase Space Reconstruction.

Reference: vf_features.pyx:hilbert_psr()

Algorithm:
  Resample signal to 50 Hz.  Compute the analytic signal (Hilbert transform).
  Form the 2-D trajectory (real part, imaginary part).  Unlike PSR, x and y
  axes use independent ranges.  Project onto 40×40 grid; HILB = occupied / 1600.
"""

import numpy as np
import scipy.signal as ss

from vftx.types import PreprocessedSignal, SegmentConfig


def compute_hilbert(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Hilbert-transform PSR feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.phase_space.grid_size`` and ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    g = cfg.phase_space.grid_size  # 40
    duration = len(samples) / sig.sampling_rate
    n_out = int(duration * 50)  # resample to 50 Hz

    x = ss.resample(samples, n_out)
    analytic = ss.hilbert(x)
    y = np.imag(analytic) # type: ignore

    x_offset = np.min(x) # type: ignore
    x_range = np.max(x) - x_offset # type: ignore
    y_offset = np.min(y)
    y_range = np.max(y) - y_offset

    # guard against flat signal on either axis
    if x_range == 0.0 or y_range == 0.0:
        return 0.0

    grid_x = ((x - x_offset) * (g - 1) / x_range).astype(np.int8)
    grid_y = ((y - y_offset) * (g - 1) / y_range).astype(np.int8)

    grid = np.zeros((g, g), dtype=np.int8)
    grid[grid_y, grid_x] = 1
    return float(np.sum(grid)) / float(g * g)
