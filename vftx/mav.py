"""
Feature [12] — MAV: Mean Absolute Value.

Reference: vf_features.pyx:mean_absolute_value()

Algorithm:
  Slide a 2-second window in 1-second steps.  In each window take the absolute
  value, normalise by the window's max, compute the mean.  Return the mean of
  all per-window means.
"""

import numpy as np

from vftx.types import PreprocessedSignal, SegmentConfig


def compute_mav(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Mean Absolute Value feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.energy.mav_window_sec`` and ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    sr = int(sig.sampling_rate)
    n_samples = len(samples)
    window_size = int(cfg.energy.mav_window_sec * sr)
    step = sr

    mavs: list[float] = []
    w_begin = 0
    w_end = window_size
    while w_end <= n_samples:
        w = np.abs(samples[w_begin:w_end])
        w_max = np.max(w)
        if w_max > 0.0:
            w = w / w_max
        pass #if
        mavs.append(float(np.mean(w)))
        w_begin += step
        w_end += step
    pass #while

    return float(np.mean(mavs)) if mavs else 0.0
pass #def
