"""
Feature [0] — TCSC: Threshold Crossing Sample Count.

Reference: vf_features.pyx:threshold_crossing_sample_counts()

Algorithm:
  Slide a 3-second Tukey-windowed sub-window over the segment in 1-second steps.
  In each window: apply Tukey window, take absolute value, normalise by max,
  count fraction of samples exceeding 20 % (threshold_ratio), express as
  percentage of window length.  Return the average across all sub-windows.
"""

import numpy as np
import scipy.signal as ss

from algo.types import PreprocessedSignal, SegmentConfig


def compute_tcsc(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Threshold Crossing Sample Count feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.threshold.tcsc_*`` and ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    sr = int(sig.sampling_rate)
    tc = cfg.threshold
    n_samples = len(samples)

    window_size = int(tc.tcsc_window_sec * sr)
    step = sr  # 1-second step
    if window_size > n_samples:
        window_size = n_samples

    tukey_win = ss.windows.tukey(window_size, alpha=0.5 / tc.tcsc_window_sec)
    counts: list[float] = []
    w_begin = 0
    w_end = window_size
    while w_end <= n_samples:
        if cfg.reference_bug_compat:
            # Reference bug: operate on a VIEW so `*= tukey` mutates the shared
            # signal in place, corrupting overlapping windows + downstream features.
            window = samples[w_begin:w_end]
        else:
            window = samples[w_begin:w_end].copy()
        window *= tukey_win
        window = np.abs(window)
        window /= np.max(window)
        counts.append(float(np.sum(window > tc.tcsc_threshold_pct)) * 100.0 / window_size)
        w_begin += step
        w_end += step

    return float(np.mean(counts)) if counts else 0.0
