"""
Feature [3] — MEA: Modified Exponential Algorithm.

Reference: vf_features.pyx:modified_exponential()

Algorithm:
  Similar to STE but uses local maxima instead of the global maximum.
  Fit an exponential E(t) = M_j · exp(−(t − t_mj) / τ) anchored at each
  local maximum M_j (τ = 0.2 s).  Count lifts: each time the signal rises
  above E(t) advance to the next local maximum and restart the envelope.
  Return lifts per second.
"""

import numpy as np
import scipy.signal as ss

from algo.types import PreprocessedSignal, SegmentConfig


def compute_mea(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Modified Exponential Algorithm feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.energy.mea_tau_sec`` and ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    sr = sig.sampling_rate
    n = len(samples)
    tau = cfg.energy.mea_tau_sec * sr  # time constant in samples

    half_peak_width = int(np.round(0.05 * sr))
    local_max_idx_arr = ss.argrelmax(samples, order=half_peak_width)[0]
    if len(local_max_idx_arr) == 0:
        return 0.0

    local_max_iter = iter(local_max_idx_arr)
    n_lifted = 0

    try:
        lm_idx = int(next(local_max_iter))
        lm_val = samples[lm_idx]
        t = lm_idx + 1
        while t < n:
            et = lm_val * np.exp(-(t - lm_idx) / tau)
            if et < samples[t]:
                # lift: find next local max beyond t
                while True:
                    lm_idx = int(next(local_max_iter))
                    if lm_idx > t:
                        break
                n_lifted += 1
                lm_val = samples[lm_idx]
                t = lm_idx + 1
            else:
                t += 1
    except StopIteration:
        pass

    return n_lifted / (n / sr)
