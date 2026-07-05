"""
Feature [2] — STE: Standard Exponential (crossing count).

Reference: vf_features.pyx:standard_exponential()

Algorithm:
  Fit an exponential envelope E(t) = M·exp(−|t−t_m| / τ) anchored at the
  global maximum M at time t_m (τ = 3 s × sampling_rate samples).  Count the
  number of times the signal crosses this envelope (from t=1 to len−2).
  Return the crossing rate in crossings per second.
"""

import numpy as np

from vftx.types import PreprocessedSignal, SegmentConfig


def compute_ste(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Standard Exponential crossing-rate feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.energy.ste_tau_sec`` and ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    sr = sig.sampling_rate
    tau = cfg.energy.ste_tau_sec * sr  # time constant in samples

    max_time = int(np.argmax(samples))
    max_amp = samples[max_time]

    t = np.arange(len(samples), dtype=np.float64)
    envelope = max_amp * np.exp(-np.abs(t - max_time) / tau)

    # detect sign changes of (samples - envelope), skipping first and last sample
    diff = samples[1:-1] - envelope[1:-1]
    prev = samples[0] - envelope[0]
    higher = prev > 0.0

    n_crosses = 0.0
    for d in diff:
        if higher:
            if d < 0.0:
                higher = False
                n_crosses += 1
        else:
            if d > 0.0:
                higher = True
                n_crosses += 1

    duration = len(samples) / sr
    return n_crosses / duration
