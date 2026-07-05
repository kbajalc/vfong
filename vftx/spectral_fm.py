"""
Feature [9] — FM: Central Frequency (spectral centroid).

Reference: vf_features.pyx:central_frequency()

Algorithm:
  Hamming-windowed FFT (positive frequencies only).
  FM = Σ(f_i · P_i) / Σ(P_i),  P_i = |FFT[i]|²
  Multiply by sampling_rate to convert from normalised units to Hz.
"""

import numpy as np
import scipy.signal as ss

from vftx.types import PreprocessedSignal, SegmentConfig


def compute_spectral_fm(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the spectral centroid frequency FM in Hz.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    sr = sig.sampling_rate
    n = len(samples)
    n_fft = int(np.ceil(n / 2))

    fft = np.fft.fft(samples * ss.windows.hamming(n)) # type: ignore
    fft_freq = np.fft.fftfreq(n)

    fft = fft[:n_fft]
    fft_freq = fft_freq[:n_fft]

    power = np.abs(fft) ** 2
    total = np.sum(power)
    if total == 0.0:
        return 0.0
    return float(np.dot(fft_freq, power) / total * sr)
