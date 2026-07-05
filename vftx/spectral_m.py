"""
Feature [7] — M: First Spectral Moment.

Reference: vf_features.pyx:spectral_features() — M component

Algorithm:
  Hamming-windowed FFT (positive half).  Find the peak frequency f_p in
  0.5–9 Hz.  Zero out amplitudes < 5 % of peak amplitude.  Upper limit:
  min(20·f_p, 100 Hz).
  M = (1/f_p) · Σ(a_i · f_i) / Σ(a_i)  for i up to the upper limit.
  Frequencies are in normalised units (cycles/sample) internally.
"""

import numpy as np
import scipy.signal as ss

from vftx.types import PreprocessedSignal, SegmentConfig


def compute_spectral_m(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the first spectral moment M feature.

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

    amplitudes = np.abs(fft).copy()

    # find peak in [0.5 Hz, 9 Hz] in normalised units
    min_idx = int(np.searchsorted(fft_freq, 0.5 / sr, side="right"))
    max_idx = int(np.searchsorted(fft_freq, 9.0 / sr, side="left"))
    peak_idx = int(np.argmax(amplitudes[min_idx:max_idx])) + min_idx
    peak_freq = fft_freq[peak_idx]
    peak_amp = amplitudes[peak_idx]

    # zero amplitudes below 5 % of peak
    amplitudes[amplitudes < 0.05 * peak_amp] = 0.0

    spec_max_freq = min(20.0 * peak_freq, 100.0 / sr)
    top_idx = int(np.searchsorted(fft_freq, spec_max_freq, side="left"))
    m_amps = amplitudes[:top_idx]
    sum_m = float(np.sum(m_amps))

    if sum_m == 0.0 or peak_freq == 0.0:
        return 0.0

    return float((1.0 / peak_freq) * np.dot(m_amps, fft_freq[:top_idx]) / sum_m)
