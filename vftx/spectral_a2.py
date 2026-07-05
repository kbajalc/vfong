"""
Feature [8] — A2: Energy Ratio (spectral concentration).

Reference: vf_features.pyx:spectral_features() — A2 component

Algorithm:
  Same FFT, peak detection, and amplitude zeroing as M.
  A2 = Σ(a_i for i in [0.7·f_p, 1.4·f_p]) / Σ(a_i for i up to upper limit).
"""

import numpy as np
import scipy.signal as ss

from vftx.types import PreprocessedSignal, SegmentConfig


def compute_spectral_a2(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the spectral energy ratio A2 feature.

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

    min_idx = int(np.searchsorted(fft_freq, 0.5 / sr, side="right"))
    max_idx = int(np.searchsorted(fft_freq, 9.0 / sr, side="left"))
    peak_idx = int(np.argmax(amplitudes[min_idx:max_idx])) + min_idx
    peak_freq = fft_freq[peak_idx]
    peak_amp = amplitudes[peak_idx]

    amplitudes[amplitudes < 0.05 * peak_amp] = 0.0

    spec_max_freq = min(20.0 * peak_freq, 100.0 / sr)
    top_idx = int(np.searchsorted(fft_freq, spec_max_freq, side="left"))
    sum_all = float(np.sum(amplitudes[:top_idx]))
    if sum_all == 0.0:
        return 0.0
    pass #if

    a2_min = int(np.searchsorted(fft_freq, 0.7 * peak_freq, side="right"))
    a2_max = int(np.searchsorted(fft_freq, 1.4 * peak_freq, side="left"))
    return float(np.sum(amplitudes[a2_min:a2_max]) / sum_all)
pass #def
