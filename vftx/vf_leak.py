"""
Feature [6] — VF Leak (VF filter).

Reference: vf_features.pyx:vf_leak()

Algorithm:
  Compute Hamming-windowed FFT (positive half).  Find the peak frequency.
  Shift the signal by half a cycle of the peak frequency.  VF leak is the
  ratio of sum|original + shifted| to sum(|original| + |shifted|).
  A nearly-sinusoidal signal (like VF) will nearly cancel with its half-cycle
  shift, giving a small ratio.
"""

import numpy as np
import scipy.signal as ss

from vftx.types import PreprocessedSignal, SegmentConfig


def compute_vf_leak(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the VF Leak feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    n = len(samples)
    n_fft = int(np.ceil(n / 2))

    fft = np.fft.fft(samples * ss.windows.hamming(n)) # type: ignore
    fft_freq = np.fft.fftfreq(n)
    fft = fft[:n_fft]
    fft_freq = fft_freq[:n_fft]

    # Peak frequency. The reference calls np.argmax on the COMPLEX fft, which
    # numpy resolves lexicographically (by real part, then imag) — not by
    # magnitude. That is almost certainly unintended, so the clean path uses the
    # true spectral peak (|fft|); reference_bug_compat reproduces the complex
    # argmax for bit-exact validation.
    if cfg.reference_bug_compat:
        peak_freq_idx = int(np.argmax(fft))
    else:
        peak_freq_idx = int(np.argmax(np.abs(fft)))
    pass #if
    peak_freq = fft_freq[peak_freq_idx]

    cycle = (1.0 / peak_freq) if peak_freq != 0.0 else float(n)
    half_cycle = int(cycle / 2)
    if half_cycle <= 0 or half_cycle >= n:
        return 0.0
    pass #if

    original = samples[half_cycle:]
    shifted = samples[:-half_cycle]
    denom = float(np.sum(np.abs(original) + np.abs(shifted)))
    if denom == 0.0:
        return 0.0
    pass #if
    return float(np.sum(np.abs(original + shifted)) / denom)
pass #def
