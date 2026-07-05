"""
Feature [16] — Amplitude: peak-to-peak mV.

Reference: signal_processing.pyx:get_amplitude()

Algorithm:
  Apply 5-order moving average, 1 Hz drift suppression, 30 Hz low-pass to raw_mv.
  Find local maxima and minima using argrelmax/argrelmin (order = 0.05 s).
  Iterate interleaved peak/valley pairs keeping track of running extrema;
  return the largest peak-to-valley difference found.
"""

import numpy as np
import scipy.signal as ss

from vftx.types import PreprocessedSignal, SegmentConfig
from vftx.preprocessing import _drift_suppression


def compute_amplitude(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the peak-to-peak amplitude in millivolts.

    Parameters
    ----------
    sig:
        Uses ``sig.raw_mv`` — amplitude is measured on the un-normalised signal.
    cfg:
        Uses ``cfg.signal.sampling_rate``.
    """
    sc = cfg.signal
    sr = sc.sampling_rate
    s = sig.raw_mv.copy()

    # same mini-pipeline as signal_processing.pyx:get_amplitude()
    order = 5
    s = np.convolve(s, np.ones(order) / order, mode="same")
    s = _drift_suppression(s, 1.0, sr)
    nyq = 0.5 * sr
    b, a = ss.butter(5, 30.0 / nyq, btype="lowpass") # type: ignore
    s = ss.filtfilt(b, a, s)

    half_peak_width = int(np.round(0.05 * sr))
    peak_idx = ss.argrelmax(s, order=half_peak_width)[0].tolist()
    valley_idx = ss.argrelmin(s, order=half_peak_width)[0].tolist()

    if not peak_idx or not valley_idx:
        return 0.0

    peak_iter = iter(peak_idx)
    valley_iter = iter(valley_idx)

    next_peak = next(peak_iter, -1)
    next_valley = next(valley_iter, -1)
    p_idx = next_peak
    v_idx = next_valley
    max_amplitude = 0.0

    while p_idx != -1 and v_idx != -1:
        peak_val = s[p_idx]
        valley_val = s[v_idx]

        if p_idx < v_idx:
            # at a peak; advance to the next valley, skipping adjacent peaks
            while next_peak < next_valley and next_peak != -1:
                next_peak = next(peak_iter, -1)
                if next_peak != -1 and s[next_peak] > peak_val:
                    peak_val = s[next_peak]
        else:
            # at a valley; advance to the next peak, skipping adjacent valleys
            while next_valley < next_peak and next_valley != -1:
                next_valley = next(valley_iter, -1)
                if next_valley != -1 and s[next_valley] < valley_val:
                    valley_val = s[next_valley]

        amplitude = abs(peak_val - valley_val)
        if amplitude > max_amplitude:
            max_amplitude = amplitude

        p_idx = next_peak
        v_idx = next_valley

    return float(max_amplitude)
