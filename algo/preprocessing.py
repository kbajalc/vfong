"""
Signal preprocessing pipeline.

Reference: signal_processing.pyx + vf_features.pyx:extract_features()

Pipeline (applied in order):
  1. Mean subtraction
  2. Min-max normalisation  → [0, 1]
  3. Moving-average smoothing (order 5, convolution)
  4. Drift suppression      — 1 Hz high-pass, custom bilinear IIR, zero-phase
  5. Butterworth low-pass   — 30 Hz, order 5, zero-phase

Amplitude (feature [16]) is computed on the raw mV signal BEFORE this pipeline.
The caller is responsible for ADC→mV conversion before calling preprocess().
"""

import numpy as np
import scipy.signal as ss

from algo.types import PreprocessedSignal, SegmentConfig


def _drift_suppression(data: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    T = 1.0 / fs
    tan_val = np.tan(cutoff_hz * np.pi * T)
    c1 = 1.0 / (1.0 + tan_val)
    c2 = (1.0 - tan_val) / (1.0 + tan_val)
    b = [c1, -c1]
    a = [1.0, -c2]
    return ss.filtfilt(b, a, data)


def preprocess(signal_mv: np.ndarray, cfg: SegmentConfig) -> PreprocessedSignal:
    """Condition a raw-mV ECG segment for feature extraction.

    Parameters
    ----------
    signal_mv:
        1-D float64 array already converted to millivolts.
    cfg:
        Uses ``cfg.signal``.

    Returns
    -------
    PreprocessedSignal
        ``raw_mv`` is the input unchanged; ``processed`` is the fully
        conditioned signal used by all features except amplitude and QRS.
    """
    sc = cfg.signal
    s = signal_mv.astype(np.float64)

    # 1. mean subtraction
    s = s - np.mean(s)

    # 2. min-max normalisation → [0, 1]
    s_min, s_max = np.min(s), np.max(s)
    if s_max != s_min:
        s = (s - s_min) / (s_max - s_min)

    # 3. moving-average smoothing
    order = sc.moving_avg_order
    s = np.convolve(s, np.ones(order) / order, mode="same")

    # 4. drift suppression (1 Hz high-pass)
    s = _drift_suppression(s, sc.highpass_hz, sc.sampling_rate)

    # 5. Butterworth low-pass
    nyq = 0.5 * sc.sampling_rate
    b, a = ss.butter(5, sc.lowpass_hz / nyq, btype="lowpass")
    s = ss.filtfilt(b, a, s)

    return PreprocessedSignal(
        raw_mv=signal_mv.astype(np.float64),
        processed=s,
        sampling_rate=sc.sampling_rate,
    )
