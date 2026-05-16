"""
Signal preprocessing pipeline.

Reference implementation: signal_processing.pyx + vf_features.pyx:extract_features()

Pipeline (applied in order):
  1. Mean subtraction
  2. Drift suppression  — 1 Hz high-pass Butterworth (zero-phase)
  3. Normalisation      — divide by standard deviation
  4. Low-pass filter    — 30 Hz zero-phase Butterworth, order 5
  5. Moving average     — order-5 smoothing

Amplitude (feature [16]) is computed on the raw mV signal BEFORE step 3.

The caller is responsible for ADC→mV conversion (subtract adc_zero, divide by
gain) before calling preprocess().  algo/ never sees raw ADC counts.
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def preprocess(signal_mv: np.ndarray, cfg: SegmentConfig) -> PreprocessedSignal:
    """Condition a raw-mV ECG segment for feature extraction.

    Parameters
    ----------
    signal_mv:
        1-D float64 array of signal samples already converted to millivolts.
        Length should be ``cfg.sampling_rate * 8`` for an 8-second segment.
    cfg:
        Extraction configuration (filter cutoffs, smoothing order, …).

    Returns
    -------
    PreprocessedSignal
        Carries both the raw mV signal (for amplitude) and the fully processed
        signal (for all other features).
    """
    # --- STUB ---
    # TODO: implement the five-step pipeline using scipy.signal
    #   Step 1: signal_mv - signal_mv.mean()
    #   Step 2: scipy.signal.butter + filtfilt (highpass at cfg.highpass_hz)
    #   Step 3: / np.std(...)
    #   Step 4: scipy.signal.butter + filtfilt (lowpass at cfg.lowpass_hz)
    #   Step 5: np.convolve with ones(cfg.moving_avg_order) / cfg.moving_avg_order
    processed = np.zeros_like(signal_mv)
    return PreprocessedSignal(
        raw_mv=signal_mv.copy(),
        processed=processed,
        sampling_rate=cfg.sampling_rate,
    )
