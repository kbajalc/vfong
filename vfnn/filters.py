"""Record-level ECG filtering for vfnn.

Run once over the whole record before windowing. Three steps, all parameterised in
milliseconds so the sampling rate is explicit:

1. A moving median of ``median_ms`` gives a first baseline estimate.
2. A moving mean of the same length smooths it. The mean is the Lynn low-pass filter
   (a triangular average, two cascaded moving averages of half the window each). The
   smoothed estimate is subtracted from the signal, which is the high-pass side.
3. The Lynn low-pass with a ``lynn_ms`` window (40 ms by default).

The Lynn filter is causal (``lfilter``) and its output is rolled back by half a window to
remove the delay, so the first few output samples are wrapped from the end. That edge
effect is kept on purpose: it is only a few samples per record.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

FS = 250           # Hz, the only sampling rate vfnn uses
MEDIAN_MS = 596    # moving-median baseline window
LYNN_MS = 40       # Lynn low-pass window


def samples(ms: float, fs: int = FS) -> int:
    """Window length in samples for a duration in milliseconds."""
    return int(ms * fs / 1000)
pass #def


def lynn_lowpass(x: np.ndarray, w: int) -> np.ndarray:
    """Lynn low-pass of window ``w`` samples (rounded down to even), unit DC gain."""
    w -= w % 2
    if w < 2:
        return np.asarray(x, dtype=float)
    pass #if

    a = np.array([1.0, -2.0, 1.0])
    b = np.zeros(w + 1)
    b[0] = 1.0
    b[w // 2] = -2.0
    b[w] = 1.0
    gain = (w // 2) ** 2
    shift = w // 2 - 1

    y = signal.lfilter(b, a, x) / gain
    return np.roll(y, -shift)
pass #def


def remove_baseline(x: np.ndarray, m: int) -> np.ndarray:
    """Subtract the smoothed moving-median baseline of ``m`` samples."""
    if m <= 1:
        return np.asarray(x, dtype=float)
    pass #if
    med = ndimage.median_filter(x, size=m, mode="nearest")
    return x - lynn_lowpass(med, m)
pass #def


def filter_record(
    x: np.ndarray,
    fs: int = FS,
    median_ms: float = MEDIAN_MS,
    lynn_ms: float = LYNN_MS,
) -> np.ndarray:
    """Full record-level conditioning: baseline removal, then the Lynn low-pass."""
    y = np.asarray(x, dtype=float)
    y = remove_baseline(y, samples(median_ms, fs))
    return lynn_lowpass(y, samples(lynn_ms, fs))
pass #def


def normalize_window(w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero mean and unit variance over the last axis of a window (or batch of windows)."""
    w = np.asarray(w, dtype=np.float32)
    mu = w.mean(axis=-1, keepdims=True)
    sd = w.std(axis=-1, keepdims=True)
    return (w - mu) / (sd + eps)
pass #def
