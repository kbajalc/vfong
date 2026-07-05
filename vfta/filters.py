"""Record-level ECG filters.

Lifted from ``ideas/VFML.ipynb`` and kept bit-for-bit equivalent. These run
once over the whole record before the sliding window, matching the real-time
``exg-core`` pipeline where baseline removal and low-pass already happen
upstream. Filtering is deliberately not applied per segment. See
``paper/PLAN.md`` Phase 2 and the open item on skipping vftx's per-segment
filtering.

The defaults are the VFML constants at 250 Hz (4 ms per sample): a Lynn
band-pass window of 48 ms and a median-baseline window of 596 ms.
"""

import numpy as np
from scipy import ndimage, signal

# Sample period in ms at 250 Hz, and the VFML filter windows in samples.
MS = 4
LYN_WIND = 48 // MS      # Lynn band-pass window (samples)
MED_WIND = 596 // MS     # median-baseline window (samples)


def lynn_filter(sig: np.ndarray, w: int) -> np.ndarray:
    """Lynn recursive band-pass filter of window ``w`` (rounded to even)."""
    w -= w % 2

    a = np.array([1, -2, 1])
    b = np.zeros(w + 1)
    b[0] = 1
    b[w // 2] = -2
    b[w] = 1
    g = (len(b) // 2) ** 2
    shift = w // 2 - 1

    sig = signal.lfilter(b, a, sig) / g
    sig = np.roll(sig, -shift)
    return sig
pass #def


def baseline_filter(sig: np.ndarray, m: int) -> np.ndarray:
    """Remove baseline wander via a moving-median plus Lynn high-pass."""
    if m <= 1:
        return sig
    pass #if
    med = ndimage.median_filter(sig, size=m, mode="nearest")
    return sig - lynn_filter(med, w=m)
pass #def


def signal_filter(sig: np.ndarray, w: int = LYN_WIND, m: int = MED_WIND, h: int = 0) -> np.ndarray:
    """Full record-level conditioning: baseline removal then Lynn band-pass.

    ``h`` optionally subtracts a moving-average high-frequency estimate.
    """
    sig = baseline_filter(sig, m)
    sig = lynn_filter(sig, w=w)
    if h > 0:
        hi = np.convolve(sig, np.ones(h) / h, mode="same")
        sig = sig - hi
    pass #if
    return sig
pass #def
