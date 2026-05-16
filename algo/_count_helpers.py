"""
Shared bandpass filter and count computation for Count1-3 features [13-15].

Reference: vf_features.pyx:auxiliary_counts()

IIR bandpass with central freq ~14.6 Hz at 250 Hz:
  FS[i] = (14·FS[i-1] − 7·FS[i-2] + (S[i] − S[i-2]) / 2) / 8
"""

import numpy as np
import scipy.signal as ss

from algo.types import PreprocessedSignal, SegmentConfig


def _aux_counts(sig: PreprocessedSignal, cfg: SegmentConfig) -> tuple[int, int, int]:
    """Return (count1, count2, count3) as raw integer sample counts.

    Replicates vf_features.pyx:auxiliary_counts() exactly, including the
    quirk that after resampling to 250 Hz the loop step stays at the original
    sampling_rate (not 250).
    """
    sr = int(sig.sampling_rate)
    n = len(sig.processed)
    samples = sig.processed.copy()

    if sr != 250:
        samples = ss.resample(samples, int(n / sr * 250))
        n = len(samples)

    # custom IIR bandpass
    fs = np.zeros(n)
    for i in range(2, n):
        fs[i] = (14.0 * fs[i - 1] - 7.0 * fs[i - 2] + (samples[i] - samples[i - 2]) / 2.0) / 8.0

    count1, count2, count3 = 0, 0, 0
    for i in range(0, n, sr):  # note: step = original sr (matches reference quirk)
        seg = fs[i: i + sr]
        if len(seg) == 0:
            continue
        fs_max = float(np.max(seg))
        fs_mean = float(np.mean(seg))
        fs_md = float(np.mean(np.abs(seg - fs_mean)))
        count1 += int(np.sum(seg >= 0.5 * fs_max))
        count2 += int(np.sum(seg >= fs_mean))
        count3 += int(np.sum((seg >= fs_mean - fs_md) & (seg <= fs_mean + fs_md)))

    return count1, count2, count3
