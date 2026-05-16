"""
Feature [10] — LZ: Lempel-Ziv Complexity.

Reference: vf_features.pyx:complexity_measure() + vf_features_native.c:lempel_ziv_complexity()

Algorithm:
  Adaptive threshold selection (3-way):
    pos_peak = max(samples), neg_peak = min(samples)
    pos_count = samples with 0 < x < 0.1·pos_peak
    neg_count = samples with 0.1·neg_peak < x < 0
    if pos_count + neg_count < 40 % of N: threshold = 0
    elif pos_count < neg_count:           threshold = 0.2·pos_peak
    else:                                 threshold = 0.2·neg_peak
  Binarise: b[i] = 1 if samples[i] > threshold else 0.
  Apply LZ76 complexity: cn / (n / log2(n)).
"""

import math

import numpy as np

from algo.types import PreprocessedSignal, SegmentConfig


def _lz76(seq: np.ndarray) -> float:
    """LZ76 complexity normalised to cn / (n / log2(n)).

    Replicates vf_features_native.c:lempel_ziv_complexity() — scans from
    offset 0 up to s_length+q_length-1 (memmem-style substring search).
    """
    n = len(seq)
    if n < 2:
        return 0.0

    cn = 1
    s_len = 1
    q_pos = s_len
    q_len = 1

    while (q_pos + q_len) <= n:
        # search seq[q_pos:q_pos+q_len] inside seq[0:s_len+q_len-1]
        haystack_end = s_len + q_len - 1
        needle = seq[q_pos: q_pos + q_len]
        found = False
        for i in range(haystack_end - q_len + 1):
            if np.array_equal(seq[i: i + q_len], needle):
                found = True
                break
        if found:
            q_len += 1
        else:
            cn += 1
            s_len += q_len
            q_pos += q_len
            q_len = 1

    return cn / (n / math.log2(n))


def compute_lz(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Lempel-Ziv complexity feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Configuration (no per-feature parameters used).
    """
    samples = sig.processed
    n = len(samples)

    pos_peak = float(np.max(samples))
    neg_peak = float(np.min(samples))

    pos_count = int(np.sum((samples > 0) & (samples < 0.1 * pos_peak)))
    neg_count = int(np.sum((samples < 0) & (samples > 0.1 * neg_peak)))

    if (pos_count + neg_count) < 0.4 * n:
        threshold = 0.0
    elif pos_count < neg_count:
        threshold = 0.2 * pos_peak
    else:
        threshold = 0.2 * neg_peak

    binary = (samples > threshold).astype(np.uint8)
    return _lz76(binary)
