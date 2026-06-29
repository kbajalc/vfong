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

    The substring search uses ``bytes.find`` (C-implemented) instead of a
    manual numpy loop: ``data.find(needle, 0, haystack_end)`` returns a match
    only when it starts at an index ``i`` with ``i + q_len <= haystack_end``,
    i.e. ``i`` in ``[0, s_len-1]`` — exactly the range the reference scans.
    The result is bit-for-bit identical to the naive loop but ~150x faster,
    which matters because IMF sequences reach 24000 bits (see imf_lz.py).
    """
    n = len(seq)
    if n < 2:
        return 0.0

    data = np.ascontiguousarray(seq, dtype=np.uint8).tobytes()
    cn = 1
    s_len = 1
    q_pos = s_len
    q_len = 1

    while (q_pos + q_len) <= n:
        # search data[q_pos:q_pos+q_len] inside data[0:s_len+q_len-1]
        haystack_end = s_len + q_len - 1
        if data.find(data[q_pos: q_pos + q_len], 0, haystack_end) != -1:
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
