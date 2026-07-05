"""
Feature [1] — TCI: Threshold Crossing Intervals.

Reference: vf_features.pyx:threshold_crossing_intervals()

Algorithm:
  Split processed signal into 1-second segments.  For each segment find the
  threshold-crossing pattern: threshold = 20 % of the segment's own maximum.
  Track (begin_silence, tail_silence, n_pulses) per segment.
  For each interior segment compute the effective pulse count using fractional
  boundary correction (partial pulses at segment edges), then TCI = 1000 ms /
  effective_pulses.  Return the mean TCI across interior segments.
"""

import numpy as np

from vftx.types import PreprocessedSignal, SegmentConfig


def _find_threshold_crossing(segment: np.ndarray, threshold_ratio: float):
    """Return (begin_silence, tail_silence, n_pulses) for one 1-second window."""
    n = len(segment)
    threshold = float(np.max(segment)) * threshold_ratio
    raised = False
    n_pulses = 0
    first_rise = -1
    last_fall = -1

    for i in range(n):
        if raised:
            if segment[i] <= threshold:
                raised = False
                last_fall = i
        else:
            if segment[i] > threshold:
                raised = True
                if n_pulses == 0:
                    first_rise = i
                n_pulses += 1

    if n_pulses > 0:
        begin_silence = first_rise
        tail_silence = 0 if last_fall == -1 else (n - last_fall)
    else:
        begin_silence = n
        tail_silence = n
    return begin_silence, tail_silence, n_pulses


def compute_tci(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Threshold Crossing Intervals feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.threshold.tci_*`` and ``cfg.signal.sampling_rate``.
    """
    samples = sig.processed
    sr = int(sig.sampling_rate)
    n_samples = len(samples)
    threshold_ratio = cfg.threshold.tci_threshold_pct

    seg_size = sr  # 1-second segments
    pulses = []
    for begin in range(0, n_samples, seg_size):
        end = begin + seg_size
        segment = samples[begin:end]
        pulses.append(_find_threshold_crossing(segment, threshold_ratio))

    n_segs = len(pulses)
    if n_segs < 3:
        return 1000.0

    tcis: list[float] = []
    for i in range(1, n_segs - 1):
        t1 = pulses[i - 1][1]   # tail_silence of previous segment
        t4 = pulses[i + 1][0]   # begin_silence of next segment
        t2, t3, n_pulses = pulses[i]

        fraction1 = float(t2) / float(t1 + t2) if (t1 + t2) > 0 else 0.0
        fraction2 = float(t3) / float(t3 + t4) if (t3 + t4) > 0 else 0.0
        effective = float(n_pulses - 1) + fraction1 + fraction2
        tcis.append(1000.0 / effective if effective > 0 else 1000.0)

    return float(np.mean(tcis)) if tcis else 1000.0
