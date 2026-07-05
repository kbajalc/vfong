"""
Feature [13] — Count1: samples >= 50 % of per-second bandpass maximum.

Reference: vf_features.pyx:auxiliary_counts() — first returned value.

Algorithm:
  Resample to 250 Hz if needed.  Apply IIR bandpass:
    FS[i] = (14·FS[i-1] − 7·FS[i-2] + (S[i] − S[i-2]) / 2) / 8
  For each 1-second window count samples FS[i] >= 0.5 · max(FS_window).
  Return total count across all windows (raw integer, not fraction).
"""

from vftx.types import PreprocessedSignal, SegmentConfig
from vftx._count_helpers import _aux_counts


def compute_count1(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Count1 feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.signal.sampling_rate``.
    """
    return float(_aux_counts(sig, cfg)[0])
