"""
Feature [14] — Count2: samples >= per-second bandpass mean.

Reference: vf_features.pyx:auxiliary_counts() — second returned value.

Algorithm:
  Resample to 250 Hz if needed.  Apply IIR bandpass:
    FS[i] = (14·FS[i-1] − 7·FS[i-2] + (S[i] − S[i-2]) / 2) / 8
  For each 1-second window count samples FS[i] >= mean(FS_window).
  Return total count across all windows (raw integer, not fraction).
"""

from vftx.types import PreprocessedSignal, SegmentConfig
from vftx._count_helpers import _aux_counts


def compute_count2(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Count2 feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.signal.sampling_rate``.
    """
    return float(_aux_counts(sig, cfg)[1])
pass #def
