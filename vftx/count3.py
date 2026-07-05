"""
Feature [15] — Count3: samples within per-second mean ± mean-deviation.

Reference: vf_features.pyx:auxiliary_counts() — third returned value.

Algorithm:
  Resample to 250 Hz if needed.  Apply IIR bandpass:
    FS[i] = (14·FS[i-1] − 7·FS[i-2] + (S[i] − S[i-2]) / 2) / 8
  For each 1-second window: md = mean(|FS - mean(FS)|);
  count samples in [mean − md, mean + md].
  Return total count across all windows (raw integer, not fraction).
"""

from vftx.types import PreprocessedSignal, SegmentConfig
from vftx._count_helpers import _aux_counts


def compute_count3(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Count3 feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.signal.sampling_rate``.
    """
    return float(_aux_counts(sig, cfg)[2])
pass #def
