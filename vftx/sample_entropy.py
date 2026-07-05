"""
Feature [11] — SpEn: Sample Entropy.

Reference: pyeeg/__init__.py:samp_entropy() called from vf_features.pyx

Algorithm:
  Use the last 5 seconds (1250 samples at 250 Hz; resample if needed).
  m=2, r=0.2·std(segment).
  Build (N-m+1)×m embedding matrix Em.  InRange[i,j] = max_k|Em[i,k]-Em[j,k]| ≤ r.
  Cm = row sums (self-pairs excluded).
  Cmp extends each template by one sample; SpEn = log(Σ Cm / Σ Cmp).
"""

import numpy as np
import scipy.signal as ss

from vftx.types import PreprocessedSignal, SegmentConfig


def _samp_entropy(x: np.ndarray, m: int, r: float) -> float:
    """Replicate pyeeg.samp_entropy(X, M, R) exactly."""
    N = len(x)
    # Em: (N-m+1, m) embedding matrix with lag 1
    Em = np.array([x[i: i + m] for i in range(N - m + 1)])
    A = np.tile(Em, (len(Em), 1, 1))          # (N-m+1, N-m+1, m)
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)
    InRange = np.max(D, axis=2) <= r
    np.fill_diagonal(InRange, False)
    Cm = InRange.sum(axis=0)

    x_tail = x[m:]                            # length N-m
    Dp = np.abs(
        np.tile(x_tail, (N - m, 1)) - np.tile(x_tail, (N - m, 1)).T
    )
    Cmp = np.logical_and(Dp <= r, InRange[:-1, :-1]).sum(axis=0)

    return float(np.log(np.sum(Cm + 1e-100) / np.sum(Cmp + 1e-100)))


def compute_sample_entropy(sig: PreprocessedSignal, cfg: SegmentConfig) -> float:
    """Return the Sample Entropy feature.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.complexity`` and ``cfg.signal.sampling_rate``.
    """
    cc = cfg.complexity
    target_sr = cc.resample_rate  # 250 Hz
    n_window = int(cc.spen_duration_sec * target_sr)  # 1250

    samples = sig.processed
    if sig.sampling_rate != target_sr:
        n_target = int(len(samples) / sig.sampling_rate * target_sr)
        samples = ss.resample(samples, n_target)

    segment = samples[-n_window:]
    r = cc.spen_r * float(np.std(segment)) # type: ignore
    return _samp_entropy(segment, cc.spen_m, r) # type: ignore
