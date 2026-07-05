"""
Features [17–21] — IMF1_LZ … IMF5_LZ: LZ complexity of EMD intrinsic mode functions.

Reference: vf_features.pyx:emd_features() + vf_features_native.c:imf_lempel_ziv_complexity()

Algorithm:
  1. Resample processed signal to 250 Hz.
  2. Normalise: (s − min(s)) / max(s)  — NOTE: divide by max, not by range.
  3. Scale to 12-bit integers: x * 2^12 → uint16.
  4. Run ptsa EMD with max_modes=5 to extract 5 IMFs.
  5. For each IMF: cast float values to uint16, encode each value as 12 bits
     MSB-first, concatenate all bits into a binary string, compute LZ76.

The 12-bit encoding: for each uint16 value v, bits are stored at positions
[i*12 ... i*12+11] as [bit11, bit10, ..., bit1, bit0] (MSB first).
"""

import numpy as np
import scipy.signal as ss

from vftx.lz import _lz76
from vftx.types import PreprocessedSignal, SegmentConfig

try:
    from ptsa.ptsa.emd import emd as _ptsa_emd
    _HAS_PTSA = True
except ImportError:
    _HAS_PTSA = False
pass #try


def _run_emd(emd_input: np.ndarray, backend: str) -> list:
    """Decompose ``emd_input`` into intrinsic mode functions (max 5).

    ``backend``:
      "ptsa"  — bundled PTSA EMD (default); matches the reference bit-for-bit.
      "pyemd" — the standard EMD-signal package; a valid but different EMD, so
                the resulting IMF_LZ features diverge from the reference.
    """
    if backend == "ptsa":
        if not _HAS_PTSA:
            raise ImportError("emd_backend='ptsa' but the bundled ptsa package is unavailable")
        pass #if
        return _ptsa_emd(emd_input, max_modes=5)  # type: ignore
    pass #if
    if backend == "pyemd":
        try:
            from PyEMD import EMD
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "emd_backend='pyemd' requires the EMD-signal package "
                "(pip install EMD-signal)") from exc
        pass #try
        return list(EMD()(emd_input.astype(np.float64), max_imf=5))
    pass #if
    raise ValueError(f"unknown emd_backend {backend!r} (expected 'ptsa' or 'pyemd')")
pass #def


def _imf_lz_complexity(imf: np.ndarray) -> float:
    """Replicate vf_features_native.c:imf_lempel_ziv_complexity().

    Each float in imf is cast to uint16, then its 12 bits are stored MSB-first
    into a flat bit array before running LZ76.
    """
    length = len(imf)
    bin_len = length * 12
    binary_seq = np.zeros(bin_len, dtype=np.uint8)
    values = imf.astype(np.uint16)
    for i in range(length):
        j = i * 12 + 11  # matching C: j starts at 11 and increments by 12
        v = int(values[i])
        for c in range(12):
            binary_seq[j - c] = (v >> c) & 1
        pass #for
    pass #for
    return _lz76(binary_seq)
pass #def


def compute_imf_lz(
    sig: PreprocessedSignal,
    cfg: SegmentConfig,
) -> tuple[float, float, float, float, float]:
    """Return LZ complexity for each of the five EMD intrinsic mode functions.

    Parameters
    ----------
    sig:
        Preprocessed signal (use ``sig.processed``).
    cfg:
        Uses ``cfg.signal.sampling_rate`` and ``cfg.complexity.emd_backend``.

    Returns
    -------
    tuple of five floats
        ``(imf1_lz, imf2_lz, imf3_lz, imf4_lz, imf5_lz)``
    """
    samples = sig.processed
    sr = sig.sampling_rate

    # 1. Resample to 250 Hz
    if sr != 250.0:
        n_out = int(len(samples) / sr * 250)
        samples = ss.resample(samples, n_out)
    pass #if

    # 2. Normalise: (s - min) / max  (not range — matches reference)
    s_min = np.min(samples) # type: ignore
    s_max = np.max(samples) # type: ignore
    if s_max == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    pass #if
    normalised = (samples - s_min) / s_max

    # 3. Scale to 12-bit uint16
    emd_input = (normalised * (2 ** 12)).astype(np.uint16)

    # 4. EMD — all 5 modes in one call (backend selectable, default ptsa)
    imfs = _run_emd(emd_input, cfg.complexity.emd_backend)

    results: list[float] = []
    for i in range(5):
        if i < len(imfs):
            results.append(_imf_lz_complexity(imfs[i].astype(np.float64)))
        else:
            results.append(0.0)
        pass #if
    pass #for

    return tuple(results)  # type: ignore[return-value]
pass #def
