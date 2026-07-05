"""Tests for the selectable EMD backend of the IMF1-5 LZ features [17-21].

Default backend is "ptsa" (bundled), which matches the reference bit-for-bit
(covered by test_agreement.py). Here we check the pluggable dispatch itself:
the "pyemd" backend runs and yields finite values (differing from ptsa, since
it is a valid but different EMD), and an unknown backend raises.
"""
import os
import sys

import numpy as np
import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from algo.imf_lz import compute_imf_lz  # noqa: E402
from algo.preprocessing import preprocess  # noqa: E402
from algo.types import ComplexityConfig, SegmentConfig, SignalConfig  # noqa: E402

_CACHE = np.load(os.path.join(PROJ, "tests", "data", "fixtures.npz"))


def _sig(seg_id="vfdb_418_0", fs=250.0):
    raw = _CACHE[f"{seg_id}__sig"]
    return preprocess(raw, SegmentConfig(signal=SignalConfig(sampling_rate=fs))), fs


def _cfg(backend, fs=250.0):
    return SegmentConfig(signal=SignalConfig(sampling_rate=fs),
                         complexity=ComplexityConfig(emd_backend=backend))


def test_ptsa_backend_is_default_and_finite():
    sig, fs = _sig()
    out = compute_imf_lz(sig, _cfg("ptsa", fs))
    assert len(out) == 5 and all(np.isfinite(v) for v in out)
    # default config uses ptsa
    assert compute_imf_lz(sig, SegmentConfig(signal=SignalConfig(sampling_rate=fs))) == out


def test_unknown_backend_raises():
    sig, fs = _sig()
    with pytest.raises(ValueError, match="unknown emd_backend"):
        compute_imf_lz(sig, _cfg("nope", fs))


def test_pyemd_backend_runs_and_differs():
    pytest.importorskip("PyEMD", reason="EMD-signal not installed")
    sig, fs = _sig()
    ptsa_out = compute_imf_lz(sig, _cfg("ptsa", fs))
    pyemd_out = compute_imf_lz(sig, _cfg("pyemd", fs))
    assert len(pyemd_out) == 5 and all(np.isfinite(v) for v in pyemd_out)
    # a different EMD -> at least one IMF_LZ value differs
    assert pyemd_out != ptsa_out
