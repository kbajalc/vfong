"""Numerical agreement of algo/ vs the reference Cython implementation.

Hermetic: reads cached signals + reference vectors from tests/data/ (produced by
tests/gen_fixtures.py). No network, libwfdb, or Cython needed at test time.

    conda activate dev
    pytest tests/test_agreement.py -v

Status (Phase 4): only `amplitude` and the QRS-zero features match today; the ~20
diverging features are marked xfail(strict=False) with the fix TODO. As each
feature is fixed, remove its index from XFAIL_FEATURES so the suite guards it.
An xfail that starts passing shows up as XPASS — a nudge to promote it.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
sys.path.insert(0, PROJ)

from algo.extract import extract_features as algo_extract  # noqa: E402
from algo.types import Features, SegmentConfig, SignalConfig  # noqa: E402

NAMES = Features.NAMES

# --- tolerances -------------------------------------------------------------
# Complexity / entropy paths legitimately differ at the ULP level; everything
# else should match the reference tightly once implemented correctly.
_COMPLEXITY = {10, 11, 17, 18, 19, 20, 21}  # LZ, SpEn, IMF1-5
_RTOL_DEFAULT, _ATOL_DEFAULT = 1e-6, 1e-9
_RTOL_COMPLEXITY, _ATOL_COMPLEXITY = 1e-3, 1e-6

# With reference_bug_compat=True, 26/27 features match the reference bit-for-bit.
# [11] spen is PERMANENTLY xfail against the reference: the reference's
# pyeeg.samp_entropy uses as_strided on a non-contiguous slice and is wrong +
# non-deterministic (no stable ground truth). algo's SpEn is correct and is
# validated independently in test_sample_entropy.py. See memory:
# reference-spen-nondeterministic.
XFAIL_FEATURES = {11}


def _tol(idx):
    if idx in _COMPLEXITY:
        return _RTOL_COMPLEXITY, _ATOL_COMPLEXITY
    return _RTOL_DEFAULT, _ATOL_DEFAULT


# --- fixture cache ----------------------------------------------------------
_NPZ = os.path.join(DATA, "fixtures.npz")
_JSON = os.path.join(DATA, "fixtures.json")

if not (os.path.exists(_NPZ) and os.path.exists(_JSON)):
    pytest.skip(
        "fixtures missing — run `python tests/gen_fixtures.py` (needs reference "
        "build + network)", allow_module_level=True)

_CACHE = np.load(_NPZ)
with open(_JSON) as _fh:
    _META = json.load(_fh)

# Cache algo output per segment so each of the 27 parametrized cases is cheap.
_ALGO = {}


def _algo_vector(seg_id, fs):
    if seg_id not in _ALGO:
        sig = _CACHE[f"{seg_id}__sig"]
        # reference_bug_compat reproduces the TCSC in-place mutation so algo can
        # be validated bit-for-bit against the (buggy) reference.
        cfg = SegmentConfig(signal=SignalConfig(sampling_rate=float(fs)),
                            reference_bug_compat=True)
        _ALGO[seg_id] = algo_extract(sig, cfg, qrs_detector=None).to_array()
    return _ALGO[seg_id]


def _params():
    out = []
    for m in _META:
        for idx in range(27):
            marks = ()
            if idx in XFAIL_FEATURES:
                marks = pytest.mark.xfail(
                    reason=f"{NAMES[idx]} diverges from reference (Phase 4 TODO)",
                    strict=False)
            out.append(pytest.param(m["id"], m["fs"], idx,
                                    id=f"{m['id']}-{idx:02d}-{NAMES[idx]}",
                                    marks=marks))
    return out


@pytest.mark.parametrize("seg_id,fs,idx", _params())
def test_feature_matches_reference(seg_id, fs, idx):
    ref = _CACHE[f"{seg_id}__ref"][idx]
    got = _algo_vector(seg_id, fs)[idx]
    rtol, atol = _tol(idx)
    assert got == pytest.approx(ref, rel=rtol, abs=atol), (
        f"{NAMES[idx]}: reference={ref!r} algo={got!r}")
