"""Loose validation of the vftx/ xqrs detector against the reference OSEA detector.

Requires the OSEA extension, built without libwfdb via:

    python setup_osea.py build_ext --inplace

If `qrs_detect` (OSEA) isn't importable, the module is skipped.

The two detectors are different algorithms — OSEA (resamples to 200 Hz, classifies
N/V/Q) vs xqrs (native rate, positions only) — so we validate LOOSE agreement, not
bit-equality: comparable beat counts and RR intervals, and (after removing a small
constant fiducial offset) good beat-position overlap for organized rhythms. VF
segments have no true QRS complexes, so position overlap there is expectedly poor.
"""
import json
import os
import sys

import numpy as np
import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

qrs_detect = pytest.importorskip(
    "qrs_detect", reason="OSEA extension not built (run: python setup_osea.py build_ext --inplace)")

from vftx.wfdb_detector import WfdbXqrsDetector  # noqa: E402

DATA = os.path.join(PROJ, "tests", "data")
_CACHE = np.load(os.path.join(DATA, "fixtures.npz"))
_META = json.load(open(os.path.join(DATA, "fixtures.json")))
_XQRS = WfdbXqrsDetector()


def _beats(sid, fs):
    sig = _CACHE[f"{sid}__sig"].astype(np.float64)
    osea_t = np.array([t for t, _ in qrs_detect.qrs_detect(sig, int(fs))]) / 200.0
    xq_t = np.array([t for t, _ in _XQRS.detect(sig, float(fs))]) / float(fs)
    return osea_t, xq_t


@pytest.mark.parametrize("m", _META, ids=[m["id"] for m in _META])
def test_beat_count_and_rr_agree(m):
    osea_t, xq_t = _beats(m["id"], m["fs"])
    assert abs(len(osea_t) - len(xq_t)) <= 2, (
        f"beat counts differ: OSEA={len(osea_t)} xqrs={len(xq_t)}")
    if len(osea_t) > 1 and len(xq_t) > 1:
        rr_o = float(np.mean(np.diff(osea_t)))
        rr_x = float(np.mean(np.diff(xq_t)))
        assert rr_x == pytest.approx(rr_o, rel=0.12), f"RR: OSEA={rr_o:.3f} xqrs={rr_x:.3f}"


@pytest.mark.parametrize("m", [m for m in _META if m["label"] != "VF"],
                         ids=[m["id"] for m in _META if m["label"] != "VF"])
def test_beat_positions_align_for_organized_rhythms(m):
    """For non-VF rhythms, after removing a constant fiducial offset, most beats
    should coincide within +/-25 ms."""
    osea_t, xq_t = _beats(m["id"], m["fs"])
    if len(osea_t) == 0 or len(xq_t) == 0:
        pytest.skip("no beats")
    offset = np.median([osea_t[np.argmin(np.abs(osea_t - t))] - t for t in xq_t])
    xq_a = xq_t + offset
    hit = sum(np.any(np.abs(osea_t - t) <= 0.025) for t in xq_a) / len(xq_a)
    assert hit >= 0.6, f"aligned position overlap {hit:.2f} < 0.6"
