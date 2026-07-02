"""Validate the QRS-derived feature computation [22-26].

Full detector-level agreement (reference OSEA vs algo xqrs) needs libwfdb, which
isn't available on the dev machine — and the detectors differ by design (xqrs
returns 'N' only, so UR/VR are always 0; OSEA classifies N/V/Q). What we CAN and
do validate here is the statistics FORMULA in compute_qrs_features against a
faithful transcription of the reference vf_features.pyx:beat_statistics().

Key detail: the reference divides RR intervals by a hardcoded 200 (OSEA resamples
to 200 Hz internally and returns 200 Hz indices). algo divides by the detector's
native sampling rate. Feeding sr=200 makes the two directly comparable.
"""
import os
import sys

import numpy as np
import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from algo.qrs_features import compute_qrs_features  # noqa: E402
from algo.types import PreprocessedSignal, SegmentConfig, SignalConfig  # noqa: E402


class _MockDetector:
    """QRSDetector returning a fixed beat list, ignoring the signal."""

    def __init__(self, beats):
        self._beats = beats

    def detect(self, signal_mv, sampling_rate):
        return self._beats


def _ref_beat_statistics(beats):
    """Faithful transcription of vf_features.pyx:beat_statistics() (+ rr_cv)."""
    rr_average = rr_std = unknown = vpc = 0.0
    rr_intervals = []
    for i in range(1, len(beats)):
        last_t, _ = beats[i - 1]
        t, bt = beats[i]
        rr_intervals.append((t - last_t) / 200.0)   # reference hardcodes 200
        if bt == "Q":
            unknown += 1
        elif bt == "V":
            vpc += 1
    if rr_intervals:
        rr_average = float(np.mean(rr_intervals))
        rr_std = float(np.std(rr_intervals))
    if len(beats) > 1:
        unknown /= (len(beats) - 1)
        vpc /= (len(beats) - 1)
    rr_cv = rr_std / rr_average if rr_average else 0.0
    return rr_average, rr_std, rr_cv, unknown, vpc


def _sig():
    z = np.zeros(1600, dtype=np.float64)
    return PreprocessedSignal(raw_mv=z, processed=z, sampling_rate=200.0)


@pytest.mark.parametrize("seed", range(6))
def test_formula_matches_reference_at_200hz(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 20))
    times = np.sort(rng.choice(np.arange(1600), size=n, replace=False))
    types = rng.choice(["N", "V", "Q"], size=n)
    beats = [(int(t), str(bt)) for t, bt in zip(times, types)]

    cfg = SegmentConfig(signal=SignalConfig(sampling_rate=200.0))
    got = compute_qrs_features(_sig(), cfg, _MockDetector(beats))
    exp = _ref_beat_statistics(beats)
    assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("beats", [[], [(10, "N")]])
def test_degenerate_beat_lists_return_zeros(beats):
    cfg = SegmentConfig()
    assert compute_qrs_features(_sig(), cfg, _MockDetector(beats)) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_ur_vr_counting_and_first_beat_skip():
    # 4 beats -> 3 classified intervals; beat[1]=V, beat[2]=Q, beat[3]=N
    beats = [(0, "N"), (200, "V"), (400, "Q"), (600, "N")]
    cfg = SegmentConfig(signal=SignalConfig(sampling_rate=200.0))
    rr, rr_std, rr_cv, ur, vr = compute_qrs_features(_sig(), cfg, _MockDetector(beats))
    assert rr == pytest.approx(1.0)      # 200 samples / 200 Hz
    assert rr_std == pytest.approx(0.0)
    assert ur == pytest.approx(1 / 3)
    assert vr == pytest.approx(1 / 3)
