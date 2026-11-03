"""vfnn.jekova: the JEKOVA5 band-pass counts, window sums and published threshold."""

import numpy as np
import pytest
from scipy import signal

from vfnn import data, jekova
from vfnn.filters import FS


def notebook_counts(seg: np.ndarray) -> tuple[int, int, int]:
    """Counts() of jekova/JEKOVA5.ipynb, transcribed, on one 1 s band-pass segment."""
    a = np.abs(np.asarray(seg, dtype=float))
    fmax, favg = np.max(a), np.mean(a)
    mdev = np.mean(np.abs(a - favg))
    return (int(np.sum(a >= fmax / 2)), int(np.sum(a >= favg)),
            int(np.sum((a >= favg - mdev) & (a <= favg + mdev))))
pass #def


def test_second_counts_match_notebook():
    x = np.random.default_rng(0).normal(size=30 * FS + 77)
    got = jekova.second_counts(x)
    bp = jekova.band_pass(x)
    want = np.array([notebook_counts(bp[s * FS:(s + 1) * FS]) for s in range(len(x) // FS)])
    assert got.shape == (30, 3)
    np.testing.assert_array_equal(got, want)
pass #def


def test_window_counts_sum_the_seconds():
    rng = np.random.default_rng(1)
    recs = [data.Rec("d", f"r{i}", rng.normal(size=40 * FS).astype(np.float32),
                     np.zeros(40 * FS, bool), np.zeros(40 * FS, bool)) for i in range(2)]
    win = data.make_windows(recs, seconds=8.0, step_s=1.0)
    got = jekova.window_counts(win)
    for i in (0, 5, len(win) - 1):
        sec = jekova.second_counts(recs[win.rec[i]].x)
        s = win.start[i] // FS
        np.testing.assert_array_equal(got[i], sec[s:s + 8].sum(axis=0))
    pass #for
pass #def


def test_misaligned_windows_rejected():
    r = data.Rec("d", "r", np.zeros(20 * FS, np.float32), np.zeros(20 * FS, bool), np.zeros(20 * FS, bool))
    win = data.make_windows([r], seconds=8.0, step_s=0.5)
    with pytest.raises(ValueError):
        jekova.window_counts(win)
    pass #with
pass #def


def test_published_threshold_is_cnt2_above_950_per_10s():
    assert jekova.published_threshold(8.0) == 761      # cnt2 > 760
    assert jekova.published_threshold(10.0) == 951     # cnt2 > 950
    assert jekova.published_threshold(4.0) == 381      # cnt2 > 380
pass #def


def test_band_pass_peaks_near_14_6_hz():
    w, h = signal.freqz(jekova.JEK_B, jekova.JEK_A, worN=4096, fs=FS)
    assert abs(w[np.argmax(np.abs(h))] - 14.6) < 0.5
pass #def
