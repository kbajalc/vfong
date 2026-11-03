"""vfnn.filters must match vfta.filters.signal_filter for the same window lengths."""

import numpy as np
import pytest

from vfnn import filters
from vfta import filters as vfta_filters


@pytest.fixture(scope="module")
def ecg():
    rng = np.random.default_rng(0)
    t = np.arange(5000) / 250
    base = 0.5 * np.sin(2 * np.pi * 0.3 * t)
    beats = np.zeros_like(t)
    beats[::200] = 1.0
    beats = np.convolve(beats, np.hanning(9), mode="same")
    return base + beats + 0.02 * rng.standard_normal(t.size)
pass #def


def test_window_lengths():
    assert filters.samples(596) == 149
    assert filters.samples(40) == 10
pass #def


def test_matches_vfta(ecg):
    mine = filters.filter_record(ecg, fs=250, median_ms=596, lynn_ms=40)
    ref = vfta_filters.signal_filter(ecg, w=10, m=149, h=0)
    assert np.allclose(mine, ref)
pass #def


def test_lynn_has_unit_dc_gain():
    y = filters.lynn_lowpass(np.ones(1000), 10)
    assert np.allclose(y[20:-20], 1.0)
pass #def


def test_baseline_is_removed():
    x = 2.0 + np.sin(2 * np.pi * 0.2 * np.arange(5000) / 250)
    y = filters.filter_record(x)
    assert abs(np.mean(y[500:-500])) < 0.05
pass #def


def test_normalize_window():
    w = filters.normalize_window(np.random.default_rng(1).normal(5, 3, (4, 2000)))
    assert np.allclose(w.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(w.std(axis=-1), 1, atol=1e-3)
pass #def
