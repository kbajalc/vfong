"""Independent correctness check for vftx/ Sample Entropy [11].

SpEn cannot be validated against the reference: the reference's
pyeeg.samp_entropy builds its embedding via numpy as_strided with hardcoded
itemsize strides, which reads wrong memory for non-contiguous input. The
reference feeds it a non-contiguous slice, so its SpEn is wrong and even varies
call-to-call (see memory: reference-spen-nondeterministic).

Instead we validate vftx._samp_entropy against pyeeg.samp_entropy on well-formed
(C-contiguous) inputs, where pyeeg is correct and deterministic. This proves vftx
faithfully implements the intended algorithm.
"""
import os
import sys

import numpy as np
import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import pyeeg  # noqa: E402
from vftx.sample_entropy import _samp_entropy  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
@pytest.mark.parametrize("n", [50, 200, 1250])
def test_matches_pyeeg_on_contiguous_input(seed, n):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.standard_normal(n))
    r = 0.2 * float(np.std(x))
    expected = pyeeg.samp_entropy(x.copy(), 2, r)  # copy: guaranteed contiguous
    got = _samp_entropy(x, 2, r)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_robust_to_non_contiguous_input():
    """vftx must give the SAME result on a non-contiguous view as on its copy —
    unlike pyeeg, whose as_strided embed breaks on non-contiguous input."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal(4000)
    view = base[-1250:]                       # typically non-contiguous slice base
    assert _samp_entropy(view, 2, 0.2 * np.std(view)) == pytest.approx(   # type: ignore
        _samp_entropy(np.ascontiguousarray(view), 2, 0.2 * np.std(view)), # type: ignore
        rel=1e-12, abs=1e-12)
