"""Per-window vftx feature extraction for the dataset build.

Two feature families are handled specially, for reasons recorded in
``paper/PLAN.md`` (Phase 2 decisions):

QRS features [22-26] (RR, RR_Std, RR_CV, UR, VR) are excluded by design. A QRS
detector is required to blank during VF/VFL (there is no QRS to detect), so a
VF/VFL detector that depended on QRS output would be circular. Our detector is
signal-only, so no QRS-derived feature is computed.

IMF_LZ [17-21] (Lempel-Ziv on EMD modes) is excluded entirely. EMD costs about
2 s/window, so it cannot keep up with a real-time 1 s window step, and the
downstream target is a real-time detector. SpEn [11] costs about 55 ms/window
(real-time feasible, but the slowest kept feature), so it is off by default to
keep the bulk overlapping-window build cheap (~9 ms/window) and can be turned on
with the ``spen`` flag when a step needs it (for example the feature screen on a
subsample).

The cbor signal is 12-bit integers at a standard gain of 200; it is converted
to millivolts here by dividing by ``GAIN``. vftx's own frequency filtering is
off (``apply_filters=False``) because the record is already filtered upstream.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from vftx.amplitude import compute_amplitude
from vftx.count1 import compute_count1
from vftx.count2 import compute_count2
from vftx.count3 import compute_count3
from vftx.hilbert import compute_hilbert
from vftx.lz import compute_lz
from vftx.mav import compute_mav
from vftx.mea import compute_mea
from vftx.preprocessing import preprocess
from vftx.psr import compute_psr
from vftx.sample_entropy import compute_sample_entropy
from vftx.spectral_a2 import compute_spectral_a2
from vftx.spectral_fm import compute_spectral_fm
from vftx.spectral_m import compute_spectral_m
from vftx.ste import compute_ste
from vftx.tci import compute_tci
from vftx.tcsc import compute_tcsc
from vftx.types import SegmentConfig, SignalConfig
from vftx.vf_leak import compute_vf_leak

GAIN = 200.0  # cbor integer counts -> millivolts

# Cheap scalar non-QRS features (~9 ms/window total), computed for every window.
_CHEAP: list[tuple[str, Callable]] = [
    ("tcsc", compute_tcsc),
    ("tci", compute_tci),
    ("ste", compute_ste),
    ("mea", compute_mea),
    ("psr", compute_psr),
    ("hilb", compute_hilbert),
    ("vf_leak", compute_vf_leak),
    ("m", compute_spectral_m),
    ("a2", compute_spectral_a2),
    ("fm", compute_spectral_fm),
    ("lz", compute_lz),
    ("mav", compute_mav),
    ("count1", compute_count1),
    ("count2", compute_count2),
    ("count3", compute_count3),
    ("amplitude", compute_amplitude),
]

_SPEN_NAME = "spen"

# JEKOVA detector columns: the three counts on the ABSOLUTE band-pass output
# (Jekova and Krasteva 2004), distinct from the signed vftx count1/2/3 features.
# They are written to the TSV for the Phase 4 cascade tuning but are not part of
# the 16-feature screen (see paper/PLAN.md and vfta/jekova.py).
_JEKOVA_NAMES = ["jc1", "jc2", "jc3"]


def feature_names(spen: bool = False, jekova: bool = False) -> list[str]:
    """Ordered feature column names for the selected options.

    ``jekova`` appends the absolute-output JEKOVA counts (jc1, jc2, jc3); the
    feature screen keeps them off (16 features), the build turns them on.
    """
    names = [n for n, _ in _CHEAP]
    if spen:
        names.append(_SPEN_NAME)
    pass #if
    if jekova:
        names.extend(_JEKOVA_NAMES)
    pass #if
    return names
pass #def


_SIGNED_COUNTS = ("count1", "count2", "count3")


def screen_features() -> list[str]:
    """The analysis feature list, with the JEKOVA absolute counts as the band-pass family.

    vftx computes the band-pass counts on the signed filter output (``count1/2/3``),
    where ``count2`` is degenerate (near 0.5 for every rhythm). The JEKOVA candidate,
    and therefore the screen and shootout, uses the paper-faithful absolute-output
    counts (``jc1/jc2/jc3``) instead; the signed counts stay in the TSV as vftx
    reference features but are not analysed. See :mod:`vfta.jekova`.
    """
    return ["jc" + n[-1] if n in _SIGNED_COUNTS else n for n in feature_names()]
pass #def


def default_config(sampling_rate: float = 250.0) -> SegmentConfig:
    """vftx config for vfta extraction: filtering off, since records are pre-filtered."""
    return SegmentConfig(signal=SignalConfig(sampling_rate=sampling_rate, apply_filters=False))
pass #def


def window_features(sig_win, cfg: SegmentConfig, spen: bool = False,
                    jekova: bool = False) -> list[float]:
    """Feature values for one window.

    ``sig_win`` is the record-level filtered signal slice in integer counts; it
    is converted to millivolts (``/ GAIN``) and preprocessed once, then each
    selected feature is computed on the shared preprocessed signal. ``jekova``
    appends the three absolute-output JEKOVA counts, computed on the same
    preprocessed signal (see :func:`vfta.jekova.abs_counts`).
    """
    pp = preprocess(np.asarray(sig_win, dtype=float) / GAIN, cfg)
    vals = [fn(pp, cfg) for _, fn in _CHEAP]
    if spen:
        vals.append(compute_sample_entropy(pp, cfg))
    pass #if
    if jekova:
        from vfta.jekova import abs_counts
        vals.extend(abs_counts(pp, cfg))
    pass #if
    return vals
pass #def
