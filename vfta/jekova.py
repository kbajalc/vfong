"""JEKOVA candidate detector (Jekova and Krasteva 2004).

The JEKOVA algorithm passes the signal through a 14.6 Hz integer IIR band-pass
and derives three per-second counts (Count1-3) from the filtered output, then
classifies the epoch with a cascade of rules over those counts. vftx exposes the
counts as three features (``count1`` [13], ``count2`` [14], ``count3`` [15]) for
the feature screen; this module consolidates them behind one band-pass pass and
carries the decision cascade the paper defines.

The published cascade (Jekova and Krasteva 2004, section "Rhythm classification
by Count parameters"), evaluated per 10 s epoch:

  R1  Count1 < 250  and Count2 > 950  and Count1*Count2/Count3 < 210  -> non-shock
  R2  250 <= Count1 < 400 and Count2 < 600 and Count1*Count2/Count3 < 210 -> non-shock
  R3  Count1 >= 250 and Count2 > 950  -> shockable
  R4  Count2 >= 1100                  -> shockable
  else -> "not classified" (their Step 5 falls back to wave detection)

So JEKOVA is not three independent thresholds: the counts combine, including the
non-linear term Count1*Count2/Count3. Two adaptations are needed here and are
documented so the tuning stays honest:

1. Count definition. The paper counts the *absolute* filter output AbsFS; vftx
   follows Hong's reimplementation, which counts the signed filter output, so our
   counts are not numerically identical to the paper's. The counts are used as
   validated (they top the feature screen); only the decision cascade is retuned.

2. Window length. The paper's constants are raw sample counts over a 10 s epoch
   (2500 samples at 250 Hz). Our windows are 8 s and 4 s, so a raw constant does
   not transfer. Counts are normalised to fractions of the window sample count
   (``Count / N``), which are comparable across window lengths; ``DEFAULT_PARAMS``
   holds the published constants divided by 2500. The cascade is then tuned on our
   data per window length (see ``vfta.tuning``), because the count definition and
   window differ from the original.

The "not classified" branch cannot use the paper's wave detection without a peak
detector, which a signal-only VF/VFL detector avoids on purpose (see the QRS note
in ``vfta.features``). Those windows fall back to a threshold on the normalised
Count3, the single strongest count in the screen; that fallback fraction is one of
the tuned parameters.
"""

from __future__ import annotations

import numpy as np

from vftx._count_helpers import _aux_counts
from vftx.types import PreprocessedSignal, SegmentConfig

COUNT_NAMES = ("count1", "count2", "count3")

# Published constants (Jekova 2004) as fractions of the 10 s / 2500-sample epoch,
# so they carry across window lengths. The ratio term Count1*Count2/Count3 also
# scales with count, so it is normalised once (/2500).
DEFAULT_PARAMS: dict[str, float] = {
    "c1_lo": 250 / 2500,    # 0.100
    "c1_hi": 400 / 2500,    # 0.160
    "c2_lo": 600 / 2500,    # 0.240
    "c2_hi": 950 / 2500,    # 0.380
    "c2_top": 1100 / 2500,  # 0.440
    "ratio": 210 / 2500,    # 0.084
    "c3_fallback": 0.50,    # substitutes the paper's Step 5 wave-detection branch
}


def counts(pp: PreprocessedSignal, cfg: SegmentConfig) -> tuple[int, int, int]:
    """Return Hong's signed (Count1, Count2, Count3) from one band-pass pass.

    These are the vftx feature counts (signed filter output). Count2 is degenerate
    on the signed output (about half the samples sit above a near-zero mean), so
    the paper's cascade needs :func:`abs_counts` instead.
    """
    return _aux_counts(pp, cfg)
pass #def


def _bandpass(samples: np.ndarray) -> np.ndarray:
    """14.6 Hz integer-coefficient IIR band-pass (Jekova 2004), at 250 Hz."""
    n = len(samples)
    fs = np.zeros(n)
    for i in range(2, n):
        fs[i] = (14.0 * fs[i - 1] - 7.0 * fs[i - 2] + (samples[i] - samples[i - 2]) / 2.0) / 8.0
    pass #for
    return fs
pass #def


def abs_counts(pp: PreprocessedSignal, cfg: SegmentConfig) -> tuple[int, int, int]:
    """Return the paper-faithful (Count1, Count2, Count3) on the absolute output.

    This follows Jekova and Krasteva 2004 exactly: band-pass, take the absolute
    filter output AbsFS, and per 1 s stage count samples in [0.5*max, max],
    [mean, max], and [mean-MD, mean+MD] (MD the mean deviation). Unlike the signed
    counts, Count2 here is informative, which is what the cascade relies on.
    """
    import scipy.signal as ss

    sr = int(pp.sampling_rate)
    samples = np.asarray(pp.processed, dtype=float)
    if sr != 250:
        samples = np.asarray(ss.resample(samples, int(len(samples) / sr * 250)), dtype=float)
        sr = 250
    pass #if

    afs = np.abs(_bandpass(samples))
    c1 = c2 = c3 = 0
    for i in range(0, len(afs), sr):
        seg = afs[i:i + sr]
        if len(seg) == 0:
            continue
        pass #if
        smax = float(np.max(seg))
        smean = float(np.mean(seg))
        smd = float(np.mean(np.abs(seg - smean)))
        c1 += int(np.sum(seg >= 0.5 * smax))
        c2 += int(np.sum(seg >= smean))
        c3 += int(np.sum((seg >= smean - smd) & (seg <= smean + smd)))
    pass #for
    return c1, c2, c3
pass #def


def count_matrix(pp_windows, cfg: SegmentConfig) -> np.ndarray:
    """Stack (Count1, Count2, Count3) for an iterable of preprocessed windows."""
    return np.array([counts(w, cfg) for w in pp_windows], dtype=float)
pass #def


def to_fractions(c1, c2, c3, window_samples: int):
    """Normalise raw counts to fractions of the window sample count.

    Returns three float arrays in [0, 1]. Making the counts window-relative is
    what lets one set of cascade thresholds apply to both the 8 s and 4 s windows.
    """
    n = float(window_samples)
    return (np.asarray(c1, dtype=float) / n,
            np.asarray(c2, dtype=float) / n,
            np.asarray(c3, dtype=float) / n)
pass #def


def decide(f1, f2, f3, params: dict[str, float]) -> np.ndarray:
    """Apply the JEKOVA cascade to normalised counts. Returns a bool array.

    ``f1``, ``f2``, ``f3`` are the counts as fractions of the window length (see
    :func:`to_fractions`). ``params`` holds the cascade thresholds (see
    :data:`DEFAULT_PARAMS`). ``True`` means shockable.

    The non-shockable rules (R1, R2) and shockable rules (R3, R4) are mutually
    exclusive on their count conditions, so rule order does not matter; windows
    matching neither fall back to a Count3 threshold in place of the paper's
    wave-detection branch.
    """
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    f3 = np.asarray(f3, dtype=float)
    ratio = f1 * f2 / np.maximum(f3, 1e-9)

    p = params
    non = (((f1 < p["c1_lo"]) & (f2 > p["c2_hi"]) & (ratio < p["ratio"]))
           | ((f1 >= p["c1_lo"]) & (f1 < p["c1_hi"]) & (f2 < p["c2_lo"]) & (ratio < p["ratio"])))
    shock = ((f1 >= p["c1_lo"]) & (f2 > p["c2_hi"])) | (f2 >= p["c2_top"])
    fallback = f3 >= p["c3_fallback"]

    return np.where(non, False, np.where(shock, True, fallback))
pass #def
