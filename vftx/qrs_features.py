"""
Features [22–26] — RR, RR_Std, RR_CV, UR, VR: QRS-derived features.

Reference: vf_features.pyx:beat_statistics()

Algorithm:
  Run QRS detection on the RAW mV signal (the detector has its own bandpass).
  From detected beat list (skipping beat index 0 — often misclassified):
    RR     = mean inter-beat interval (seconds at sig.sampling_rate)
    RR_Std = std of inter-beat intervals
    RR_CV  = RR_Std / RR
    UR     = fraction of 'Q' (unknown) beats
    VR     = fraction of 'V' (VPC) beats

NOTE on sampling rate: the reference OSEA detector resamples to 200 Hz
internally and returns sample indices at 200 Hz.  If using an OSEA-compatible
detector wrapper, divide beat intervals by 200 rather than sig.sampling_rate.
For detectors that return indices in the signal's native sampling rate, no
adjustment is needed.  This implementation divides by sig.sampling_rate.
"""

from __future__ import annotations

import numpy as np

from vftx.types import PreprocessedSignal, QRSDetector, SegmentConfig


def compute_qrs_features(
    sig: PreprocessedSignal,
    cfg: SegmentConfig,
    detector: QRSDetector,
) -> tuple[float, float, float, float, float]:
    """Return all five QRS-derived features.

    Parameters
    ----------
    sig:
        Uses ``sig.raw_mv`` (detector needs raw un-normalised signal).
    cfg:
        Uses ``cfg.signal.sampling_rate``.
    detector:
        Any object satisfying the ``QRSDetector`` protocol.

    Returns
    -------
    tuple of five floats
        ``(rr, rr_std, rr_cv, ur, vr)``
    """
    beats = detector.detect(sig.raw_mv, sig.sampling_rate)
    if len(beats) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    pass #if

    sr = sig.sampling_rate
    rr_intervals: list[float] = []
    unknown = 0
    vpc = 0

    for i in range(1, len(beats)):
        prev_time, _ = beats[i - 1]
        beat_time, beat_type = beats[i]
        rr_intervals.append((beat_time - prev_time) / sr)
        if beat_type == "Q":
            unknown += 1
        elif beat_type == "V":
            vpc += 1
        pass #if
    pass #for

    rr = float(np.mean(rr_intervals)) if rr_intervals else 0.0
    rr_std = float(np.std(rr_intervals)) if rr_intervals else 0.0
    rr_cv = rr_std / rr if rr != 0.0 else 0.0

    n_classified = len(beats) - 1  # skip first beat (index 0)
    ur = unknown / n_classified if n_classified > 0 else 0.0
    vr = vpc / n_classified if n_classified > 0 else 0.0

    return rr, rr_std, rr_cv, ur, vr
pass #def
