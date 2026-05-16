"""
Features [22–26] — RR, RR_Std, RR_CV, UR, VR: QRS-derived features.

Reference: vf_features.pyx:beat_statistics()

Algorithm summary:
  Run QRS detection on the RAW mV signal (not the processed signal — the detector
  has its own internal bandpass filter).  The OSEA detector requires a 5-second
  warm-up pass (discarded) before the full 8-second pass to overcome initialisation
  latency.  Beat classification: 'N' normal, 'V' VPC, 'Q' unknown.

  From the detected beat list (skipping beat index 0 — often misclassified):
    RR     = mean RR interval in milliseconds
    RR_Std = standard deviation of RR intervals
    RR_CV  = RR_Std / RR  (coefficient of variation)
    UR     = unknown-beat ratio  = count('Q') / (len(beats) - 1)
    VR     = VPC-beat ratio      = count('V') / (len(beats) - 1)
"""

from __future__ import annotations

import numpy as np

from algo.types import PreprocessedSignal, QRSDetector, SegmentConfig


def compute_qrs_features(
    sig: PreprocessedSignal,
    cfg: SegmentConfig,
    detector: QRSDetector,
) -> tuple[float, float, float, float, float]:
    """Return all five QRS-derived features.

    Parameters
    ----------
    sig:
        Preprocessed signal — uses ``sig.raw_mv`` (detector needs raw signal).
    cfg:
        Extraction configuration (``cfg.sampling_rate``).
    detector:
        Any object satisfying the ``QRSDetector`` protocol.

    Returns
    -------
    tuple of five floats
        ``(rr, rr_std, rr_cv, ur, vr)``
    """
    # --- STUB ---
    return 0.0, 0.0, 0.0, 0.0, 0.0
