"""
Main feature extraction entry point.

Usage::

    from vftx.extract import extract_features
    from vftx.types import SegmentConfig
    import numpy as np

    signal_mv = np.random.randn(2000)          # 8 s at 250 Hz, already in mV
    cfg = SegmentConfig(sampling_rate=250.0)
    features = extract_features(signal_mv, cfg)
    array_27 = features.to_array()             # shape (27,), float64

The caller is responsible for:
  - ADC → mV conversion (subtract adc_zero, divide by gain)
  - Verifying the segment is 8 seconds and artifact-free
  - Providing an appropriate QRSDetector if QRS features are needed

QRS features (indices 22–26) require a detector instance.  Pass
``qrs_detector=None`` to skip them (they will be 0.0 in the result).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vftx.amplitude import compute_amplitude
from vftx.count1 import compute_count1
from vftx.count2 import compute_count2
from vftx.count3 import compute_count3
from vftx.hilbert import compute_hilbert
from vftx.imf_lz import compute_imf_lz
from vftx.lz import compute_lz
from vftx.mav import compute_mav
from vftx.mea import compute_mea
from vftx.preprocessing import preprocess
from vftx.psr import compute_psr
from vftx.qrs_features import compute_qrs_features
from vftx.sample_entropy import compute_sample_entropy
from vftx.spectral_a2 import compute_spectral_a2
from vftx.spectral_fm import compute_spectral_fm
from vftx.spectral_m import compute_spectral_m
from vftx.ste import compute_ste
from vftx.tci import compute_tci
from vftx.tcsc import compute_tcsc
from vftx.types import Features, QRSDetector, SegmentConfig
from vftx.vf_leak import compute_vf_leak


def extract_features(
    signal_mv: np.ndarray,
    cfg: SegmentConfig,
    qrs_detector: Optional[QRSDetector] = None,
) -> Features:
    """Extract all 27 features from one 8-second ECG segment.

    Parameters
    ----------
    signal_mv:
        1-D float64 array of ECG samples in millivolts.  Length should be
        ``cfg.sampling_rate * 8``.  The signal must already be converted from
        ADC counts to millivolts by the caller.
    cfg:
        Extraction configuration.  Defaults match the reference Cython
        implementation exactly.
    qrs_detector:
        Implementation of the ``QRSDetector`` protocol.  If ``None``, QRS-
        derived features [22–26] are left at 0.0.

    Returns
    -------
    Features
        Dataclass with all 27 named feature fields populated.
    """
    sig = preprocess(signal_mv, cfg)

    # --- Time domain ---
    tcsc = compute_tcsc(sig, cfg)
    tci = compute_tci(sig, cfg)
    ste = compute_ste(sig, cfg)
    mea = compute_mea(sig, cfg)

    # --- Phase space ---
    psr = compute_psr(sig, cfg)
    hilb = compute_hilbert(sig, cfg)

    # --- Frequency domain ---
    vf = compute_vf_leak(sig, cfg)
    m = compute_spectral_m(sig, cfg)
    a2 = compute_spectral_a2(sig, cfg)
    fm = compute_spectral_fm(sig, cfg)

    # --- Complexity ---
    lz = compute_lz(sig, cfg)
    spen = compute_sample_entropy(sig, cfg)

    # --- Time domain (cont.) ---
    mav = compute_mav(sig, cfg)
    c1 = compute_count1(sig, cfg)
    c2 = compute_count2(sig, cfg)
    c3 = compute_count3(sig, cfg)
    amp = compute_amplitude(sig, cfg)

    # --- EMD complexity ---
    imf1, imf2, imf3, imf4, imf5 = compute_imf_lz(sig, cfg)

    # --- QRS-derived (optional) ---
    rr = rr_std = rr_cv = ur = vr = 0.0
    if qrs_detector is not None:
        rr, rr_std, rr_cv, ur, vr = compute_qrs_features(sig, cfg, qrs_detector)

    return Features(
        tcsc=tcsc, tci=tci, ste=ste, mea=mea,
        psr=psr, hilb=hilb,
        vf_leak=vf, m=m, a2=a2, fm=fm,
        lz=lz, spen=spen,
        mav=mav, count1=c1, count2=c2, count3=c3,
        amplitude=amp,
        imf1_lz=imf1, imf2_lz=imf2, imf3_lz=imf3, imf4_lz=imf4, imf5_lz=imf5,
        rr=rr, rr_std=rr_std, rr_cv=rr_cv, ur=ur, vr=vr,
    )
