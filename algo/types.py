"""
Shared data types for the algo/ feature extraction package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SegmentConfig:
    """All parameters that govern feature extraction for one 8-second segment.

    Defaults match the reference implementation in vf_features.pyx exactly.
    """
    # Signal
    sampling_rate: float = 250.0        # Hz — must reflect the actual recording rate

    # Preprocessing
    highpass_hz: float = 0.5            # drift-suppression high-pass cutoff
    lowpass_hz: float = 30.0            # anti-alias low-pass cutoff
    moving_avg_order: int = 5           # moving-average smoothing order

    # TCSC [0]
    tcsc_threshold_pct: float = 0.20    # ±20 % of peak-to-peak amplitude
    tcsc_window_sec: float = 3.0        # moving window length
    tcsc_step_sec: float = 1.0          # window step

    # TCI [1]
    tci_threshold_pct: float = 0.20     # 20 % of peak-to-peak amplitude
    tci_window_sec: float = 1.0         # 1-second analysis windows

    # STE [2]
    ste_tau_sec: float = 3.0            # exponential decay time constant

    # MEA [3]
    mea_tau_sec: float = 0.2            # exponential decay time constant

    # PSR [4] and HILB [5]
    psr_delay_sec: float = 0.5          # time-delay embedding delay
    psr_grid_size: int = 40             # N×N phase-space grid

    # MAV [12]
    mav_window_sec: float = 2.0         # sliding window length

    # Count1–3 [13–15] and SpEn [11] — require 250 Hz; signal is resampled if needed
    count_target_rate: float = 250.0

    # SpEn [11]
    spen_duration_sec: float = 5.0      # use last 5 s of segment
    spen_m: int = 2                     # embedding dimension
    spen_r: float = 0.2                 # tolerance (fraction of std)

    # Asystole gate — used upstream, not inside algo/
    asystole_threshold_mv: float = 0.15


# ---------------------------------------------------------------------------
# Intermediate preprocessing result
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedSignal:
    """Output of the preprocessing pipeline.

    Carries both forms of the signal that downstream feature extractors need:
    - ``raw_mv``: after ADC→mV conversion only — used by amplitude [16]
    - ``processed``: fully conditioned (drift-suppressed, normalised, filtered,
      smoothed) — used by all other features
    """
    raw_mv: np.ndarray          # shape (N,), float64, millivolts, un-normalised
    processed: np.ndarray       # shape (N,), float64, normalised + filtered
    sampling_rate: float        # Hz (may differ from SegmentConfig if resampled)


# ---------------------------------------------------------------------------
# Feature result
# ---------------------------------------------------------------------------

@dataclass
class Features:
    """All 27 features for one 8-second ECG segment.

    Field names and index order match the canonical ordering in CLAUDE.md:
      TCSC(0) TCI(1) STE(2) MEA(3) PSR(4) HILB(5) VF(6) M(7) A2(8) FM(9)
      LZ(10) SpEn(11) MAV(12) Count1(13) Count2(14) Count3(15)
      Amplitude(16) IMF1_LZ(17)…IMF5_LZ(21) RR(22) RR_Std(23) RR_CV(24)
      UR(25) VR(26)
    """
    # --- Time domain ---
    tcsc: float = 0.0           # [0]  Threshold Crossing Sample Count
    tci: float = 0.0            # [1]  Threshold Crossing Intervals
    ste: float = 0.0            # [2]  Short-Time Energy
    mea: float = 0.0            # [3]  Modified Exponential Algorithm
    # --- Phase space ---
    psr: float = 0.0            # [4]  Phase Space Reconstruction
    hilb: float = 0.0           # [5]  Hilbert-transform PSR
    # --- Frequency domain ---
    vf_leak: float = 0.0        # [6]  VF Leak
    m: float = 0.0              # [7]  Spectral parameter M
    a2: float = 0.0             # [8]  Energy ratio A2
    fm: float = 0.0             # [9]  Central frequency FM
    # --- Complexity ---
    lz: float = 0.0             # [10] Lempel-Ziv complexity
    spen: float = 0.0           # [11] Sample Entropy
    # --- Time domain (cont.) ---
    mav: float = 0.0            # [12] Mean Absolute Value
    count1: float = 0.0         # [13] Count1 — samples in 50–100 % amplitude range
    count2: float = 0.0         # [14] Count2 — samples above mean
    count3: float = 0.0         # [15] Count3 — samples within mean ± mean-deviation
    amplitude: float = 0.0      # [16] Peak-to-peak mV (raw, pre-normalisation)
    # --- EMD complexity ---
    imf1_lz: float = 0.0        # [17] LZ complexity of EMD mode 1
    imf2_lz: float = 0.0        # [18] LZ complexity of EMD mode 2
    imf3_lz: float = 0.0        # [19] LZ complexity of EMD mode 3
    imf4_lz: float = 0.0        # [20] LZ complexity of EMD mode 4
    imf5_lz: float = 0.0        # [21] LZ complexity of EMD mode 5
    # --- QRS-derived ---
    rr: float = 0.0             # [22] Mean RR interval (ms)
    rr_std: float = 0.0         # [23] Std of RR intervals (ms)
    rr_cv: float = 0.0          # [24] Coefficient of variation = rr_std / rr
    ur: float = 0.0             # [25] Unknown-beat ratio
    vr: float = 0.0             # [26] VPC-beat ratio

    def to_array(self) -> np.ndarray:
        """Return all 27 features as a float64 array in canonical index order."""
        return np.array([
            self.tcsc, self.tci, self.ste, self.mea,
            self.psr, self.hilb,
            self.vf_leak, self.m, self.a2, self.fm,
            self.lz, self.spen,
            self.mav, self.count1, self.count2, self.count3,
            self.amplitude,
            self.imf1_lz, self.imf2_lz, self.imf3_lz, self.imf4_lz, self.imf5_lz,
            self.rr, self.rr_std, self.rr_cv, self.ur, self.vr,
        ], dtype=np.float64)

    NAMES: tuple[str, ...] = field(default_factory=lambda: (
        "tcsc", "tci", "ste", "mea",
        "psr", "hilb",
        "vf_leak", "m", "a2", "fm",
        "lz", "spen",
        "mav", "count1", "count2", "count3",
        "amplitude",
        "imf1_lz", "imf2_lz", "imf3_lz", "imf4_lz", "imf5_lz",
        "rr", "rr_std", "rr_cv", "ur", "vr",
    ))


# ---------------------------------------------------------------------------
# QRS detector protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class QRSDetector(Protocol):
    """Interface for any QRS detector used by qrs_features.py.

    Implementations:
    - OseaDetector  — wraps the original C OSEA library via ctypes (reference)
    - NeuroKitDetector — pure Python via neurokit2 (production replacement)
    """

    def detect(
        self,
        signal_mv: np.ndarray,
        sampling_rate: float,
    ) -> list[tuple[int, str]]:
        """Detect QRS complexes and classify each beat.

        Parameters
        ----------
        signal_mv:
            ECG signal in millivolts, raw (un-normalised).
        sampling_rate:
            Samples per second.

        Returns
        -------
        List of ``(sample_index, beat_type)`` tuples in ascending sample order.
        Beat types follow OSEA convention: ``'N'`` normal, ``'V'`` VPC,
        ``'Q'`` unknown/unclassified, etc.
        """
        ...
