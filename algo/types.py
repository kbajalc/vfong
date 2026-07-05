"""
Shared data types for the algo/ feature extraction package.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Configuration — one sub-config per concern
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    """Physical signal properties and preprocessing filter parameters."""
    sampling_rate: float = 250.0    # Hz — must match the actual recording
    highpass_hz: float = 1.0        # drift-suppression high-pass cutoff (reference: 1 Hz)
    lowpass_hz: float = 30.0        # anti-alias low-pass cutoff
    moving_avg_order: int = 5       # moving-average smoothing kernel size


@dataclass
class ThresholdConfig:
    """Parameters for threshold-crossing features: TCSC [0] and TCI [1]."""
    # TCSC [0]
    tcsc_threshold_pct: float = 0.20    # ±20 % of peak-to-peak amplitude
    tcsc_window_sec: float = 3.0        # Tukey-windowed sub-window length
    tcsc_step_sec: float = 1.0          # sub-window step

    # TCI [1]
    tci_threshold_pct: float = 0.20     # 20 % of peak-to-peak amplitude
    tci_window_sec: float = 1.0         # 1-second analysis windows


@dataclass
class EnergyConfig:
    """Parameters for energy-envelope features: STE [2], MEA [3], MAV [12]."""
    ste_tau_sec: float = 3.0        # STE exponential decay time constant
    mea_tau_sec: float = 0.2        # MEA exponential decay time constant
    mav_window_sec: float = 2.0     # MAV sliding window length


@dataclass
class PhaseSpaceConfig:
    """Parameters for phase-space features: PSR [4] and HILB [5]."""
    delay_sec: float = 0.5          # time-delay embedding lag
    grid_size: int = 40             # N×N occupancy grid (both axes)


@dataclass
class ComplexityConfig:
    """Parameters for complexity and count features: LZ [10], SpEn [11], Count1–3 [13–15].

    Count1–3 and SpEn require 250 Hz; the signal is resampled to
    ``resample_rate`` before computing these features.
    """
    resample_rate: float = 250.0    # target rate for count and SpEn features

    # SpEn [11]
    spen_duration_sec: float = 5.0  # use last N seconds of segment
    spen_m: int = 2                 # embedding dimension
    spen_r: float = 0.2             # tolerance as fraction of signal std

    # IMF1-5 LZ [17-21] — EMD backend for the intrinsic mode functions.
    # "ptsa"  = bundled PTSA (default): matches the reference bit-for-bit.
    # "pyemd" = the standard EMD-signal package (pip install EMD-signal); a valid
    #           but different EMD, so IMF_LZ values diverge from the reference.
    emd_backend: str = "ptsa"


@dataclass
class SegmentConfig:
    """Top-level configuration for one 8-second ECG segment.

    Compose sub-configs to override specific sections::

        cfg = SegmentConfig(
            signal=SignalConfig(sampling_rate=360.0),
            phase_space=PhaseSpaceConfig(grid_size=64),
        )

    All defaults match the reference Cython implementation (vf_features.pyx)
    exactly.
    """
    signal: SignalConfig = field(default_factory=SignalConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    phase_space: PhaseSpaceConfig = field(default_factory=PhaseSpaceConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)

    # When True, reproduce a known reference bug for bit-exact validation: the
    # reference TCSC multiplies overlapping windows of the shared preprocessed
    # signal by a Tukey window *in place* (vf_features.pyx:84), corrupting its own
    # later windows and every feature computed after it. The clean default leaves
    # each feature working on the uncorrupted signal. See algo/PLAN.md "Phase 4".
    reference_bug_compat: bool = False


# ---------------------------------------------------------------------------
# Intermediate preprocessing result
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedSignal:
    """Output of the preprocessing pipeline.

    Carries both signal forms that downstream extractors need:

    ``raw_mv``
        After ADC→mV conversion only, before normalisation.
        Used by: amplitude [16], QRS detector input [22–26].
    ``processed``
        Fully conditioned: drift-suppressed, std-normalised, low-pass filtered,
        moving-average smoothed.
        Used by: all other features.
    """
    raw_mv: np.ndarray      # shape (N,), float64, millivolts, un-normalised
    processed: np.ndarray   # shape (N,), float64, normalised + filtered
    sampling_rate: float    # Hz — from SignalConfig.sampling_rate


# ---------------------------------------------------------------------------
# Feature result
# ---------------------------------------------------------------------------

@dataclass
class Features:
    """All 27 features for one 8-second ECG segment.

    Field names and index order match the canonical ordering in CLAUDE.md::

      TCSC(0) TCI(1) STE(2) MEA(3) PSR(4) HILB(5)
      VF(6) M(7) A2(8) FM(9) LZ(10) SpEn(11)
      MAV(12) Count1(13) Count2(14) Count3(15) Amplitude(16)
      IMF1_LZ(17) IMF2_LZ(18) IMF3_LZ(19) IMF4_LZ(20) IMF5_LZ(21)
      RR(22) RR_Std(23) RR_CV(24) UR(25) VR(26)
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

    NAMES: ClassVar[tuple[str, ...]] = (
        "tcsc", "tci", "ste", "mea",
        "psr", "hilb",
        "vf_leak", "m", "a2", "fm",
        "lz", "spen",
        "mav", "count1", "count2", "count3",
        "amplitude",
        "imf1_lz", "imf2_lz", "imf3_lz", "imf4_lz", "imf5_lz",
        "rr", "rr_std", "rr_cv", "ur", "vr",
    )

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


# ---------------------------------------------------------------------------
# QRS detector protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class QRSDetector(Protocol):
    """Interface for any QRS detector used by qrs_features.py.

    Concrete implementations in this package:
    - ``WfdbXqrsDetector``  — wfdb.processing.xqrs_detect; all beats typed 'N'
    Other wrappers (outside this package):
    - ``OseaDetector``      — wraps the C OSEA library; N/V/Q classification
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
            Raw (un-normalised) ECG in millivolts.
        sampling_rate:
            Samples per second.

        Returns
        -------
        list of (sample_index, beat_type) tuples, ascending by sample_index.
        Beat types follow OSEA convention: ``'N'`` normal, ``'V'`` VPC,
        ``'Q'`` unknown/unclassified.
        """
        ...
