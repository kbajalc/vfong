# algo/ — Pure-Python Feature Extraction: Implementation Plan

## Goal

A clean, pure-Python reimplementation of the 27-feature ECG segment feature extractor
originally written as Cython extensions (`vf_features.pyx`, `signal_processing.pyx`,
`vf_features_native.c`). The existing code is kept intact as the **reference implementation**
and is never modified.

The new package lives in `algo/` and depends only on standard scientific Python libraries
(numpy, scipy, optionally neurokit2 / PyEMD / antropy). No Cython, no C extensions.

---

## Guiding principles

- **Type-annotated throughout** — every function signature carries full type hints
- **API first, correctness second** — stubs before implementations; tests before optimisation
- **One file per feature** — each of the 27 features (or natural group) has its own module,
  making it easy to review, test, and replace independently
- **Config-driven** — all tunable parameters live in `SegmentConfig`; defaults match the
  reference implementation exactly
- **Dependency injection for QRS detection** — `QRSDetector` is a `typing.Protocol`;
  the OSEA C implementation and pure-Python alternatives are interchangeable

---

## Package layout

```
algo/
  __init__.py
  PLAN.md               ← this file
  types.py              ← SegmentConfig, PreprocessedSignal, Features, QRSDetector
  preprocessing.py      ← signal conditioning pipeline
  extract.py            ← main entry point: extract_features()

  # Time-domain features
  tcsc.py               ← [0]  TCSC  Threshold Crossing Sample Count
  tci.py                ← [1]  TCI   Threshold Crossing Intervals
  ste.py                ← [2]  STE   Short-Time Energy
  mea.py                ← [3]  MEA   Modified Exponential Algorithm
  mav.py                ← [12] MAV   Mean Absolute Value
  count1.py             ← [13] Count1
  count2.py             ← [14] Count2
  count3.py             ← [15] Count3
  amplitude.py          ← [16] Peak-to-peak amplitude (raw mV)

  # Phase-space features
  psr.py                ← [4]  PSR   Phase Space Reconstruction
  hilbert.py            ← [5]  HILB  Hilbert-transform PSR

  # Frequency-domain features
  vf_leak.py            ← [6]  VF    VF Leak (Kuo & Dillman 1978)
  spectral_m.py         ← [7]  M     Spectral centroid-like parameter (Barro 1989)
  spectral_a2.py        ← [8]  A2    Energy ratio around peak frequency
  spectral_fm.py        ← [9]  FM    Central frequency (Dzwonczyk 1990)

  # Complexity features
  lz.py                 ← [10] LZ    Lempel-Ziv complexity
  sample_entropy.py     ← [11] SpEn  Sample Entropy

  # EMD complexity features (one file — all five require the same EMD decomposition)
  imf_lz.py             ← [17–21] IMF1_LZ … IMF5_LZ

  # QRS-derived features (one file — all five require the same QRS detection pass)
  qrs_features.py       ← [22–26] RR, RR_Std, RR_CV, UR, VR
```

---

## Phase 1 — API definition and stub implementations  *(current)*

**Deliverables:**
- `types.py` fully defined: `SegmentConfig`, `PreprocessedSignal`, `Features` (all 27
  fields), `QRSDetector` protocol
- `preprocessing.py` stub: correct signature, returns zeroed `PreprocessedSignal`
- All 27 feature module stubs: correct signatures, `return 0.0`
- `extract.py`: `extract_features()` wires everything together end-to-end, returning a
  `Features` instance (all zeros at this stage)
- `algo/PLAN.md` (this file)

**Acceptance criteria:**
```python
from algo.extract import extract_features
from algo.types import SegmentConfig
import numpy as np

sig = np.random.randn(2000)          # 8 s at 250 Hz
cfg = SegmentConfig(sampling_rate=250.0)
f = extract_features(sig, cfg)
assert len(f.to_array()) == 27
```

---

## Phase 2 — Difficulty assessment and library selection

For each feature module, document:
- Algorithm summary (3–5 lines)
- Reference implementation location (`vf_features.pyx` line numbers)
- Proposed Python/scipy implementation approach
- Best candidate library if applicable
- Difficulty rating: **Easy** / **Medium** / **Hard**
- Known gotchas or edge cases

This assessment will be added as a docstring header in each feature file and summarised
in a table here.

**Preliminary difficulty estimate:**

| Feature | Difficulty | Notes |
|---------|-----------|-------|
| amplitude | Easy | `np.ptp()` on raw mV |
| mav | Easy | `np.convolve` sliding mean |
| count1–3 | Easy | Array comparisons and sums |
| lz | Easy | Pure Python or `antropy.lziv_complexity` |
| ste | Easy | Exponential decay weights, dot product |
| mea | Easy | Same as STE with shorter τ |
| vf_leak | Medium | Hamming FFT, ratio of band energies |
| spectral_m/a2/fm | Medium | FFT-based, need to match Hamming window and bin logic |
| tcsc | Medium | Tukey window, ±20 % thresholds, overlapping 3 s windows |
| tci | Medium | Pulse-pair tracking across 1 s windows |
| psr | Medium | 2D histogram on delay-embedded signal |
| hilbert | Medium | `scipy.signal.hilbert`, same grid logic as PSR |
| sample_entropy | Medium | `antropy.sample_entropy` or direct pyEEG port |
| imf_lz | Medium | `PyEMD` for EMD + LZ per mode |
| qrs_features | Hard | Need a validated QRS detector; OSEA replacement required |

---

## Phase 3 — Testing strategy

Each feature needs a numerical agreement test against the reference Cython implementation.

**Approach:**
1. Build the reference Cython extensions (`make build`)
2. Run `extract_one.py` on a set of known WFDB segments to capture reference feature vectors
3. Implement a `tests/test_<feature>.py` that:
   - Loads the same signal segment
   - Calls the algo/ implementation
   - Asserts `abs(algo_value - reference_value) < tolerance`
4. Tolerance: `1e-6` for deterministic features; `1e-3` for entropy/complexity features
   where numerical paths may differ

**Test segments to use:**
- `mitdb/100` segment 0 (NSR — exercises most features normally)
- `vfdb/422` segment at sample 385788 (coarse VF — high-amplitude, high-frequency)
- A fine-VF segment from mghdb (low amplitude, stresses the amplitude/threshold features)
- An asystole segment (near-zero signal, exercises edge cases)

**QRS detector validation:**
- Capture OSEA beat detections from `qrs_test.py` for the test segments
- The Python QRS detector must agree within ±10 ms on beat positions for NSR segments

---

## Phase 4+ — Feature implementation (one PR per module)

Order of implementation (easy → hard):
1. amplitude, count1–3, mav (trivial numpy)
2. lz, ste, mea (simple signal processing)
3. vf_leak, spectral_m, spectral_a2, spectral_fm (FFT-based)
4. tcsc, tci (windowed threshold crossing)
5. psr, hilbert (phase-space)
6. sample_entropy (port from pyeeg or use antropy)
7. imf_lz (requires EMD library integration)
8. qrs_features (requires QRS detector decision and validation)

---

## Open decisions

- [ ] **QRS detector**: neurokit2 vs wfdb-python XQRS vs biosppy vs pure Pan-Tompkins port?
      Decision deferred to Phase 2 assessment.
- [ ] **EMD library**: bundled PTSA vs `PyEMD` (pip-installable, actively maintained)?
      PTSA is already in the repo; PyEMD is cleaner. Decide in Phase 2.
- [ ] **antropy vs manual**: `antropy` provides `lziv_complexity` and `sample_entropy`;
      verify they match the reference implementation's specific parameter choices.
