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

## Phase 1 — API definition and stub implementations  *(complete)*

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

## Phase 2 — Difficulty assessment and library selection  *(complete)*

### Critical corrections to earlier documentation

Reading `vf_features.pyx` and `signal_processing.pyx` line-by-line revealed several
discrepancies between the reference code and the description in CLAUDE.md / SUMMARY.md:

| What docs said | What the code actually does |
|---|---|
| Normalise by **standard deviation** | Normalise by **min-max**: `(x - min) / (max - min)` → [0, 1] |
| Drift suppression at **0.5 Hz** | Cutoff is **1 Hz** in both `extract_features()` and `get_amplitude()` |
| Amplitude = `(max - min) / 2` | Amplitude = **max peak-to-valley pair** found by iterating argrelmax/argrelmin |
| Drift suppression = high-pass Butterworth | Custom **1-pole bilinear IIR** via `tan(fc·π·T)`; implemented with `filtfilt` |

These affect our `preprocessing.py` and `amplitude.py` implementations directly.
The algo/ package follows the **code**, not the docs.

---

### Feature-by-feature assessment

#### Preprocessing (`preprocessing.py`)
- **Reference**: `signal_processing.pyx:drift_supression`, `butter_lowpass_filter`, `moving_average`; `vf_features.pyx:extract_features` lines 567–577
- **Pipeline** (in order):
  1. Mean subtraction: `x - mean(x)`
  2. **Min-max normalisation** to [0, 1]: `(x - min) / (max - min)` ← not std!
  3. Moving average: `np.convolve(x, ones(5)/5, mode='same')`
  4. Drift suppression at **1 Hz**: custom 1-pole IIR `b=[c1,-c1], a=[1,-c2]` with `filtfilt`, where `c1=1/(1+tan(fc·π·T))`, `c2=(1-tan)/(1+tan)`
  5. Butterworth low-pass at 30 Hz: `scipy.signal.butter(5, 30/nyq) + filtfilt`
- **Implementation**: all pure numpy/scipy — **Easy**

#### Amplitude [16] (`amplitude.py`)
- **Reference**: `signal_processing.pyx:get_amplitude` lines 48–94
- Uses `src_samples` (raw mV, unconverted), NOT the preprocessed signal
- Applies its OWN pipeline: moving_average(5) → drift_suppression(1 Hz) → butter_lowpass(30 Hz)
- Finds local maxima/minima with `scipy.signal.argrelmax(order=round(0.05·sr))`
- Iterates peak-valley pairs, tracks maximum `|peak - valley|`
- Returns the single largest peak-to-valley amplitude across all pairs
- **Gotcha**: the peak/valley iteration is stateful (tracks `next_peak`, `next_valley`); edge case when sequence runs out mid-pair
- **Implementation**: scipy.signal.argrelmax + careful iteration — **Medium**

#### TCSC [0] (`tcsc.py`)
- **Reference**: `vf_features.pyx:threshold_crossing_sample_counts` lines 58–97
- 3-second moving window, step=1 second
- Each window: apply `scipy.signal.windows.tukey(N, alpha=0.5/window_duration)` → abs → normalise by max → fraction > 0.2 threshold → ×100
- Returns mean of all windows
- **Gotcha**: `alpha = 0.5 / 3.0 ≈ 0.167`; step is exactly `sampling_rate` samples
- **Implementation**: pure numpy/scipy — **Easy**

#### TCI [1] (`tci.py`)
- **Reference**: `vf_features.pyx:threshold_crossing_intervals` lines 132–167, `find_threshold_crossing` lines 100–128
- Per 1-second segment: threshold = `max(segment) × 0.20` (max of that segment, not global!)
- Returns `(begin_silence, tail_silence, n_pulses)` where begin_silence = first_rise_sample, tail_silence = samples after last fall
- TCI per interior segment = 1000 / effective_n_pulses
  - effective_n_pulses = (n_pulses−1) + fraction1 + fraction2
  - fraction1 = t2/(t1+t2), fraction2 = t3/(t3+t4)
  - t1 = prev_segment tail_silence, t2 = curr begin_silence, t3 = curr tail_silence, t4 = next begin_silence
- Skip first and last 1-second segments (boundary artefacts)
- Returns mean TCI; returns 1000.0 if < 3 segments
- **Gotcha**: threshold per segment (not global); careful index tracking for t1/t4
- **Implementation**: pure Python with numpy helper — **Medium**

#### STE [2] (`ste.py`)
- **Reference**: `vf_features.pyx:standard_exponential` lines 170–198
- E(t) = M × exp(−|t − t_peak| / (tc × sr)), where M = global max, t_peak = argmax
- Count zero-crossings between signal and E(t), normalise by duration (seconds)
- **Implementation**: vectorisable with numpy — **Easy**

#### MEA [3] (`mea.py`)
- **Reference**: `vf_features.pyx:modified_exponential` lines 201–240
- Finds local maxima with `argrelmax(order=round(0.05·sr))`
- Exponential E(t) = local_max × exp(−(t − t_local_max) / (tau × sr))
- Counts "lifts": times when signal rises above current exponential → jump to next local max
- Returns n_lifts / duration
- **Gotcha**: iterator-based; StopIteration terminates loop cleanly
- **Implementation**: pure Python loop, scipy.signal.argrelmax — **Medium**

#### PSR [4] (`psr.py`)
- **Reference**: `vf_features.pyx:phase_space_reconstruction` lines 273–290
- Delay-embed: x_t vs x_{t+delay} where delay = round(0.5 × sr) samples
- Single axis range: min/max of the FULL signal (both axes use same range)
- Map to 40×40 grid: `idx = (x - min) × 39 / range`
- Count occupied cells / 1600
- **Implementation**: 3 lines of numpy — **Easy**

#### HILB [5] (`hilbert.py`)
- **Reference**: `vf_features.pyx:hilbert_psr` lines 296–319
- Resample to 50 Hz first (for speed): `n_samples = int(duration × 50)`
- Analytic signal: `scipy.signal.hilbert(resampled)` → imaginary part = Hilbert transform
- x and y axes use **separate** min/max ranges (unlike PSR which uses one)
- Map to 40×40 grid, count occupied cells / 1600
- **Implementation**: scipy.signal.resample + hilbert — **Easy**

#### VF Leak [6] (`vf_leak.py`)
- **Reference**: `vf_features.pyx:vf_leak` lines 323–343; shares FFT with M/A2/FM
- Input: original time-domain `samples` + pre-computed Hamming FFT
- `peak_freq = fft_freq[argmax(fft)]` (normalised, not Hz)
- half_cycle = int((1/peak_freq) / 2) samples in the **time domain** (not spectral!)
- Result = `sum(|x[half_cycle:] + x[:-half_cycle]|) / sum(|x[half_cycle:]| + |x[:-half_cycle:]|)`
- **Gotcha**: if half_cycle ≤ 0 or ≥ N → return 0.0
- **Gotcha**: shares FFT with M, A2, FM — see "Spectral grouping" decision below
- **Implementation**: numpy + shared FFT — **Medium**

#### Spectral M [7] (`spectral_m.py`) and A2 [8] (`spectral_a2.py`)
- **Reference**: `vf_features.pyx:spectral_features` lines 347–403; computed together
- Both use same Hamming FFT; peak in 0.5–9 Hz searched in **normalised** freq units (0.5/sr to 9.0/sr)
- Amplitudes < 5% of peak set to zero before computing M/A2
- **M**: `(1/peak_freq) × sum(amp_i × freq_i) / sum(amp_i)` up to `min(20×peak_freq, 100/sr)`
- **A2**: `sum(amp in [0.7·peak_freq, 1.4·peak_freq]) / sum(amp up to spec_max_freq)`
- Both use normalised freq units throughout — must convert back when needed
- **Gotcha**: `spectral_features()` is computed once; M and A2 are returned together
- **Implementation**: numpy searchsorted + dot product — **Medium**

#### FM [9] (`spectral_fm.py`)
- **Reference**: `vf_features.pyx:central_frequency` lines 409–415; shares FFT
- FM = `sum(freq_i × |fft_i|²) / sum(|fft_i|²) × sampling_rate`
- Full spectrum (no band restriction), power-weighted centroid converted back to Hz
- **Implementation**: one numpy dot product — **Easy**

#### LZ [10] (`lz.py`)
- **Reference**: `vf_features.pyx:complexity_measure` lines 437–460; `vf_features_native.c:lempel_ziv_complexity`
- **Adaptive threshold** (not just mean!):
  - pos_count = samples where 0 < s < 0.1×pos_peak
  - neg_count = samples where 0.1×neg_peak < s < 0
  - If (pos_count + neg_count) < 40% of N → threshold = 0.0 (binary at zero)
  - Elif pos_count < neg_count → threshold = 0.2 × pos_peak
  - Else → threshold = 0.2 × neg_peak
- Binarise: `bin_str = (samples > threshold).astype(uint8)`
- LZ C algorithm: scan left-to-right, count new substrings via `memmem()`
  - Initial C(n)=1, scan with sliding pointers s and q
  - Normalise: C(n) / (N / log2(N))
- **Implementation**: Pure Python LZ is ~20 lines; `antropy.lziv_complexity` may differ in normalisation — must verify. Adaptive threshold must be ported exactly.
- **Difficulty**: **Medium**

#### SpEn [11] (`sample_entropy.py`)
- **Reference**: `vf_features.pyx:sample_entropy` line 422–429; `pyeeg/__init__.py:samp_entropy` lines 539–602
- Uses last 1250 samples (= last 5 s at 250 Hz after resampling)
- Tolerance r = 0.2 × std(last_1250_samples)
- pyeeg uses Chebyshev (max-norm) distance, NOT Euclidean
- Vectorised: builds N×N distance tensor, counts matches, `log(Cm/Cmp)`
- `antropy.sample_entropy` uses the same definition — likely a direct match
- **Gotcha**: pyeeg adds `1e-100` to numerator/denominator before log (avoid log(0)); must replicate or match result
- **Difficulty**: **Medium**

#### MAV [12] (`mav.py`)
- **Reference**: `vf_features.pyx:mean_absolute_value` lines 247–266
- 2-second windows, step = `sampling_rate` samples (1-second step)
- Per window: `abs(window) / max(abs(window))`, then `mean()`
- Returns mean across all windows
- **Gotcha**: normalises per window (not globally) — cannot vectorise trivially
- **Implementation**: numpy loop — **Easy**

#### Count1–3 [13–15] (`count1.py`, `count2.py`, `count3.py`)
- **Reference**: `vf_features.pyx:auxiliary_counts` lines 464–498
- Resample to 250 Hz if needed
- Custom IIR bandpass filter: `FS[i] = (14·FS[i-1] − 7·FS[i-2] + (S[i]−S[i-2])/2) / 8`
  - Equivalent `scipy.signal.lfilter` coefficients: `b = [1/16, 0, -1/16]`, `a = [1, -14/8, 7/8]`
- Per 1-second block of filtered signal:
  - Count1: `sum(fs_segment ≥ 0.5 × max(fs_segment))`
  - Count2: `sum(fs_segment ≥ mean(fs_segment))`
  - Count3: `sum(|fs_segment - mean| ≤ mean_abs_deviation)`
- Accumulate raw counts across all blocks (integer, not normalised)
- **Gotcha**: the filter uses the unfiltered `samples` (after main preprocessing), not the raw signal
- **Implementation**: scipy.signal.lfilter + numpy comparisons — **Medium**

#### IMF LZ [17–21] (`imf_lz.py`)
- **Reference**: `vf_features.pyx:emd_features` lines 504–524; `vf_features_native.c:imf_lempel_ziv_complexity`
- Resample to 250 Hz; normalise to [0,1]; scale to uint16 range (×2^12)
- Run `ptsa.ptsa.emd.emd(samples, max_modes=5)` → list of 5 IMF arrays
- For each IMF: convert each float to 12-bit integer, concatenate all bits into binary string of length N×12, then run standard LZ on that string
- **This is completely different from the regular LZ** — it works on a 12× longer bit string encoding the actual IMF values
- `PyEMD` (pip package) can replace PTSA; need to verify it produces matching IMFs
- **Difficulty**: **Hard** (12-bit encoding trick + EMD library validation)

#### QRS features [22–26] (`qrs_features.py`)
- **Reference**: `vf_features.pyx:beat_statistics` lines 527–555; `qrs_detect.pyx`
- OSEA always resamples to 200 Hz internally; RR intervals in units of 1/200 s → ×1000 for ms
- Skips beat[0] for UR and VR counts (often misclassified as 'Q')
- rr_cv = rr_std / rr_avg (0 if rr_avg=0)
- **QRS detector options**:
  - `neurokit2.ecg_peaks()` — best maintained, multiple algorithms
  - `wfdb.processing.xqrs_detect()` — WFDB-native, good for PhysioNet data
  - Both lack per-beat classification ('N'/'V'/'Q') that OSEA provides for UR/VR
- **Difficulty**: **Hard** (UR/VR require beat-type classification; no Python QRS detector produces OSEA-compatible labels)

---

### Summary table

| Feature | File | Difficulty | Key library | Gotcha |
|---------|------|-----------|-------------|--------|
| Preprocessing | `preprocessing.py` | Easy | scipy.signal | min-max norm, 1 Hz cutoff, custom IIR |
| Amplitude [16] | `amplitude.py` | Medium | scipy.signal.argrelmax | own preprocessing pipeline on raw_mv |
| TCSC [0] | `tcsc.py` | Easy | scipy.signal.windows.tukey | alpha=0.167 |
| TCI [1] | `tci.py` | Medium | — | per-segment threshold, fractional pulses |
| STE [2] | `ste.py` | Easy | numpy | vectorisable |
| MEA [3] | `mea.py` | Medium | scipy.signal.argrelmax | iterator-based loop |
| PSR [4] | `psr.py` | Easy | numpy | single axis range for both x and y |
| HILB [5] | `hilbert.py` | Easy | scipy.signal.hilbert | separate x/y axis ranges; resample to 50 Hz |
| VF Leak [6] | `vf_leak.py` | Medium | numpy | time-domain shift, shared FFT |
| M [7] | `spectral_m.py` | Medium | numpy | normalised freq units |
| A2 [8] | `spectral_a2.py` | Medium | numpy | shares FFT + peak logic with M |
| FM [9] | `spectral_fm.py` | Easy | numpy | single dot product |
| LZ [10] | `lz.py` | Medium | pure Python | adaptive threshold, not just mean |
| SpEn [11] | `sample_entropy.py` | Medium | antropy | last 5 s only; Chebyshev distance |
| MAV [12] | `mav.py` | Easy | numpy | per-window normalisation |
| Count1 [13] | `count1.py` | Medium | scipy.signal.lfilter | custom IIR filter |
| Count2 [14] | `count2.py` | Medium | scipy.signal.lfilter | same IIR |
| Count3 [15] | `count3.py` | Medium | scipy.signal.lfilter | same IIR |
| IMF1–5 LZ [17–21] | `imf_lz.py` | Hard | PyEMD | 12-bit encoding, EMD validation |
| RR/RR_Std/RR_CV/UR/VR [22–26] | `qrs_features.py` | Hard | neurokit2 or wfdb | no Python detector gives N/V/Q labels |

---

## Phase 3 — Feature implementation  *(complete)*

All 27 features are implemented across three difficulty groups:

| Group | Files | Key implementation notes |
|-------|-------|--------------------------|
| Easy | `preprocessing.py`, `tcsc.py`, `ste.py`, `psr.py`, `hilbert.py`, `spectral_fm.py`, `mav.py` | Pure numpy/scipy, direct translations |
| Medium | `amplitude.py`, `tci.py`, `mea.py`, `vf_leak.py`, `spectral_m.py`, `spectral_a2.py`, `lz.py`, `sample_entropy.py`, `_count_helpers.py`, `count1.py`, `count2.py`, `count3.py` | Custom IIR, adaptive LZ threshold, pyeeg SpEn replicated exactly |
| Hard | `imf_lz.py`, `qrs_features.py` | ptsa EMD + 12-bit MSB-first LZ encoding; QRS detector protocol |

### Known limitations and gotchas for testing

1. **LZ is pure Python** — O(n²) substring search; expect ~seconds per segment at 2000 samples.
   Can be accelerated with a bytes-based approach if needed.

2. **IMF LZ requires `ptsa`** — bundled in the repo at `ptsa/ptsa/emd.py`.  Falls back to
   all-zeros if the import fails.  `ptsa.emd` and `PyEMD` may produce slightly different IMFs
   (different sifting stopping criteria); test against the reference to confirm agreement.

3. **QRS features — sample indices at native SR** — `WfdbXqrsDetector` returns indices in
   the signal's native sampling rate, which is what `qrs_features.py` expects (divides by
   `sig.sampling_rate`).  An `OseaDetector` wrapper must convert its 200 Hz internal indices
   back to native SR before returning.  `WfdbXqrsDetector` classifies all beats as `'N'`,
   so **UR and VR are always 0.0** with this detector.

4. **IMF normalisation uses `/ max`, not `/ range` — this is a reference-code bug we
   replicate** — `imf_lz.py` uses `(s − min(s)) / max(s)`.  After the main preprocessing
   pipeline the signal is zero-mean (drift suppression removes DC), so `max(s) ≈ 0.4` and
   `min(s) ≈ −0.4`.  Dividing by `max` (not `max − min ≈ 0.8`) stretches the normalised
   result to roughly [0, 2] instead of [0, 1].  The downstream `× 2^12` then produces
   uint16 values up to ~8192 (13-bit range), not 4096.  The reference comment says
   "normalize to 0−1" but the code doesn't achieve that.  We match the code, not the
   comment.

5. **Count1–3 loop step stays at original `sampling_rate` after resampling to 250 Hz** —
   faithful replication of the reference quirk; irrelevant when input is already 250 Hz.

---

## Phase 4 — Testing strategy  *(next)*

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

## Phase 5+ — Optimisation and integration

Candidates for follow-up:
- Replace pure-Python LZ with a `bytes`-based or Cython port if speed is a bottleneck
- Validate `ptsa.emd` vs `PyEMD` on the test segments; switch if PTSA diverges
- Add `OseaDetector` / `NeuroKitDetector` concrete classes outside `algo/` for QRS
- Wire `algo/extract.py` into the main `feature_extraction.py` driver as an optional backend

---

## Open decisions

- [x] **EMD library**: chose bundled **PTSA** (`ptsa/ptsa/emd.py`) — no pip dependency,
      matches the reference.  PyEMD is a drop-in fallback if PTSA proves inaccurate.
- [x] **Sample entropy**: replicated **pyeeg** directly in `sample_entropy.py` — no
      `antropy` dependency; exact match with the reference is guaranteed.
- [x] **QRS detector**: **`wfdb.processing.xqrs_detect`** via `algo/wfdb_detector.py`.
      All beats returned as `'N'`; UR and VR are always 0.0 until a beat classifier is
      added.  OSEA also only distinguishes N vs V (with Q for unclassified); xqrs omits
      that distinction entirely for now.
- [ ] **LZ speed**: pure-Python O(n²) implementation — optimisation **deferred**.
      Correct results are more important than speed at this stage.
