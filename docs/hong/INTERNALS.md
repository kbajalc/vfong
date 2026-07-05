# Feature Extraction — Internals Reference

This document covers everything that happens inside
`vf_features.extract_features()` (`hong/vf_features.pyx:560`): the preprocessing
pipeline, all 27 feature functions, the structure of the return value, and the
layout of the pickled `.dat` files that are the output of
`hong/feature_extraction.py`.

> **Paths:** the Cython sources described here now live under `hong/`. For the
> pure-Python reimplementation and its per-feature notes see `algo/` and
> `algo/PLAN.md`; feature-by-feature agreement with this reference is validated
> in `tests/`.

---

## Overview of call structure

```
vf_features.extract_features(src_samples, sampling_rate, features_to_extract)
│
│  src_samples  : float64 ndarray, already in mV (ADC zero subtracted, divided by gain)
│  sampling_rate: int, Hz (360 for mitdb, 250 for vfdb/cudb, 128 for edb, varies for mghdb)
│  features_to_extract: set of feature name strings (all 27 by default)
│
├── [PREPROCESSING — modifies a copy; src_samples kept raw]
│   ├── mean subtraction
│   ├── min-max normalisation → [0, 1]
│   ├── moving_average(order=5)            signal_processing.pyx
│   ├── drift_supression(cutoff=1 Hz)      signal_processing.pyx
│   └── butter_lowpass_filter(cutoff=30 Hz) signal_processing.pyx
│
├── [QRS DETECTION — on raw src_samples]
│   └── qrs_detect(src_samples, sampling_rate)   qrs_detect.pyx → osea/ C library
│       returns: list of (beat_sample_at_200Hz, beat_type_char)
│
├── [FEATURE GROUP: time-domain / morphology]  — on preprocessed samples
│   ├── threshold_crossing_sample_counts()  → TCSC  [0]
│   ├── threshold_crossing_intervals()      → TCI   [1]
│   ├── standard_exponential()              → STE   [2]
│   ├── modified_exponential()              → MEA   [3]
│   ├── phase_space_reconstruction()        → PSR   [4]
│   └── hilbert_psr()                       → HILB  [5]
│
├── [FEATURE GROUP: spectral]  — on preprocessed samples (FFT computed once, shared)
│   ├── vf_leak()                           → VF    [6]
│   ├── spectral_features()[0]              → M     [7]
│   ├── spectral_features()[1]              → A2    [8]
│   └── central_frequency()                → FM    [9]
│
├── [FEATURE GROUP: complexity]  — on preprocessed samples
│   ├── complexity_measure()  → C: lempel_ziv_complexity()  → LZ    [10]
│   └── sample_entropy()      → pyeeg.samp_entropy()        → SpEn  [11]
│
├── [FEATURE GROUP: morphology continued]
│   ├── mean_absolute_value()               → MAV    [12]
│   └── auxiliary_counts()                  → Count1 [13], Count2 [14], Count3 [15]
│
├── [FEATURE: amplitude]  — on raw src_samples (separate preprocessing)
│   └── get_amplitude()    signal_processing.pyx   → Amplitude [16]
│
├── [FEATURE GROUP: EMD complexity]  — on preprocessed samples
│   └── emd_features()  → ptsa.emd() + C: imf_lempel_ziv_complexity()
│       → IMF1_LZ [17], IMF2_LZ [18], IMF3_LZ [19], IMF4_LZ [20], IMF5_LZ [21]
│
└── [FEATURE GROUP: QRS-derived statistics]  — on beats list from qrs_detect()
    └── beat_statistics(beats)
        → RR [22], RR_Std [23], RR_CV [24], UR [25], VR [26]

return (result, beats, amplitude)
   result   : array.array('d') of 27 floats, indices 0-26 as above
   beats    : list of (sample_at_200Hz, type_char) tuples
   amplitude: float64 scalar, mV
```

---

## Preprocessing pipeline

The preprocessed signal used for most features is built from a **copy** of
`src_samples`. The original `src_samples` array is preserved and passed separately
to the QRS detector and the amplitude function.

All preprocessing is in `signal_processing.pyx` and uses `scipy.signal.filtfilt`
(zero-phase, no group delay) wherever filtering is applied.

| Step | Operation | Code | Effect |
|------|-----------|------|--------|
| 1 | Mean subtraction | `samples - np.mean(samples)` | Removes DC offset |
| 2 | Min-max normalisation | `(s - min) / (max - min)` | Scales to [0, 1] |
| 3 | 5-point moving average | `np.convolve(s, ones(5)/5, mode='same')` | Light smoothing, reduces high-frequency noise |
| 4 | 1 Hz high-pass (drift suppression) | First-order IIR via `filtfilt` | Removes baseline wander below 1 Hz |
| 5 | 30 Hz low-pass (Butterworth order 5) | `scipy.signal.butter` via `filtfilt` | Removes high-frequency EMG and powerline noise |

After step 5 the signal is a normalised, filtered, zero-mean ECG waveform.

### Amplitude preprocessing (separate path)

`Amplitude` is computed on `src_samples` via its own internal preprocessing in
`signal_processing.get_amplitude()`:

1. 5-point moving average
2. 1 Hz drift suppression
3. 30 Hz Butterworth low-pass

This replicates steps 3–5 but skips mean subtraction and normalisation so that
the signal retains its physical mV scale. All other features work on the
normalised [0,1] signal.

---

## QRS detection

Runs on `src_samples` (raw mV, unfiltered) via `qrs_detect.qrs_detect()`.

Internally the OSEA C library (`osea/bdac.c`) does its own bandpass
filtering. The signal is resampled to 200 Hz before being fed to the detector.
The detector is run **twice** on the same segment:

1. **Warm-up pass** — fed until 8 beats are detected, allowing the adaptive
   thresholds to stabilise.
2. **Detection pass** — the full segment is fed again; this time beat times and
   types are recorded.

A global `threading.Lock` in `qrs_detect.pyx` serialises calls because the OSEA
C library is not reentrant.

**Return value:** `list[(beat_sample: int, beat_type: str)]`

- `beat_sample` is the sample index **at 200 Hz** (not the original sampling rate)
- `beat_type` is a single character matching the ANSI/AAMI beat codes:
  `N` normal, `V` ventricular premature, `Q` unclassifiable, `S` supraventricular,
  etc.

The beat list is used by `beat_statistics()` for the RR-interval features (indices
22–26) and stored on `SegmentInfo.detected_beats` by `feature_extraction.py`.

---

## The 27 features

All features except Amplitude operate on the **preprocessed** (normalised,
filtered) signal. Amplitude uses the raw mV signal. All return a scalar `float64`
unless noted.

### Group 1 — Time-domain / morphology

#### [0] TCSC — Threshold Crossing Sample Count

**Function:** `threshold_crossing_sample_counts(samples, sampling_rate, window_duration=3.0, threshold_ratio=0.2)`

Divides the segment into overlapping 3-second windows that advance by 1 second.
For each window:

1. Applies a Tukey window (cosine taper α = 0.5/3 ≈ 0.17) to suppress edge effects.
2. Takes the absolute value and normalises by the window maximum.
3. Counts samples above `threshold_ratio` (0.2 = 20 % of peak) as a percentage
   of the window length.

Returns the **mean percentage** across all windows.

| Rhythm | Typical range | Reason |
|--------|--------------|--------|
| VF | high (>48 for high specificity, 25–35 for high sensitivity) | Rapid irregular oscillations keep many samples above threshold |
| Normal sinus | low | Wide QRS complexes separated by quiet baselines |

**Unit:** percent (0–100)

---

#### [1] TCI — Threshold Crossing Interval

**Function:** `threshold_crossing_intervals(samples, sampling_rate, threshold_ratio=0.2)`

Divides the segment into 1-second windows. For each window it finds:

- `t2`: position of the first threshold rise (begin silence before the pulse)
- `t3`: position of the last threshold fall
- `t1`: tail silence of the *previous* window
- `t4`: begin silence of the *next* window

Effective number of pulses in the window = `(n_pulses - 1) + fraction1 + fraction2`
where fractions account for the partial pulses at the window boundaries.

`TCI = 1000 / effective_pulses`

Returns the **mean TCI in milliseconds** across all interior windows (first and
last windows are skipped because they lack boundary context).

| Rhythm | Typical range | Reason |
|--------|--------------|--------|
| Normal sinus | high (>400 ms) | One QRS per second → 1 pulse → long interval |
| VF | low | High-frequency oscillations → many pulses per second → short interval |

**Unit:** milliseconds

---

#### [2] STE — Standard Exponential

**Function:** `standard_exponential(samples, sampling_rate, time_constant=3)`

Finds the global maximum of the signal (`M` at time `tm`). Fits an exponential
envelope `E(t) = M × exp(−|t − tm| / tc)` with `tc = 3 × sampling_rate` samples.
Counts the number of times the ECG signal **crosses** this envelope.

Returns **crossings per second** = `n_crosses / duration`.

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | high | Rapid irregular oscillations cross the envelope many times |
| NSR | low | One dominant QRS peak, few crossings |

**Unit:** crossings per second

---

#### [3] MEA — Modified Exponential

**Function:** `modified_exponential(samples, sampling_rate, time_constant=0.2)`

Similar to STE but uses a *local* rather than global maximum. The envelope is
re-anchored at every local maximum (`tc = 0.2 s`). Counts how many times the
signal rises above the falling exponential from the most recent local peak.

Returns **lifts per second** (each re-anchor of the envelope is one "lift").

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | high | Many local maxima close together, envelope re-anchored frequently |
| NSR | lower | Fewer, well-separated local maxima |

**Unit:** lifts per second

---

#### [4] PSR — Phase Space Reconstruction

**Function:** `phase_space_reconstruction(samples, sampling_rate, delay=0.5)`

Constructs a 2-D phase-space portrait using a 0.5-second time delay:

- x-axis: `x(t)`
- y-axis: `x(t + 0.5 s)`

Both axes are discretised into a 40 × 40 grid (1600 cells). Returns the
**fraction of cells visited** = `occupied_cells / 1600`.

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | high (>0.15) | Chaotic, space-filling trajectory visits many cells |
| NSR | low | Periodic orbit confined to a narrow band |

**Unit:** fraction (0–1)

---

#### [5] HILB — Hilbert-transform PSR

**Function:** `hilbert_psr(samples, sampling_rate)`

An analytic-signal variant of PSR:

1. Downsamples to 50 Hz for speed.
2. Computes the Hilbert transform of the downsampled signal via `scipy.signal.hilbert`.
3. Constructs a 2-D portrait where x = `x(t)` and y = `Im{analytic_signal(t)}`
   (the Hilbert transform).
4. Discretises on a 40 × 40 grid, returns `occupied_cells / 1600`.

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | high (>0.15) | Irregular analytic signal fills many grid cells |
| NSR | low | Near-sinusoidal analytic signal traces a tight ellipse |

**Unit:** fraction (0–1)

---

### Group 2 — Spectral

The FFT is computed **once** and shared across VF, M, A2, and FM:

```
fft      = FFT(samples × Hamming_window)[0 : N/2]   (one-sided)
fft_freq = fftfreq(N)[0 : N/2]                       (normalised, ×Fs → Hz)
```

#### [6] VF — VF Leak

**Function:** `vf_leak(samples, fft, fft_freq)`

Exploits the near-sinusoidal nature of VF. Finds the **peak frequency** in the
FFT, then shifts the signal by half a cycle at that frequency. If the signal is
nearly sinusoidal, `original + shifted ≈ 0` because the half-cycle shift inverts
the phase.

`VF_leak = Σ|x(t) + x(t − half_cycle)| / Σ(|x(t)| + |x(t − half_cycle)|)`

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | low (<26/64 ≈ 0.41) | Signal nearly cancels when shifted by half period |
| NSR | high | Broadband; shifted copy does not cancel |

**Unit:** ratio (0–1, lower = more sinusoidal)

---

#### [7] M — First Spectral Moment

**Function:** `spectral_features(fft, fft_freq, sampling_rate)[0]`

Computed jointly with A2. Steps:

1. Find peak frequency `fp` in 0.5–9 Hz band.
2. Zero out components below 5 % of peak amplitude.
3. Compute `M = (1/fp) × Σ(ai × fi) / Σ(ai)` over frequencies up to
   `min(20×fp, 100 Hz)`.

`M` is the normalised first moment: how spread the spectral energy is relative
to the dominant frequency.

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | low (≤1.55) | Energy concentrated near peak frequency → low moment |
| NSR | high | Broadband spectrum → energy spread → high moment |

**Unit:** dimensionless ratio

---

#### [8] A2 — Spectral Concentration

**Function:** `spectral_features(fft, fft_freq, sampling_rate)[1]`

Fraction of total spectral power in the band `[0.7 fp, 1.4 fp]` (the ±40 %
band around the dominant frequency):

`A2 = Σ amplitude[0.7fp:1.4fp] / Σ amplitude[0:100Hz]`

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | high (≥0.45) | Most power concentrated in a narrow band |
| NSR | low | Power spread across many harmonics |

**Unit:** fraction (0–1)

---

#### [9] FM — Central Frequency

**Function:** `central_frequency(fft, fft_freq, sampling_rate)`

The power-weighted mean frequency (median frequency of the power spectrum):

`FM = Σ(fi × Pi) / Σ(Pi)`

where `Pi = |FFT(i)|²`.

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | 3–10 Hz | VF energy concentrated in low frequencies |
| NSR | higher | QRS complex has significant energy above 10 Hz |

**Unit:** Hz

---

### Group 3 — Complexity

#### [10] LZ — Lempel-Ziv Complexity

**Function:** `complexity_measure(samples)` → C function `lempel_ziv_complexity()`
(`vf_features_native.c`)

1. Finds an adaptive binarisation threshold based on the ratio of near-zero
   samples to total samples (three cases: threshold = 0 if signal is mostly
   near-zero; else ±20 % of the dominant peak).
2. Converts the signal to a binary string `S` (1 if sample > threshold, 0
   otherwise).
3. Passes the binary string to the C implementation of the Lempel-Ziv 1976
   algorithm, which counts the number of new subsequences needed to parse `S`.

The returned value is the normalised LZ complexity.

| Rhythm | Value | Reason |
|--------|-------|--------|
| NSR | low (<0.15) | Periodic signal → highly compressible binary string |
| VT | 0.15–0.486 | Semi-regular |
| VF | high (>0.486) | Irregular waveform → low compressibility |

**Unit:** normalised complexity (0–1)

---

#### [11] SpEn — Sample Entropy

**Function:** `sample_entropy(samples)` → `pyeeg.samp_entropy()`

Sample entropy measures the likelihood that sequences of `m` consecutive
samples that are similar remain similar when extended by one sample (lower
entropy → more predictable).

Parameters (from the referenced paper by Haiyan Li 2009):
- Uses the **last 1250 samples** of the preprocessed segment (= 5 seconds at
  250 Hz; the first 3 seconds are discarded to reduce transient effects)
- Template length `m = 2`
- Tolerance `r = 0.2 × std(segment)`

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | high (>0.25) | Irregular → matching templates rare → high entropy |
| NSR | low | Periodic → templates match frequently → low entropy |

**Unit:** non-negative float (unbounded above, typically 0–3)

---

### Group 4 — Morphology (continued)

#### [12] MAV — Mean Absolute Value

**Function:** `mean_absolute_value(samples, sampling_rate, window_duration=2.0)`

Uses a 2-second window with 1-second step (50 % overlap). For each window:

1. Takes the absolute value.
2. Normalises by the window's maximum absolute value.
3. Computes the mean.

Returns the **mean of per-window means**.

| Rhythm | Value | Reason |
|--------|-------|--------|
| VF | high (close to 0.5–0.6) | Normalised oscillations fill the window uniformly |
| NSR | lower | Tall narrow QRS peaks dominate; baseline is near zero |

**Unit:** fraction (0–1)

---

#### [13–15] Count1, Count2, Count3 — Auxiliary Counts

**Function:** `auxiliary_counts(samples, sampling_rate)` — returns a tuple `(int, int, int)`

These features are designed for signals at **250 Hz**. If the input is at a
different rate it is resampled first.

A custom IIR bandpass filter centred at 14.6 Hz (–3 dB at 13–16.5 Hz) is
applied:

```
FS[i] = (14·FS[i-1] − 7·FS[i-2] + (S[i] − S[i-2]) / 2) / 8
```

For each 1-second block of the filtered signal `FS`:

| Feature | Range counted | What it captures |
|---------|--------------|-----------------|
| **Count1** | `FS ≥ 0.5 × max(FS)` | Samples near peak — VF has many, NSR has few |
| **Count2** | `FS ≥ mean(|FS|)` | Samples above mean — measures duty cycle |
| **Count3** | `mean − MD ≤ FS ≤ mean + MD` | Samples in one mean-deviation band — measures regularity |

where `MD = mean(|FS − mean(FS)|)` (mean absolute deviation).

Each count is an **integer sum** across all 1-second blocks in the segment
(not normalised by duration).

**Unit:** integer sample count (typically hundreds to thousands for an 8-second
segment at 250 Hz)

---

#### [16] Amplitude

**Function:** `signal_processing.get_amplitude(src_samples, sampling_rate)`

Computed on the **raw mV signal** after its own preprocessing (moving average,
drift suppression, 30 Hz low-pass — no normalisation). Locates all local maxima
and minima using `scipy.signal.argrelmax` / `argrelmin` with a half-peak width
of `0.05 × sampling_rate` samples (50 ms). Iterates through alternating
peak/valley pairs and returns the **maximum peak-to-peak amplitude** found.

Clinical thresholds used elsewhere in the pipeline:

| Threshold | Meaning |
|-----------|---------|
| < 0.15 mV | Operational asystole (`asystole_check.py`) |
| ≤ 0.20 mV | Fine VF (AHA threshold) |
| > 0.20 mV | Coarse VF (AHA threshold) |

**Unit:** mV (physical, not normalised)

---

### Group 5 — EMD complexity

#### [17–21] IMF1_LZ … IMF5_LZ — EMD Intrinsic Mode Function LZ Complexity

**Function:** `emd_features(samples, sampling_rate, imf_modes)` → C function
`imf_lempel_ziv_complexity()` (`vf_features_native.c`)

Empirical Mode Decomposition (EMD) decomposes the signal into Intrinsic Mode
Functions (IMFs), ordered from highest to lowest frequency. VF energy tends to
concentrate in the first 1–2 IMFs while normal sinus energy is distributed
differently.

Steps:

1. Resample to 250 Hz if needed.
2. Normalise to [0, 1] then scale to 12-bit integers (`× 4096`).
3. Call `ptsa.ptsa.emd.emd(signal, max_modes=5)` to produce 5 IMFs.
4. Compute LZ complexity of each IMF via the C function
   `imf_lempel_ziv_complexity()` (a variant of the LZ algorithm tuned for
   continuous-valued rather than binary sequences).

Returns an `array.array('d')` of 5 values, one per IMF in order IMF1–IMF5.
IMF1 is the highest-frequency mode.

| Feature | Rhythm | Typical behaviour |
|---------|--------|------------------|
| IMF1_LZ | VF | High — VF IMF1 is irregular |
| IMF1_LZ | VT | Lower — more regular high-frequency component |
| IMF5_LZ | NSR | Low — low-frequency residual is smooth |

**Unit:** LZ complexity value (bounded, non-negative)

---

### Group 6 — QRS-derived statistics

These five features are derived from the `beats` list returned by
`qrs_detect()`. All beat times in the list are at **200 Hz** resolution.

#### [22] RR — Mean RR Interval

Mean of all successive beat-to-beat intervals.

`RR = mean(beat[i].time − beat[i-1].time) / 200`  (converted to seconds)

Returns 0.0 if fewer than 2 beats were detected.

**Unit:** seconds

---

#### [23] RR_Std — RR Standard Deviation

Standard deviation of the RR interval series. Returns 0.0 if fewer than 2 beats.

**Unit:** seconds

---

#### [24] RR_CV — RR Coefficient of Variation

`RR_CV = RR_Std / RR`

Normalises variability by rate. Returns 0.0 if RR = 0.

**Unit:** dimensionless ratio

---

#### [25] UR — Unknown Beat Ratio

Fraction of beats classified by OSEA as type `Q` (unclassifiable):

`UR = count(type=='Q') / (total_beats − 1)`

The first beat is excluded from the denominator because OSEA frequently
misclassifies it during initialisation.

**Unit:** fraction (0–1)

---

#### [26] VR — Ventricular Premature Beat Ratio

Fraction of beats classified as type `V` (premature ventricular contraction):

`VR = count(type=='V') / (total_beats − 1)`

**Unit:** fraction (0–1)

---

## Return value of `extract_features()`

```python
return result, beats, amplitude
```

| Name | Type | Shape | Contents |
|------|------|-------|----------|
| `result` | `array.array('d')` | 27 elements | Feature values in canonical index order (see table below) |
| `beats` | `list` | variable | `[(sample_at_200Hz: int, type: str), ...]` from OSEA |
| `amplitude` | `float64` | scalar | Peak-to-peak amplitude in mV from raw signal |

### Canonical feature index table

| Index | Name | Group | Unit |
|-------|------|-------|------|
| 0 | TCSC | Time-domain | % |
| 1 | TCI | Time-domain | ms |
| 2 | STE | Time-domain | crossings/s |
| 3 | MEA | Time-domain | lifts/s |
| 4 | PSR | Time-domain | fraction 0–1 |
| 5 | HILB | Time-domain | fraction 0–1 |
| 6 | VF | Spectral | ratio 0–1 |
| 7 | M | Spectral | dimensionless |
| 8 | A2 | Spectral | fraction 0–1 |
| 9 | FM | Spectral | Hz |
| 10 | LZ | Complexity | normalised 0–1 |
| 11 | SpEn | Complexity | non-negative float |
| 12 | MAV | Morphology | fraction 0–1 |
| 13 | Count1 | Morphology | integer count |
| 14 | Count2 | Morphology | integer count |
| 15 | Count3 | Morphology | integer count |
| 16 | Amplitude | Morphology | mV |
| 17 | IMF1_LZ | EMD complexity | LZ value |
| 18 | IMF2_LZ | EMD complexity | LZ value |
| 19 | IMF3_LZ | EMD complexity | LZ value |
| 20 | IMF4_LZ | EMD complexity | LZ value |
| 21 | IMF5_LZ | EMD complexity | LZ value |
| 22 | RR | QRS-derived | seconds |
| 23 | RR_Std | QRS-derived | seconds |
| 24 | RR_CV | QRS-derived | dimensionless |
| 25 | UR | QRS-derived | fraction 0–1 |
| 26 | VR | QRS-derived | fraction 0–1 |

When `--update-features` is used (partial recompute), skipped features are
written as `0.0` to `result` at their canonical index position and later
overwritten by values from the existing file in `output_results()`.

---

## Pickled `.dat` file structure

The output file (e.g. `features/features_s8.dat`) is a raw binary stream
written by two sequential `pickle.dump()` calls:

```
[ pickle object 1: x_data    ]
[ pickle object 2: x_data_info ]
```

Read back with:

```python
import pickle, numpy as np

with open("features/features_s8.dat", "rb") as f:
    x_data      = pickle.load(f)   # list of array.array('d'), one per segment
    x_data_info = pickle.load(f)   # list of SegmentInfo objects

x_data      = np.array(x_data)      # shape: (N, 27),  dtype: float64
x_data_info = np.array(x_data_info) # shape: (N,),     dtype: object
```

This is exactly what `vf_features.load_features(path)` does.

### x_data — shape (N, 27)

A float64 matrix. Row `i` is the feature vector for segment `i`.
Column order is the canonical index table above.

Segments are in the order they were yielded by `DataSet.get_samples()`:
mitdb records sorted alphabetically → vfdb → cudb → edb → mghdb (7 hardcoded
records), and within each database by record name then by time offset.

### x_data_info — shape (N,) object array of SegmentInfo

Each element is a `vf_data.SegmentInfo` instance. The full set of attributes
at the time the file is written is:

| Attribute | Type | Source | Description |
|-----------|------|--------|-------------|
| `record_name` | str | `vf_data.py` | `"db/record"` e.g. `"mitdb/100"` |
| `sampling_rate` | int | WFDB header | Original sampling rate in Hz |
| `begin_time` | int | segmentation | Start sample index in the original record |
| `end_time` | int | segmentation | End sample index (exclusive) = `begin_time + segment_size` |
| `n_beats` | int | annotation | Number of annotated beats in the window (from WFDB `.atr`). 0 for mghdb or rhythms without beat annotations |
| `rhythm` | str | annotation | Rhythm label, possibly corrected. Examples: `"(N"`, `"(VF"`, `"(VF,coarse"`, `"(VT,rapid"`, `"(ASYS"` |
| `detected_beats` | list | `feature_extraction.py` | OSEA QRS beats as `[(sample_at_200Hz, type_char), ...]`. Added after feature computation, not present in `vf_data.DataSet` |
| `amplitude` | float | `feature_extraction.py` | Peak-to-peak amplitude in mV from `get_amplitude()`. Same value as `x_data[i, 16]` (the Amplitude feature). Added after feature computation |

#### Derived methods on SegmentInfo

```python
info.get_duration()    # float — (end_time - begin_time) / sampling_rate  (seconds)
info.get_heart_rate()  # float — (n_beats / duration) * 60                (BPM)
                       #         uses annotated n_beats, not detected_beats
                       #         returns 0.0 if duration == 0
```

#### Rhythm label values

Labels after `vf_classify.initialize_aha_labels()` has been called (which
happens in `vf_tests.py`, not during extraction):

| Label | AHA class | Meaning |
|-------|-----------|---------|
| `(VF,coarse` | Shockable | VF with amplitude > 0.2 mV |
| `(VT,rapid` | Shockable | VT with heart rate ≥ 180 BPM |
| `(VF,fine` | Intermediate | VF with amplitude ≤ 0.2 mV |
| `(VT,slow` | Intermediate | VT with heart rate < 180 BPM |
| `(N`, `(NSR`, `(AFIB`, etc. | Non-shockable | All other rhythms |
| `(ASYS` | Excluded by default | Asystole (excluded via `--exclude-rhythms`) |
| `X` | Excluded | Manually marked as corrupt in correction file |
| `C` | Correct and confirmed | Correction file entry meaning "leave as-is" |

At extraction time the labels are the raw corrected annotation strings. The
coarse/rapid/fine/slow suffixes are added later by `initialize_aha_labels()`.
