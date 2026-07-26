# JEKOVA (2004) algorithm: reconstructed step-by-step specification

Source paper: Jekova I and Krasteva V, "Real time detection of ventricular
fibrillation and tachycardia", *Physiol. Meas.* **25** (2004) 1167-1178.
Verified transcription: [JEKOVA-2004.md](JEKOVA-2004.md).

The paper describes the method in prose and leans on external references (Thakor
1990, Kuo and Dillman 1978, Minami 1999) for several design choices without
pinning down the numeric details a re-implementation needs: filter orders,
whether a pass is causal or zero-phase, exact count ranges, loop granularity.
This document reconstructs the algorithm as an unambiguous, implementable
sequence. Where the paper leaves a gap, the resolution is stated with its
reasoning, and cross-checked against the two implementations in this repo:

- `hong/vf_features.pyx:auxiliary_counts()` (the 2016 reference; signed counts).
- `vfta/jekova.py` (the paper pipeline; `abs_counts()` is the paper-faithful path).

A final section lists every point where those implementations deviate from a
literal reading of the paper, so the reconstruction stays honest.

---

## 0. Signal conventions

- Sampling rate: **250 Hz**, 12-bit resolution. Every constant in the paper (the
  count thresholds, the band-pass coefficients) is expressed for 250 Hz. A
  re-implementation at another rate must resample to 250 Hz first (both repo
  implementations do this before the band-pass).
- Analysis unit: a **10 s epoch** (2500 samples). Decisions are taken at the end
  of each epoch; epochs are subsequent, non-overlapping.
- Amplitude is handled in physical units (µV / mV), so the ADC-to-mV conversion
  (subtract ADC zero, divide by gain) happens before Step 1.

The block diagram (figure 1 of the paper) is the authoritative control flow. The
stages below follow it top to bottom.

---

## 1. Preprocessing filtration

Paper text (section 2.2.1): "(i) two successive first-order high-pass filters
with 1 Hz cut-off frequency ...; (ii) a second-order 30 Hz Butterworth low-pass
filter ...; (iii) a notch filter to eliminate powerline interference."

Reconstructed, with the gaps resolved:

| Sub-step | Filter | Order | Cutoff | Pass direction | Notes |
|---|---|---|---|---|---|
| 1a | High-pass | first order, **cascaded twice** | 1 Hz each | **forward only** | combined -3 dB ≈ 1.4 Hz (paper's "equivalent 1.4 Hz"; two cascaded 1 Hz first-order sections put the joint -3 dB near 1.55 Hz, which the paper rounds to 1.4) |
| 1b | Butterworth low-pass | **second order** | 30 Hz | **forward only** | "following the approach of Thakor *et al* 1990" |
| 1c | Notch | (unspecified in paper) | 50 Hz | forward only | powerline rejection; a second-order IIR notch at 50 Hz is the natural choice |

**Forward-only vs filtfilt.** The paper is an AED real-time algorithm: it
"analyses an overall 10 s signal epoch and takes a decision for shock delivery in
the end of the same epoch" and is "implemented in a real-time operating device by
a standard microcontroller". Zero-phase filtering (`filtfilt`) needs the whole
segment and a backward pass, which is acceptable for a fixed 10 s block but is not
what the phrase "two successive first-order high-pass filters" describes. The
paper's wording (a cascade of causal first-order sections) and its real-time
framing both point to **single forward passes**. Treat every filter in Step 1 as
causal and forward.

**Order of Butterworth: second order.** Stated explicitly ("a second-order 30 Hz
Butterworth low-pass filter"). Do not use a higher order; the Thakor 1990
reference it follows also uses a low-order low-pass.

**Why a high-pass near 1.4 Hz rather than 0.67 Hz.** The paper defends the higher
cut-off: it stays inside the acceptable defibrillator-monitor bandwidth, does not
attenuate VF/VT, and gives faster recovery after high-amplitude noise and
defibrillation artefacts plus better baseline-drift suppression.

The preprocessed signal from Step 1 is the single input shared by both analysis
branches. The count branch (band-pass in Step 4, then counts in Steps 5-7) and the
wave-detection branch (Step 8) each take the Step 1 output as their input; they are
two independent branches off the same preprocessed signal, not a chain. In
particular, the band-pass in Step 4 filters the preprocessed signal, and the wave
detector in Step 8 works on the preprocessed signal directly (the paper: "the
analysis of the 'Not classified' rhythms is performed on the ECG signal passed
through preprocessing filtration"), so the wave detector does **not** see the
band-pass output.

> Repo divergence (see section 9): neither `hong/` nor `vftx/` implements this
> exact Jekova front end. Both run Hong's own pipeline (mean subtraction, min-max
> normalisation, order-5 moving average, a 1 Hz bilinear high-pass, and an
> order-5 Butterworth low-pass, all zero-phase via `filtfilt`). The `vfta/`
> paper pipeline filters once per record with the exg-core Lynn band-pass +
> median baseline instead. Only the band-pass (Step 4) and the counts (Step 6)
> are reproduced faithfully.

---

## 2. Noise detection

First decision diamond after preprocessing. An epoch is rejected as **Noise** when
either amplitude or slope is uncharacteristic of ECG:

- **Amplitude limit**: set from the dynamic range of the input amplifier and the
  AD converter, to catch extreme artefacts such as AD-converter saturation. The
  paper does not give a number because it is hardware-specific; use the converter's
  full-scale value.
- **Slew-rate limit**: a signal is "noise" when its maximum slew rate exceeds
  **400 µV ms⁻¹**. At 250 Hz (4 ms/sample) that is a sample-to-sample difference
  above 1600 µV = 1.6 mV. Compute `max(|S_i - S_{i-1}|)` over the epoch and compare
  against 1.6 mV.

If either limit is exceeded, output **Noise** and stop. Otherwise continue.

---

## 3. Asystole detection (amplitude gate)

Second decision diamond: `max(SignalAmplitude) < 150 µV`.

Signals whose peak amplitude is below **150 µV** are not analysed and are labelled
**Asystoly**. This is a peak amplitude on the preprocessed epoch (the paper writes
`Max(SignalAmplitude)`), not a peak-to-peak span. If below 150 µV, output
**Asystoly** and stop. Otherwise continue to the band-pass branch.

---

## 4. Band-pass digital filtration (integer-coefficient IIR)

The discriminating filter. Central frequency **14.6 Hz**, pass band 13 to 16.5 Hz
(-3 dB). It passes supraventricular and ventricular QRS complexes (which have
components up to ~14-20 Hz) and suppresses VF/ventricular-flutter energy (below
~7-10 Hz), so a large filter output means "organised complexes present" and a
small, peakless output means "fibrillation".

Difference equation (paper eq. 1, valid at 250 Hz):

$$FS_i = \frac{14\,FS_{i-1} - 7\,FS_{i-2} + \tfrac{1}{2}(S_i - S_{i-2})}{8}.$$

Here `S_i` is the input sample (the preprocessed signal from Step 1) and `FS_i` is
the band-pass output.

**This is a two-pole recursive filter run forward only.** It is causal by
construction (each output depends on the two previous outputs); there is no
backward pass and no `filtfilt`. Initialise `FS_0 = FS_1 = 0` and iterate from
`i = 2`. Both repo implementations do exactly this
(`hong/vf_features.pyx:480`, `vfta/jekova.py:_bandpass`).

Transfer function and sanity check:

$$H(z) = \frac{\tfrac{1}{2}(1 - z^{-2})}{8 - 14 z^{-1} + 7 z^{-2}}.$$

- Numerator zeros at z = +1 (DC) and z = -1 (Nyquist), so DC and 125 Hz are
  nulled: a band-pass shape.
- Poles at z = 0.875 ± j0.3307, magnitude 0.9354, angle 0.361 rad. The pole angle
  maps to 0.361 / (2π) × 250 ≈ **14.4 Hz**, matching the stated 14.6 Hz centre.
  The pole radius 0.935 gives the narrow ~3.5 Hz band.

The integer coefficients (14, 7, 8, and the /2) are the point of the design: they
were reduced from floating-point coefficients so the filter needs only shifts and
adds on an AED microcontroller.

---

## 5. Absolute value of the filter output

Take `AbsFS_i = |FS_i|`. All three counts in Step 6 are defined on this rectified
output, not on the signed `FS`.

> Repo divergence (see section 9): Hong's reference and `vftx` count on the
> **signed** `FS`, which makes Count2 degenerate (about half the samples sit above
> a near-zero mean for every rhythm). `vfta/jekova.py:abs_counts()` restores the
> paper's absolute-value definition and is the version the cascade should use.

---

## 6. Count parameters (Count1, Count2, Count3)

Each count is the number of `AbsFS` samples falling in an amplitude band. The
bands' reference levels (`max`, `mean`, `MD`) are recomputed **per 1 s interval**,
and the counts are **accumulated across the ten 1 s intervals** of the epoch. So
each count is a single integer per 10 s epoch, in the range 0 to 2500.

For each 1 s window (250 samples) of `AbsFS` compute:

- `smax = max(AbsFS)` over the window,
- `smean = mean(AbsFS)` over the window,
- `MD = mean(|AbsFS - smean|)` over the window (mean absolute deviation).

Then accumulate:

| Count | Band (per 1 s window) | Rule |
|---|---|---|
| Count1 | `0.5·smax` to `smax` | `count1 += sum(AbsFS >= 0.5*smax)` |
| Count2 | `smean` to `smax` | `count2 += sum(AbsFS >= smean)` |
| Count3 | `smean - MD` to `smean + MD` | `count3 += sum((AbsFS >= smean-MD) & (AbsFS <= smean+MD))` |

Intuition (paper section 3): organised rhythms have sharp peaks, so few samples
reach the Count1/Count2 upper bands but many cluster in the narrow `mean ± MD`
band, giving small Count1/Count2 and large Count3. Fibrillation is peakless, so
more samples enter the Count1/Count2 bands and the `mean ± MD` band is wider but
less populated, giving larger Count1/Count2 and smaller Count3.

**`MD` is the mean absolute deviation.** Section 2.2.4 defines `MD` as "mean
deviation" while section 3 loosely calls the Count3 band a "median deviation
around the mean". The method definition wins: it is the mean of `|AbsFS - mean|`,
which is what both implementations compute (`hong/vf_features.pyx:489`,
`vfta/jekova.py:114`). Do not use a median.

**Window granularity.** The reference (`hong`) computes `max/mean/MD` on 1 s
windows exactly. Note the loop-step quirk documented in section 9 that only
matters at sampling rates other than 250 Hz.

---

## 7. Rhythm classification cascade (Count1, Count2, Count3)

Evaluate these rules on the three per-epoch counts. The non-shockable rules and
the shockable rules are mutually exclusive on their count conditions, so rule
order does not change the outcome; an epoch matching none falls through to the
wave-detection branch (Step 8).

| Rule | Condition | Output |
|---|---|---|
| R1 | `Count1 < 250` and `Count2 > 950` and `Count1·Count2/Count3 < 210` | Non-shockable |
| R2 | `250 <= Count1 < 400` and `Count2 < 600` and `Count1·Count2/Count3 < 210` | Non-shockable |
| R3 | `Count1 >= 250` and `Count2 > 950` | Shockable |
| R4 | `Count2 >= 1100` | Shockable |
| else | (none of the above) | Not classified -> Step 8 |

The composite term `Count1·Count2/Count3` is the strongest single discriminator
(figure 4c): near 30 for non-shockable, near 360 for shockable, threshold 210. The
thresholds (Count1: 250, 400; Count2: 600, 950, 1100; ratio: 210) were fixed by
descriptive statistics of the count distributions followed by iterative tuning on
the hardest borderline cases (branch blocks and sub-180 bpm VT on the
non-shockable side; sub-3 Hz VF and above-180 bpm VT on the shockable side).

> Repo note: because `vftx` counts on the signed output and the `vfta/` windows
> are 8 s / 4 s rather than 10 s, `vfta/tuning.py` re-derives these thresholds on
> normalised counts (`Count / N`). `DEFAULT_PARAMS` in `vfta/jekova.py` is exactly
> the published constants divided by 2500 as the starting point. The published
> integer thresholds above are the correct values only for the paper's own 10 s,
> 250 Hz, absolute-count pipeline.

---

## 8. Wave-detection branch (parameter `Period`) for "Not classified" epochs

Only epochs that fall through the cascade reach this branch. It runs on the
**preprocessed** signal from Step 1 (not on `FS`), detects individual waves, and
derives a rate parameter `Period`.

### 8.1 Peak / wave detection

A "peak" here is a **local extremum** (a turning point of the signal), not the
threshold-crossing sample. The threshold is only an amplitude qualifier: crossing
it opens a candidate excursion, and the peak is the extremum reached inside that
excursion (the local maximum for a positive peak, the local minimum for a negative
peak). The peak's value is what feeds `MPP`/`MNP`, the threshold update, and the
0.1 s refractory comparison, so recording the extremum (not the crossing) matters.

1. Find the first peak: scan until the signal crosses the starting threshold of
   **±150 µV**, then track to the turning point of that excursion (local maximum
   above +150 µV, or local minimum below -150 µV). That turning point is the peak;
   its value is the peak amplitude.
2. After a peak is found, refresh the threshold to the greater of `0.25 ×
   (peak amplitude)` and 150 µV for positive peaks (and the more-negative of
   `0.25 × (peak amplitude)` and -150 µV for negative peaks). The threshold is
   refreshed after every detected peak.
3. Half-wave rule. Having found a positive peak, search for the next peak. If it is
   negative and within **1 s** of the positive peak, a half-wave is detected. If
   the next peak is again positive (the intervening negative peak stayed below the
   negative threshold), keep the larger of the two positives and keep searching for
   a negative peak. The procedure is symmetric for a leading negative peak.
4. Same-polarity refractory. If a new peak of the same polarity is within **0.1 s**
   of the previous same-polarity peak, discard the smaller (in absolute value).
5. After all peaks are validated, compute `MPP` = mean amplitude of all positive
   peaks and `MNP` = mean amplitude of all negative peaks. Keep the peak set
   belonging to the larger of `MPP`, `MNP` for the rate estimate.

### 8.2 Period estimate

- **Regular case:** if more than **87.5%** of the kept peaks lie within **75% to
  125%** of their mean (`MPP` or `MNP`), then

  `Period = (length of the 10 s interval) / (number of detected waves)`.

- **Irregular case:** otherwise use the Kuo and Dillman (1978) VF-filter estimate
  (paper eq. 2):

$$\text{Period} = 2\pi \frac{\sum_{i=1}^{m} |S_i|}{\sum_{i=2}^{m} |S_i - S_{i-1}|},$$

  with `m` the number of samples in the 10 s segment. This is a smooth,
  peak-free mean-period estimate for signals without clean waves (no FFT, no
  zero-crossings, no peak finding), which is exactly why it is the fallback when
  the peak-based regular case fails.

  **What it computes.** The numerator is the signal's average absolute amplitude
  (its L1 size); the denominator is its total variation (average absolute
  one-sample step), which grows with how fast the waveform wiggles. Their ratio is
  large when the signal moves slowly relative to its size (long period) and small
  when it oscillates fast (short period); `Σ|ΔS|/Σ|S|` is a cheap time-domain
  estimate of the mean angular frequency, and Period is its reciprocal times 2π.

  For a pure sinusoid `S = A·sin(ωt)` the estimate is exact: `mean|S| = (2/π)A`
  and `mean|ΔS| ≈ (2/π)A·ω/f_s`, so the amplitude cancels and
  `2π·Σ|S|/Σ|ΔS| = f_s·T`. So **eq. 2 returns the period in samples**; convert to
  a rate with `rate = 60·f_s / Period_samples` bpm (that rate is what Step 8.3
  compares against 180 bpm), or divide by `f_s` for seconds.

  This is only the *period* half of Kuo and Dillman's VF-filter: their full method
  then delays the signal by half this period and measures the residual
  `S(t) + S(t − T/2)` (near zero for a narrowband, sinusoid-like VF). Jekova
  borrows only the period. The full leakage measure lives separately in this repo
  as the vftx `vf_leak` feature [6].

### 8.3 Wave-branch outputs

- **Asystoly** if no waves were detected in the **last 5 s** of the 10 s epoch.
- **Non-shockable** if the rate from `Period` is **below 180 bpm**.
- **Shockable** if `Period` yields a rate **above 180 bpm**.

Convert `Period` (in seconds per beat) to rate with `60 / Period`; the 180 bpm
boundary is `Period = 1/3 s`.

> Repo divergence (see section 9): the `vfta/` signal-only detector has no peak
> detector on purpose, so it does not implement Step 8. `vfta/jekova.py:decide()`
> substitutes a single threshold on the normalised Count3 (`f3 <= c3_fallback`,
> shockable when Count3 is low) for the whole wave-detection branch, tuned per
> window length.

---

## 9. Where the repo implementations differ from a literal paper reading

Keep these in mind when comparing repo output to the published sensitivity
(95.93%) and specificity (94.38%):

1. **Preprocessing front end.** Neither `hong/` nor `vftx/` runs Jekova's Step 1
   (two 1 Hz first-order high-passes + 30 Hz second-order Butterworth + 50 Hz
   notch, all forward). `hong/` runs its own zero-phase pipeline; `vfta/` filters
   once per record with the exg-core Lynn band-pass + median baseline. Only the
   band-pass (Step 4) is Jekova's.

2. **Signed vs absolute counts.** `hong/vf_features.pyx:auxiliary_counts()` and
   `vftx`'s `count1/2/3` features count the **signed** `FS`, not `|FS|`. This makes
   Count2 (`sum(FS >= mean)`) roughly 0.5 of the window for every rhythm and
   therefore uninformative. `vfta/jekova.py:abs_counts()` restores the paper's
   `AbsFS` definition, which is what the cascade needs.

3. **Loop-step quirk at non-250 Hz.** After resampling to 250 Hz, the reference
   still steps the 1 s count loop by the *original* `sampling_rate`
   (`hong/vf_features.pyx:484`, mirrored in `vftx/_count_helpers.py:39`). At 250 Hz
   this is correct (1 s = 250 samples). At any other native rate the "1 s" window
   length no longer matches the resampled array, so the windows are mis-sized.
   Feed 250 Hz data to avoid it.

4. **Window length and threshold scaling.** The published integer thresholds
   (Step 7) are counts over 2500 samples. The paper pipeline (`vfta/`) uses 8 s and
   4 s windows, so it normalises counts to `Count / N` and retunes the cascade
   (`vfta/tuning.py`). Do not apply the raw 250/400/600/950/1100/210 constants to
   any window that is not 10 s at 250 Hz.

5. **No wave-detection branch in `vfta/`.** Step 8 (peaks, `Period`, Kuo-Dillman)
   is replaced by a Count3 fallback threshold. Epochs the cascade cannot resolve
   are decided by `f3 <= c3_fallback` rather than by heart rate.

---

## 10. Reference implementation skeleton (250 Hz, paper-faithful)

Pulled together from the resolved steps above; this is the "clean" Jekova path,
matching `vfta/jekova.py:abs_counts()` for Steps 4-6.

```
def jekova_classify(epoch_uV, fs=250):        # epoch_uV: 2500-sample 10 s epoch, microvolts
    x = preprocess(epoch_uV, fs)              # Step 1: 2x 1 Hz HP (fwd) -> 30 Hz 2nd-order Butter (fwd) -> 50 Hz notch

    if max_slew(x, fs) > 1.6:  return "Noise"      # Step 2: >400 uV/ms == >1.6 mV/sample at 250 Hz
    if amp_saturated(x):       return "Noise"      # Step 2: AD full-scale
    if peak_amplitude(x) < 150:return "Asystoly"   # Step 3: <150 uV

    fs_out = bandpass_14p6(x)                  # Step 4: FS[i]=(14 FS[i-1]-7 FS[i-2]+(S[i]-S[i-2])/2)/8, forward only
    afs = abs(fs_out)                          # Step 5

    c1 = c2 = c3 = 0                           # Step 6: per 1 s, summed over 10 s
    for w in one_second_windows(afs, fs):      # 250-sample windows
        mx, mn = max(w), mean(w)
        md = mean(abs(w - mn))
        c1 += count(w >= 0.5*mx)
        c2 += count(w >= mn)
        c3 += count((w >= mn-md) & (w <= mn+md))

    ratio = c1 * c2 / c3                        # Step 7
    if c1 < 250 and c2 > 950 and ratio < 210:              return "Non-shockable"
    if 250 <= c1 < 400 and c2 < 600 and ratio < 210:       return "Non-shockable"
    if c1 >= 250 and c2 > 950:                             return "Shockable"
    if c2 >= 1100:                                         return "Shockable"

    period = wave_period(x, fs)                # Step 8 (on preprocessed x, not afs)
    if no_waves_last_5s(x, fs):  return "Asystoly"
    return "Non-shockable" if 60.0/period < 180 else "Shockable"
```

---

## 11. Implementation plan: paper-faithful JEKOVA, then an EXG front end

The goal is a JEKOVA path that (a) reproduces the paper's own front end first, so
the counts and cascade are validated against a known reference, and only then (b)
swaps in the EXG-core preprocessing (median baseline + Lynn band-pass) and checks
whether the counts stay discriminative. Everything is built forward-only and
block-independent, so the same code runs as a real-time stream and as a batch pass
over the record corpus.

This is a clean standalone implementation: it does not reuse the `vfta/` code. Every
filter, including the band-pass, is a plain `(b, a)` IIR run once over the whole
input signal with `scipy.signal.lfilter` (forward only). There is no per-window
re-filtering and no explicit filter-state (`zi`) handling: the single whole-signal
pass is the batch form of a real-time stream, and streaming state, if ever needed,
is a later concern, not part of this plan.

Design invariant for the whole plan: **all filtering is causal, single forward
pass, applied once over the whole input signal, never re-run per overlapping
window.** That is both the real-time target and the precondition for the 1 s count
reuse (section 6): with one continuous filter pass, each 1 s cell's `AbsFS` is
fixed, so a window's counts are just the rolling sum of its 8 (for 8 s) or 4 (for
4 s) per-second partials.

### Phase A: paper-faithful front end (forward-only)

- **A1. Preprocessing (Step 1), causal cascade.** Implement the paper front end as
  four forward `lfilter` stages, each a `(b, a)` pair: two first-order 1 Hz
  high-passes, then a second-order 30 Hz Butterworth low-pass (`scipy.signal.butter`),
  then a 50 Hz notch (`scipy.signal.iirnotch`), all at 250 Hz (resample first if the
  source differs). No `filtfilt`, no `zi`.
- **A2. 14.6 Hz integer band-pass (Step 4).** Implement eq. (1) directly as a
  `(b, a)` IIR and apply with `lfilter` (forward only):

  ```
  b = [1/16, 0, -1/16]          # 0.5*(1 - z^-2) / 8
  a = [1, -14/8, 7/8]           # 8 - 14 z^-1 + 7 z^-2, normalised by a[0]
  FS = lfilter(b, a, S)         # FS[i] = 1.75 FS[i-1] - 0.875 FS[i-2] + (S[i]-S[i-2])/16
  ```

  This is eq. (1) verbatim, so no custom recursion loop and no `vfta` dependency.
- **A3. AbsFS + per-second partial counts (Steps 5-6).** Compute `AbsFS = |FS|`
  once over the whole signal. For each 1 s cell store the partial `(count1, count2,
  count3)` from `max/mean/MD` of that cell.
- **A4. Window totals by rolling sum.** A window's counts are the sum of its
  consecutive per-second partials. One pass gives every overlapping window at ~1/8
  the band-pass cost.
- **A5. Cascade + fallback (Steps 7-8).** Apply the cascade. For "not classified"
  windows, either implement the Step 8 wave/`Period` branch or use a normalised
  Count3 fallback; make the choice a config flag so both can be measured.
- **A6. Validation.** On 250 Hz / 10 s epochs, confirm the cascade with the
  published integer thresholds reproduces the paper's shock/no-shock logic, and
  that filtering the whole signal in one pass matches a block-wise forward pass
  (identical away from the cold start), which is what makes the design stream-safe.

Deliverable: a standalone `JekovaFaithful` producing per-window counts and a
decision, counts taken on `AbsFS` (not Hong's signed output), with no moving
average and no min-max in front (the divergences flagged in section 9).

### Phase B: EXG-core front end variant

- **B1. Swap only Step 1.** Replace the paper preprocessing with the EXG-core
  filters (median-baseline removal + Lynn band-pass, reimplemented cleanly to match
  the exg-core pipeline), leaving A2-A5 unchanged. Both front ends feed the same
  14.6 Hz band-pass and the same count logic, so any difference in counts is
  attributable to the front end alone.
- **B2. Cheap pre-check: front-end frequency response.** Before any label study,
  overlay the two front ends' magnitude responses across 13 to 16.5 Hz (the count
  band). If both pass that band with similar gain and near-linear phase, the counts
  should transfer; if the Lynn band-pass reshapes or phase-distorts that region,
  expect the count distributions to move. This one plot predicts most of the
  outcome for little cost.
- **B3. Matched corpus.** Emit both front ends' counts for the same window set
  (same records, same window grid) so every comparison in Phase C is paired.

### Phase C: are the counts still relevant? Comparison methodology

Raw count equality is not the target: the counts are scale-relative (thresholds
are `0.5*max`, `mean`, `mean +/- MD` within each 1 s), so any linear gain
difference between front ends cancels, and the band-pass zeroes DC so offsets
cancel too. What must survive is the **shape of the per-class distributions and
their separation**. Assess it at three levels.

1. **Distribution shape (reproduce paper figure 4).** For each front end, overlay
   histograms of `Count1`, `Count2`, and `Count1*Count2/Count3` for shockable vs
   non-shockable (using the shockable/non-shockable annotation). Visual first pass:
   are the two classes still separated, and do the modes sit where the paper's do.
2. **Separation preserved (single-count discrimination).** For each count under
   each front end compute point-biserial correlation, single-count AUC, and mutual
   information. The counts stay relevant if each count's AUC under the EXG front
   end is within a small tolerance of the paper front end.
   Also quantify class separation directly with the KS statistic and the
   Wasserstein (earth-mover) distance between the shockable and non-shockable
   count distributions, per front end.
3. **Transferability (do thresholds carry over).** On the paired corpus:
   - Spearman rank correlation of each count between the two front ends (high
     monotonic agreement means a rescaled threshold transfers);
   - Bland-Altman on the normalised counts (`Count/N`) to expose systematic bias;
   - Run the cascade under each front end, both with the paper thresholds and with
     a single retuning (threshold sweep / operating-point search); report Se/Sp per
     front end and Cohen's kappa on the window-level shock/no-shock decision between
     the two.

Decision rule for "sufficiently similar": the EXG front end is acceptable if (a)
each count's AUC is within tolerance of the paper front end, and (b) one retuning
of the cascade thresholds recovers Se/Sp within tolerance. Exact numeric agreement
of the counts is neither expected nor required; monotonic, separable, retunable is
the bar.

If a count fails the bar, B2's frequency-response plot is the first diagnostic:
a divergence there localises the cause to how the EXG front end treats the 13 to
16.5 Hz band, which is the only band the counts actually see.
