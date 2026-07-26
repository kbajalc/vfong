# JEKOVA findings

What we learned reimplementing the Jekova and Krasteva (2004) shockable-rhythm
detector in `JEKOVA2.ipynb` and evaluating it on five PhysioNet databases. The
short version: the algorithm's discriminative content is a single count feature
on the 14.6 Hz band-pass output. Everything else in the published design is
real-time implementation and operating-range handling that adds nothing
measurable to discrimination on these data.

## Setup

The pipeline reads ECG through `pxg`, filters per record, slides overlapping
1 s-step windows at 8 s, and writes one TSV per record with per-window counts
`c1`, `c2`, `c3` and the VF-overlap label `vfb`. A window is shockable when at
least 85% of it overlaps a VF, VFL, VT or WF episode (`FRAC = 0.85`), and
non-shockable when it does not overlap one at all. The counts are the paper's
Step 5-6 features on the absolute band-pass output, accumulated over the window.

Evaluation is per record with a pooled TOTAL, reported as F1 (the positive class
is a small minority in every database).

| database | 8 s windows | shockable windows |
|---|---:|---:|
| mitdb | 86112 | 111 |
| ahadb | 140383 | 5346 |
| cudb | 17465 | 3493 |
| edb | 646560 | 0 |
| vfdb | 45958 | 2022 |

edb has no shockable windows, so it contributes only false positives and its F1
is 0 or undefined throughout.

## 1. The cascade reduces to a single c2 threshold

The published cascade has four rules over `c1`, `c2` and the ratio
`c1·c2/c3`. On our data it collapses to one threshold on `c2`.

| detector | pooled F1 |
|---|---:|
| full 4-rule cascade | 70.4 |
| `c2 > C2_HI` only (one feature, one threshold) | 69.9 |
| cascade minus the c1 gate | 69.9 |

Per database the full cascade and `c2`-only are indistinguishable: mitdb
94.0 / 94.0, ahadb 96.4 / 96.3, cudb 79.3 / 79.2, vfdb 37.6 / 36.8. The
non-shockable rules R1 and R2, which carry the `c1·c2/c3` ratio the paper calls
its strongest discriminator, never change an output. They require `c1` low while
`c2` high, a near-contradictory combination (5 to 230 windows in the whole
corpus), and even when they fire they emit "non-shockable", which scores the same
as the "not classified" default. A grid search over all six thresholds confirms
only `C1_LO` and `C2_HI` affect the result; 180 of ~3900 grid points tie at the
maximum, and the published values sit at that maximum.

## 2. What c2 actually measures

`c2` is the count of samples in the window whose rectified band-pass amplitude is
at or above the window's own mean, `c2 = #{ |FS| >= mean(|FS|) }`. Normalised, it
is the fraction of samples above the local mean, so absolute amplitude cancels and
what remains is the shape of the amplitude distribution over time.

Physiologically it is an "absence of isoelectric baseline" measure. Organised
rhythm is bursty: quiet baseline between beats, brief beat-locked bursts. Those
bursts pull the mean up, so most samples fall below it and `c2` is low.
Fibrillation is continuous activity with no quiet baseline, so the amplitude is
spread more evenly and more samples clear the mean, giving high `c2`. On 8 s
windows the median fraction above the mean is about 0.34 for VF and 0.22 for
organised rhythm; that gap is the entire discriminator.

`c3` (fraction of samples in the mean±MD band) is the same quantity read from the
other side. Its correlation with `c2` is -0.89, and combining the two (AND or OR)
gives 68.8 versus 68.7 for `c2` alone. `c1` (fraction above half the window
maximum) is a third correlated variant. The three counts are one idea measured
three ways.

## 3. The band-pass is the load-bearing part

Removing the 14.6 Hz band-pass and counting on the preprocessed signal directly
drops the best global-threshold pooled F1 from ~70 to ~55, and even a per-database
best threshold reaches only mitdb 75.9 and vfdb 25.3, versus 94 and 37.6 with the
band-pass.

The band-pass does two things that matter. First, scale normalisation: it makes
count magnitudes comparable across records and databases so one global threshold
transfers, which is why the with-band-pass cascade reaches 94 to 96 on clean data
with a single threshold. Second, tail separation: on mitdb it lifts `c2` AUC from
0.998 to 1.000. That 0.002 sounds trivial, but with 111 shockable windows among
86112 the tail overlap it removes is enough false positives to move F1 from 76 to
94. The counting is cheap; the band-pass is what makes the counts separable.

## 4. The front-end filter choice barely matters

We compared the exg-core Lynn band-pass front end against the paper's own Step-1
sequence (two 1 Hz high-passes, 30 Hz Butterworth low-pass, 50 Hz notch). Best
`c2` global threshold gives 69.9 with the Lynn front end and 68.7 with the
original Jekova sequence. The difference is within noise and shows up mostly on
mitdb (c2 F1 88 versus 94), a tail-alignment effect. Swapping in the paper's exact
preprocessing does not unlock anything.

## 5. Step 8 (the wave branch) does not help

The paper resolves "not classified" windows with a wave-detection branch. We built
it two ways.

The irregular fallback (Kuo and Dillman eq. 2, `Period = 2π·Σ|S|/Σ|ΔS|`) is a
mean-frequency estimate, not a wave rate. On a QRS-containing signal it measures
edge sharpness: median implied rate 269 bpm for non-VF versus 254 bpm for VF, the
wrong order, both above the 180 bpm boundary. Wired as a fallback it flags almost
everything shockable and pooled F1 collapses to 56.8.

The regular-case peak/wave detector (Section 8.1: adaptive ±threshold, half-wave
pairing, refractory, dominant-polarity count) does work once the amplitude floor
is set correctly. A fixed absolute floor fails because the preprocessed amplitude
varies about 3x across records; a low fixed floor (the causal choice for a
real-time device, since the threshold then adapts to 0.25·peak) gives realistic
counts: mitdb sinus 10 waves per 8 s, cudb VF 36. The resulting wave rate has AUC
0.82 to 0.99. It is a real discriminator, but redundant with `c2`: `c2` beats it
on every database (0.94 to 1.00 versus 0.82 to 0.99), and `c2 + wave-rate` equals
`c2` alone. "No isoelectric baseline" and "many fast waves" are two views of the
same continuous ventricular activity.

Used as a fallback for "not classified" windows, the wave branch is net-negative:
at the paper's 180 bpm pooled F1 is 56.8, and even at a tuned 220 bpm it is 68.2,
below the 70.4 baseline, helping only cudb while adding false positives elsewhere.
Used as a veto (require fast waves to confirm a shock) it is worse, because real
VF often has a low wave rate (coarse or slow VF), so the veto removes true
positives and F1 falls toward 30 to 57.

## 6. Where the real headroom is, and why it is hard

A single `c2` threshold tuned per record (an oracle, using labels) reaches pooled
F1 88.0: mitdb 96.1, ahadb 98.1, cudb 90.5, vfdb 66.0. All of the headroom above
the ~70 global operating point is per-record threshold adaptation, not more
features and not the wave branch.

That headroom resists label-free realisation. `c2` is amplitude-invariant (its
reference is the window's own mean) but not baseline-invariant: rhythm morphology
and noise shift the distribution shape per record, which is the drift a fixed
global threshold cannot track. Simple deterministic per-record schemes all lose to
the global baseline: percentile subtraction 49, ratio normalisation 39, per-record
Otsu 60, all below 70. They fail for two reasons. In records with little or no VF
any per-record split invents false positives, and in VF-heavy records the non-VF
baseline estimate is contaminated by VF. Estimating a record's non-VF reference
without labels is the open problem.

The two hardest databases have distinct causes. vfdb over-calls on genuine non-VF
(6019 of 6263 false positives are pure non-VF, not label-boundary artifacts), a
per-record `c2` offset problem. cudb carries coarse or slow VF (record cu20's VF
reads ~75 bpm by wave rate and sits below the count threshold), which no count or
wave threshold catches because the rhythm is genuinely low-frequency.

## Conclusion

JEKOVA's discriminative core is one physiologically cheap feature: the fraction of
14.6 Hz band-pass samples above the window's local mean (`c2`), an absence-of-
isoelectric-baseline measure. The band-pass in front of it is essential and does
the real work of scale normalisation and tail separation. The remaining published
machinery (the c1 gate, the `c1·c2/c3` ratio, `c3`, the multi-threshold cascade,
integer filter coefficients, the Step 8 wave branch) is correlated restatement,
real-time implementation detail, or operating-range handling, and none of it moves
pooled F1 off ~70 on these databases. The only lever with real headroom is
per-record threshold adaptation, worth ~18 points of pooled F1, for which we found
no label-free deterministic realisation.
