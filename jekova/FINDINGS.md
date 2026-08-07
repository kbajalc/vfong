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
The TSVs also carry five amplitude columns in millivolts, added while investigating
the false positives in section 7: `jabs` and `jmax` (sum and peak of the rectified
band-pass output), `sabs` (sum of the rectified preprocessed signal), and `smin`/`smax`
(its signed extrema).

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

## 7. Amplitude: what the counts throw away

Every threshold inside `Counts()` is derived from the segment's own `smax`, `smean`
and `md`, so the counts are self-normalised and the amplitude cancels completely. A
low-amplitude agonal trace with no isoelectric baseline therefore scores exactly like
coarse VF.

cudb/cu28 is the clearest case. The record is 508 s and its only annotated shockable
episode is the last 12.7 s (`WF`, 496.2 s to the end), yet the cascade calls 145
windows shockable: 3 true positives and 142 false ones, in two long runs at 180-280 s
and 313-372 s. Reading the signal explains why. Until 180 s the record is a 35 bpm
bradycardia with 3.5 mV QRS complexes and a flat baseline. At 180 s the amplitude
collapses about eightfold to a continuous 0.4 mV undulation with no baseline left,
and it never recovers. The mean rectified preprocessed amplitude over those false
positive windows is 0.057 mV against 0.371 mV for the record's own true negatives.
Spectrally the collapsed stretch carries 83-89% of its power below 3 Hz with a peak
near 1.7 Hz, while the annotated `WF` at the end has 43% of its power in 3-10 Hz. The
counts cannot see any of this. This is a dying heart, and the correct output is
non-shockable.

### A fixed amplitude floor cannot fix it

The obvious remedy is a millivolt floor, and on its own it does not work. Median mean
`|FS|` over the true shockable windows, by database:

| database | median | 5th percentile |
|---|---:|---:|
| cudb | 0.243 mV | 0.100 |
| ahadb | 0.154 | 0.036 |
| mitdb | 0.114 | 0.088 |
| vfdb | 0.051 | 0.032 |

cu28's false positive stretch sits at 0.034 mV, below vfdb's median true VF and inside
ahadb's lower tail. A floor at 0.05 mV clears most of cu28 but costs ahadb 694 false
negatives and takes vfdb from F1 37.6 to 24.7. Fine VF is real, and lead and database
gains differ, so no single absolute level separates the two populations. Normalising
per record against the record's own 90th percentile fails for a related reason: ahadb
records go into sustained VF, so the reference is set by the VF itself and late,
finer VF gets vetoed (ahadb false negatives 260 to 898 at a ratio of 0.25).

### Two amplitude gates tried, neither kept

We tried gating the cascade's shockable calls on amplitude, and in the end kept
neither variant. The code in JEKOVA4 is the plain published cascade; this section
records what we tried and why it did not stay.

The first version was a running reference: `AmpRef()`, a causal peak-hold with a
120 s half-life over each record's per-second peak-to-peak amplitude, with a window
gated on the ratio of its own amplitude to that reference. It worked (cudb F1 79.32
to 80.84-81.09 across two variants, ahadb 96.39 to 96.75-96.77) and it was built to
answer section 6's open problem, per-record adaptation without labels. cu28's ratio
read about 1.0 through the healthy bradycardia, 0.05 to 0.14 across the collapsed
stretch, and 0.86 again at the annotated `WF`, so a 0.15 threshold separated the two
cleanly. The second version dropped the running reference and gated on a single
absolute floor, `jabs >= 0.02` mV, on the window's mean `|FS|`. That also worked, for
most of the same false positives, but for less: cudb F1 79.32 to 80.28, ahadb 96.39
to 96.73, and per record cu28 142 false positives to 85 against 26 for the running
reference, cu31 56 to 37 against 21.

Both were removed. The running reference carried a second stateful column (`aref`), a
hand-tuned half-life, and a ratio threshold that needed retuning whenever the
underlying statistic changed (0.25 to 0.15 moving from a per-window mean to a
per-second peak-to-peak), for a gain of roughly 0.5 to 0.8 points of pooled F1 over
the floor alone. The floor alone was simpler but earns less, and either way the gate
does not close cu28: even the running-reference variant left 26 of its windows
amplitude-indistinguishable from real fine VF elsewhere. Neither addresses cudb's
false negatives, which are the larger half of its error budget (818 against 577 false
positives); the records that dominate them (cu20 with 223, cu30 with 141, cu12 with
97) carry slow or coarse VF that no amplitude statistic reaches. Weighed against that,
we chose to keep JEKOVA4 at the plain published cascade and treat amplitude gating as
a documented dead end rather than carry code whose benefit is a few points on one
database.

## 8. Revisiting cudb with the full feature set in JEKOVA4

JEKOVA4 renamed `c1`/`c2`/`c3` to `cnt1`/`cnt2`/`cnt3`, precomputes `cnrt` (the
`cnt1·cnt2/cnt3` ratio) once instead of inline, and adds six raw amplitude stats per
window: `smax`, `savg`, `sdev` on the Lynn-filtered signal, and `fmax`, `favg`,
`mdev` on the band-pass output (the same `max`, `mean`, mean-absolute-deviation that
section 7's `jabs`/`jmax` were built from, kept this time even after the gate was
dropped). With cudb's F1 still stuck around 79, the question was whether this larger
feature set, used honestly rather than as a hand-tuned cascade, has more to give.

The renamed columns reproduce section 1 exactly: `cnt2 >= C2_HI` alone scores 79.22
on cudb against the cascade's 79.32, the same equivalence as before under new names.

Univariate AUC of every column against the shockable label, cudb:

| feature | AUC | | feature | AUC |
|---|---:|---|---|---:|
| cnt2 | 0.948 | | fmax | 0.622 |
| cnt3 | 0.945 | | mdev | 0.637 |
| cnrt | 0.933 | | favg | 0.561 |
| cnt1 | 0.898 | | savg | 0.705 |
| smax | 0.525 | | sdev | 0.549 |

The six raw amplitude stats are individually weak on cudb (AUC 0.52 to 0.70). `cnt2`
and `cnt3` still dominate, matching section 2's account of `c2`/`c3`.

From the same six columns we built five dimensionless ratios that need no new
state, each is just two numbers from the current window: `purity = favg/savg`,
`peaked_f = fmax/favg`, `peaked_s = smax/savg`, `disp_f = mdev/favg`,
`disp_s = sdev/savg`. One is a real find: `disp_f`, the relative dispersion of the
band-pass amplitude within the window, reaches AUC 0.933 on cudb, competitive with
`cnt2`. Its own best single threshold scores F1 78.18, and a 2D grid search over
`cnt2` and `disp_f` together reaches 80.09 against the cascade's 79.32, a gain of
0.8 points.

To check whether any combination of these fifteen columns does better than that, we
trained decision trees on cudb with grouped cross-validation (`GroupKFold` by record,
so no record's windows appear in both train and test), sweeping depth 3 to 8 and four
feature sets: the paper's four columns, those plus the six raw amplitude stats, those
plus the five ratios, and all fifteen together. Every combination lands between 73 and
78, at or below the hand-tuned cascade. A model free to combine all fifteen columns
however it likes cannot beat four fixed thresholds. That is strong evidence the
within-window feature set is exhausted: the ceiling is not a feature-selection problem.

The real headroom is exactly where section 6 already found it. An oracle that picks
each record's best `cnt2` threshold using the labels reaches F1 90.75 on cudb (`cnt3`
gives 90.73), an 11-point gap that has nothing to do with which column is used and
everything to do with knowing where to draw the line per record.

We tried to close that gap without labels, normalising `cnt2` against a causal
reference built from its own history, the same idea as section 7's amplitude gate but
applied to the count instead:

| reference | cudb F1 |
|---|---:|
| baseline cascade | 79.3 |
| ratio to a 300 s trailing median of `cnt2` | 63.7 |
| ratio to the whole record's median `cnt2` | 61.8 |

Both are worse than doing nothing. cudb's records are short (8 min 20 s) and often
spend a large fraction of that time in VF, so any unsupervised baseline built from the
record's own `cnt2` history is contaminated by the VF it is supposed to be measured
against, the same failure mode section 6 documented for amplitude normalisation, now
confirmed for the count directly. Looking at the worst false-negative records directly
shows the size of the offset an oracle would need: cu20's own best `cnt2` threshold is
559 against the global 740, cu12's is 615, cu10's is 622. Real, and each one bigger
than anything a within-record estimate recovered.

Given the constraint of a single window's worth of signal and no labels, cudb sits
close to its ceiling around F1 79 to 80. That matches the original paper's own note
that cudb gave the lowest accuracy of every database it tested, for the same
underlying reason: short records with a high VF fraction leave no clean stretch to
calibrate a per-record baseline from.

## 9. Checking against the vfta paper's feature study

`paper/PAPER.md` runs a separate, larger study on the same idea: 16 signal-only
features across five candidate detectors, screened and tuned on the vfta pipeline's
own build of mitdb, ahadb, cudb and vfdb. Its JEKOVA candidate is the same band-pass
count family as JEKOVA4, so it is worth checking whether anything it found closes
cudb's gap here.

The paper's headline result is that tuning moves the published cascade from F1 0.670
to 0.847 (pooled across its four databases), by grid-searching all six cascade
constants instead of accepting the published values, and by adding a fallback rule on
Count3 for windows the cascade leaves "not classified", which the published cascade
defaults to non-shockable. That second point matches a gap we found directly in
JEKOVA4's own cudb data: of the 818 false negatives under the published cascade, 807
sit in the unclassified bucket, not in a rule that actively calls them wrong.

Neither lever moves cudb specifically. Applying the paper's own tuned thresholds to
JEKOVA4's cudb data, unmodified, makes it worse, F1 79.32 to 75.48, because the
tuning that helps the pooled mix trades sensitivity for specificity in a direction
cudb cannot afford. Re-tuning all six constants plus a Count3 fallback from scratch,
this time restricted to cudb alone (a 40,000-point random search over the parameter
space), reaches F1 80.38, the same ceiling section 8 found by every other route.
Targeted single-feature fallbacks land in the same place: a Count3 fallback alone on
the unclassified bucket gives 80.07, the ratio term 80.21. The paper's own reported
gain is real, but it belongs to the easier databases in its pool; cudb was already
near its ceiling under the published cascade, and no re-parameterisation of the same
count family moves it.

The paper's other twelve features split cleanly into two groups by its own screen
(section 4.2 there). Eight sit close behind the band-pass counts, all AUC 0.90 to
0.964: TCSC, MAV, HILB, PSR, A2, VF_LEAK, M, TCI, a threshold-crossing count, a mean
amplitude, two phase-space fills, and three spectral descriptors. None of them was
implemented here; HILB needs a Hilbert transform and a phase-space grid, and the
spectral three need an FFT power spectrum, real machinery beyond anything JEKOVA4
currently computes. But the paper's own feature-feature correlation
heatmap gives a reason to expect little from them here: this whole group correlates
with the band-pass counts, so it mostly re-measures the same "departure from
baseline" property `cnt2` already captures rather than adding independent evidence,
and the paper's own shootout already ranks every one of them behind JEKOVA on
discrimination. We tried a cheap proxy for the one property none of JEKOVA4's
existing columns capture, whether the window's activity looks organised over time
rather than merely how much of it clears a threshold: the standard deviation of
`cnt2` across the eight 1-second slices inside each 8-second window. It scores AUC
0.527 on cudb, no better than chance, which is consistent with the correlation
argument rather than against it.

The remaining five, LZ, FM, MEA, STE, and the raw peak-to-peak Amplitude, are the
ones the paper's screen ranks at the bottom (AUC 0.54 to 0.80), the complexity and
amplitude-shape families Amann et al. found fail wherever specificity must stay high
[COMP55-2005]. That matches section 8's own finding for JEKOVA4's raw amplitude
columns (`smax`, `savg`, `sdev`, `fmax`, `favg`, `mdev`, AUC 0.52 to 0.70 on cudb):
two independent studies, on different pipelines, agree that amplitude shape and
complexity are the weak end of this feature space.

## Conclusion

JEKOVA's discriminative core is one physiologically cheap feature: the fraction of
14.6 Hz band-pass samples above the window's local mean (`c2`), an absence-of-
isoelectric-baseline measure. The band-pass in front of it is essential and does
the real work of scale normalisation and tail separation. The remaining published
machinery (the c1 gate, the `c1·c2/c3` ratio, `c3`, the multi-threshold cascade,
integer filter coefficients, the Step 8 wave branch) is correlated restatement,
real-time implementation detail, or operating-range handling, and none of it moves
pooled F1 off ~70 on these databases. The one thing the published design leaves out
is amplitude, which the self-normalised counts discard by construction, and section 7
shows that gating on it does recover a few points on the worst database, though we
judged the gain too small to keep in JEKOVA4. Section 8 pushed further, adding six
raw amplitude columns and engineered ratios and testing them honestly with grouped
cross-validation, and none of it moves cudb past the ~79 to 80 ceiling a bare `cnt2`
threshold already reaches. The headroom is real (11 points on cudb alone) and it is
entirely per-record calibration, not feature choice, and it stays open: every
label-free attempt to estimate that calibration from the same data has made things
worse, because the records that need it most do not have enough non-VF signal to
estimate it from. Section 9 checked this against `paper/PAPER.md`'s larger, separate
study of sixteen signal-only features: its own tuning of the same cascade family
lifts pooled F1 from 0.670 to 0.847, but applied to JEKOVA4's cudb data as published
it makes cudb worse, and re-tuned specifically for cudb it caps at the same ~80 every
other route in this file finds. Two independent studies on two independent pipelines
converge on the same ceiling for the same reason.
