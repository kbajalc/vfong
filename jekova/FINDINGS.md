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

## 10. Richer level statistics, shape features and VF-filter leakage (JEKOVA6)

Goal: a cheap, causal marker of suspected VF/VFL for human review, still filters and counting statistics only. Section 6 left per-record calibration as the open headroom. This section tests whether a better count statistic or one extra independent channel closes part of it. Code is `jekova/JEKOVA6.ipynb` (the detector and a sample-level eval); the sweeps ran offline from per-second statistics cached for all five databases.

### Protocol

All numbers use 8 s windows, shockable = reference overlap >= 85%, wfon bracket mode. Sample F1 counts seconds of VF covered by marks. Episode scoring: a reference episode is caught when a mark overlaps it for >= 1 s, a mark is false when it lies more than 4 s from every reference episode, false marks per hour use non-VF hours. Anything fitted was fitted leave-one-database-out (thresholds and weights chosen on the other three of mitdb, ahadb, cudb, vfdb; criterion = mean of sample F1 and episode F1, averaged over the training databases). edb is never used for fitting; it gets the threshold fitted on all four.

The published cascade reproduces here at pooled sample F1 69.9, episode F1 57.1, recall 91/97 (>= 5 s), 83/85 (>= 8 s), 70/70 (>= 15 s), edb 0.04 false marks/h. Per database my second-resolution scorer gives mitdb 93.7 (not 93.1), ahadb 97.0, cudb 80.6, vfdb 37.1.

### What was tried

About 110 per-window features were screened by AUC (cheap, per database), then the best went through threshold sweeps. Variants evaluated at some depth: about 100 single-feature thresholds (70 thresholds each), 8 feature sets x 2 models (logistic regression, depth-2 tree) leave-one-database-out, 864 hysteresis / gap-merge / minimum-length settings (two level statistics), and 7 finalists x 2 gap options x 40 thresholds. Roughly 10,000 evaluated settings in all, most of them cheap threshold points.

**Level statistic (idea A, A-C).** Counting at a different reference level helps a little. Plain single-threshold leave-one-database-out, pooled sample F1: mean 70.2 (published level), RMS 72.2, L3-mean 73.6, 0.5 x mean 73.1, L4 73.2. A max-fraction count (>= 0.25 x max) 69.8. Median, trimmed mean, geometric mean and the 10th to 75th percentiles are worse or, for the median and percentiles, constant by construction (the count at the median is always half the samples). Other envelopes (smoothed |x|, Teager-Kaiser, first difference, squared) do not beat |x| and the smoothed one is clearly worse. Why RMS helps: a higher bar rejects the broad low-amplitude background of vfdb and cudb, at the cost of mitdb (87 to 94 depending on level).

**Shape statistics.** Crest factor, form factor and kurtosis of the band output have inverted AUC 0.90 to 0.99, about as good as c2 on mitdb and ahadb and 0.86 to 0.95 on cudb and vfdb. They are cheap (running sums of |x|, x^2, x^4, max) but largely redundant with the count: adding crest to the count+leakage model changes pooled F1 by -0.4 to +0.4 and lowers episode F1 once gap-merged.

**Other bands (idea D).** 2.5 to 7.5 Hz, 3 to 6 Hz and 6.5 to 30 Hz counts and band-energy ratios have AUC 0.80 to 0.99, not better than the 14.6 Hz band. The 2.5 to 7.5 Hz count and the 6.5 to 30 Hz count come out in a similar league to c2 (wt count at mean: cudb 82.6 held out, vfdb 36.4) but none helps vfdb. Not kept.

**VF-filter leakage (idea E).** Kuo-Dillman leakage computed on the Lynn-filtered signal (no band-pass) over 8 s, with the half period taken from `T = 2*pi*sum|V|/sum|dV|`, clipped to 8..60 samples. Low leakage means quasi-sinusoidal. Inverted AUC: mitdb 0.999, ahadb 0.996, cudb 0.89, vfdb 0.89. Alone it gives high episode sensitivity (95% pooled) but broad marks, sample F1 only 55.0. On the band-passed 2.5 to 7.5 Hz signal it is slightly worse. As a second feature next to the RMS count it is the one clear gain.

**Hysteresis and gap handling (idea G).** On a plain RMS or L3 score, high/low hysteresis, gap filling and minimum length trade sample F1 against episode F1 without a stable leave-one-database-out gain (mean criterion 75.6 with hysteresis vs 76.1 plain at RMS; 71.1 vs 74.0 at L3). Filling gaps of 15 s or less is kept only because it helps once the leakage channel is in (below).

**Not retried** (listed in the task as known to fail): label-free per-record calibration, persistence of window decisions, shorter windows, amplitude gates. Idea F (a reference-window selector from QRS trains) was not built, so it is still open.

### Result

Leave-one-database-out, gap filling optional (chosen on the training databases). Rows are the published cascade and the finalists; gap 15 means short gaps filled.

| detector | mitdb F1 | ahadb F1 | cudb F1 | vfdb F1 | pooled F1 | episode F1 | recall >= 5 s | >= 8 s | >= 15 s | false/h (4 DBs) | edb false/h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| published cascade | 93.7 | 97.0 | 80.6 | 37.1 | 69.9 | 57.1 | 91/97 | 83/85 | 70/70 | 1.74 | 0.04 |
| cnt at mean, one threshold | 92.2 | 97.0 | 80.4 | 39.9 | 72.6 | 51.2 | 78/97 | 76/85 | 68/70 | 1.42 | 0.04 |
| cnt at RMS | 91.4 | 96.3 | 81.2 | 43.5 | 74.4 | 57.0 | 87/97 | 81/85 | 70/70 | 1.27 | 0.01 |
| cnt at L3 mean | 85.8 | 95.8 | 80.8 | 44.5 | 74.2 | 62.4 | 95/97 | 85/85 | 70/70 | 1.34 | 0.01 |
| LR(cnt RMS, leakage) | 93.1 | 96.5 | 82.6 | 46.5 | 75.5 | 73.5 | 95/97 | 84/85 | 70/70 | 0.97 | 0.00 |
| **LR(cnt RMS, leakage) + gap 15 s** | 92.2 | 97.1 | 83.2 | 45.6 | **75.4** | **75.1** | 95/97 | 84/85 | 70/70 | **0.55** | **0.00** |
| LR(cnt RMS, leakage, crest) + gap 15 s | 92.5 | 97.0 | 83.5 | 46.0 | 75.8 | 69.7 | 95/97 | 84/85 | 70/70 | 0.95 | 0.00 |
| leakage alone + gap | 82.5 | 76.0 | 72.1 | 32.3 | 55.0 | 76.4 | 92/97 | 80/85 | n/a | 0.71 | 0.00 |

Episode F1 and false marks per hour per database for the bold row: mitdb 80.0 / 0.00, ahadb 92.5 / 0.03, cudb 89.9 / 2.58, vfdb 61.0 / 2.63 (published: cudb 10.0/h and vfdb 7.2/h). The small recall of sub-5 s bursts also improves, 42/74 caught versus 24/74.

Final parameters (fitted on the four databases pooled; the held-out numbers above come from refits without the evaluated database): `score = 0.022395 * crms - 4.0570 * leak - 7.6289`, mark when score > 3.379, fill gaps <= 15 s. `crms` is the 8 s sum of the per-second count of band-pass samples >= that second's RMS (max 2000). Applied as fixed constants to the four databases (in-sample) it gives pooled sample F1 75.1, episode F1 65.8, 0.53 false marks/h, recall 91/97, 82/85, 69/70. The episode F1 spread (65.8 in-sample at one threshold, 73.5 to 75.1 held out) is mostly vfdb, where small threshold moves swap many short marks. Treat episode F1 on vfdb as +-10.

### What this does and does not show

- The gain is real but modest on sample F1 (+5.5 pooled, mostly vfdb +8 and cudb +3) and large on mark quality (episode F1 57 to 75, false marks 1.7 to 0.55 per hour, cudb 10 to 2.6, vfdb 7 to 2.6). A reviewer sees about a third of the false marks, and sub-5 s VFL recall goes from 32% to 57%.
- The leakage channel is independent evidence in the sense section 9 expected: the count measures "no isoelectric baseline", leakage measures "single dominant frequency". The RMS level does what the oracle needed only in part: it lifts the bar where the background is high, but does not calibrate per record. Per-record headroom (oracle F1 ~88) is still mostly open.
- mitdb loses 1.5 to 2 points of sample F1 against the cascade (small VF mass, 6 episodes), which fails the "do not trade databases" test narrowly on that one; ahadb is unchanged (97.0 vs 97.1), cudb and vfdb gain. Pooled across all four it passes both pooled sample F1 and episode F1, and edb false marks go down (0 vs 8 marks over 180 h).
- Known weak records are unchanged in kind: the slow/coarse VF in cu20 and ahadb 8206 are still the main misses, and vfdb remains low-PPV (45.6 held out) because its non-VF background looks fibrillatory to every count.
- Selection used pooled LODO with a mixed criterion; with four databases and one fitted threshold per fold there is selection noise of a couple of points. The ranking leakage-plus-count over count alone holds in every fold's episode F1 but ahadb and mitdb are within noise.

### Cost for a causal Go port

Everything is a running sum on the 250 Hz Lynn output and the band-pass output, with per-second accumulators and an 8 s ring.

| piece | cost |
|---|---|
| RMS count | per second: one sum of squares, one second pass to count |x| >= rms (needs the 250 samples buffered once per second, same as cnt2 today) |
| leakage | running sums of |v| and |dv| per second; then for each of 53 lags the per-second sums of |v_i + v_(i-h)| and |v_i|+|v_(i-h)|, which is 53 x 2 multiply-free passes per second at 250 Hz (about 26,000 adds/s), or compute only the lag for the current period estimate once per second from the buffered 8 s (about 2,000 samples x 2) |
| score | 3 multiplies per window |
| marks | threshold and a 15 s gap fill, which delays a mark's end by at most 15 s |

The cheaper leakage variant (compute at the window's own half period from the buffered window, one pass per second) is what I would port. I did not test it numerically against the 53-lag version except through the notebook code, which agrees with the cached values to 1e-6 on cu20 and 0.05 on vfdb 418 (float32 rounding at a lag boundary).

### Follow-up: simple window baselines, TCSC, leakage without a period, spectral concentration

After the main sweep we ran four small, deliberately simple experiments to see what the counts and shape statistics do without the extra machinery. All of them score a single threshold on one feature, chosen leave-one-database-out (on the other three of mitdb, ahadb, cudb, vfdb, mean sample F1), with the same sample and episode scorer as above. Nothing here beat the RMS count plus leakage detector; this part is mostly a record of what does not work and why we think so.

#### A. Window-level statistics on the Lynn-filtered signal, no Jekova band-pass

Sliding 5 s window, 1 s stride, on the Lynn-filtered signal only. Features per window: mean |x|, RMS, cnt2 as the fraction of the window's samples at or above the window's own mean |x|, the same count at RMS and at the L3 mean, and the crest factor.

AUC inside each database (mitdb, ahadb, cudb, vfdb): mean 0.79, 0.76, 0.70, 0.36; RMS 0.54, 0.64, 0.60, 0.28; cnt2 0.975, 0.968, 0.83, 0.85; count at RMS 0.98, 0.97, 0.82, 0.85; crest (inverted) 0.98, 0.94, 0.78, 0.84. Amplitude alone is useless and in vfdb it points the wrong way, which agrees with section 7.

Leave-one-database-out, single threshold:

| feature | pooled sample F1 | episode F1 | false marks/h | edb false/h |
|---|---|---|---|---|
| cnt2 (count at mean) | 43.0 | 49.1 | 11.5 | 4.4 |
| count at RMS | 37.7 | 56.4 | 8.0 | 0.73 |
| count at L3 mean | 33.3 | 56.9 | 7.2 | n/a |
| crest factor | 40.6 | 62.4 | 7.2 | n/a |
| mean \|x\| | 25.2 | 39.8 | 8.0 | n/a |
| RMS | 16.9 | 27.0 | 12.1 | n/a |

The AUC looks fine and the F1 does not. Mitdb has 36k VF samples in 21M, and ahadb is similar, so a small false-positive rate ruins precision; in mitdb the best own threshold for the RMS count gives F1 64 and the best global threshold 9.5. Without the 14.6 Hz band-pass the signal still holds QRS, T wave and baseline residue, so "evenly spread activity" is not specific to VF. The band-pass is the load-bearing step, as section 3 says. This is the floor: a window-level count on the Lynn signal is not a detector.

#### B. The same window statistics on the Jekova band-pass output

Same windows, applied to the band-pass output. Single threshold, leave-one-database-out:

| window | level | sample F1 (mitdb / ahadb / cudb / vfdb) | pooled F1 | episode F1 | false marks/h | recall >= 5 s |
|---|---|---|---|---|---|---|
| 5 s | mean | 83 / 95 / 80 / 38 | 70.2 | 50.9 | 3.0 | 86/97 |
| 5 s | (RMS + mean)/2 | 85 / 95 / 79 / 42 | 72.0 | 55.3 | 2.9 | 93/97 |
| 5 s | RMS | 86 / 95 / 78 / 44 | 72.6 | 57.1 | 3.1 | 94/97 |
| 5 s | L3 mean | 72 / 95 / 72 / 44 | 70.4 | 61.4 | 3.7 | 96/97 |
| 5 s | crest | 62 / 71 / 54 / 39 | 56.0 | 57.2 | 6.7 | 91/97 |
| 8 s | mean | 91 / 96 / 81 / 38 | 71.0 | 51.7 | 1.6 | 78/97 |
| 8 s | (RMS + mean)/2 | 89 / 96 / 79 / 41 | 72.2 | 57.8 | 1.6 | 89/97 |
| 8 s | RMS | 88 / 96 / 78 / 43 | 72.5 | 60.5 | 1.7 | 94/97 |
| 8 s | L3 mean | 78 / 94 / 70 / 44 | 69.4 | 66.2 | 2.2 | 95/97 |
| 8 s | crest | 48 / 66 / 51 / 38 | 52.2 | 58.7 | 4.5 | 89/97 |

The question was whether a level computed over the whole window instead of per 1 s chunk helps. It does not. At 8 s the mean count gives pooled F1 71.0, the same as the cascade (69.9) within noise, and the RMS count gives 72.5, slightly below the per-second RMS version (74.4). The chunk-boundary jitter we worried about, where a QRS split across two chunks lowers both means, costs nothing measurable.

Every level statistic lands on one plateau (pooled F1 70 to 73, episode F1 51 to 60 for the counts) and moves along the same trade-off: a higher level gives up sample F1 on mitdb and cudb for recall and episode F1 on vfdb. The (RMS + mean)/2 blend sits between its two ends on every column and is not worth an extra statistic. The best per-database thresholds are always above the global one, so what limits all of these is calibration, not the choice of level. 8 s beats 5 s on false marks (1.6 against 3.0 per hour) at about equal sample F1.

#### C. TCSC (threshold crossing sample count)

TCSC as in `vftx/tcsc.py`: 3 s Tukey sub-windows, step 1 s, fraction of |x| above `th * max`, averaged over the six sub-windows of an 8 s window. Single threshold, leave-one-database-out, 8 s windows:

| signal, th | pooled sample F1 | episode F1 | episode sens | false marks/h | recall >= 5 s, >= 8 s |
|---|---|---|---|---|---|
| Jekova 0.1 | 68.3 | 63.0 | 67 | 1.57 | 93/97, 84/85 |
| Jekova 0.2 | 68.4 | 65.8 | 74 | 1.49 | 95/97, 85/85 |
| Jekova 0.3 | 69.5 | 62.0 | 67 | 1.74 | 94/97, 84/85 |
| Jekova 0.4 | 65.2 | 60.8 | 77 | 2.74 | 94/97, 83/85 |
| Jekova 0.5 | 54.8 | 64.5 | 73 | 2.75 | 90/97, 79/85 |
| Lynn only 0.2 | 56.8 | 68.5 | 95 | 2.65 | 95/97, 84/85 |

On the Jekova output TCSC at 0.2 has the cascade's sample F1 and a better episode picture (episode F1 65.8 against 57.1, recall 95/97 and 85/85), the best single-count result on the episode metrics. On the Lynn signal alone the episode numbers look good (sensitivity 95%) but sample F1 is poor: the marks are in the right places and over-extend.

Replacing the max by the window RMS (a Tukey-tapered window, fraction above `m * RMS`) was tried at 5 s and 8 s:

| signal, m | window | pooled sample F1 | episode F1 | false marks/h |
|---|---|---|---|---|
| Jekova 0.5 | 8 s | 71.3 | 48.9 | 1.75 |
| Jekova 0.75 | 8 s | 71.8 | 51.5 | 1.40 |
| Jekova 1.0 | 5 s / 8 s | 72.4 / 72.5 | 56.2 / 59.2 | 3.2 / 1.75 |
| Jekova 1.25 | 5 s / 8 s | 70.7 / 71.1 | 64.2 / 66.2 | 3.8 / 2.0 |
| Jekova 1.5 | 5 s / 8 s | 48.2 / 47.4 | 63.1 / 62.1 | 8.0 / 5.1 |
| Jekova 2.0 | 5 s / 8 s | 2.4 / 1.8 | 16.1 / 16.4 | 47.8 / 24.6 |
| Lynn only, best (0.75) | 5 s / 8 s | 45.1 / 48.5 | 58.4 / 63.5 | 7.7 / 3.9 |

At m = 1.0 this is the plain RMS count: pooled F1 72.4 to 72.5, within noise of the count at RMS, so the Tukey taper changes nothing measurable. m of 1.5 and above leaves too few samples above the bar in a VF window. The RMS-normalized version is no better than the max-normalized one, perhaps because the maximum follows burst amplitude that the RMS partly absorbs (a hypothesis; we did not test it). The Lynn-only TCSC variants are poor for the same reason as in A. We did not run the edb check for TCSC.

#### D. Leakage without the period estimate

Concern: `T = 2*pi*sum|V| / sum|dV|` assumes one clean sine. Noise inflates `sum|dV|`, so T comes out too short, and in a harmonic-rich or noisy window the half period `h` is simply wrong. The period-based version still tested best, so this is a design objection that the data do not confirm as a failure. To remove it we took the minimum leakage over the lags 12 to 47 samples instead of reading the lag from T:

| detector (gap 15 s) | pooled sample F1 | episode F1 | false marks/h | recall >= 5 s | sub-5 s bursts | edb false/h |
|---|---|---|---|---|---|---|
| RMS count + period-based leakage | 75.4 | 75.1 | 0.55 | 95/97 | 42/74 | 0.000 |
| RMS count + minimum leakage | 75.5 | 68.0 | 0.62 | 94/97 | 23/74 | 0.006 |
| minimum leakage alone | 52.5 | 71.6 | 0.88 | 92/97 | 70/74 | 0.011 |

Sample F1 is unchanged, so the period is not needed for the main gain. Episode F1 drops by 7 points (vfdb 51 against 61, ahadb 90 against 93), and short bursts are caught half as often. A plausible reason is that a minimum over 36 lags is biased low for every window, noise included, since some lag cancels by chance; the correctly read period gives a sharper test. The vfdb episode figures move by about +-10 with small threshold changes, so the size of the gap is uncertain though its direction is consistent. The minimum is the version to port if the period estimate is judged too fragile, and it costs more (36 to 53 lag sums per second). Autocorrelation at the half period is a close relative of it; we did not run it.

#### E. Spectral concentration (frequency-domain), dropped

A dense bank of DFT bins (2.5 to 8 Hz in 0.25 Hz steps) over the 8 s window, built from causal per-second partial sums. Features: the peak bin's share of the window energy, the peak and its two neighbours, and the energy share of the whole band. AUC inside each database (mitdb, ahadb, cudb, vfdb): Lynn signal peak +-1 bin 0.92, 0.74, 0.65, 0.81; Jekova output peak +-1 bin 0.96, 0.93, 0.73, 0.88; Jekova output band share 0.95, 0.96, 0.75, 0.88. Period-based leakage (inverted) is 0.999, 0.996, 0.89, 0.89.

| detector (gap 15 s) | pooled sample F1 | episode F1 | false marks/h | edb false/h |
|---|---|---|---|---|
| concentration alone, Lynn | 35.9 | 42.7 | 4.2 | 2.8 |
| concentration alone, Jekova | 47.9 | 61.0 | 2.1 | 0.64 |
| RMS count + concentration, Lynn | 74.7 | 58.0 | 0.69 | 0.011 |
| RMS count + concentration, Jekova | 74.6 | 57.7 | 0.73 | 0.017 |
| RMS count + period-based leakage | 75.4 | 75.1 | 0.55 | 0.000 |

Next to the RMS count it adds nothing: sample F1 is the count's own 74.4 to 74.7, episode F1 stays at 58, and short-burst recall is 6/74 against 42/74 with leakage. An 8 s window of VF is not a clean sine, so its energy spreads over many bins and the peak is often a baseline or T-wave component on the Lynn signal. It only works at all on the band-passed signal, where the time-domain filter is doing the selection. Our working conclusion is that a frequency-domain feature does not suit this signal: the A2, M and FM group of the vftx list belongs to the same family and was not carried forward. Leakage measures something different, that the waveform repeats at its own half period, which is why it helps and concentration does not.

### What we would not repeat

- Any level statistic for the count (mean, RMS, L3, a blend, a trimmed or geometric mean), at 1 s, 5 s or 8 s: all land on the same pooled F1 plateau of 70 to 75 and trade vfdb for mitdb and cudb.
- Median and percentile counts: constant by construction.
- Counts on the Lynn-only signal: far below the floor; the band-pass is required.
- Crest factor, form factor and kurtosis as stand-alone detectors: AUC is high, F1 is not (pooled 52 to 56 on the Jekova output, 41 on Lynn).
- TCSC with either normalization as a replacement for the count: it shifts the trade-off toward episode metrics without improving sample F1.
- Hysteresis and minimum-length rules on a single score: no stable leave-one-database-out gain.
- Spectral concentration: see E.

### Open

Idea F (reference windows picked by a confident narrow-QRS train) was not built. It is the next cheapest thing to try for the per-record calibration gap. A leakage taken at the dominant spectral peak (as in vftx `vf_leak`) is no longer worth building after E. Autocorrelation at the half period, as a period-free relative of minimum leakage, was not run.
