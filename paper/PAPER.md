# Detection of Ventricular Tachyarrhythmias

<!--
Manuscript skeleton. Each section opens with an editorial note in a blockquote:
what the section contains and which phase (see PLAN.md) fills it. Delete the notes
as sections are written. Prose follows STYLE.md.
-->

## Abstract

> To write last (Phase 5). One paragraph: the problem (detecting VT, VFL, VF from short ECG
> segments), what we do (rank classical deterministic features on VFDB + CUDB + AHADB +
> MITDB by how well they separate shockable rhythms, then tune the strongest into a
> detector), the headline result (top feature and its tuned F1 / Se / Sp), and the scope
> limit (deterministic methods only, for a regulated downstream use). No learned classifier.

## 1. Introduction

<!-- First pass, Phase 1. Reuses DRAFT.md Introduction and Related Work and THESIS.md,
     em dashes and AI tells removed. COMP5/COMP55 figures to be re-checked against the
     source papers in a later pass. -->

### 1.1 The ventricular tachyarrhythmias

The heart beats because electrical impulses travel through it in a coordinated sequence. When
that sequence breaks down in the ventricles, the heart's main pumping chambers, the result is
a group of dangerous rhythms called ventricular tachyarrhythmias. This paper looks at three of
them: ventricular tachycardia (VT), ventricular flutter (VFL), and ventricular fibrillation
(VF). They form a continuum of increasingly disorganized electrical activity, and where each
sits on that continuum is what makes it easy or hard to detect.

VT is a fast but still organized rhythm, typically above 100 bpm, arising in the ventricles
rather than the normal conduction path. The ECG still shows distinct QRS complexes, though
they are wide and abnormal in shape, and the rhythm is regular or nearly so. Because
individual beats stay identifiable, standard QRS detectors can follow VT, as long as they can
tell it apart from other fast rhythms. Rates above 180 bpm are considered shockable
[JEKOVA-2004].

VFL sits between VT and VF. The ventricular rate is very fast (200–350 bpm) and the ECG
becomes a continuous, smooth, sine-wave-like oscillation in which individual beats can no
longer be separated. VFL is usually short-lived and unstable, tending either to revert to a
more organized rhythm or to deteriorate into VF.

VF is the most severe form. The ventricles are activated chaotically by many independent
wavefronts at once, and the ECG shows rapid, irregular, low-amplitude activity with no P
waves, QRS complexes, or T waves. There is no effective pumping, and without immediate
defibrillation the patient dies within minutes. On the surface ECG, VF can resemble noise or
motion artifact, which adds to the difficulty of recognizing it.

### 1.2 Why detection matters

These arrhythmias are a leading cause of sudden cardiac death (SCD). In the United States
about 325,000 people suffer an out-of-hospital cardiac arrest each year, and VF is the initial
rhythm in up to 75% of those cases [MODERN-2024]. SCD accounts for roughly half of all
cardiovascular deaths [MODERN-2024]. Survival depends on how fast treatment arrives: for every
minute without defibrillation the chance of survival falls by about 10% [MODERN-2024]. Rapid
and reliable automatic detection is therefore central to any device that has to decide whether
to treat.

Automated external defibrillators (AEDs) exist for this purpose. They recognize a shockable
rhythm and deliver a shock without needing a trained clinician [JEKOVA-2004, COMP5-2000]. The
detection has to be both highly sensitive, so it never misses a true event, and highly
specific, so it never shocks unnecessarily. Missing a VF episode is fatal, and an
inappropriate shock is harmful. That double requirement is what makes the detection problem
genuinely hard [COMP4-1993].

### 1.3 Why detection is hard

Each rhythm poses a different challenge. VT is organized enough for beat-by-beat analysis, but
it has to be told apart from fast normal rhythms, paced rhythms, and bundle branch blocks that
also produce wide, abnormal-looking QRS complexes. VFL and VF cannot be handled by counting
beats at all. A QRS detector works by emphasizing slope, amplitude, and temporal isolation, so
during VFL, when the ECG is a continuous narrow-band oscillation, it either misses peaks or
over-counts them, and during VF, when the signal is irregular and can look like noise, beat
detection stops being meaningful. Detecting VFL and VF is therefore a matter of characterizing
the signal segment as a whole: its dominant frequency, its regularity, its amplitude
distribution, and how concentrated its spectrum is. Several other conditions make this harder,
because muscle artifact, electrode noise, and atrial flutter or fibrillation share surface
features with VF and cause false positives [COMP4-1993]. Data scarcity adds to the problem,
since recordings that capture the onset and progression of VF are rare and the number of
labeled episodes is small.

### 1.4 Aim and scope

This paper measures which classical, deterministic signal-processing features best separate
the shockable ventricular tachyarrhythmias from other rhythms, then selects and tunes a single
detector built on the strongest feature. Five candidate detectors are compared on the same
databases under the same evaluation, and judged on both discrimination and computational cost.
The winning detector gets one further test: whether it can also separate flutter from
fibrillation, which have very different signatures.

The scope stays inside classical, deterministic methods on purpose. The downstream goal is to
extend an existing beat-detection and classification engine (exg-core) toward a regulatory
submission, and machine-learning and deep-learning detectors need a different validation and
approval path. Learned methods are reviewed in section 2 and named as future work, but they
are not evaluated here. The reference implementation of every feature is the vftx package, a
validated pure-Python reimplementation of the 27 features from Hong's thesis [HONG-2016].

## 2. Background and related work

Algorithms for VT, VFL, and VF detection span several decades and moved from simple counting
rules to spectral and phase-space analysis. The progression was shaped partly by the hardware
of early AEDs, which had little memory and no floating-point unit, so counting and threshold
operations were the only practical option at first. This review groups the methods by their
underlying principle rather than by date, following the feature categories Hong used to
organize his 27-feature set [HONG-2016]. Amann et al.'s comparative study anchors the review:
across the methods they tested, the best-performing features worked in the time domain, the
spectral parameters used energy distribution but not phase information, and the
complexity-based methods performed poorly wherever specificity had to stay above 80%
[COMP55-2005]. That finding guides which methods this paper carries forward as candidates.

### 2.1 Threshold-crossing measures

The simplest family asks how often the signal leaves the baseline. In normal sinus rhythm the
ECG rests near the isoelectric line and crosses a threshold briefly once per beat, while during
VF it is in constant motion and crosses continuously. The zero-crossing rate is the earliest
such feature, but it is sensitive to baseline drift [COMP4-1993]. Threshold crossing intervals
(TCI) use an adaptive threshold set at 20% of the local maximum and measure the mean time
between upward crossings, which falls sharply in VF [COMP4-1993, COMP5-2000]. Threshold
crossing sample count (TCSC) counts the fraction of samples above a normalized threshold over a
3-second cosine-tapered window: a normal ECG spends most of its time near baseline and gives a
low count, VF does not and gives a high one. On the full MIT-BIH and CU databases without
preselection, TCSC gave the best area under the ROC curve among the classical algorithms
compared in that study [TCSC-2009].

### 2.2 Amplitude distribution and shape

A second time-domain family looks at the distribution or shape of the amplitude rather than the
crossing rate. In sinus rhythm the amplitude is concentrated near zero with narrow peaks at the
QRS complexes; in VF it is spread more evenly across the range. The standard and modified
exponential algorithms (STE and MEA) count how the ECG intersects a decaying exponential
envelope, capturing that property without an explicit threshold. On standard databases MEA
reached an IROC of about 82%, better than STE's 67% [COMP55-2005].

### 2.3 Complexity and entropy

A third family asks how predictable the signal is. Sinus rhythm is quasi-periodic and
predictable, VF is chaotic. Sample entropy and approximate entropy measure the chance that a
short pattern that recurs also recurs one sample longer, and the complexity measure applies
Lempel-Ziv complexity to a thresholded binary version of the signal, scoring high for VF and
low for sinus rhythm [COMP5-2000, COMP55-2005]. These measures are sensitive to noise, and
Amann et al. found they performed poorly wherever specificity had to stay above 80%
[COMP55-2005]. Hong added a related measure for a different job: Lempel-Ziv complexity computed
on the intrinsic mode functions of an empirical mode decomposition, which separates VF from VT
better than the same measure on the raw signal [HONG-2016].

### 2.4 Spectral measures

VF produces a narrow-band signal roughly between 4 and 9 Hz, while organized rhythms carry
energy at the heart-rate fundamental and its harmonics. The VF-filter (VFLEAK) estimates the
mean signal period, adds each sample to the one half a period away, and measures the residual
leakage, which is near zero for a sine-like signal and near one for a wideband one
[COMP4-1993, COMP5-2000]. The SPEC algorithm windows the segment, takes its FFT, and extracts
descriptors of where the spectral energy sits, a normalized mean frequency together with
band-energy ratios, which tell a narrow-band VF spectrum from a harmonic-rich sinus spectrum
[SPEC-1989, COMP5-2000]. Both reach an IROC near 87–89% on standard databases, with very high
specificity but a sensitivity that depends on threshold tuning [COMP55-2005]. Jekova and
Krasteva's band-pass approach filters near 14.6 Hz, a band where normal QRS complexes carry
energy but VF does not, and reports about 96% sensitivity and 94% specificity on the combined
AHA and MIT databases [JEKOVA-2004].

### 2.5 Phase-space measures

A signal can be plotted against a time-delayed or Hilbert-transformed copy of itself, which
turns the recording into a trajectory in a plane. Periodic signals trace a tight, repeating
loop, and chaotic signals fill a large area. The Hilbert-transform method (HILB) overlays the
trajectory on a 40 by 40 grid and measures the fraction of boxes it visits, which is low for
sinus rhythm and high for VF, and which depends on the shape of the trajectory rather than its
amplitude. On combined databases with more than 330,000 decisions, HILB reached an IROC of
about 95%, the highest of the classical algorithms in that study [HILB-2005].

### 2.6 Learned methods, and why they are out of scope here

Recent work puts a classifier on top of several features at once. VFPred combines empirical
mode decomposition with a support vector machine and reports sensitivity near 99.99% and
specificity near 98.40% on the MIT-BIH and CU databases with 5-second windows [VFPRED-2018].
Hong's thesis aggregates all 27 features with a support vector machine and meets the AHA
performance goals, and notes that even a linear classifier on the same features performs well
[HONG-2016]. Convolutional and recurrent networks have reported sensitivity and specificity
above 95% on benchmark databases [MODERN-2024, DEEP-2023]. These methods are strong, but they
need large annotated training sets and a validation and regulatory path that differs from that
of a deterministic detector. This paper takes Hong's 27-feature set as its candidate pool and
its category structure, and treats the learned classifiers as future work rather than
evaluating them here.

## 3. Materials and methods

### 3.1 Databases

Four PhysioNet databases are used, read through a common reader at a uniform 250 Hz. For each
record the reader selects the best available channel, preferring lead II (MLII, ML2, or II)
and otherwise the nearest lead (V5, V2, and so on), rather than a fixed channel index.

The MIT-BIH Malignant Ventricular Arrhythmia Database (VFDB) holds 22 two-channel records of
about 35 minutes, from patients with sustained VT, VFL, and VF. Its rhythm annotations mark
the ventricular episodes this study targets [VFDB].

The Creighton University Ventricular Tachyarrhythmia Database (CUDB) holds 35 single-channel
records of 8 minutes, also from patients with sustained ventricular arrhythmias. Its episodes
are short, which is one reason for the 4-second window test [CUDB].

The MIT-BIH Arrhythmia Database (MITDB) holds 48 two-channel records of about 30 minutes,
spanning normal sinus rhythm and a range of non-shockable rhythms: atrial fibrillation and
flutter, bundle branch blocks, paced rhythms, and ectopic beats. It is the main source of
non-shockable diversity and carries few ventricular episodes. Its native 360 Hz is resampled
to 250 Hz [MITDB].

The American Heart Association Database (AHADB) holds 30-minute two-channel records and
requires a licence from ECRI. Its records carry beat annotations and VF markers but no
rhythm-episode labels, so only the 8-series ventricular records (8201 to 8210, the ones with VF
brackets) are used. In those records the bracketed spans are shockable and the surrounding
background is non-shockable. This is the same AHA ventricular material used by Jekova and
Krasteva [JEKOVA-2004, AHADB].

Together the four give rich shockable content (VFDB, CUDB, and the AHADB subset) against a
broad non-shockable background (MITDB), matching the combination used in the benchmark
literature [COMP55-2005].

### 3.2 Signal preprocessing

Preprocessing runs in two stages, once per record and once per window.

Per record, the signal is conditioned in a single pass: a moving-median baseline removal
(596 ms window) followed by a Lynn recursive band-pass (48 ms window). This matches the
real-time exg-core pipeline, where baseline removal and band-limiting run once as the signal
streams in, not per analysis window. Filtering the whole record once, rather than each
overlapping window, also avoids repeating the same work on the seven seconds that two adjacent
8-second windows share.

Per window, the integer signal is converted to millivolts by dividing by the standard gain of
200. It is then conditioned for the feature functions by the steps the feature math needs:
mean subtraction, min-max normalisation to unit peak-to-peak, order-5 moving-average
smoothing, and a re-centering to zero mean. The frequency filtering that the reference
pipeline applies (a 1 Hz high-pass and a 30 Hz low-pass [COMP55-2005]) is turned off here,
because the record is already band-limited from the per-record stage. The zero-mean
re-centering stands in for the high-pass, so features that assume an oscillation around zero
(the leakage, spectral, and phase-space measures) behave correctly.

All records are unified at 250 Hz (MITDB resampled from 360 Hz), so every feature works at one
sampling rate.

### 3.3 Segmentation and labeling

Each record is scanned with overlapping windows at a 1-second step, at two lengths: 8 seconds
(2000 samples), the length most of the benchmark algorithms were defined on, and 4 seconds
(1000 samples), which catches the short episodes CUDB and MITDB contain. Each window position
produces one row, written to a per-record tab-separated file.

A row holds identifiers (database, record, window start and end), episode-duration labels,
beat counts, and the feature columns. The episode labels record how many samples of the window
each annotated rhythm covers: normal sinus, bigeminy, trigeminy, VT, VFL, the bracketed VF
onset marker, VF, atrial flutter, atrial fibrillation, and an "other" bucket. The beat counts
tally the annotation marks in the window by type. Storing durations and counts, rather than one
collapsed label, keeps the raw evidence in the file, so the target definitions below can change
without rebuilding.

Each window carries two labels. The rhythm label names the dominant episode when one covers at
least 90% of the window, otherwise MIX; it gives clean, homogeneous examples and keeps VT, VFL,
and VF as distinct classes for the sub-analysis. The shockable label follows the benchmark
convention [COMP55-2005, JEKOVA-2004]: a window is shockable when shockable episodes (VT, VFL,
VF) cover at least 90% of it, non-shockable when no shockable episode is present, and MIX only
when a shockable episode partially straddles the window. An unannotated background window is
therefore non-shockable, which is how these algorithms are scored over a whole recording, and
MIX is limited to shockable-boundary transitions. Because VFL sits between VT and VF and the
databases annotate it separately, the three VFL configurations (VFL shockable, VFL
non-shockable, VFL excluded) are all available from the same file. The clean windows are used
for the feature analysis and tuning; evaluation uses all windows, since a deployed detector
cannot skip boundary windows.

Per-window decisions become reference episode annotations by majority voting. Each sample
belongs to several overlapping windows and takes the majority label across them; contiguous
runs of the same label collapse into episodes with start and end times. This smooths isolated
errors and produces the interval output the standard WFDB tools compare against. The window and
class counts from the build are reported in Results (4.1).

### 3.4 Feature set and candidate detectors

Each window is described by 16 signal-only features, computed by a validated pure-Python
reimplementation of the feature set from Hong's thesis [HONG-2016]. They fall into the same
categories as the review in section 2.

| Feature | Category | What it measures |
|---|---|---|
| TCSC | threshold crossing | fraction of samples whose amplitude exceeds a normalised threshold; high in VF |
| TCI | threshold crossing | mean interval between threshold crossings; short in VF |
| STE | amplitude shape | intersections of the signal with a standard decaying exponential envelope |
| MEA | amplitude shape | same idea with a modified envelope that separates VF better [COMP55-2005] |
| MAV | amplitude shape | mean absolute amplitude of the window |
| PSR | phase space | fraction of a grid filled by the time-delay phase-space trajectory |
| HILB | phase space | fraction of a grid filled by the Hilbert-transform phase-space trajectory [HILB-2005] |
| VF_LEAK | spectral | residual after half-period cancellation; low for a sine-like VF/VFL signal [VFLEAK-1978] |
| M | spectral | first-moment spectral parameter of the amplitude spectrum |
| A2 | spectral | energy in a band around the dominant frequency; high for narrow-band VF |
| FM | spectral | amplitude-weighted mean frequency of the spectrum |
| LZ | complexity | Lempel-Ziv complexity of the binarised signal; high for disordered VF |
| Count1 | band-pass | samples in the upper half of the 14.6 Hz band-pass output [JEKOVA-2004] |
| Count2 | band-pass | samples above the mean of the band-pass output [JEKOVA-2004] |
| Count3 | band-pass | samples within the mean plus or minus the mean deviation of the band-pass output [JEKOVA-2004] |
| Amplitude | amplitude | peak-to-peak amplitude in millivolts |

Sample entropy (how predictable the signal is) is also available but off by default, because at
about 55 ms/window it is the slowest of the kept features; it is computed only when a step
needs it.

Two feature families are excluded, and the reasons matter for the paper's claim to be a
signal-only, real-time detector. QRS-derived features (mean RR interval and beat-type ratios)
are dropped because a QRS detector must blank during VF and VFL: there is no QRS to detect, so
a VF/VFL detector that consumed QRS output would be circular and would fail exactly when it is
needed. The EMD-based IMF-LZ features are dropped because empirical mode decomposition costs
about 2 seconds per window, which cannot keep up with a real-time 1-second step. Both families
exist in the reference set but have no place in a signal-only real-time detector.

Five candidate detectors are compared, each one algorithm computed from one or a few of the
features above plus a threshold decision. Each maps onto a distinct method family.

| Candidate | Feature(s) | Family | Reference |
|---|---|---|---|
| TCSC | tcsc | threshold crossing | [TCSC-2009] |
| VFLEAK | vf_leak | spectral, leakage | [VFLEAK-1978] |
| SPEC | m, a2, fm | spectral, FFT descriptors | [SPEC-1989] |
| HILB | hilb | phase space | [HILB-2005] |
| JEKOVA | count1, count2, count3 | 14.6 Hz band-pass | [JEKOVA-2004] |

TCSC, VFLEAK, and SPEC come from the benchmark literature. HILB is added as the strongest
single classical algorithm [HILB-2005]. The fifth slot first held MEA (the amplitude-shape
family), but the feature screen (section 4.2) ranked MEA poorly and ranked JEKOVA's band-pass
counts at the top, so JEKOVA takes the slot: it is a strong published detector (about 96%
sensitivity and 94% specificity) that uses only integer arithmetic, which suits the real-time
target [JEKOVA-2004]. Complexity and entropy measures are left out of the candidate set because
they perform poorly wherever specificity must stay above 80% [HONG-2016, COMP55-2005]. Each
candidate's decision threshold is set and tuned in Phase 3 and Phase 4; the tuned values and
the per-window compute cost are reported in Results.

### 3.5 Screen and candidate shootout

The feature analysis runs in two passes on the clean windows (shockable versus non-shockable,
with the MIX transition windows dropped): a broad screen that ranks every feature on its own,
then a shootout that compares the five candidate detectors on discrimination and compute cost.
Working on the clean set keeps the ranking honest, since the boundary windows have no single
correct label. The screen scores each of the 16 signal-only features against the binary
shockable label by three complementary measures, chosen so that a feature has to look good
from three different angles (a linear one, an information-theoretic one, and a decision one) to
be trusted.

Point-biserial correlation is the Pearson correlation coefficient between a continuous feature
and the binary label, which is the ordinary product-moment correlation with the label coded as
0 and 1. It ranges over [-1, 1] and is signed, so it shows the direction of the effect: a
positive value means the feature rises for shockable rhythms, a negative value means it falls.
It is simple, familiar, and cheap to compute over the whole set, and its sign is what tells us
which way to orient a threshold. Its weakness is that it only sees linear, monotonic
association: a feature that separates the classes through a non-monotonic or non-linear
relationship can score near zero even when it is highly informative, so it cannot be the only
measure.

Mutual information measures how much knowing the feature reduces uncertainty about the label,
in the information-theoretic sense, and is zero only when feature and label are statistically
independent. Unlike correlation it makes no assumption of linearity or monotonicity, so it
captures any form of dependence, which is why it is included as a guard against the blind spot
of the correlation. It is estimated here with a nearest-neighbour method on a random subsample
of the windows, both to bound the cost of the density estimate and because that estimator does
not need the data binned in advance. It is non-negative and unbounded, so it is read
comparatively (higher means more informative) rather than against a fixed scale.

Single-feature AUC is the area under the receiver operating characteristic curve obtained when
the label is decided by thresholding that one feature and sweeping the threshold across its
whole range. It equals the probability that a randomly chosen shockable window scores higher
than a randomly chosen non-shockable one, so 0.5 is chance and 1.0 is perfect separation; a
feature that separates the classes in the opposite direction gives an AUC below 0.5, so the
value is oriented to max(AUC, 1 - AUC) and reported in [0.5, 1]. It is threshold-free, which
makes it the natural match to a single-threshold detector, and it is directly comparable to
the ROC-based figures the benchmark papers report [COMP55-2005, TCSC-2009]. Of the three it is
the closest proxy for how the feature will behave as an actual detector.

Other feature-selection scores were considered and not added, because they would not give a
fourth independent view. The Fisher score and the Kolmogorov-Smirnov statistic both rank
features by class separation in a way that single-feature AUC already captures monotonically,
and ReliefF is an instance-based, neighbour-weighted measure whose extra cost buys little here,
where each detector is ultimately a single global threshold rather than a local rule. A
feature-feature Pearson correlation matrix over the clean set is computed alongside the screen
to expose redundancy: features that carry the same information cluster into visibly correlated
blocks, which explains why several strong single features do not add up to independent evidence.

The shootout then treats each candidate as a deployable detector rather than a bare feature.
Every candidate is a single feature (its primary feature, the strongest of the group by
oriented AUC) plus one threshold, and its discrimination is reported three ways: the oriented
single-feature AUC and mutual information from the screen, plus the best F1 score reached at
any single threshold. F1, the harmonic mean of precision and recall, is used because the
classes are heavily imbalanced (far more non-shockable than shockable windows), a setting in
which plain accuracy is misleading; it is computed here as the maximum F1 over all thresholds
and both orientations, so it reads as the best operating point a single threshold on that
feature can reach. Full multi-threshold tuning of the multi-feature detectors is deferred to
Phase 4; in the shootout each is fairly represented by its best sub-feature.

Alongside discrimination each candidate carries a rough compute cost, reported as milliseconds
per 1000 windows for readability. It is timed through the same feature path the dataset build
uses (preprocess the window once, then compute the feature), and charged at the primary
feature, which carries the detector's one shared heavy step exactly once: JEKOVA's 14.6 Hz
band-pass, or SPEC's power spectrum. Summing the sub-features would redo that step and overstate
a detector that a real implementation computes once. These are pure-Python, single-thread
measurements on one machine, given only for relative scale between candidates, not as absolute
or portable timings; an embedded C implementation would be far faster, and only the ordering
between candidates carries over. The winner is argued from both axes together. Because the
downstream target is a real-time, embedded extension of exg-core, a cheaper candidate that
trails the best discrimination by a small margin can still be preferred, so the shootout reports
the two axes side by side rather than collapsing them into one score.

### 3.6 Candidate tuning and flutter-vs-fibrillation test

> Phase 4. Tune two candidates, not one: TCSC, the established classical design already
> confirmed in practice (find a reference), and JEKOVA, a newer but still simple, real-time
> friendly design that topped the screen. For each, sweep its decision threshold(s), build the
> ROC curve, and pick operating points at both window lengths and the three VFL configurations.
> JEKOVA needs a small grid search over its three count thresholds; TCSC a single-threshold ROC
> sweep as in COMP55-2005. Then the sub-analysis on the leading candidate: does its feature
> separate VFL from VF? VFL is fast, near-sinusoidal, regular; VF is disorganized; VT and VFL
> are usually short transient phases toward VF. Use cheap features for the split (spectral
> concentration and regularity, for example a2, vf_leak, the phase-space fill); Hong's IMF-LZ
> answer to VF-vs-VT is out, EMD being too slow for real time. Cite TCSC-2009, COMP55-2005,
> JEKOVA-2004, HONG-2016.

### 3.7 Evaluation metrics

> Phase 4. F1 (headline), Se, Sp, PPV, Acc, G-Mean. TP/FP/TN/FN as durations in ms. Define
> each briefly. Note ROC is used only for the winner tuning, not the whole shootout.

## 4. Results

### 4.1 Dataset composition

The build produced 167,783 windows at 8 seconds and 168,243 at 4 seconds across the four
databases, stepping by 1 second and keeping the AHADB 8-series only. Under the benchmark
labeling most windows are non-shockable, as expected from recordings that are mostly background
rhythm with embedded ventricular events: 148,689 non-shockable, 16,036 shockable, and 3,058 MIX
at 8 seconds. The shorter window straddles fewer episode boundaries, so at 4 seconds MIX falls
to 1,851 while the shockable count rises slightly to 16,483. The per-database split shows where
each class comes from: the shockable windows are supplied by VFDB, CUDB, and the AHADB subset,
while MITDB contributes almost none but dominates the non-shockable background.

| Database | NON (8 s) | SHOCK (8 s) | MIX (8 s) | NON (4 s) | SHOCK (4 s) | MIX (4 s) |
|---|---|---|---|---|---|---|
| VFDB | 37,417 | 6,835 | 1,772 | 37,889 | 7,096 | 1,127 |
| CUDB | 13,432 | 3,513 | 590 | 13,704 | 3,664 | 307 |
| AHADB (8-series) | 12,371 | 5,447 | 102 | 12,427 | 5,480 | 53 |
| MITDB | 85,469 | 241 | 594 | 85,889 | 243 | 364 |
| All | 148,689 | 16,036 | 3,058 | 149,909 | 16,483 | 1,851 |

The separate rhythm label, which keeps the ventricular classes distinct for the
flutter-versus-fibrillation analysis, shows how uneven the shockable material is inside itself.
At 8 seconds the clean ventricular windows split into 10,569 VF, 4,937 VT, and only 474 VFL,
against 77,932 normal-sinus and 41,313 other non-shockable windows (the remaining windows are
MIX). Flutter is by far the rarest of the three, which is expected given how short-lived VFL is,
and it sets the main limit on the VFL-versus-VF sub-analysis in section 4.4. The three VFL
configurations (flutter shockable, non-shockable, or excluded) are all derived from this same
labeling without rebuilding, since the file stores episode durations rather than one collapsed
label.

### 4.2 Feature screen and candidate shootout

The screen was run on the 8-second clean set (16,036 shockable, 148,689 non-shockable). The
16 features rank as follows, by oriented single-feature AUC, with point-biserial correlation
and mutual information alongside.

| Feature | Point-biserial | AUC | Mutual information |
|---|---|---|---|
| Count3 | -0.722 | 0.986 | 0.237 |
| TCSC | 0.623 | 0.964 | 0.185 |
| MAV | 0.643 | 0.964 | 0.189 |
| Count1 | 0.714 | 0.957 | 0.203 |
| HILB | 0.614 | 0.954 | 0.183 |
| PSR | 0.616 | 0.944 | 0.167 |
| A2 | 0.656 | 0.940 | 0.172 |
| VF_LEAK | -0.605 | 0.924 | 0.154 |
| M | -0.331 | 0.923 | 0.145 |
| TCI | -0.229 | 0.900 | 0.115 |
| LZ | 0.336 | 0.797 | 0.063 |
| FM | -0.276 | 0.781 | 0.051 |
| MEA | 0.292 | 0.778 | 0.049 |
| STE | 0.286 | 0.732 | 0.040 |
| Count2 | -0.048 | 0.573 | 0.012 |
| Amplitude | 0.142 | 0.538 | 0.066 |

The three measures agree on the overall ordering. The band-pass count Count3 leads on all three
(AUC 0.986), and the top of the table is filled by the threshold-crossing, band-pass, amplitude,
and phase-space families, with the spectral features close behind. The complexity measure LZ,
the amplitude-shape measures MEA and STE, and the raw peak-to-peak Amplitude sit at the bottom,
and Count2 is near chance on its own. This confirms the two decisions the candidate set rests
on: the band-pass counts are the strongest single features, which is why JEKOVA replaced MEA in
the fifth slot, and the complexity and amplitude-shape measures discriminate poorly, matching
Amann et al.'s finding that they fail wherever specificity must stay high [COMP55-2005]. The
feature-feature correlation heatmap (Figure, from PAPER.ipynb) shows the strong features are not
independent: the threshold-crossing, band-pass, and phase-space measures form a correlated
block, so they largely re-measure the same underlying property (how much of the window departs
from baseline) rather than adding separate evidence.

The shootout compares the five candidates as detectors. At 8 seconds:

| Detector | Primary feature | AUC | Mutual information | Best F1 | Cost (ms/1000 win) |
|---|---|---|---|---|---|
| JEKOVA | count3 | 0.986 | 0.236 | 0.843 | 693 |
| TCSC | tcsc | 0.964 | 0.186 | 0.707 | 47 |
| HILB | hilb | 0.954 | 0.182 | 0.723 | 55 |
| SPEC | a2 | 0.940 | 0.172 | 0.719 | 55 |
| VFLEAK | vf_leak | 0.924 | 0.154 | 0.692 | 55 |

JEKOVA leads every discrimination column by a clear margin, most visibly on best F1 (0.843
against 0.71 or below for the rest), so the hypothesis that TCSC would top the shootout does not
hold: TCSC is strong and comes second on AUC, but the band-pass detector is better. The cost
column tells the other half of the story. TCSC is the cheapest by an order of magnitude (47 ms
per 1000 windows against JEKOVA's 693), because it is a single normalised-threshold count over
the window, whereas JEKOVA has to run the sample-by-sample recursive band-pass filter first.
The three FFT and leakage detectors sit together near 55 ms. On the discrimination-versus-cost
scatter (Figure, from PAPER.ipynb) JEKOVA sits at the top right (best, most expensive) and TCSC
at the far left (cheapest, second-best), with the others clustered between them, which is
exactly the tradeoff that makes the winner an argued choice rather than a lookup of the top
score.

The 4-second rerun holds the picture. JEKOVA still leads (AUC 0.982, best F1 0.823), the
ordering of the rest barely moves (HILB and TCSC swap by a hair on AUC), and every cost drops
with the shorter window (JEKOVA 350, TCSC 29 ms per 1000 windows), so the ranking is not an
artifact of the 8-second length.

| Detector | AUC (8 s) | F1 (8 s) | Cost (8 s) | AUC (4 s) | F1 (4 s) | Cost (4 s) |
|---|---|---|---|---|---|---|
| JEKOVA | 0.986 | 0.843 | 693 | 0.982 | 0.823 | 350 |
| TCSC | 0.964 | 0.707 | 47 | 0.955 | 0.681 | 29 |
| HILB | 0.954 | 0.723 | 55 | 0.960 | 0.731 | 47 |
| SPEC | 0.940 | 0.719 | 55 | 0.938 | 0.707 | 39 |
| VFLEAK | 0.924 | 0.692 | 55 | 0.926 | 0.680 | 38 |

Because the two axes point at different candidates (JEKOVA on discrimination, TCSC on cost), both
are carried into Phase 4 and tuned in full, and the choice between a single detector and a
combination of the two is left to that stage and to future work.

### 4.3 Tuned candidates

> Phase 4 output. For each tuned candidate (TCSC and JEKOVA): a ROC figure and an
> operating-point table (F1/Se/Sp/PPV/Acc/G-Mean) under the three VFL configurations and both
> window lengths, with the grid-searched thresholds (JEKOVA's three counts, TCSC's single
> threshold, per COMP55-2005) and a comparison to each algorithm's published numbers. The final
> choice of a single detector, or an ensemble of them, is left to future work, as it goes beyond
> the scope of this paper.

### 4.4 Flutter vs fibrillation, for the winner

> Phase 4 output. Whether the winning feature separates VFL from VF, and how well. Table or
> figure. Tie back to their different signatures (VFL near-sinusoidal and regular, VF
> disorganized) and their transient nature.

## 5. Discussion

> Phase 5. What the shootout says about which signal properties carry the discrimination, and
> why the winner wins once cost is counted. How the tuned detector compares to the literature
> and what its limits are. What the VFL-vs-VF result implies. Threats to validity (database
> imbalance, annotation quality, window choice). Keep claims tied to the numbers.

## 6. Conclusion and future work

> Phase 5. Short. The selected feature and its tuned performance. The path forward: the
> deterministic detector as a component to extend exg-core toward an FDA submission, and the
> learned methods (EMD + SVM, CNN) as the future-work direction that needs a separate
> validation and regulatory track.

## References

> Phase 5. Merge the DRAFT.md reference list unchanged, plus the addition below. Keep the
> DRAFT grouping (Databases, Comparative studies, Algorithm papers).

[HONG-2016] Hong, Jen-Yee. Detecting Life-Threatening Arrhythmia with Machine Learning
Algorithms. Master Thesis, Department of Computer Science and Information Engineering,
National Taiwan University, July 2016.
