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

> Phase 2. VFDB, CUDB, AHADB, MITDB: what each contributes, sampling rates, why the
> combination. Pull from DRAFT.md "Databases". Note the AHADB licence and the record
> selection settled in Phase 2. Cite MITDB, CUDB, VFDB, AHADB.

### 3.2 Signal preprocessing

> Phase 2. The pipeline as implemented in vftx: ADC to mV, mean subtraction, normalization,
> moving-average smoothing, 1 Hz high-pass drift suppression, 30 Hz Butterworth low-pass.
> Unify at 250 Hz (resample MITDB). Cite COMP55-2005.

### 3.3 Segmentation and labeling

> Phase 2. Overlapping windows with a 1 s step, at two window lengths: 8 s for the benchmark
> (matches the windows most of these algorithms were defined on) and 4 s for the short-episode
> test (MITDB carries brief episodes). Rhythm labels from the annotations; derive the shockable
> class and the VT/VFL/VF sub-labels. Majority-vote smoothing turns per-window decisions into
> reference episode labels. The three VFL configurations. Endpoint vs coverage labeling for
> transition windows.

### 3.4 Feature set and candidate detectors

> Phase 3. The 27 vftx features, grouped by category as in section 2, with a compact table
> (name, index, what it measures); point to vftx as the validated reference implementation.
> Then the five candidate detectors (TCSC, VFLEAK, SPEC, plus HILB and MEA), each a feature
> plus a decision rule, with a note on each one's compute cost. Say why these five: Amann's
> finding (via Hong) that time-domain features perform best motivates MEA, HILB is the
> strongest classical single algorithm, and complexity/entropy is left out because it performs
> poorly above 80% specificity. Cite TCSC-2009, VFLEAK-1978, SPEC-1989, HILB-2005, COMP55-2005,
> HONG-2016.

### 3.5 Screen and candidate shootout

> Phase 3. Two passes. The broad screen: score each of the 27 features against the shockable
> label with point-biserial correlation, mutual information, and single-feature AUC; this
> confirms the two added candidates and flags redundancy (feature-feature correlation). The
> shootout: for each of the five candidates report discrimination (same scores plus F1 at a
> swept threshold) and a rough compute cost per window, at 8 s, with the leaders rerun at 4 s.
> State how the winner is chosen: discrimination weighed against cost, not the top score alone.

### 3.6 Winner tuning and flutter-vs-fibrillation test

> Phase 4. Take the winning candidate (hypothesis: TCSC). Threshold sweep, ROC construction,
> operating-point selection at both window lengths and the three VFL configurations. Then the
> winner-only sub-analysis: does the winning feature also separate VFL from VF, which have very
> different signatures? Note that VT and VFL are usually short transient phases toward VF. If
> the winning feature cannot, fall back to the IMF-LZ features (LZ on EMD modes), which Hong
> added for exactly the VF-vs-VT distinction after Xia et al. 2014. Cite TCSC-2009,
> COMP55-2005, HONG-2016.

### 3.7 Evaluation metrics

> Phase 4. F1 (headline), Se, Sp, PPV, Acc, G-Mean. TP/FP/TN/FN as durations in ms. Define
> each briefly. Note ROC is used only for the winner tuning, not the whole shootout.

## 4. Results

### 4.1 Dataset composition

> Phase 2 output. Table: segment counts per database, per class, and per window length (8 s
> and 4 s), per VFL configuration.

### 4.2 Feature screen and candidate shootout

> Phase 3 output. The 27-feature screen (ranking table: correlation, mutual information, AUC),
> then the five-candidate shootout with discrimination and compute cost side by side. State
> whether TCSC leads as hypothesized, and which candidate wins on the discrimination-vs-cost
> tradeoff. Supporting figures: mutual-information bar chart, feature-feature correlation
> heatmap, discrimination-vs-cost scatter.

### 4.3 Tuned winner

> Phase 4 output. ROC figure for the winning feature; operating-point table with F1/Se/Sp/
> PPV/Acc/G-Mean under the three VFL configurations and both window lengths; comparison to the
> published numbers for that algorithm.

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
