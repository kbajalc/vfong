# Detection of Ventricular Tachyarrhythmias

## Abstract

Life-threatening ventricular tachyarrhythmias (ventricular tachycardia, flutter, and
fibrillation) are a leading cause of sudden cardiac death, and detecting them from short ECG
segments quickly and reliably is central to any monitoring or treatment device that has to act
on them. This paper measures which classical, deterministic signal-processing features best
separate shockable from non-shockable rhythms, then tunes the two strongest candidates into
detectors. Using a validated pure-Python reimplementation of 27 features from Hong's thesis, we
build a labeled sliding-window dataset from four PhysioNet databases (VFDB,
CUDB, AHADB, MITDB) at 8-second and 4-second windows, screen 16 signal-only features against the
shockable label, and run a five-candidate shootout (TCSC, VFLEAK, SPEC, HILB, JEKOVA) judged on
both discrimination and compute cost. The 14.6 Hz band-pass counts of the JEKOVA algorithm lead
the screen (single-feature AUC 0.986) and the shootout; tuning its published decision cascade
reaches F1 0.847 (sensitivity 0.898, specificity 0.976), and the reproduced published cascade
gives sensitivity 0.973 and specificity 0.900, close to the original report. TCSC is the cheapest
candidate, about fourteen times faster, and second on discrimination (F1 0.706), so both are
carried forward as an accuracy-first and a cost-first option. The winning feature does not
separate flutter from fibrillation (AUC 0.603), which needs a regularity feature. The study stays
deterministic on purpose: it is the feature-selection step for extending a beat-detection engine
toward a regulatory submission, so learned classifiers are reviewed and named as future
work but not evaluated.

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

Reproducibility is a difficulty of its own. These algorithms are almost never published with a
reference implementation, only with a prose description and, at best, a formula, so their exact
behaviour has to be reconstructed and re-validated before it can be compared fairly against
others or reused. Rebuilding a trusted, validated implementation of the feature set is therefore
a precondition for the comparison this paper makes.

### 1.4 Aim and scope

This paper measures which classical, deterministic signal-processing features best separate
the shockable ventricular tachyarrhythmias from other rhythms, then tunes the two strongest
candidates into detectors. Five candidate detectors are compared on the same databases under the
same evaluation, and judged on both discrimination and computational cost. The two that lead on
those axes are then tuned in full, and the leading detector gets one further test: whether it can
also separate flutter from fibrillation, which have very different signatures.

The scope stays inside classical, deterministic methods on purpose. The downstream goal is to
extend an existing beat-detection and classification engine toward a regulatory
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
ECG rests near the isoelectric line and departs from it only briefly at each beat, when the QRS
complex crosses the threshold twice, once as it rises and once as it falls, while during VF the
signal is in constant motion and crosses continuously. The zero-crossing rate is the earliest
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
real-time pipeline, where baseline removal and band-limiting run once as the signal
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
| jc1 | band-pass | samples in the upper half [0.5·max, max] of the absolute 14.6 Hz band-pass output [JEKOVA-2004] |
| jc2 | band-pass | samples above the mean of the absolute band-pass output [JEKOVA-2004] |
| jc3 | band-pass | samples within the mean plus or minus the mean deviation of the absolute band-pass output; low in VF [JEKOVA-2004] |
| Amplitude | amplitude | peak-to-peak amplitude in millivolts |

The three band-pass counts (jc1, jc2, jc3) are computed on the absolute filter output, as
Jekova and Krasteva specify [JEKOVA-2004]. The vftx reference set also computes them on the
signed filter output (count1, count2, count3), and those columns are kept in the data, but the
signed second count is degenerate (about half the samples sit above a near-zero mean for every
rhythm), so the JEKOVA candidate, and the feature screen, use the absolute-output counts. Sample
entropy (how predictable the signal is) is also available but off by default, because at about
55 ms/window it is the slowest of the kept features; it is computed only when a step needs it.

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
| JEKOVA | jc1, jc2, jc3 | 14.6 Hz band-pass | [JEKOVA-2004] |

TCSC, VFLEAK, and SPEC come from the benchmark literature. HILB is added to bring in the
phase-space family, which the first three do not cover; it is a natural choice for that slot
because it was the strongest single classical algorithm in Amann et al.'s comparative study
(IROC about 95%) [HILB-2005, COMP55-2005], though on our data it lands mid-pack. The fifth slot
first held MEA (the amplitude-shape family), but the feature screen (section 4.2) ranked MEA
poorly and ranked JEKOVA's band-pass counts at the top, so JEKOVA takes the slot: it is a strong
published detector (about 96% sensitivity and 94% specificity) that uses only integer arithmetic,
which suits the real-time target [JEKOVA-2004]. Complexity and entropy measures are left out of the candidate set because
they perform poorly wherever specificity must stay above 80% [HONG-2016, COMP55-2005]. Each
candidate's decision threshold is set and tuned in Phase 3 and Phase 4; the tuned values and
the per-window compute cost are reported in Results.

### 3.5 Feature screen

The feature analysis runs in two passes: a broad screen that ranks every feature on its own,
then the shootout of section 3.6 that compares the candidate detectors. The screen works on the
clean windows (shockable versus non-shockable, with the MIX transition windows dropped), since
that keeps the ranking honest, the boundary windows having no single correct label. It scores
each of the 16 signal-only features against the binary shockable label by three complementary
measures, chosen so that a feature has to look good from three different angles (a linear one, an
information-theoretic one, and a decision one) to be trusted.

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

### 3.6 Candidate shootout

The shootout treats each candidate as a deployable detector rather than a bare feature.
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
downstream target is a real-time, embedded extension of existing detector and classifier, 
a cheaper candidate that trails the best discrimination by a small margin can still be preferred, 
so the shootout reports the two axes side by side rather than collapsing them into one score.

### 3.7 Candidate tuning and flutter-vs-fibrillation test

The two candidates that lead on the two axes of the shootout are tuned in full: TCSC, the
cheapest detector and the established threshold-crossing design [TCSC-2009], and JEKOVA, the
strongest on discrimination and still real-time friendly [JEKOVA-2004]. Both are tuned on the
clean set, separately for each window length (the feature distributions shift with window
length, so a threshold tuned at 8 seconds may not be optimal at 4 seconds) and under each of the
three VFL configurations.

TCSC is a single feature with a single threshold. The threshold is swept across the whole range
of the feature, and the operating point that maximises F1 is reported, together with the full
ROC curve, in the manner of the COMP55-2005 single-threshold analysis. This is the same sweep
that produced the shootout AUC, now read at a chosen operating point rather than as an area.

JEKOVA is not a single threshold but the cascade of count rules from the original paper
[JEKOVA-2004]: two rules that declare a segment non-shockable, two that declare it shockable,
and a combined term jc1·jc2/jc3, with anything unmatched left "not classified" for a later
wave-detection stage. Two adjustments make the published cascade transfer to this study, both
recorded so the tuning stays honest. First, the counts use the absolute band-pass output
(section 3.4), not vftx's signed counts, because the signed second count is degenerate and the
cascade relies on exactly that count's spread. Second, the window length: the paper's constants
are raw sample counts over a 10-second epoch, so they do not transfer to an 8 or 4-second window;
the counts are normalised to fractions of the window sample count, which are comparable across
window lengths, and the published constants are expressed on the same fractional scale. The
cascade thresholds are then grid-searched for best F1, with the published-constant cascade
reported as a baseline. The "not classified" branch cannot use the paper's wave detection without
a peak detector, which a signal-only detector avoids by design (section 3.4), so those windows
fall back to a threshold on jc3, which is low for shockable rhythms; that fallback is one of the
grid-searched parameters.

The winning detector then gets one further test: does a cheap feature separate flutter from
fibrillation? The two are physiologically distinct, VFL being a fast, regular, near-sinusoidal
oscillation and VF being disorganised, so spectral-concentration and regularity measures should
carry the split. Using the rhythm label, the VFL and VF windows are scored by the oriented AUC
of each candidate feature (spectral concentration a2, leakage vf_leak, the phase-space fills
psr and hilb, and the band-pass count jc3). Hong's answer to the related VF-versus-VT question was
Lempel-Ziv complexity on the empirical-mode-decomposition modes [HONG-2016], but that is out of
scope here because EMD is too slow for the real-time target (section 3.4). Flutter is rare in
these databases (its window count is reported in section 4.1), so this result is indicative
rather than definitive.

### 3.8 Evaluation metrics

Each detector produces a binary decision per window, which is compared against the window's
shockable label to give the four confusion counts: true positives (TP, shockable windows
correctly flagged), false positives (FP, non-shockable windows wrongly flagged), true negatives
(TN), and false negatives (FN, missed shockable windows). All the metrics below are functions of
these four counts, and all are reported at the tuned operating point.

Sensitivity (Se = TP / (TP + FN)) is the fraction of shockable windows caught; for an AED it is
the safety-critical metric, since a false negative is a missed shock. Specificity
(Sp = TN / (TN + FP)) is the fraction of non-shockable windows correctly passed over; a low
specificity means inappropriate shocks. Positive predictive value (PPV = TP / (TP + FP)) is the
fraction of flagged windows that were truly shockable, which matters here because the classes
are imbalanced (far more non-shockable windows), so even a high specificity can still leave many
false positives per true positive. Accuracy (Acc = (TP + TN) / total) is the overall fraction
correct, reported for completeness but weak under class imbalance, where predicting the majority
class alone already scores high.

The headline metric is F1, the harmonic mean of PPV and sensitivity
(F1 = 2·PPV·Se / (PPV + Se)). It rewards a detector only when both are high, and unlike accuracy
it is not inflated by the large non-shockable majority, which makes it the right single number
for tuning on this imbalanced problem. The G-Mean, the geometric mean of sensitivity and
specificity (sqrt(Se·Sp)), is reported alongside as a balance measure that, unlike F1, is
symmetric in the two classes and so penalises sacrificing either one. The ROC curve (sensitivity
against 1 minus specificity as the threshold sweeps) is used only for the winner tuning in this
section, not for the whole shootout, where a single AUC per candidate already summarised the
threshold-free separation.

## 4. Results

### 4.1 Dataset composition

The build produced 167,783 windows at 8 seconds and 168,243 at 4 seconds across the four
databases, stepping by 1 second and keeping the AHADB 8-series only. Under the benchmark
labeling most windows are non-shockable, as expected from recordings that are mostly background
rhythm with embedded ventricular events: 148,689 non-shockable, 16,036 shockable, and 3,058 MIX
at 8 seconds. The shorter window straddles fewer episode boundaries, so at 4 seconds MIX falls
to 1,851 while the shockable count rises slightly to 16,483. The per-database split shows where
each class comes from: the shockable windows are supplied by VFDB, CUDB, and the AHADB subset,
while MITDB contributes little but dominates the non-shockable background.

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
and it sets the main limit on the flutter-versus-fibrillation sub-analysis in section 4.5. The
three VFL configurations (flutter shockable, non-shockable, or excluded) are all derived from
this same labeling without rebuilding, since the file stores episode durations rather than one
collapsed label.

### 4.2 Feature screen

The screen was run on the 8-second clean set (16,036 shockable, 148,689 non-shockable). The
16 features rank as follows, by oriented single-feature AUC, with point-biserial correlation
and mutual information alongside.

| Feature | Point-biserial | AUC | Mutual information |
|---|---|---|---|
| jc2 | 0.722 | 0.986 | 0.237 |
| jc3 | -0.710 | 0.985 | 0.229 |
| jc1 | 0.784 | 0.974 | 0.222 |
| TCSC | 0.623 | 0.964 | 0.185 |
| MAV | 0.643 | 0.964 | 0.189 |
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
| Amplitude | 0.142 | 0.538 | 0.066 |

The three measures agree on the overall ordering. The three band-pass counts lead on all three
metrics (AUC 0.986, 0.985, 0.974 for jc2, jc3, jc1), and the rest of the top is filled by the
threshold-crossing, amplitude, and phase-space families, with the spectral features close behind.
The complexity measure LZ, the amplitude-shape measures MEA and STE, and the raw peak-to-peak
Amplitude sit at the bottom. This confirms the two decisions the candidate set rests on: the
band-pass counts are the strongest single features, which is why JEKOVA replaced MEA in the fifth
slot, and the complexity and amplitude-shape measures discriminate poorly, matching Amann et
al.'s finding that they fail wherever specificity must stay high [COMP55-2005]. The
feature-feature correlation heatmap (Figure, from PAPER.ipynb) shows the strong features are not
independent: the threshold-crossing, band-pass, and phase-space measures form a correlated
block, so they largely re-measure the same underlying property (how much of the window departs
from baseline) rather than adding separate evidence.

### 4.3 Candidate shootout

The shootout compares the five candidates as detectors. At 8 seconds:

| Detector | Primary feature | AUC | Mutual information | Best F1 | Cost (ms/1000 win) |
|---|---|---|---|---|---|
| JEKOVA | jc2 | 0.986 | 0.237 | 0.846 | 1318 |
| TCSC | tcsc | 0.964 | 0.186 | 0.707 | 94 |
| HILB | hilb | 0.954 | 0.182 | 0.723 | 109 |
| SPEC | a2 | 0.940 | 0.172 | 0.719 | 114 |
| VFLEAK | vf_leak | 0.924 | 0.154 | 0.692 | 111 |

JEKOVA leads every discrimination column by a clear margin, most visibly on best F1 (0.846
against 0.72 or below for the rest), so the hypothesis that TCSC would top the shootout does not
hold: TCSC is strong and comes second on AUC, but the band-pass detector is better. The cost
column tells the other half of the story. TCSC is the cheapest by more than an order of magnitude
(94 ms per 1000 windows against JEKOVA's 1318), because it is a single normalised-threshold count
over the window, whereas JEKOVA has to run the sample-by-sample recursive band-pass filter first.
The three FFT and leakage detectors sit together near 110 ms. On the discrimination-versus-cost
scatter (Figure, from PAPER.ipynb) JEKOVA sits at the top right (best, most expensive) and TCSC
at the far left (cheapest, second-best), with the others clustered between them, which is
exactly the tradeoff that makes the winner an argued choice rather than a lookup of the top
score. The cost numbers are pure-Python timings on one machine and shift with load, so only their
ratios carry meaning: JEKOVA costs about fourteen times TCSC at both window lengths.

The 4-second rerun holds the picture. JEKOVA still leads (AUC 0.982, best F1 0.826), the
ordering of the rest barely moves (HILB and TCSC swap by a hair on AUC), and every cost roughly
halves with the shorter window, so the ranking is not an artifact of the 8-second length.

| Detector | AUC (8 s) | F1 (8 s) | Cost (8 s) | AUC (4 s) | F1 (4 s) | Cost (4 s) |
|---|---|---|---|---|---|---|
| JEKOVA | 0.986 | 0.846 | 1318 | 0.982 | 0.826 | 660 |
| TCSC | 0.964 | 0.707 | 94 | 0.955 | 0.681 | 55 |
| HILB | 0.954 | 0.723 | 109 | 0.960 | 0.731 | 91 |
| SPEC | 0.940 | 0.719 | 114 | 0.938 | 0.707 | 77 |
| VFLEAK | 0.924 | 0.692 | 111 | 0.926 | 0.680 | 74 |

Because the two axes point at different candidates (JEKOVA on discrimination, TCSC on cost), both
are carried into Phase 4 and tuned in full, and the choice between a single detector and a
combination of the two is left to that stage and to future work.

### 4.4 Tuned candidates

Both candidates were tuned on the clean set at each window length and under the three VFL
configurations. The operating points below are for the shockable configuration (flutter counts
as shockable); the other two configurations move every number by less than a percentage point,
because flutter is rare, so only the shockable configuration is tabulated here.

| Window | Detector | Se | Sp | PPV | F1 | Acc | G-Mean |
|---|---|---|---|---|---|---|---|
| 8 s | TCSC (tuned) | 0.827 | 0.945 | 0.617 | 0.706 | 0.933 | 0.884 |
| 8 s | JEKOVA (published) | 0.973 | 0.900 | 0.511 | 0.670 | 0.907 | 0.936 |
| 8 s | JEKOVA (tuned) | 0.898 | 0.976 | 0.802 | 0.847 | 0.969 | 0.936 |
| 4 s | TCSC (tuned) | 0.778 | 0.944 | 0.605 | 0.681 | 0.928 | 0.857 |
| 4 s | JEKOVA (published) | 0.962 | 0.893 | 0.497 | 0.656 | 0.900 | 0.927 |
| 4 s | JEKOVA (tuned) | 0.875 | 0.973 | 0.783 | 0.826 | 0.964 | 0.923 |

Two things stand out. First, the published JEKOVA cascade, reproduced on our absolute-output
counts with its constants only rescaled to the window length, already reaches Se 0.973 and
Sp 0.900 at 8 seconds. That is close to the Se 0.959 and Sp 0.944 the original paper reports on
the AHA and MIT databases [JEKOVA-2004], which is a strong independent check that the cascade and
the absolute counts were reproduced correctly, given that our counts follow Hong's band-pass and
our evaluation scores every window of every recording rather than curated 10-second episodes. The
small specificity gap (our 0.900 against the paper's 0.944) is the expected cost of scoring the
full continuous recording, including the noisy MITDB background, without the paper's separate
noise and asystole gates.

Second, tuning moves JEKOVA along its operating curve: the grid search trades a little
sensitivity (0.973 to 0.898) for a large gain in specificity (0.900 to 0.976) and precision
(0.511 to 0.802), which lifts F1 from 0.670 to 0.847, the best of any detector at either window
length. The tuned thresholds are the same at 8 and 4 seconds (the fraction normalisation makes
them transfer), which is a useful robustness property for deployment. TCSC tuned to its best-F1
threshold reaches F1 0.706 at 8 seconds, below tuned JEKOVA on every metric except that both keep
specificity high; its strength is elsewhere, in cost (section 4.3). The ROC figure (from
PAPER.ipynb) shows the TCSC sweep as a curve with the two JEKOVA cascade points marked: the tuned
point sits up and to the left of the published one, and both JEKOVA points sit above the TCSC
curve, so at matched specificity JEKOVA reaches higher sensitivity. All numbers drop by one to
three points at 4 seconds, as expected from the shorter evidence window, without changing the
ordering.

For reproducibility, the concrete thresholds are these. TCSC flags a window shockable when its
sample-count feature exceeds 45.8 at 8 seconds, and 47.1 at 4 seconds. JEKOVA's cascade is
expressed as fractions of the window sample count, so one set of thresholds applies to both
window lengths. The published constants, Jekova's 10-second raw counts of 250 and 400 for the
Count1 gates, 600, 950, and 1100 for the Count2 gates, and 210 for the Count1·Count2/Count3 term,
correspond after division by the 2500-sample epoch to fractions of 0.10 and 0.16, 0.24, 0.38, and
0.44, and 0.084. The grid search moved these to 0.16 and 0.20 for the Count1 gates, 0.24, 0.30,
and 0.35 for the Count2 gates, and 0.08 for the ratio term, with the unclassified-branch fallback
flagging shockable when the Count3 fraction is at or below 0.55. In effect the tuned cascade
raises the lower Count1 and Count2 gates and lowers the shockable Count2 gate, which is what
trades sensitivity for the specificity and precision gain above.

Reading the two axes together (section 4.3 and this table), JEKOVA is the accuracy choice and
TCSC the cost choice. Which one, or which combination, a deployment should carry is left to
future work, since it depends on the embedded compute budget and the required sensitivity floor,
both outside the scope of this feature study.

### 4.5 Flutter vs fibrillation

The winning detector, JEKOVA, is built on the band-pass counts, which barely separate flutter
from fibrillation: on the VFL and VF windows the jc3 fraction gives an oriented AUC of only 0.603
at 8 seconds, close to chance. This is expected, since the count measures how much 14.6 Hz band
energy is absent, which both flutter and fibrillation share, so it cannot tell the two apart. 
Cheap spectral-concentration and regularity features do better, though only modestly.
Scoring VF against VFL by oriented AUC on the flutter and fibrillation windows (section 4.1) at
8 seconds, the threshold-crossing count TCSC leads at 0.735, followed by the phase-space fill PSR
at 0.722 and the leakage measure VF_LEAK at 0.706, with the spectral concentration A2 at 0.682;
the 4-second window gives the same ordering (TCSC 0.740, VF_LEAK 0.706, PSR 0.697). The
distribution figure (from PAPER.ipynb) shows the separation is real but with heavy overlap, which
matches the physiology: flutter is the fast, regular, near-sinusoidal precursor and fibrillation
the disorganised end state, but the two form a continuum and flutter often degrades into
fibrillation within the same episode. As noted in section 4.1, flutter is rare in these
databases, so this is an indicative result rather than a tuned flutter-versus-fibrillation
detector. The practical reading
is that a deployed detector would need a second, regularity-oriented feature (TCSC or the
phase-space fill) on top of the band-pass count to attempt the flutter split, and even then only
partially.

## 5. Discussion

### 5.1 Which signal property carries the discrimination

The screen and shootout point to a single property above all others: how much of the window's
energy sits outside the narrow band where a normal ECG puts its sharp features. The three
band-pass counts top the screen (AUC 0.986, 0.985, 0.974), ahead of the threshold-crossing
count TCSC (0.964), the mean absolute amplitude MAV (0.964), and the phase-space fill HILB
(0.954). These leading features are, at bottom, measuring the same thing from different angles:
a normal ECG rests near baseline for most of the window with brief sharp excursions at the QRS
complexes, whereas a shockable rhythm is in continuous motion. The correlation heatmap confirms
it, since the threshold-crossing, band-pass, and phase-space features form one correlated block,
so combining them adds little independent evidence. The spectral leakage and FFT descriptors
(VF_LEAK 0.924, A2, M) trail slightly and measure a related property, spectral concentration.
The features that measure something genuinely different, signal complexity (LZ 0.797) and
amplitude shape (MEA 0.778, STE 0.732), discriminate worst, which matches Amann et al.'s finding
that complexity measures fail wherever specificity must stay high [COMP55-2005]. The useful
signal for the shockable decision is therefore concentrated on one axis, and the candidate that
reads that axis most cleanly wins.

### 5.2 Discrimination against cost

The hypothesis going in was that TCSC, the established cheap design, would top the shootout. It
did not. JEKOVA leads every discrimination metric (F1 0.846 against 0.72 or below), because its
14.6 Hz band-pass isolates exactly the property above and its counts read it directly. But it is
the most expensive candidate, about fourteen times TCSC, because the recursive filter runs sample
by sample where TCSC is a single normalized-threshold pass. This gap is the crux for the
downstream embedded target: on an embedded processor a fourteenfold compute cost can weigh more
than a fifteen-point F1 gap, so the paper deliberately keeps the two axes separate rather than
folding them into one score. Both detectors go forward, JEKOVA as the accuracy-first option and
TCSC as the cost-first one, and the deployment choice is left to the compute budget and the
required sensitivity floor, which are outside a feature study.

### 5.3 Comparison with the literature

The tuned JEKOVA reaches F1 0.847 (Se 0.898, Sp 0.976) at 8 seconds. More telling than the tuned
number is the reproduction check: the published cascade, run on our absolute counts with only its
constants rescaled to the window length (section 4.4 lists the published and tuned threshold
values), gives Se 0.973 and Sp 0.900, close to the Se 0.959 and Sp 0.944 the original paper
reports [JEKOVA-2004], despite a different count implementation and a stricter evaluation that
scores every window of every recording rather than curated 10-second episodes. The small specificity gap is consistent with scoring the continuous MITDB background
without the paper's separate noise and asystole gates. TCSC's tuned F1 of 0.706 sits below JEKOVA
but keeps a solid specificity (0.945); the difference between them is precision (0.617 against
0.802), that is, false positives, which the band-pass suppresses better. Neither single-feature
detector reaches the AHA sensitivity goal here, but both are scored per window without the
majority-vote episode smoothing a deployed system would add, so these are lower bounds rather than
final operating numbers.

### 5.4 Flutter versus fibrillation

The winning detector's own feature cannot make this split (jc3 AUC 0.603), which is expected,
since the band-pass measures a property that flutter and fibrillation share, the absence of
14.6 Hz energy. Regularity and phase-space features do modestly better (TCSC 0.735, PSR 0.722,
VF_LEAK 0.706), but with heavy distribution overlap. The practical implication is that separating
flutter from fibrillation needs a second, regularity-oriented feature on top of the band-pass
count, and even then the two rhythms form a physiological continuum that resists a clean boundary.
This matters little for the shock decision, where both are shockable, but it limits any attempt to
characterize the episode more finely.

### 5.5 Threats to validity

Three limits qualify these results. First, class imbalance: non-shockable windows outnumber
shockable ones roughly nine to one, which inflates accuracy and depresses precision, so the paper
leads with F1 and G-Mean and the absolute PPV figures should be read against that imbalance.
Second, coverage: flutter is represented by only a few hundred windows (section 4.1), so the flutter-versus-
fibrillation result is indicative rather than settled, and the AHADB contribution is limited to
its 8-series records, which narrows its non-shockable diversity. Third, the evaluation is
per-window at a 1-second step with no majority-vote episode reconstruction, so the confusion
counts are windows rather than episode durations in milliseconds; the published algorithms were
scored on episode-level decisions, which smooth isolated errors, so the per-window numbers here
are conservative by comparison. The 8-second versus 4-second comparison, by contrast, is robust:
the ranking barely moves and the tuned JEKOVA thresholds are identical at both lengths, so the
choice of window is not driving the conclusions.

## 6. Conclusion and future work

Across a screen of 16 signal-only features and a five-candidate shootout on four PhysioNet
databases, the property that best separates shockable from non-shockable rhythms is how much of
the window departs from the narrow band of a normal ECG, and the detector that reads it best is
the 14.6 Hz band-pass of the JEKOVA algorithm (single-feature AUC 0.986, tuned F1 0.847 with
sensitivity 0.898 and specificity 0.976). TCSC is a cheaper second, about one-fourteenth the
compute cost at F1 0.706, so the two are offered as an accuracy-first and a cost-first candidate
rather than a single winner. The winning feature does not separate flutter from fibrillation
(AUC 0.603); that split needs a regularity-oriented feature.

The contribution is the feature-selection groundwork for extending an existing beat-detection
engine toward a regulatory submission: a reproducible screen and shootout over classical,
deterministic features, and two tuned deterministic detectors as concrete candidates. Staying
deterministic was a deliberate scope choice tied to that regulatory path. The learned methods
reviewed in section 2, empirical-mode decomposition with a support vector machine [VFPRED-2018]
and convolutional or recurrent networks [MODERN-2024, DEEP-2023], report higher numbers but need
a separate validation and approval track, so they remain future work. On the deterministic side,
the immediate next steps are a majority-vote episode layer to turn the per-window decisions into
episode-level output comparable to the published benchmarks, and a regularity feature to attempt
the flutter-versus-fibrillation split for the winning detector.

## References

### Databases

**[MITDB]** Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. https://physionet.org/content/mitdb/1.0.0/

**[CUDB]** Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. https://physionet.org/content/cudb/1.0.0/

**[VFDB]** Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. https://physionet.org/content/vfdb/1.0.0/

**[AHADB]** Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. https://physionet.org/content/ahadb/1.0.0/

### Comparative studies

**[COMP4-1993]** Clayton RH, Murray A, Campbell RW. Comparison of four techniques for recognition of ventricular fibrillation from the surface ECG. Med Biol Eng Comput. 1993 Mar;31(2):111-7. doi: https://doi.org/10.1007/BF02446668  PMID: 8331990.

**[COMP5-2000]** Jekova I. Comparison of five algorithms for the detection of ventricular fibrillation from the surface ECG. Physiol Meas. 2000 Nov;21(4):429-39. doi: https://doi.org/10.1088/0967-3334/21/4/301  PMID: 11110242.

**[COMP55-2005]** Amann, A., Tratnig, R. & Unterkofler, K. Reliability of old and new ventricular fibrillation detection algorithms for automated external defibrillators. BioMed Eng OnLine 4, 60 (2005). https://doi.org/10.1186/1475-925X-4-60

**[MODERN-2024]** Fira, Monica & Costin, Hariton & Liviu, Goras. (2024). Ventricular Fibrillation Prediction and Detection: A Comprehensive Review of Modern Techniques. Applied Sciences. 14. 11167.  https://doi.org/10.3390/app142311167

**[DEEP-2023]** Ansari Y, Mourad O, Qaraqe K, Serpedin E. Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017-2023. Front Physiol. 2023 Sep 15;14:1246746. doi: 10.3389/fphys.2023.1246746. PMID: 37791347; PMCID: PMC10542398. https://pmc.ncbi.nlm.nih.gov/articles/PMC10542398/

### Algorithm papers

**[VFLEAK-1978]** Kuo S and Dillman R 1978 Computer detection of ventricular fibrillation Proc. Computers in Cardiology 1978 (Long Beach, CA: IEEE Computer Society Press) pp 347–9

**[SPEC-1989]** Barro S, Ruiz R, Cabello D, Mira J. Algorithmic sequential decision-making in the frequency domain for life threatening ventricular arrhythmias and imitative artefacts: a diagnostic system. J Biomed Eng. 1989;11:320–8. doi: https://doi.org/10.1016/0141-5425(89)90067-8

**[EMD-1998]** Huang, Norden & Shen, Zheng & Long, Steven & Wu, Manli & Shih, Hsing & Zheng, Quanan & Yen, Nai-Chyuan & Tung, Chi-Chao & Liu, Henry. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences. 454. 903-995. https://doi.org/10.1098/rspa.1998.0193.

**[EMD-2010]** A. Zeiler, R. Faltermeier, I. R. Keck, A. M. Tomé, C. G. Puntonet and E. W. Lang, "Empirical Mode Decomposition - an introduction," The 2010 International Joint Conference on Neural Networks (IJCNN), Barcelona, Spain, 2010, pp. 1-8, doi: https://doi.org/10.1109/IJCNN.2010.5596829.

**[JEKOVA-2004]** Jekova I, Krasteva V. Real time detection of ventricular fibrillation and tachycardia. Physiol Meas. 2004 Oct;25(5):1167-78. doi: https://doi.org/10.1088/0967-3334/25/5/007 PMID: 15535182.

**[HILB-2005]** Amann, Anton & Tratnig, R. & Unterkofler, Karl. (2005). A new ventricular fibrillation detection algorithm for automated external defibrillators. Computers in Cardiology. 32. 559 - 562. doi: https://doi.org/10.1109/CIC.2005.1588162

**[TIME-2007]** Amann A, Tratnig R, Unterkofler K. Detecting ventricular fibrillation by time-delay methods. IEEE Trans Biomed Eng. 2007 Jan;54(1):174-7. doi: https://doi.org/10.1109/TBME.2006.880909  PMID: 17260872.

**[TCSC-2009]** Arafat, M.A., Chowdhury, A.W. & Hasan, M.K. A simple time domain algorithm for the detection of ventricular fibrillation in electrocardiogram. SIViP 5, 1–10 (2011). https://doi.org/10.1007/s11760-009-0136-1

**[VFPRED-2018]** A Fusion of Signal Processing and Machine Learning techniques in Detecting Ventricular Fibrillation from ECG Signals. https://ar5iv.labs.arxiv.org/html/1807.02684

**[HONG-2016]** Hong, Jen-Yee. Detecting Life-Threatening Arrhythmia with Machine Learning Algorithms. Master Thesis, Department of Computer Science and Information Engineering, National Taiwan University, July 2016.
