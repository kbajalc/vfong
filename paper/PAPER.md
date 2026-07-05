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

> Phase 1. Non-technical and concise. Pull and trim from DRAFT.md sections "Project
> Statement" and "Introduction". Keep it readable at prep-year level.

### 1.1 The ventricular tachyarrhythmias

> What VT, VFL, and VF are, in plain terms, and where each sits on the continuum from
> organized to chaotic ventricular activity. One short paragraph each.

### 1.2 Why detection matters

> Clinical importance: leading cause of sudden cardiac death, survival falls with every
> minute to defibrillation. Cite MODERN-2024. Keep the numbers, drop the puffery.

### 1.3 Why detection is hard

> The technical difficulty: QRS detectors assume separable beats, which VFL and VF break;
> VF can resemble noise and artifact; VT must be told apart from other fast rhythms;
> labeled data is scarce. This paragraph sets up why a feature-selection study is useful.
> Cite COMP4-1993, COMP5-2000.

### 1.4 Aim and scope

> One paragraph. State the aim: measure which classical deterministic features best separate
> the shockable rhythms and each of VT/VFL/VF, then tune one detector on the strongest. State
> the scope limit plainly: deterministic methods only, because the downstream use (extending
> the exg-core engine toward an FDA submission) needs that. Learned methods are reviewed and
> named as future work, not evaluated here.

## 2. Background and related work

> Phase 1. Condense DRAFT.md "Related Work" into a short organized tour. Use the feature
> categories from Hong's thesis for structure. One compact paragraph per category describing
> what it measures and naming representative methods. No hanging equations; put any formula
> the reader truly needs inline and short, or defer to the cited paper.

### 2.1 Threshold-crossing measures

> ZCR, TCI, TCSC. What "how often does the signal leave the baseline" captures. Cite
> COMP4-1993, COMP5-2000, TCSC-2009.

### 2.2 Amplitude distribution and shape

> STE, MEA, band and histogram features. Cite COMP55-2005.

### 2.3 Complexity and entropy

> Sample entropy, Lempel-Ziv complexity. What "how predictable is the signal" captures.
> Cite COMP5-2000, COMP55-2005.

### 2.4 Spectral measures

> VF-filter leakage, dominant frequency, SPEC descriptors, band-pass approach. Cite
> SPEC-1989, JEKOVA-2004, COMP55-2005.

### 2.5 Phase-space measures

> Hilbert phase-space box-counting. Cite HILB-2005.

### 2.6 Learned methods, and why they are out of scope here

> Short. Name EMD + SVM and CNN/LSTM approaches, note their reported performance, and state
> that they need a different validation and regulatory path, so they are future work. Cite
> VFPRED-2018, MODERN-2024, DEEP-2023. Position Hong-2016 as the source of the 27-feature
> set and category structure this paper uses. Cite HONG-2016.

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
