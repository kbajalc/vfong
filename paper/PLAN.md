# Paper plan

The dataset-construction strategy for the next phase (read through `pxg.cbor` instead of
WFDB, one TSV feature file per record under `work/vft8/`, code in `vfta/`, filter once at the
record level) is folded into "Phase 2: Dataset and preprocessing" below.

Working plan for the paper in `paper/PAPER.md`. This file defines the scope, the phases,
and the decisions that shape the manuscript. `paper/DRAFT.md` is the earlier, broader draft
and stays as a source of text and references we pull from. It is not the target document.

Prose in the paper and in this plan follows `STYLE.md` (no em dashes, sentence case
headings, straight quotes). Run the STYLE self-check before committing any prose.

## What this paper is

A focused study for the "Introduction to Bioinformatics" exam (bioinformatics in the broad
sense of processing data of biological origin, here the ECG), at the level of a preparation
year for a Masters in Bioinformatics. The paper does one thing well: it measures which
classical, deterministic signal-processing features best separate the life-threatening
ventricular tachyarrhythmias (VT, VFL, VF) from other rhythms, then selects and tunes one
detector built on the strongest feature.

The work has a second, practical purpose. It is the feature-selection step for extending an
existing beat-detection and classification engine (the `exg-core` project) toward an FDA
submission. That regulatory path is why the paper stays inside classical, deterministic
methods: machine-learning and deep-learning detectors need a very different validation and
approval process, so they are out of scope here and appear only in the literature review and
as future work.

Non-goals: no SVM, no CNN, no other learned classifier in the experiments; no attempt at a
full benchmark reproduction of every published algorithm; no clinical validation.

## What changed from DRAFT.md

`DRAFT.md` grew into a broad benchmarking and reproducibility study that ended in a CNN. It
carried long method write-ups with inline equations and a large decisions log. The refined
paper is smaller and has a single spine:

- The introduction is non-technical and concise: what the ventricular tachyarrhythmias are,
  why they matter (life-threatening), and why VT, VFL, and VF are hard to detect and hard to
  tell apart from each other and from other rhythms.
- The literature review is a short, organized tour of known methods. It uses the feature
  categories from Hong's thesis to give structure and describes what each category measures,
  without hanging equations or full derivations.
- The core is an experiment: a small set of candidate detectors (the three from `DRAFT.md`
  plus two more, five in total) computed with `vftx` across the databases and ranked on
  shockable vs non-shockable discrimination. A broader screen of the 27 `vftx` features
  supports the choice of the two added candidates.
- The close is a single tuned detector: pick the winning candidate by weighing discrimination
  against computational cost, sweep its threshold, and report a ROC curve and operating points
  in the style of COMP55-2005. For the winner only, we also test whether it can separate
  flutter (VFL) from fibrillation (VF).

References carry over from `DRAFT.md` unchanged, plus one addition: Hong's thesis, now a
primary source (see [HONG-2016] below).

## Resolved decisions

These fix the shape of the experiments and the Results section. The `DRAFT.md` decisions
Q1-Q8 still inform the details (VFL handling, window lengths, metrics); the three below are
the ones that changed or were newly settled for this paper.

| Topic | Decision |
|---|---|
| Benchmark target | All candidates are benchmarked on shockable (VT/VFL/VF) vs non-shockable. This matches the published purpose of these algorithms and keeps the shootout comparable to the literature. |
| Winner sub-analysis | The winning candidate gets one extra test: can it separate flutter (VFL) from fibrillation (VF), which have very different signatures? See "Why VFL vs VF, for the winner only" below. |
| Candidate set | Five detectors: the three from `DRAFT.md` (VFLEAK, SPEC, TCSC) plus HILB (phase space) and MEA (amplitude shape), chosen from Hong's thesis on the grounds below. See "Candidate detectors and winner selection". |
| Winner selection | Data-driven, weighing discrimination against computational cost. Not the single best score if it is far more expensive. TCSC is the stated hypothesis for the winner (cheap and strong). |
| Databases | Full set: VFDB + CUDB + AHADB (licensed, available) + MITDB, matching the COMP55-2005 combination for comparability. |
| Windowing | Overlapping windows with a 1 s step, and a majority-vote smoothing to turn per-window decisions into reference episode labels (the `DRAFT.md` post-processing). Two window lengths, 8 s (benchmark, matches the original papers) and 4 s (short-episode test, MITDB). Both run for all five candidates if the dataset build allows; confirmed in Phase 2. |
| Proposed method (was DRAFT Q9) | Not a new algorithm. The contribution is the candidate shootout plus a tuned deterministic detector on the winning feature. No learned classifier. |

Carried from `DRAFT.md` and still in force: VFL kept as a separate class with the three
shockable configurations (VFL-as-VF, VFL-as-non-VF, VFL-excluded); F1 as the headline metric
with Se, Sp, PPV, Acc, and G-Mean alongside; TP/FP/TN/FN reported as durations in ms.
Difference from `DRAFT.md`: ROC is no longer fully deferred. It is computed for the final
candidate-tuning step only (Phase 4), not for the whole benchmark grid.

### Candidate detectors and winner selection

The candidate set has five detectors, each a single feature plus a decision rule. Three come
straight from `DRAFT.md`:

- TCSC (Arafat 2009), threshold crossing, cheap. The hypothesized winner.
- VFLEAK (Kuo and Dillman 1978), leakage ratio, cheap.
- SPEC (Barro 1989), FFT spectral descriptors, expensive.

Two more are added, chosen from what Hong's thesis reports about which feature families
actually work (thesis section 2.3, quoting Amann et al. 2005):

- HILB (Amann 2005), phase-space box-counting. Amann's own algorithm and the strongest single
  classical algorithm in the literature (IROC around 95%), at moderate cost (one FFT-based
  convolution). This adds the phase-space family, which none of the first three cover.
- MEA (modified exponential), amplitude distribution and shape, cheap. Hong reports Amann's
  finding that the best-performing features work in the time domain, so a strong, cheap
  time-domain shape feature is a natural second addition. It adds the distribution-and-shape
  family.

Two families are left out of the candidate set on purpose. Complexity and entropy measures
(LZ, SpEn) are excluded because Amann found they perform poorly wherever specificity is above
80% (thesis section 2.3), which is exactly the region an AED-grade detector must live in. So
the five candidates span threshold crossing (TCSC), spectral (VFLEAK, SPEC), phase space
(HILB), and distribution and shape (MEA), and skip complexity, with a documented reason.

The point of five is to make the winner choice an argued one, not a coin flip. Cost matters
because the downstream target is an embedded, real-time extension of `exg-core`. Hong's own
discussion is the evidence here: threshold-crossing features were the cheapest, FFT-based
spectral analysis was not a bottleneck, but the complexity measures, sample entropy above
all, were computation-intensive. So Phase 3 reports both discrimination (correlation, mutual
information, single-feature AUC, F1 at a swept threshold) and a rough compute cost per window,
and Phase 4 argues the winner from both. A cheaper candidate that trails the best score by a
small margin can still win.

### Why VFL vs VF, for the winner only

The shockable-vs-non-shockable benchmark is the right target for the shootout, because that
is what these algorithms were published to do. But VFL and VF have very different signal
signatures: VFL is a fast, near-sinusoidal, regular oscillation, while VF is disorganized and
irregular. VT and VFL are usually short, transient phases that often, but not always, lead
into VF. For the winning detector we therefore add one analysis: does its feature also
separate VFL from VF?

Hong's thesis points at the tool for this. To tell VF from VT in his multiclass setting, Hong
added Lempel-Ziv complexity computed on the EMD intrinsic mode functions (the IMF-LZ features
[17-21]), following Xia et al. 2014, because that separates the rhythms better than LZ on the
raw signal (thesis section 2.3). So if the winning feature does not separate VFL from VF on
its own, the IMF-LZ features are the documented fallback for that sub-analysis. This is
reported for the winner only, not for the whole candidate set.

### Dataset and feature-extraction decisions (Phase 2)

Settled for the dataset build in `vfta/`:

- vftx filtering off. The record is filtered once upstream by the `vfta` `SignalFilter`
  (Lynn band-pass plus median baseline), matching the real-time `exg-core` pipeline, so
  vftx's own frequency filtering (drift suppression and low-pass, preprocessing steps 4-5) is
  not used. It is now optional via `SignalConfig.apply_filters` (default `True` for the
  reference suite; `vfta` sets it `False`). Mean subtraction, normalisation, and
  moving-average smoothing still run, since the feature math needs them.
- Signal units. The cbor signal is 12-bit integers at a standard gain of 200. Convert to
  millivolts per segment for feature extraction by dividing by 200.
- QRS features skipped for now. RR, RR_Std, RR_CV, UR, VR [22-26] are left out of this
  dataset. Neither OSEA nor xqrs is used: the target algorithm is `exg-core`, and its EXG
  beat annotations will be exposed as the beat source and wired in later. vftx already skips
  these features when no detector is passed.

## Open items to confirm during the phases

- AHADB record use. Only some AHADB records carry ventricular arrhythmias; many have none, so
  the full database is overkill for the benchmark. Jekova used the AHA ventricular series
  (A8001-A8010) [JEKOVA-2004]; the cbor `ahadb/RECORDS` is already curated (about 79 active
  records, the 8200-series being the ventricular ones) and `CborDatabase` skips the commented
  rest. Confirm the benchmark subset in Phase 3. The full annotated AHADB is kept for the
  Phase 4 fine-tuning.
- Whether both window lengths (8 s and 4 s) run for all five candidates, or 4 s runs only for
  the leaders, depends on how heavy the overlapping-window dataset build turns out to be.
  Decide in Phase 2 when the dataset is constructed.

## Phases

Each phase names its goal, its inputs, its deliverables, and the paper section it feeds.
Phases are sequential but the write-up (Phase 5) folds in results as each earlier phase
lands.

### Phase 0: Scope and skeleton (this document)

Goal: agree the refined scope, the decisions above, and the manuscript skeleton.
Deliverables: this `PLAN.md`; the section skeleton in `PAPER.md`.
Status: done once both files are reviewed.

### Phase 1: Introduction and literature review

Goal: write the non-technical introduction and the organized methods review.
Inputs: `DRAFT.md` Introduction and Related Work; `docs/texts/` for the cited papers;
Hong's thesis for the feature categories.
Work: trim the introduction to the rhythms, their clinical importance, and the detection
difficulty. Reorganize the methods review around Hong's feature categories (threshold
crossing, distribution and shape, complexity and entropy, spectral, phase space), one short
paragraph per category, no hanging equations. Add the Hong citation.
Deliverables: `PAPER.md` sections 1 and 2 drafted.

### Phase 2: Dataset and preprocessing

Goal: build the labeled segment dataset as on-disk feature files, one per record, so the
later analysis never touches the raw ECG again. Dataset construction is the heavy part of the
work: it takes time and needs intermediate results saved to disk. VFPred is the model for this
staging, and part of the strategy is already prototyped in `ideas/VFML.ipynb`.

Data source and parallelism. Do not read through WFDB directly: its C library is slow and not
thread-safe, which blocks record-level parallelism. Read instead through `pxg.cbor`, which
serves the same WFDB signals and annotations verbatim from a custom format; the data is
already available in `work/cbor`. Databases are processed one at a time, parallel at the record
level, so each record (for example `mitdb/100`) produces a single feature file (for example
`work/vft8/mitdb/100.tsv` for the 8 s window).

TSV layout. Each row is one segment: beat counts, episode lengths, labels, and all 27 `vftx`
features. The exact columns follow `ideas/VFML.ipynb`. Once every record is written, the
analysis in Phases 3 and 4 works on the TSV corpus alone.

Windowing and filtering. Following `VFML.ipynb`: read the whole record, apply the initial
filtering once at the record level (baseline removal and low-pass), then form 8 s and 4 s
segments stepping by 1 s (the stride is configurable). Filters are not applied per segment.
This matches the intended `exg-core` integration, where baseline removal and low-pass run once
in a real-time pipeline, not per window. `VFML.ipynb` has the filter implementations in Python.
Because filtering moves to the record level, the per-segment filtering inside `vftx` is
skipped. See the open item on this below.

Code. Dataset construction lives in `vfta/` as Python modules plus notebooks for orchestration
and visualization. The first step is to read `ideas/VFML.ipynb`, then extract and rewrite the
relevant parts so that, given a database name, one function produces all the TSV files in
parallel from `pxg.cbor`. Feature extraction wiring comes after a working pipeline exists.

Then: label each window and derive the shockable class and the VT/VFL/VF sub-labels; set up
the majority-vote smoothing into reference episode labels; settle the AHADB record selection;
gauge build cost to decide whether both window lengths run for all five candidates; report
window counts per database, per class, and per window length.
Deliverables: the on-disk TSV corpus; the dataset composition table for Results section 4.1;
`PAPER.md` sections 3.1-3.3 drafted.

### Phase 3: Feature screen and candidate shootout

Goal: rank the candidate detectors on shockable vs non-shockable, and argue the winner choice.
Inputs: the Phase 2 dataset; the 27 `vftx` features; the five candidate detectors.
Work: two passes. First, a broad screen: run `vftx` over all segments, and score each of the
27 features against the shockable label with point-biserial correlation, mutual information,
and single-feature AUC. This screen confirms the two candidates to add (see the open items).
Second, the shootout: for each of the five candidate detectors report its discrimination
(the same scores plus F1 at a swept threshold) and a rough compute cost per window, at the
8 s window, and rerun the leaders at 4 s. Produce the ranking tables, the supporting figures
(per-feature distributions, a mutual-information bar chart, a feature-feature correlation
heatmap for redundancy), and the discrimination-vs-cost comparison.
Deliverables: screen and shootout tables and figures for Results sections 4.2-4.3;
`PAPER.md` sections 3.4-3.5 drafted.

### Phase 4: Winner tuning and VFL-vs-VF test

Goal: pick the winner, tune it, and test the flutter-vs-fibrillation split.
Inputs: the Phase 3 shootout (discrimination and cost).
Work: choose the winner by weighing discrimination against compute cost (hypothesis: TCSC).
Sweep its threshold, build the ROC curve, and pick operating points. Report F1, Se, Sp, PPV,
Acc, and G-Mean at the chosen point, with TP/FP/TN/FN as durations in ms, under the three VFL
configurations and at both window lengths. Compare the tuned operating point against the
published numbers for that algorithm. Then the winner-only sub-analysis: test whether the
winning feature separates VFL from VF, and report that result.
Deliverables: the ROC figure, the operating-point table, and the VFL-vs-VF result for
Results section 4.4; `PAPER.md` sections 3.6-3.7 drafted.

### Phase 5: Write-up, figures, and finalization

Goal: finish the manuscript.
Work: fold all results into `PAPER.md`; write Discussion and Conclusion; place figures and
tables; merge the references (DRAFT set plus Hong); state the ML and deep-learning direction
as future work and connect it to the `exg-core` and FDA path; run the STYLE self-check.
Deliverables: complete `PAPER.md`.

## How vftx feeds the paper

`vftx` is the validated, pure-Python reimplementation of all 27 features (Phase 4 of the
implementation work is complete; see `vftx/PLAN.md`). It gives the reproducible feature math
the experiments run on. The candidate detectors the paper needs (a threshold and decision
rule on top of a feature) are thin wrappers to add during Phases 3-4:

| Candidate detector | vftx feature | Note |
|---|---|---|
| TCSC (Arafat 2009) | `tcsc` [0] | Crossing count; add the `N_d` threshold, retuned at 250 Hz. Cheap. |
| VFLEAK (Kuo and Dillman 1978) | `vf_leak` [6] | Leakage ratio; add the threshold. Cheap. |
| SPEC (Barro 1989) | `m` [7], `a2` [8], `fm` [9] | Spectral descriptors; add FSMN/A1/A2/A3 and Jekova thresholds. Expensive (FFT per window). |
| HILB (Amann 2005) | `hilb` [5] | Phase-space box-count; add the threshold. Moderate cost. |
| MEA (modified exponential) | `mea` [3] | Amplitude envelope; add the threshold. Cheap. |

The five candidates already map onto existing `vftx` features, so no new feature math is
needed, only the thin threshold-and-decision wrappers and the cost measurement. The IMF-LZ
features [17-21], the fallback for the winner's VFL-vs-VF sub-analysis, are also already in
`vftx`.

## New reference to add

[HONG-2016] Hong, Jen-Yee. Detecting Life-Threatening Arrhythmia with Machine Learning
Algorithms. Master Thesis, Department of Computer Science and Information Engineering,
National Taiwan University, July 2016.
