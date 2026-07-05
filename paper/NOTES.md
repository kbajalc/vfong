# Ventricular tachyarrhythmias paper: status and working notes

Takeover report written 2026-07-05, when the paper-writing project was merged into the
`exg-vfong` repo and joined with the `vftx` reference-implementation work. This is the
documented starting point for continuing the paper effort. For the implementation status
see `vftx/PLAN.md`; for the manuscript see `paper/DRAFT.md`.

## 1. What this project is

`paper/DRAFT.md`, titled "Detection of Ventricular Tachyarrhythmias", is a benchmarking and
reproducibility study of VT, VFL, and VF detection from short ECG segments. Its stated gap
is that the literature is extensive but lacks reference implementations, which holds back
reproducibility and comparative evaluation. The plan is to reimplement the core
temporal and spectral benchmark metrics, assemble annotated databases, run a uniform
evaluation, and eventually propose a new method (possibly a CNN on time-domain and/or
spectral representations).

`vftx/` is our clean, validated reference-implementation engine, the concrete answer to that
reproducibility gap.

## 2. Repository map (paper-relevant)

| Path | Role |
|---|---|
| `paper/DRAFT.md` | The manuscript (state in section 3). |
| `paper/NOTES.md` | This file: status and working rules. |
| `docs/papers/` | Symlink to the external literature corpus (`~/Desktop/VFL`). The literature source of truth (PDFs and SLIDES). Maintained externally. |
| `docs/texts/` | Extracted `.md`/`.html` of papers, a cache to prefer over reprocessing PDFs. About 19 papers: COMP4/5/55, EMD-1998/2010, HONG, JEKOVA, SPEC, TCSC, VFPRED, HILB, HYPO, TAYLOR, TIME, MODERN, GORAN. |
| `vftx/` | Clean pure-Python feature and metric implementations. The reference implementation (Phase 4 complete, see section 3). |
| `hong/` | Original 2016 NTU thesis Cython implementation (Hong); the reference that `vftx/` is validated against. |
| `tests/` | `vftx`-vs-`hong` agreement suite plus unit tests (hermetic). |
| `vfpred/` | VFPred (Ibtehaz 2019: EMD + DTFT + SVM) reference repo. Reference and scratch only for now. |
| `ideas/` | Prior exploration: notebooks (EMD, TCSC, TCSCX, VFFT, VFML, JEKOVA, FIRST) plus a `vta/` module (det/dsp/fft/vis). Reference and scratch only. |
| `work/` | Symlink to `../exg-work`. Off-limits: never scan unless explicitly asked. |
| `exg` | Shell wrapper to `../exg-core` tooling. |

## 3. Current status

### vftx implementation: Phase 4 complete

26 of 27 features reproduce Hong's reference bit-for-bit (via `reference_bug_compat`). SpEn
is correct but validated independently, because the reference is non-deterministic. QRS
features are validated. The EMD backend is selectable (default `ptsa`). Suite:
`pytest tests/` gives 166 passed, 5 xfailed. Full detail is in `vftx/PLAN.md`.

### paper/DRAFT.md: mature draft, benchmarks defined

- Sections present: Project Statement, Introduction, Related Work (method groups A to F plus
  a summary table), Materials and Methods (Databases, Preprocessing, Evaluation, Benchmark
  Algorithms), References, Decisions, Feedback.
- Benchmark algorithms defined: VFLEAK (Kuo and Dillman 1978), SPEC (Barro 1989, with
  Jekova-adapted thresholds), TCSC (Arafat 2009). Each has formulas, thresholds, and
  published performance.
- Databases: primary set is VFDB + CUDB + AHADB (licensed); the extended set adds MITDB.
- Primary metric is F1, with Se, Sp, PPV, Acc, and G-Mean. TP/FP/TN/FN are reported as
  duration in ms. ROC/IROC is deferred.
- Decisions Q1 to Q8 are resolved. Q9 (the proposed new method) is resolved in `PLAN.md`.
- Advisor feedback is logged: Gusev, 2026-03-06. The original timeline (Structure section,
  in Macedonian) ran Phases 1 to 5, February to April 2026.

### paper/PLAN.md and paper/PAPER.md: refined scope, phases defined

The paper was refocused (2026-07-05) into a smaller, exam-level study, and the plan and the
manuscript skeleton now live in their own files. `DRAFT.md` stays as a source of text and
references, not the target document.

- `PLAN.md` holds the refined scope, the resolved decisions, and six phases (0 to 5).
- `PAPER.md` is the manuscript skeleton on a standard academic template, each section
  carrying an editorial note on what it contains and which phase fills it.
- Scope: rank a set of classical, deterministic detectors by how well they separate the
  shockable rhythms, then tune the winner. No ML in the experiments (the downstream target is
  an FDA-oriented extension of `exg-core`, which needs a different validation path); ML is
  reviewed and named as future work only. This resolves Q9: the contribution is the candidate
  shootout plus a tuned deterministic detector, not a new algorithm.
- Candidate set (five): TCSC, VFLEAK, SPEC (from `DRAFT.md`), plus HILB and MEA, the two
  additions chosen from Hong's thesis (Amann's finding that time-domain features perform best,
  HILB the strongest classical algorithm, complexity left out for poor performance above 80%
  specificity). Winner chosen by discrimination weighed against compute cost; TCSC is the
  hypothesis.
- Windowing: overlapping windows, 1 s step, majority-vote smoothing into episode labels; two
  lengths, 8 s (benchmark) and 4 s (short-episode test).
- Databases: full VFDB + CUDB + AHADB + MITDB set.
- Winner-only sub-analysis: VFL vs VF, with IMF-LZ [17-21] as the documented fallback (Hong,
  after Xia et al. 2014).
- Hong's thesis is now a cited source: [HONG-2016] in `PLAN.md` and `PAPER.md`.

## 4. How the pieces connect

The paper's three benchmark detectors map directly onto vftx feature computations.

| Paper benchmark | vftx feature(s) | Note |
|---|---|---|
| TCSC (Arafat 2009) | `tcsc` [0] | vftx computes the crossing count; the paper adds the `N_d` decision threshold (retuned at 250 Hz). |
| VFLEAK (Kuo and Dillman 1978) | `vf_leak` [6] | vftx computes the leakage ratio; the paper adds the threshold. |
| SPEC (Barro 1989) | `m` [7], `a2` [8], `fm` [9] | Related spectral descriptors; the paper needs FSMN/A1/A2/A3 plus Jekova thresholds. |

So vftx gives the reproducible feature math, and the paper needs standalone detector wrappers
(decision rules and thresholds) on top, plus the AHADB database and the F1/duration
evaluation harness. `vfpred/` and `ideas/vta/` are candidate sources for additional
algorithms and DSP, kept as reference only until we decide per algorithm.

## 5. Working rules (retained from the paper's original NOTES, paths updated for the merge)

Literature is the source of truth. Anchor research to `docs/papers/` (the external VFL
corpus). Suggest outside papers only sparingly and clearly framed as out of scope.

Cache PDFs. Whenever a PDF is read (from `docs/papers/` or elsewhere), extract its full
content to `docs/texts/<base>.md`. On later sessions prefer `docs/texts/<base>.html` when it
exists, then `docs/texts/<base>.md`, over reprocessing the PDF.

Respect the off-limits directory. Never scan or read `./work/` (which points to
`../exg-work`) unless explicitly asked. It is large and unrelated, and reading it wastes
context.

## 6. Open questions and next steps

Next up is Phase 1 (introduction and literature review). See `PLAN.md` for the full phase
breakdown. The items below are what remains open or unbuilt.

The `exg-*` ecosystem is not yet documented: `exg-core` (the tool behind the `exg`
wrapper), `exg-work` (the off-limits `work/`), and `exg-rad` (in `.vscode/settings.json`
`extraPaths`). Their relevance to the vftx and paper work is unconfirmed.

Detector wrappers to build (Phases 3 to 4): the five candidates (TCSC, VFLEAK, SPEC, HILB,
MEA) are thin threshold-and-decision rules on top of existing vftx features, plus a compute
cost measurement per window.

AHADB integration is new. vftx is validated on a small diverse set (mitdb, vfdb, cudb, edb,
mghdb); the paper needs the full VFDB + CUDB + AHADB + MITDB set assembled in Phase 2, with
the AHADB record selection still to settle.

The evaluation harness is still to build (Phases 3 to 4): the feature screen and the
candidate shootout (correlation, mutual information, single-feature AUC, F1 at a swept
threshold, plus compute cost), then the winner ROC tuning, over overlapping 1 s-step windows
at 8 s and 4 s, across the three VFL configurations, with F1 and duration-based TP/FP/TN/FN.

Open decisions for later phases: exact AHADB record selection (Phase 2); whether both window
lengths run for all five candidates or 4 s only for the leaders (Phase 2).
