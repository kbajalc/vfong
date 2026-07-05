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
- Decisions Q1 to Q8 are resolved. Q9 (the proposed new method) is still open.
- Advisor feedback is logged: Gusev, 2026-03-06. The original timeline (Structure section,
  in Macedonian) ran Phases 1 to 5, February to April 2026.

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

The `exg-*` ecosystem is not yet documented: `exg-core` (the tool behind the `exg`
wrapper), `exg-work` (the off-limits `work/`), and `exg-rad` (in `.vscode/settings.json`
`extraPaths`). Their relevance to the vftx and paper work is unconfirmed.

Q9, the proposed method, is still open, and it affects what Materials and Methods should
describe now versus later.

We need to decide whether to add standalone VFLEAK/SPEC/TCSC detector wrappers (thresholds
and decision rules) on top of vftx, as the paper's reproducible benchmarks.

Database sets need reconciling. The paper's primary set is VFDB + CUDB + AHADB, while vftx is
currently validated on a small diverse set (mitdb, vfdb, cudb, edb, mghdb). AHADB integration
is new.

The evaluation harness is still to build: F1 plus duration-based TP/FP/TN/FN over sliding
windows at 4 s (and each algorithm's native window), across the three VFL configurations.
