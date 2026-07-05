# Ventricular Tachyarrhythmias — Paper Project: Status & Working Notes

*Takeover report written 2026-07-05, when the paper-writing project was merged into the
`exg-vfong` repo and joined with the `vftx` reference-implementation work. This is the
documented starting point for continuing the paper effort. For the implementation status
see `vftx/PLAN.md`; for the manuscript see `paper/DRAFT.md`.*

---

## 1. What this project is

`paper/DRAFT.md` — **"Detection of Ventricular Tachyarrhythmias"** — is a benchmarking /
reproducibility study of VT / VFL / VF detection from short ECG segments. Its stated gap:
*the literature is extensive but lacks reference implementations, hindering reproducibility
and comparative evaluation.* The plan is to (re)implement the core temporal/spectral
benchmark metrics, assemble annotated databases, run a uniform evaluation, and eventually
propose a new method (possibly a CNN on time-domain and/or spectral representations).

**`vftx/` is our clean, validated reference-implementation engine** — the concrete answer
to that reproducibility gap.

---

## 2. Repository map (paper-relevant)

| Path | Role |
|---|---|
| `paper/DRAFT.md` | The manuscript (state in §3). |
| `paper/NOTES.md` | This file — status + working rules. |
| `docs/papers/` | **Symlink** → external literature corpus (`~/Desktop/VFL`). The literature **source of truth** (PDFs + SLIDES). Maintained externally. |
| `docs/texts/` | Extracted `.md`/`.html` of papers (cache — prefer over reprocessing PDFs). ~19 papers: COMP4/5/55, EMD-1998/2010, HONG, JEKOVA, SPEC, TCSC, VFPRED, HILB, HYPO, TAYLOR, TIME, MODERN, GORAN… |
| `vftx/` | Clean pure-Python feature/metric implementations. **The reference implementation** (Phase 4 complete — see §3). |
| `hong/` | Original 2016 NTU thesis Cython implementation (Hong) — the reference `vftx/` is validated against. |
| `tests/` | `vftx`-vs-`hong` agreement suite + unit tests (hermetic). |
| `vfpred/` | VFPred (Ibtehaz 2019: EMD + DTFT + SVM) reference repo — **reference/scratch only** for now. |
| `ideas/` | Prior exploration: notebooks (EMD, TCSC, TCSCX, VFFT, VFML, JEKOVA, FIRST) + `vta/` module (det/dsp/fft/vis) — **reference/scratch only**. |
| `work/` | **Symlink** → `../exg-work`. **OFF-LIMITS** — never scan unless explicitly asked. |
| `exg` | Shell wrapper → `../exg-core` tooling. |

---

## 3. Current status

### vftx (implementation) — **Phase 4 complete**
26/27 features reproduce Hong's reference **bit-for-bit** (via `reference_bug_compat`);
SpEn is correct but validated independently (reference is non-deterministic); QRS features
validated; EMD backend is selectable (default `ptsa`). Suite: `pytest tests/` → 166 passed,
5 xfailed. Full detail in `vftx/PLAN.md`.

### paper/DRAFT.md — **mature draft, benchmarks defined**
- Sections: Project Statement, Introduction, Related Work (method Groups A–F + summary
  table), Materials & Methods (Databases, Preprocessing, Evaluation, Benchmark Algorithms),
  References, Decisions, Feedback.
- **Benchmark algorithms defined:** **VFLEAK** (Kuo & Dillman 1978), **SPEC** (Barro 1989,
  Jekova-adapted thresholds), **TCSC** (Arafat 2009). Each with formulas, thresholds, and
  published performance.
- **Databases:** primary = VFDB + CUDB + **AHADB** (licensed); extended adds MITDB.
- **Primary metric:** F1 (+ Se, Sp, PPV, Acc, G-Mean); TP/FP/TN/FN as duration in ms;
  ROC/IROC deferred.
- **Decisions:** Q1–Q8 resolved; **Q9 (the proposed new method) still open**.
- Advisor feedback logged: **Gusev, 2026-03-06**. Original timeline (Structure section, in
  Macedonian) ran Phases 1–5, Feb–Apr 2026.

---

## 4. How the pieces connect (the load-bearing link)

The paper's three benchmark **detectors** map directly onto **vftx feature computations**:

| Paper benchmark | vftx feature(s) | Note |
|---|---|---|
| TCSC (Arafat 2009) | `tcsc` [0] | vftx computes the crossing count; paper adds the `N_d` decision threshold (retuned at 250 Hz). |
| VFLEAK (Kuo & Dillman 1978) | `vf_leak` [6] | vftx computes the leakage ratio ℓ; paper adds the ℓ₀ threshold. |
| SPEC (Barro 1989) | `m` [7], `a2` [8], `fm` [9] | Related spectral descriptors; paper needs FSMN/A1/A2/A3 + Jekova thresholds. |

So **vftx gives reproducible feature math; the paper needs standalone detector wrappers
(decision rules + thresholds) on top**, plus the AHADB database and the F1/duration
evaluation harness. `vfpred/` and `ideas/vta/` are candidate sources for additional
algorithms/DSP, kept reference-only until we decide per-algorithm.

---

## 5. Working rules (retained from the paper's original NOTES, paths updated for the merge)

- **Literature = source of truth:** anchor research to `docs/papers/` (the external VFL
  corpus). Suggest outside papers only sparingly and clearly framed as out-of-scope.
- **PDF caching:** whenever a PDF is read (from `docs/papers/` or elsewhere), extract its
  full content to `docs/texts/<base>.md`. On later sessions prefer `docs/texts/<base>.html`
  (when present), then `docs/texts/<base>.md`, over reprocessing the PDF.
- **Off-limits:** never scan or read `./work/` (→ `../exg-work`) unless explicitly asked —
  it's large and unrelated, and will waste context.

---

## 6. Open questions / next steps

- **`exg-*` ecosystem** — `exg-core` (tool behind the `exg` wrapper), `exg-work` (the
  off-limits `work/`), and `exg-rad` (in `.vscode/settings.json` `extraPaths`). Their
  relevance to the vftx/paper work is not yet confirmed.
- **Q9 — proposed method** (paper) still open; affects what M&M describes now vs. later.
- **Benchmark wrappers** — decide whether to add standalone VFLEAK/SPEC/TCSC detector
  wrappers (thresholds + decision rules) on top of vftx, as the paper's reproducible
  benchmarks.
- **Database reconciliation** — paper primary set is VFDB + CUDB + AHADB; vftx is currently
  validated on a small diverse set (mitdb/vfdb/cudb/edb/mghdb). AHADB integration is new.
- **Evaluation harness** — F1 + duration-based TP/FP/TN/FN over sliding windows at 4 s (+
  each algorithm's native window), three VFL configurations.
