# Project History

## Origins

The repository was created by **Hong Jen Yee (PCMan)** — a Taiwanese open-source developer
best known as the author of PCManFM (the default file manager for LXDE), libfm, and a number
of core Linux desktop components. This project is unrelated to those efforts: it is the
software artifact of his 2016 NTU master's thesis, *"Detecting Life-Threatening Arrhythmia
with Machine Learning Algorithms"*, submitted to the Graduate Institute of Biomedical
Electronics and Bioinformatics, National Taiwan University, July 2016.

The git history begins on **28 May 2016** with a single enormous grafted root commit
(`4945776`) — the whole codebase was added in one shot. This means the project was already
fully developed before it was put under version control, or an earlier private repository was
squashed. The root commit introduced 204 files (~49,000 lines) including all Cython
extensions, the OSEA QRS detector, bundled PTSA/pyeeg libraries, the full PhysioNet record
lists, a literature survey (`dblp/`) and pre-computed univariate feature-selection CSV results.

---

## Phase 1 — Foundation burst (28–31 May 2016)

In the same week the repository was created, Hong added four more pieces of infrastructure:

- **Distributed feature extraction via Pyro** (`a73defb`): master/slave cluster mode where a
  Pyro nameserver coordinates worker processes on multiple machines. This suggests he was
  running the feature extraction on a university computing cluster, not a laptop.
- **QRS detector fix** (`9f6a3ad`): the detector was being called with normalised signals; it
  needs raw ADC samples. Fundamental correctness fix.
- **First beat skip** (`045d190`): the OSEA detector initialisation artefact was polluting the
  first detected beat; it is now excluded from the unknown-beat ratio feature.
- **README** (`7ee3fd2`): brief usage instructions, now superseded by `docs/EXECUTABLES.md`.

---

## Phase 2 — LZ complexity and native C (1 June 2016)

`0022678` — Lempel-Ziv complexity had been implemented in Python/Cython. Hong rewrote it in
plain C (`vf_features_native.c`) for speed and simultaneously fixed a subtle initialisation
bug where `C(n)` was being set to 0 instead of 1, making LZ values slightly incorrect for
short sequences.

One day later (`b6c0abe`), he extended LZ to all five EMD intrinsic mode functions (IMF1–5),
adding five new features to the 27-feature set. The EMD decomposition comes from the bundled
PTSA library (`ptsa/ptsa/emd.py`).

---

## Phase 3 — Feature selection experiments (2–4 June 2016)

Three commits (`5a35524`, `a104291`, `70cd55c`) attempted to add Recursive Feature Elimination
(RFE) via `sklearn.feature_selection.RFE`. The commit messages read "try to support RFE" and
"try to implement filter-type feature selection" — exploratory work that was ultimately not
included in the thesis conclusions. A merge-reports utility (`merge_reports.py`) was added to
consolidate CSV outputs from multiple parallel cluster runs.

---

## Phase 4 — TCI rewrite and AHA reporting (6–7 June 2016)

The most concentrated bug-fix period. TCI (Threshold Crossing Intervals) had an
implementation error that was corrected across four commits (`8244ba6` → `e351c23`). The final
version avoids computing TCI for the boundary seconds of each 8-second segment, which would
otherwise be biased by incomplete windows.

`50400b5` added AHA-compliant performance reporting — the three-class (shockable / intermediate
/ non-shockable) accuracy breakdown with the specific sensitivity and specificity thresholds
required by the AHA guidelines. This was the key methodological contribution of the thesis:
prior work typically reported only binary VF/non-VF classification and could not be compared
against AHA standards.

---

## Phase 5 — Classifier refactoring and error analysis (11–14 June 2016)

`06253b8` introduced the `VFClassifier` class in `vf_classify.py`, wrapping both RFE-based and
filter-type feature selection into a common interface. Multiple bug fixes followed:
- Feature scaling was applying the wrong transform direction (`b63ed27`)
- AHA evaluation metrics were off-by-one in certain edge cases (`17bf424`)
- Error analysis tooling (`error_analysis.py`) was extended to produce per-class confusion
  detail

---

## Phase 6 — Statistics and cross-validation polish (14–17 June 2016)

The last cluster of original commits tightened the evaluation loop: correct test-score
calculation (`29cf65a`), refactored statistics output, and bug fixes in the error-analysis
reporting. By 17 June the codebase was essentially feature-complete for the thesis.

---

## Phase 7 — EMD visualisation (12 July 2016)

The final two commits from Hong (`924168f`, `e497922`) added `demo_emd.py` — a standalone
visualisation script that applies EMD to a sample ECG segment and plots the resulting intrinsic
mode functions. This likely corresponds to Figure 3.x in the thesis. The last line in the
original history is a cosmetic fix: adjusting the font size of the EMD plot title.

After this the repository went silent for almost **ten years**.

---

## Phase 8 — Revival and takeover (15 May 2026)

The repository was forked/cloned and two commits were pushed by **Kostadin Bajalcaliev**:

**"Takeover"** (`972921e`) — modernisation for a 2026 Python environment:
- Added `Makefile` with `build` / `trace` / `debug` / `release` / `clean` / `check` targets
- Added `pyproject.toml` (PEP 517 build metadata)
- Rewrote `setup.py` to use `cythonize()` properly, compile `vf_features_native.c` as a
  separate source, and support the new build targets
- Extracted the inline C code out of the old `vf_features.c` into `vf_features_native.c`
- Minor compatibility patches to `signal_processing.pyx`, `qrs_detect.pyx`, `pyeeg/__init__.py`
- Added `.vscode/` workspace configuration

**"Documents"** (`4b48055`) — converted the thesis PDF to Markdown (`docs/THESIS.md`) using an
automated tool, added raw output files (`THESIS.html`, `docs/xtract.py`), and seeded initial
documentation stubs (`CMDS.md`, `NOTES.md`, `PAPER.md`).

---

## Phase 9 — Documentation branch (15 May 2026, `develop`)

A `develop` branch was cut from the takeover point and used for documentation:

- `docs/README.md` — full executable reference (later renamed `EXECUTABLES.md`)
- `docs/EXTRACT.md` — detailed analysis of `feature_extraction.py` (renamed `FEATURE_EXTRACTION.md`)
- `docs/FEATURES.md` — all 27 feature algorithms with call tree (renamed `FEATURE_INTERNALS.md`)
- `docs/DEBUG.md` — debugging guide: `make trace` for pdb/VS Code, `make debug` for gdb/cygdb
- `extract_one.py` — single-segment extraction script for interactive debugging
- Makefile `trace` and `debug` targets that compile Cython extensions with `linetrace=True`
  and `gdb_debug=True` respectively

---

## Phase 10 — Bug fixes and THESIS.md restoration (15–16 May 2026, `fixes` → `codex`)

A `fixes` branch (now renamed `codex`) was cut from `develop` and merged the documentation
work with two categories of change:

**Code fixes** (`bc720e7`):
- `info.resample_rate` → `info.sampling_rate` — AttributeError on Python 3.x
- Per-record checkpointing in `feature_extraction.py` — crash recovery without re-processing
  completed records
- `sklearn.cross_validation` → `sklearn.model_selection`, `sklearn.grid_search` →
  `sklearn.model_selection` — deprecated APIs removed in sklearn 0.20+

**CLAUDE.md** (`b12ae5c`) — project context file for LLM-assisted development, documenting
the build system, data pipeline, architecture, and debugging workflow.

**THESIS.md restoration** (commits `fef51a8` through `b0bd897`) — the automated PDF→Markdown
conversion introduced several artefacts that were corrected manually:
- Chinese title page and abstract removed from body; English-only header added
- Contents table rebuilt as a linked Markdown list with page-number references
- List of Figures and List of Tables rebuilt with `<a id>` anchors in the document body
- 18 page-break word-split artefacts repaired (e.g. `previ-\n\nous` → `previous`)
- References normalised to a consistent `- [N]` bullet list
- All 50 reference anchors added; 99 inline citations linked to them
- Reference numbers bolded (`**[N]**`)

**Docs reorganisation** (`2b88a84`):
- Scratch files deleted: `CMDS.md`, `NOTES.md`, `PAPER.md`, `xtract.py`
- Descriptive renames: `README.md` → `EXECUTABLES.md`, `EXTRACT.md` →
  `FEATURE_EXTRACTION.md`, `FEATURES.md` → `FEATURE_INTERNALS.md`
- `SUMMARY.md` written from scratch as a structured reference covering the AHA class
  scheme, five datasets, all 27 features with canonical indices, thesis→code mapping, and
  known weak points

---

## Branch summary

| Branch | Status | Purpose |
|--------|--------|---------|
| `master` | frozen | Hong's original thesis code, last touched July 2016 |
| `develop` | frozen | Documentation additions, May 2026 |
| `fixes` / `codex` | active | Bug fixes, modernisation, documentation quality (same commits, renamed) |
