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
substantially developed before it was put under version control, or an earlier private
repository was squashed. The root commit introduced 204 files (~49,000 lines) including all
Cython extensions, the OSEA QRS detector, bundled PTSA/pyeeg libraries, the full PhysioNet
record lists, a DBLP literature survey (`dblp/`), pre-computed univariate feature-selection
CSV results across 10 features, and the first `README`. The root commit message itself signals
an early architectural choice: switching from XGBoost to `sklearn.GradientBoostingClassifier`
because it "works better for me".

---

## Phase 1 — First fixes and distributed infrastructure (28–31 May 2016)

Within hours of the root commit, Hong pushed five fixes and the distributed-computing layer:

### QRS detector was receiving the wrong signal (`9f6a3ad`)

`beat_statistics()` was passing the preprocessed, normalised `samples` array to the OSEA
detector. OSEA has its own internal bandpass filter and expects raw ADC values. The fix changed
the argument to `src_samples` (the unconverted, un-normalised signal). Without this, every
QRS-derived feature (RR interval, RR standard deviation, unknown-beat ratio, VPC ratio) was
computed from detector output that was itself operating on a distorted input.

### First beat skip in unknown-beat ratio (`045d190`)

In `beat_statistics()`, the ratio of unknown (`'Q'`-type) beats was computed over
`len(beats)`. The first beat detected by OSEA is frequently misclassified as `'Q'` due to
initialisation latency (the algorithm needs a few cycles to stabilise its template). The fix
changed the denominator to `len(beats) - 1` and skipped beat index 0, with an inline comment:
*"the first beat is often wrongly classified as 'Q' type."*

### Pyro4 distributed feature extraction (`a73defb`, `fc1b5ae`, `e3f0813`, `2badcd7`, `cc111a1`)

Feature extraction over 84,000 segments is expensive. Hong implemented an optional
master/slave architecture using Pyro4 (Python Remote Objects):

- A `Server` class exposes `next_segment()` and `add_results()` via the Pyro nameserver.
  Workers call `next_segment()` to pull the next unit of work, compute features locally, then
  return results via `add_results()`.
- `feature_extraction.py` gained `-l/--listen-port` (start as master) and `-m/--master-uri`
  (connect as slave) flags.
- A threading `Lock` protects the shared segment iterator in `next_segment()` so multiple
  workers cannot pull the same segment concurrently (`e3f0813`).
- A variable initialisation ordering bug in the generator caused `UnboundLocalError` on the
  first call (`2badcd7`). Fixed by moving `idx = -1` and `segment = None` outside the
  conditional.
- The master's `get_args()` method was added (`cc111a1`) so slaves fetch configuration
  (`resample_rate`, `update_features`) from the master rather than requiring matching CLI
  flags on every worker.
- Hostname resolution was fixed (`fc1b5ae`) to use
  `socket.gethostbyname(socket.gethostname())` after the Pyro daemon rejected the implicit
  default.

The Pyro layer strongly suggests Hong was running extraction jobs on a university HPC cluster
rather than a single workstation.

### Statistics for regression coefficients (`25a1e29`)

Added `statistics.py` — a utility to analyse the regression coefficients or feature
importances output by the classifiers, helping understand which of the 27 features carried
the most signal.

---

## Phase 2 — Gradient boosting fix and LZ complexity to C (29 May – 1 June 2016)

### Gradient boosting result output bug (`71fc6f8`)

`GradientBoostingClassifier` was being instantiated with `warm_start=True`, and its results
were not being written to the CSV feature-importance output. The fix added
`gradient_boosting` to the reporting branch alongside `random_forest` and `adaboost`.

### max_recall scorer (`8d9141f`)

Added `max_recall_score()` and `f1_binary_score()` as custom sklearn scorer objects for use
in grid-search cross-validation. The default `accuracy` scorer was misleading on the
heavily class-imbalanced dataset (NSR dominates); recall-oriented scoring better reflects
AED safety requirements where missing a shockable rhythm is catastrophic.

### LZ complexity rewritten in C with bug fix (`0022678`)

Lempel-Ziv complexity had been computed in Python using `bytearray` and string search.
Hong rewrote it in plain C using `memmem()` for substring matching and compiled it with
`-O3`. The C version is roughly 10× faster. Simultaneously he fixed an algorithmic error:
the initial value of the complexity counter `C(n)` was 0 instead of 1 (as specified by
Lempel and Ziv 1976), causing all LZ values to be slightly underestimated. The result was
stored in `vf_features.c` (later split to `vf_features_native.c` in 2026).

### IMF LZ extended to modes 2–4 (`b6c0abe`)

The original implementation computed LZ complexity only for IMF1 (lowest frequency mode)
and IMF5 (highest). Hong extended this to all five modes, adding IMF2_LZ, IMF3_LZ, IMF4_LZ
to the feature set, bringing the total to 27. Storage changed from a dynamically appended
list to a pre-allocated array indexed by `imf_mode - 1`, avoiding repeated allocations.

---

## Phase 3 — Recursive Feature Elimination experiments (2–4 June 2016)

Hong spent three days exploring whether the 27 features could be pruned without losing
performance — a reasonable question when deployment targets include AEDs with limited CPU.

### Initial RFE implementation (`5a35524`)

Added `-r / --rfe-n-features` flag to `vf_tests.py`. The RFE loop manually eliminated the
lowest-ranked feature at each iteration: for linear models it ranked by absolute value of
`coef_`; for tree models by `feature_importances_`. Two new scoring functions — `f1_macro`
and `f1_binary` — were added to give the grid-search a non-accuracy objective.

### RFE bug fix — wrong termination condition (`70cd55c`)

The loop `break` was placed outside the `if` that checked feature count, causing premature
termination. Fixed by moving `if len(surviving_features) <= args.rfe_n_features: break`
inside the conditional. Also renamed the flag from `--rfe_iters` to `--rfe-n-features` to
clarify it specifies a target count, not an iteration budget.

### RFE feature-rank output (`a104291`)

Separated the RFE loop into a training phase (eliminate on full dataset) and a reporting
phase (test at each step). Added `output_feature_ranks()` and `eliminated_features` tracking
so the CSV output showed which feature was dropped at each iteration and what CV score
remained.

### merge_reports.py and expanded SVC search space (`0cd49c8`)

Parallel cluster runs produce separate CSV files. `merge_reports.py` combines them with
automatic averaging. The linear SVC parameter grid was widened: `C` range extended to
`logspace(-6, 2)` (from `-5`) for finer granularity at small regularisation values.

### Filter-type feature selection (`d0b008a`)

Added `-k / --feature-rank` flag accepting a predefined elimination order. This lets a
previously computed RFE ranking be replayed without re-running the expensive RFE loop,
enabling faster ablation studies.

---

## Phase 4 — TCI rewrite (6 June 2016)

The Threshold Crossing Intervals feature (TCI, feature index 1) required the most sustained
debugging of any single feature — seven commits in a single day.

### The original bug (`8244ba6`)

The initial implementation used a windowed approach that produced incorrect pulse counts near
segment boundaries. Hong rewrote it as a pulse-tracking algorithm: first enumerate all
threshold-crossing pairs (rise + fall = one pulse), then for each 1-second window count how
many pulses fall within it. Variable names in the rewrite made the intent explicit:
`dist_to_prev_pulse`, `dist_to_segment_begin`.

### Boundary condition: skip first and last seconds (`d0098cb`)

Even with correct pulse tracking, the first and last 1-second windows of an 8-second segment
are unreliable — a pulse that started before the window began or ends after it will have
truncated `t1/t2/t3/t4` distances. The fix adds a guard that computes TCI only for interior
windows (indices 1 through 6 of 8).

### Final refactor (`e351c23`)

After three intermediate passes, the TCI function was reduced from ~127 lines to ~73 while
maintaining correctness. This is the version that reached the thesis.

---

## Phase 5 — AHA reporting (6–7 June 2016, same day as TCI)

On the same day as the TCI overhaul, Hong wired in AHA-compliant performance reporting.

### Three-class AHA output (`50400b5`, `dd4c86f`, `17bf424`)

`vf_tests.py` previously reported generic multi-class accuracy. The new output added explicit
AHA fields:

- `AHA_Se[shockable]` — sensitivity for the combined coarse VF + rapid VT class (AHA
  minimum: 90 % for coarse VF, 75 % for rapid VT)
- `AHA_Sp[non_shockable]` — specificity for NSR (AHA minimum: 99 %)
- Per-rhythm breakdown (sensitivity for shockable/intermediate; specificity for
  non-shockable)

The intermediate class is excluded from the AHA sensitivity/specificity calculation — it is
reported separately. This is the correct AHA methodology: only shock/no-shock decisions are
evaluated against binding thresholds.

`dd4c86f` split `output_aha_result()` from `output_multiclass_result()` after discovering
edge cases in result field merging. `17bf424` fixed a divide-by-zero that occurred when a
rhythm class had zero true-positive samples in a given test iteration.

---

## Phase 6 — Feature scaling bug and VFClassifier refactor (11–13 June 2016)

### Critical feature scaling bug (`b63ed27`)

`vf_tests.py` was calling `preprocessing.scale()` on the concatenated train+test dataset.
This is a classic data-leakage error: the scaler learns the mean and standard deviation of
the test set, which inflates reported performance. The fix switched to
`MinMaxScaler(feature_range=(-1, 1))` fit only on the training fold, then applied to the
test fold via `.transform()`. This is likely the most impactful correctness fix in the
post-commit history.

### VFClassifier class (`06253b8`)

`vf_classify.py` gained a `VFClassifier` class that encapsulates the full
train/evaluate/report pipeline:
- Stores `eliminated_features`, `cv_scores`, `estimators[]`, and `params[]` across RFE
  iterations
- `set_filter_fs_order()` allows chaining SVM-RFE output directly into a filter-type
  elimination order
- Supports multiple classifiers under a common interface (SVC-RBF, SVC-linear, logistic
  regression, random forest, gradient boosting, MLP variants)

A short-lived helper script `svm_rbf_filter_fs.py` was added then removed (`eb196e5`) once
its functionality was absorbed into the class.

### RFE iteration cap (`eb196e5`)

Without a cap, the RFE loop could eliminate features down to 1, which was
not useful for reporting. Added a minimum feature count to halt elimination.

### Accuracy added to CSV output (`89f6d71`)

The CSV report gained an overall accuracy column alongside per-class sensitivity/specificity,
giving a single-number summary alongside the AHA-specific metrics.

---

## Phase 7 — Statistics polish and correct test-score calculation (13–17 June 2016)

### Test score was calculated on the wrong data (`29cf65a`)

`VFClassifier` was computing the test-set score using the training scaler but not correctly
propagating the subset of selected features to the test evaluation step. After RFE, the
surviving feature indices were not being applied consistently. The fix ensured that both the
scaler and the feature mask from the training fold were applied identically to the test fold.

### statistics.py refactoring (`a60c8e0`, `6988489`, `f4c5d79`)

`statistics.py` and `error_analysis.py` produced per-class confusion breakdowns for
post-hoc analysis of misclassified segments. Three cleanup commits reduced duplication,
fixed calculation errors (incorrect denominator in sensitivity for the intermediate class),
and improved the output format. By 17 June the pipeline was stable enough to run the
100-iteration 70/30 experiments reported in the thesis.

---

## Phase 8 — EMD visualisation and submission (12 July 2016)

The final two commits from Hong (`924168f`, `e497922`) added `demo_emd.py` — a standalone
script that loads a WFDB record, extracts one 8-second segment, applies EMD via the bundled
PTSA library, and plots all five intrinsic mode functions with the original signal. The last
commit adjusts the font size of the plot title. This corresponds to the EMD illustration
figures in Chapter 3 of the thesis.

After this the repository went silent for almost **ten years**.

---

## Phase 9 — Revival and takeover (15 May 2026)

The repository was adopted by **Kostadin Bajalcaliev** with two commits:

**"Takeover"** (`972921e`) — modernisation for a 2026 Python/Cython environment:
- Added `Makefile` with `build` / `trace` / `debug` / `release` / `clean` / `check` targets
- Added `pyproject.toml` (PEP 517 build metadata)
- Rewrote `setup.py` to use `cythonize()` properly, compile `vf_features_native.c` as a
  separate source, and support the new build targets
- Extracted the inline C from the old generated `vf_features.c` into the dedicated
  `vf_features_native.c`
- Minor compatibility patches to `signal_processing.pyx`, `qrs_detect.pyx`, `pyeeg/__init__.py`
- Added `.vscode/` workspace configuration

**"Documents"** (`4b48055`) — converted the thesis PDF to Markdown (`docs/THESIS.md`) using an
automated tool, added raw output files (`THESIS.html`, `docs/xtract.py`), and seeded initial
documentation stubs (`CMDS.md`, `NOTES.md`, `PAPER.md`).

---

## Phase 10 — Documentation branch (15 May 2026, `develop`)

A `develop` branch was cut from the takeover point and used for documentation:

- `docs/README.md` → `EXECUTABLES.md` — full executable reference with CLI flags
- `docs/EXTRACT.md` → `FEATURE_EXTRACTION.md` — detailed analysis of `feature_extraction.py`
- `docs/FEATURES.md` → `FEATURE_INTERNALS.md` — all 27 feature algorithms with call tree
- `docs/DEBUG.md` — debugging guide: `make trace` for pdb/VS Code, `make debug` for gdb/cygdb
- `extract_one.py` — single-segment extraction script for interactive debugging
- Makefile `trace` and `debug` targets (Cython `linetrace=True` and `gdb_debug=True`)

---

## Phase 11 — Bug fixes and THESIS.md restoration (15–16 May 2026, `codex`)

A `fixes` branch (now renamed `codex`) merged the documentation work with:

**Code fixes** (`bc720e7`):
- `info.resample_rate` → `info.sampling_rate` — AttributeError on Python 3.x
- Per-record checkpointing in `feature_extraction.py` — crash recovery without reprocessing
- `sklearn.cross_validation` / `sklearn.grid_search` → `sklearn.model_selection` — removed
  in sklearn 0.20

**CLAUDE.md** (`b12ae5c`) — project context file for LLM-assisted development.

**THESIS.md restoration** (`fef51a8` – `b0bd897`) — artefacts from the automated PDF
conversion corrected: English-only title page, linked Markdown ToC, `<a id>` anchors for
figures and tables, 18 page-break word-split repairs, normalised References section, 50
reference anchors, 99 inline citation links, bold reference numbers.

**Docs reorganisation** (`2b88a84`) — scratch files removed; descriptive renames applied;
`SUMMARY.md` written from scratch as a structured reference.

---

## Branch summary

| Branch | Status | Purpose |
|--------|--------|---------|
| `master` | frozen | Hong's original thesis code, last touched July 2016 |
| `develop` | frozen | Documentation additions, May 2026 |
| `codex` | active | Bug fixes, modernisation, documentation quality |
