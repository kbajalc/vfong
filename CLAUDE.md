# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline (2016 NTU master thesis) for detecting life-threatening cardiac arrhythmias (VF/VT) in 8-second ECG segments. The system reads PhysioNet WFDB records, extracts 27 handcrafted signal-processing features via Cython extensions, and trains/evaluates scikit-learn classifiers using AHA reporting rules (shockable / intermediate / non-shockable).

The repo also hosts a paper-writing project (`paper/`). `paper/PLAN.md` is the working plan
(refined scope, resolved decisions, the phases); `paper/PAPER.md` is the manuscript skeleton;
`paper/DRAFT.md` is the earlier broad draft kept as a source of text and references;
`paper/NOTES.md` is the takeover status report.

---

## Writing style

Prose we write in this repo (the paper in `paper/`, the docs, commit messages, any narrative
text) follows the rules in `STYLE.md`. The goal is human-friendly writing without mechanical
AI tells. The flagship rule: do not use em dashes (`—`); use a comma, a colon, parentheses,
or two sentences instead. Read `STYLE.md` before writing or editing paper or doc text, and
run its self-check before committing prose.

---

## Repository layout

```
hong/     Hong's original Cython implementation (the reference): all .pyx +
          vf_features_native.c, setup.py, Makefile, thesis Python drivers
          (feature_extraction.py, vf_tests.py, extract_one.py, …), .sh scripts,
          file_lists/, corrections_s8.txt, README.
vftx/     Clean pure-Python reimplementation of all 27 features (see below).
tests/    Agreement + unit tests (validate vftx/ against hong/).
ptsa/  pyeeg/  osea/   Vendored third-party deps, shared by hong/ and vftx/.
docs/     Project documentation.  setup_ref.py / setup_osea.py  build hong/'s
          reference extensions for the tests (kept at root, sources point into hong/).
```

Hong's build and thesis commands below run from inside `hong/` (e.g. `cd hong && make
build`, `python hong/feature_extraction.py …`). The `.so` files still build to the repo
root so `import vf_features` works with the root on `sys.path`.

---

## Build

### Prerequisites

**System:** The WFDB C library (`libwfdb-dev`) must be installed before building. On Debian/Ubuntu:
```bash
sudo apt-get install libwfdb-dev
# or build from source: https://physionet.org/content/wfdb/
```

**Python packages:**
```bash
pip install Cython numpy scipy matplotlib joblib scikit-learn wfdb
# Pyro4 only needed for distributed mode
# wfdb required for vftx/ pure-Python feature extraction (xqrs detector)
```

### Makefile targets

Run from inside `hong/` (`cd hong`), where the Makefile and `setup.py` live:

```bash
make build      # normal optimised release build (-O3)
make trace      # debug build for pdb/VS Code stepping through .pyx files (-O0 -g, linetrace)
make debug      # debug build for gdb/cygdb inspection of C internals (-O0 -g, gdb_debug)
make release    # restore optimised build after a debug session
make clean      # delete all .so, generated .c, .pyx.gdb, and build/
make check      # smoke test: demo_emd.py + qrs_test.py on mitdb/111
make fextract   # run full feature extraction (requires WFDB data + time)
```

After any `.pyx` edit, always rebuild before running or debugging.

### WFDB data path

Scripts use the `$WFDB` environment variable or the WFDB library's default search path. The original cluster setup sourced `/usr/local/bin/setwfdb`. Set it to point at a directory containing `mitdb/`, `vfdb/`, `cudb/`, `edb/`, `mghdb/` sub-directories.

```bash
export WFDB=/path/to/physionet/data
```

Record lists are in `file_lists/<db>.txt` (one record name per line, sorted).

---

## Full pipeline

```
WFDB records
    ↓  wfdb_reader.pyx  (C WFDB library → Python)
    ↓  vf_data.pyx      (DataSet / Record / SegmentInfo / Segment)
    ↓  feature_extraction.py  (-o features/features_s8.dat)
    ↓  vf_tests.py / vf_classify.py  (train + evaluate classifiers)
    ↓  CSV results in aha/
```

### 1. Feature extraction

```bash
python feature_extraction.py -o features/features_s8.dat -s 8 -j -1
```

Iterates all databases (mitdb, vfdb, cudb, edb, mghdb), splits each record into non-overlapping 8-second artifact-free segments, computes 27 features per segment using joblib multiprocessing. Writes a pickle containing two sequential objects: `x_data` (`list[array.array('d', [27 floats])]`) and `x_data_info` (`list[SegmentInfo]`).

Per-record checkpointing is enabled by default when `-o` is given. Checkpoint dir: `<output>.ckpt/<db>__<record>.pkl`. Resume a crashed run simply by re-running the same command. Use `--no-checkpoint` to disable.

Key flag: `-u FEATURE_NAME ...` recomputes only the named features, merging results back into an existing `.dat` file (used to update one feature without rerunning everything).

### 2. Classifier evaluation

```bash
python vf_tests.py -i features/features_s8.dat -m logistic_regression -s custom -o results.csv
```

Available models: `logistic_regression`, `random_forest`, `adaboost`, `gradient_boosting`, `svc_linear`, `svc_poly`, `svc_rbf`, `mlp1`, `mlp2`.

`test_classifiers.sh` runs the full grid across models and scoring functions, writing CSV results to `aha/`.

### 3. Single-segment debug extraction

```bash
python extract_one.py -r mitdb/100 -n 0          # segment by index
python extract_one.py -r vfdb/422 -b 385788       # segment by begin sample
python extract_one.py -r mitdb/100                # list all segments
```

No parallelism, no joblib. Set a breakpoint on the `vf_features.extract_features()` call in `run_extraction()` to step through the full pipeline.

---

## Code architecture

### Cython extensions (`hong/setup.py`)

| Extension | Sources | Role |
|---|---|---|
| `wfdb_reader` | `wfdb_reader.pyx` + `libwfdb` | Thin Cython wrapper around the WFDB C library: `read_signals()`, `read_annotations()`, `read_info()` |
| `vf_data` | `vf_data.pyx` | `DataSet`, `Record`, `SegmentInfo`, `Segment`; dataset iteration and label correction; loads record lists from `file_lists/` |
| `qrs_detect` | `qrs_detect.pyx` + `osea/*.c` + `libwfdb` | Wraps the OSEA QRS detector (EP Limited); resamples to 200 Hz internally; **not thread-safe** (uses a module-level lock) |
| `vf_features` | `vf_features.pyx` + `vf_features_native.c` | All 27 feature computations; `extract_features(signals, sampling_rate, feature_names_set)` is the main entry point |
| `signal_processing` | `signal_processing.pyx` | Shared DSP utilities used by `vf_features`: drift suppression, Butterworth filter, moving average |

`vf_features_native.c` implements `lempel_ziv_complexity` and `imf_lempel_ziv_complexity` in plain C (no Cython).

### Pure Python modules

- `vf_classify.py` — classifier wrappers, AHA label logic (`initialize_aha_labels`, VF coarse/fine split, VT rapid/slow split), estimator definitions
- `vf_eval.py` — `BinaryClassificationResult`, `MultiClassificationResult`, custom scorers
- `vf_tests.py` — CLI driver for cross-validated classifier experiments; writes per-iteration CSV rows
- `feature_extraction.py` — joblib parallel driver; per-record checkpointing; `output_results()` writes the `.dat` pickle
- `pyeeg/__init__.py` — bundled PyEEG library (sample entropy `samp_entropy`)
- `ptsa/` — bundled PTSA library (empirical mode decomposition `emd`)

### `vftx/` — pure-Python feature extraction package

`vftx` = **V**entricular **T**achyarrhythmias **F**eatures. A clean reimplementation of
all 27 features with no Cython or C dependencies. The Cython extensions (in `hong/`) are
kept intact as the reference; `vftx/` is a parallel implementation.

| Module | Role |
|---|---|
| `vftx/types.py` | `SegmentConfig`, `PreprocessedSignal`, `Features` (all 27 fields), `QRSDetector` protocol |
| `vftx/preprocessing.py` | 5-step signal conditioning pipeline |
| `vftx/extract.py` | `extract_features(signal_mv, cfg, qrs_detector?)` — main entry point |
| `vftx/wfdb_detector.py` | `WfdbXqrsDetector` — concrete `QRSDetector` using `wfdb.processing.xqrs_detect`; all beats typed `'N'` (UR=VR=0) |
| `vftx/<feature>.py` | One file per feature or natural group (`imf_lz.py` for IMF1–5, `qrs_features.py` for RR/UR/VR) |
| `vftx/_count_helpers.py` | Shared IIR bandpass filter for Count1–3 |
| `vftx/PLAN.md` | Implementation plan, difficulty assessment, known gotchas |

Dependencies: `numpy`, `scipy`, `wfdb` (xqrs detector), `ptsa/` (bundled, default EMD
backend for IMF features). The EMD backend is selectable via
`SegmentConfig.complexity.emd_backend`: `"ptsa"` (default, matches the reference) or
`"pyemd"` (the pip-installable `EMD-signal` package — a valid but different EMD, so IMF_LZ
[17–21] diverge from the reference by up to ~26%).

#### Validating `vftx/` against the reference (Phase 4 — complete)

Status and full findings: `vftx/PLAN.md` "Phase 4" section. **26/27 features match the
reference bit-for-bit; SpEn is correct but unmatchable (non-deterministic reference).**

Run the hermetic suite (no network/libwfdb/Cython needed — uses cached fixtures):

```bash
conda activate dev                            # Python 3.14
pip install pytest                            # one-time
pytest tests/                                 # 146 passed, 5 xfailed (SpEn ×5)
```

To regenerate fixtures or the side-by-side table (needs the reference build + network):

```bash
pip install Cython setuptools wfdb            # one-time
python setup_ref.py build_ext --inplace       # builds signal_processing + vf_features .so
python tests/gen_fixtures.py                   # refresh tests/data/fixtures.{npz,json}
python tests/compare_reference.py              # side-by-side reference vs vftx table
```

Key mechanism — **`SegmentConfig.reference_bug_compat`** (default `False`): `vftx/` is a
clean/correct reimplementation by default; the flag reproduces reference bugs so the suite
can bit-match. Bugs found and gated: TCSC multiplies overlapping windows of the shared
signal by a Tukey window *in place* (corrupts all downstream features — the master cause);
vf_leak uses `argmax` on the complex FFT. Config fix (clean): `highpass_hz` 0.5→1.0.
SpEn: reference `pyeeg.samp_entropy` uses `as_strided` on non-contiguous input → wrong +
non-deterministic; vftx's SpEn is correct and validated independently in
`tests/test_sample_entropy.py`. Also: LZ76 rewritten with `bytes.find` (bit-identical,
~150× faster). `tests/_refstub/qrs_detect.py` stubs OSEA so `vf_features` imports without
libwfdb (reference QRS features come out 0). The real OSEA detector is also buildable
without libwfdb via `python setup_osea.py build_ext --inplace` (WFDB headers stubbed in
`tests/_osea/`); `tests/test_qrs_osea.py` then loosely validates xqrs vs OSEA (skips if the
`.so` isn't built).

### Key data structures

**`SegmentInfo`** (defined in `vf_data.pyx`): `db_name`, `record_name`, `begin_time` (samples), `end_time`, `sampling_rate`, `rhythm` (WFDB annotation string e.g. `"(VF"`, `"(N"`). After extraction, two fields are added dynamically: `detected_beats` (`list[(sample, beat_type_str)]`) and `amplitude` (`float64`, peak-to-peak mV).

**Feature array**: `array.array('d', [27 floats])`. Canonical index order:
`TCSC(0) TCI(1) STE(2) MEA(3) PSR(4) HILB(5) VF(6) M(7) A2(8) FM(9) LZ(10) SpEn(11) MAV(12) Count1(13) Count2(14) Count3(15) Amplitude(16) IMF1_LZ(17)…IMF5_LZ(21) RR(22) RR_Std(23) RR_CV(24) UR(25) VR(26)`

**Pickle `.dat` format**: two sequential `pickle.dump` calls — `x_data` first, `x_data_info` second. Load with `vf_features.load_features(path)`.

### Database conventions

| DB | channel | annotator |
|---|---|---|
| mitdb, vfdb, cudb, edb | 0 | `atr` |
| mghdb | 1 | `ari` |

mghdb records are hardcoded in `vf_data.pyx` (7 records: mgh040, mgh041, mgh229, mgh236, mgh044, mgh046, mgh122); all others come from `file_lists/<db>.txt`.

### `extract_features()` preprocessing pipeline

ADC→mV conversion (subtract ADC zero, divide by gain) happens before this function is called.
Inside `vf_features.pyx:extract_features()` the pipeline is (in order):

1. **Mean subtraction** — `x − mean(x)`
2. **Min-max normalisation** → [0, 1]: `(x − min) / (max − min)`
3. **Moving-average smoothing** — order 5, via convolution
4. **Drift suppression** — 1 Hz high-pass using a custom 1-pole bilinear IIR (`filtfilt`)
5. **Butterworth low-pass** — 30 Hz, order 5 (`filtfilt`)

After step 4 the signal is zero-mean (the high-pass removes DC).

**Amplitude** [16] is computed on the raw mV signal via a separate call to
`signal_processing.pyx:get_amplitude()`, which applies the same steps 3–5 then finds
the largest peak-to-valley difference using `argrelmax`/`argrelmin`.
It is **not** simply `(max − min) / 2`.

**QRS detection** (Cython pipeline only) runs twice on the raw mV signal: a 5-second
warm-up pass (results discarded) then a full pass — this compensates for OSEA's
initialisation latency.  The `vftx/` package uses `WfdbXqrsDetector` instead.

---

## Debugging

See `docs/hong/DEBUG.md` for the full guide. Quick reference:

```bash
make trace    # enables pdb / VS Code breakpoints inside .pyx files
make debug    # enables gdb / cygdb for C-level inspection
make release  # restore speed for production runs
```

VS Code `launch.json` already has three configurations for `extract_one.py` and `feature_extraction.py`. **`justMyCode: false` is required** to step into `.pyx` frames. Required extension: **Cython+** (`cython-collective.cython-plus`).

---

## Branches

| Branch | Purpose |
|---|---|
| `master` | Original thesis code — frozen at Hong's last commit (July 2016) |
| `develop` | Documentation additions (`docs/`) — frozen |
| `codex` | Active development: bug fixes, modernisation, documentation |

Applied on `codex`: `info.resample_rate` → `info.sampling_rate` AttributeError; per-record checkpointing; sklearn deprecated API updates (`cross_validation`/`grid_search` → `model_selection`); THESIS.md restoration; docs reorganisation; `vftx/` pure-Python feature extraction package (all 27 features). See `docs/hong/HISTORY.md` for the full narrative.

---

## Docs

- `docs/hong/HISTORY.md` — project history: original author background, phase-by-phase narrative of all commits from May 2016 to present
- `docs/hong/SUMMARY.md` — thesis summary: AHA class definitions, dataset details, all 27 features with indices, thesis→code mapping table
- `docs/hong/EXECUTABLES.md` — every runnable script with CLI flags and usage examples
- `docs/hong/FEATURES.md` — detailed `feature_extraction.py` execution flow and all CLI flags
- `docs/hong/INTERNALS.md` — all 27 feature functions: algorithms, call tree, preprocessing pipeline, `.dat` file format
- `docs/hong/DEBUG.md` — debugging guide (pdb, gdb, VS Code, `make trace`/`debug`/`release`)
- `docs/hong/THESIS.md` / `docs/hong/THESIS.pdf` — full 2016 NTU thesis with figures, tables, formulas, clickable ToC and references
