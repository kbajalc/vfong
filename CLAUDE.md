# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline (2016 NTU master thesis) for detecting life-threatening cardiac arrhythmias (VF/VT) in 8-second ECG segments. The system reads PhysioNet WFDB records, extracts 27 handcrafted signal-processing features via Cython extensions, and trains/evaluates scikit-learn classifiers using AHA reporting rules (shockable / intermediate / non-shockable).

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
pip install Cython numpy scipy matplotlib joblib scikit-learn
# Pyro4 only needed for distributed mode
```

### Makefile targets

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

### Cython extensions (`setup.py`)

| Extension | Sources | Role |
|---|---|---|
| `wfdb_reader` | `wfdb_reader.pyx` + `libwfdb` | Thin Cython wrapper around the WFDB C library: `read_signals()`, `read_annotations()`, `read_info()` |
| `vf_data` | `vf_data.pyx` | `DataSet`, `Record`, `SegmentInfo`, `Segment`; dataset iteration and label correction; loads record lists from `file_lists/` |
| `qrs_detect` | `qrs_detect.pyx` + `osea20-gcc/*.c` + `libwfdb` | Wraps the OSEA QRS detector (EP Limited); resamples to 200 Hz internally; **not thread-safe** (uses a module-level lock) |
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

Inside `vf_features.pyx:extract_features()`:
1. Convert ADC units → mV (subtract ADC zero, divide by gain)
2. Drift suppression (high-pass at 0.5 Hz)
3. Normalise by standard deviation
4. Butterworth low-pass at 30 Hz
5. Moving-average smoothing (order 5)

Amplitude is computed separately on the raw mV signal (before normalisation) as `(max − min) / 2`.

QRS detection runs twice: a warm-up pass over the first 5 seconds (results discarded) then a full pass — this compensates for the OSEA detector's initialisation latency.

---

## Debugging

See `docs/DEBUG.md` for the full guide. Quick reference:

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

Applied on `codex`: `info.resample_rate` → `info.sampling_rate` AttributeError; per-record checkpointing; sklearn deprecated API updates (`cross_validation`/`grid_search` → `model_selection`); THESIS.md restoration; docs reorganisation. See `docs/HISTORY.md` for the full narrative.

---

## Docs

- `docs/HISTORY.md` — project history: original author background, phase-by-phase narrative of all commits from May 2016 to present
- `docs/SUMMARY.md` — thesis summary: AHA class definitions, dataset details, all 27 features with indices, thesis→code mapping table
- `docs/EXECUTABLES.md` — every runnable script with CLI flags and usage examples
- `docs/FEATURE_EXTRACTION.md` — detailed `feature_extraction.py` execution flow and all CLI flags
- `docs/FEATURE_INTERNALS.md` — all 27 feature functions: algorithms, call tree, preprocessing pipeline, `.dat` file format
- `docs/DEBUG.md` — debugging guide (pdb, gdb, VS Code, `make trace`/`debug`/`release`)
- `docs/THESIS.md` / `docs/THESIS.pdf` — full 2016 NTU thesis with figures, tables, formulas, clickable ToC and references
