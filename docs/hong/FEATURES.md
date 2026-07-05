# feature_extraction.py — Detailed Analysis

> **Paths:** this describes the original thesis code, now under `hong/`
> (`hong/feature_extraction.py`, `hong/vf_features.pyx`, …); run from inside
> `hong/`. The pure-Python reimplementation is `vftx/` (see `vftx/PLAN.md`).

## Purpose

`feature_extraction.py` is the first mandatory step of the thesis pipeline.
It walks every ECG database, slices each continuous rhythm into non-overlapping
8-second segments, computes 27 features per segment, and writes the result to a
single binary file that all downstream scripts read.

---

## CLI Parameters

```
python feature_extraction.py [OPTIONS]
```

| Flag | Long form | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `-o` | `--output` | str | *(none)* | Output file path for the pickled feature data (e.g. `features/features_s8.dat`). **Required in practice**; omitting it makes the script compute everything and discard the result. |
| `-s` | `--segment-duration` | int | `8` | Segment length in seconds. The thesis uses 8. |
| `-r` | `--resample-rate` | int | `None` | Target sampling rate (Hz). If given, each segment is resampled before feature extraction. **Note: this flag is currently broken** — the resampling check reads `info.resample_rate` which does not exist on `SegmentInfo`; the correct attribute is `info.sampling_rate`. This will raise `AttributeError` in every worker when `-r` is used. |
| `-j` | `--jobs` | int | `-1` | Number of parallel worker processes. `-1` means "use all available CPU cores" (joblib default). |
| `-v` | `--verbose` | flag | False | Print per-segment feature values and detected beats to stdout. |
| `-u` | `--update-features` | str+ | `None` | Names of specific features to recompute. All other feature values are copied from an existing output file. Useful for re-running only a subset of the 27 features without reprocessing everything. Accepts one or more feature names from the canonical list (see Feature Names below). |
| `-c` | `--correction-file` | str | `None` | Path to a label correction file (e.g. `corrections_s8.txt`). Overrides incorrect rhythm labels in the original dataset annotations before segmentation. |
| `-l` | `--listen-port` | int | `None` | *Distributed mode — master.* Starts a Pyro4 RPC server on the given port. Slave processes connect to it to pull segments and return results. The master blocks until all slaves have finished, then writes the output file. |
| `-m` | `--master-uri` | str | `None` | *Distributed mode — slave.* URI of the Pyro4 master server (printed by the master at startup). The slave fetches segments from the master, computes features locally using joblib, and sends results back. A slave does **not** write any output file. |

### Typical invocations

```bash
# Single machine, all cores, default 8-second segments
python feature_extraction.py -o features/features_s8.dat -s 8

# With label corrections
python feature_extraction.py -o features/features_s8.dat -s 8 -c corrections_s8.txt

# Recompute only two features into an existing file
python feature_extraction.py -o features/features_s8.dat -u IMF1_LZ IMF2_LZ

# Distributed: start master on port 9090
python feature_extraction.py -o features/features_s8.dat -s 8 -l 9090

# Distributed: start slave pointing at master
python feature_extraction.py -m PYRO:obj_...@192.168.1.10:9090
```

---

## Where the WFDB Files Are Read From

### Record lookup — `file_lists/`

`vf_data.get_records(db_name)` reads the list of record IDs from:

```
file_lists/<db_name>.txt   (one record name per line, sorted)
```

The five databases used and their record counts:

| Database | File | Records |
|----------|------|---------|
| mitdb | `file_lists/mitdb.txt` | 48 |
| vfdb | `file_lists/vfdb.txt` | 22 |
| cudb | `file_lists/cudb.txt` | 35 |
| edb | `file_lists/edb.txt` | 90 |
| mghdb | `file_lists/mghdb.txt` | 250 (but only 7 are used — hardcoded) |

For mghdb the record list is **not** read from the file; instead `vf_data.DataSet.get_samples()` has a hardcoded list of 7 records: `mgh040`, `mgh041`, `mgh229`, `mgh236`, `mgh044`, `mgh046`, `mgh122`.

### Physical file location — WFDB path

Record IDs are assembled into paths like `"mitdb/100"`, `"vfdb/422"`, etc. and
passed to `wfdb_reader.read_signals()` and `wfdb_reader.read_annotations()`.
These call the WFDB C library functions `isigopen()` / `annopen()`, which resolve
file paths using the **`WFDB` environment variable** (the WFDB search path), not
the current working directory.

The WFDB library will look for:

```
$WFDB/mitdb/100.hea   (header)
$WFDB/mitdb/100.dat   (signal data)
$WFDB/mitdb/100.atr   (annotations)
```

The environment variable is set by running `setwfdb` (called in
`test_classifiers.sh` as `. /usr/local/bin/setwfdb`). If `WFDB` is not set the C
library falls back to a system default (often `/usr/database`). If neither
location exists, `isigopen()` returns 0 channels silently and the record produces
no segments.

**The datasets are not included in this repository.** They must be downloaded
separately from PhysioNet and configured via `WFDB`.

### Channel and annotator selection

| Database | Channel | Annotator | Notes |
|----------|---------|-----------|-------|
| mitdb | 0 | `atr` | Lead II is channel 0 |
| vfdb | 0 | `atr` | |
| cudb | 0 | `atr` | |
| edb | 0 | `atr` | |
| mghdb | 1 | `ari` | Lead II is channel 1 in mghdb |

After reading, `wfdb_reader` subtracts ADC zero and divides by gain to convert
raw integer ADC samples to millivolts (float64). All downstream code operates in
mV.

---

## Execution Flow

```
main()
  │
  ├── [slave mode: -m]  connect to Pyro4 master
  │     └── pull segments from master → parallel_extract_features() → push results back
  │
  ├── [master mode: -l]  start Pyro4 server, block until all slaves done
  │     └── output_results() → write output file
  │
  └── [default: single machine]
        ├── parallel_extract_features(args, load_all_segments(args))
        │     │
        │     │   joblib.Parallel(n_jobs, backend="multiprocessing")
        │     │   ┌─────────────────────────────────────────────────┐
        │     │   │  worker process N                               │
        │     │   │  extract_features(idx, segment, ...)            │
        │     │   │    → vf_features.extract_features(...)          │
        │     │   │    ← (idx, feature_array, segment_info)         │
        │     │   └─────────────────────────────────────────────────┘
        │     └── returns full list of (idx, features, info) tuples
        │
        └── output_results() → write output file
```

### Step 1 — Segment generation: `load_all_segments(args)`

A Python **generator** that yields `(idx, Segment)` tuples.

Internally calls `vf_data.DataSet.get_samples(segment_duration)`, which:

1. Iterates over the five databases in order: mitdb → vfdb → cudb → edb → mghdb.
2. For each database loads each record (`wfdb_reader.read_signals` + `read_annotations`).
3. Calls `record.get_artifact_free_rhythms()` which parses annotations and returns
   continuous artifact-free rhythm spans, excluding `NOISE`, isolated artifacts
   (`|`), and noisy quality sections (`~` with `sub_type != 0`).
4. For each rhythm, applies database-specific exclusion rules:
   - vfdb: skips `(VT` (no beat annotations → heart rate unknown)
   - cudb: skips `(N` (NSR labels in cudb are unreliable)
   - mghdb: includes only VF and VT rhythms
5. Slices each rhythm span into non-overlapping windows of `segment_size` samples
   (`= round(duration × sampling_rate)`). **The last partial window is always
   discarded** (`range(begin, end - segment_size, segment_size)`).
6. Applies any label corrections from `corrections_s8.txt`.
7. Yields a `Segment(info=SegmentInfo, signals=np.ndarray)`.

The generator is **lazy** — records are loaded one at a time as the joblib
dispatcher consumes the generator. No full dataset is ever held in memory at once
on the main process side.

### Step 2 — Parallel dispatch: `parallel_extract_features()`

```python
Parallel(n_jobs=args.jobs, backend="multiprocessing", max_nbytes=2048)(
    delayed(extract_features)(idx, segment, ...) for idx, segment in data_generator
)
```

- Uses the `"multiprocessing"` backend (separate OS processes, bypasses the GIL).
- `max_nbytes=2048` — numpy arrays larger than 2 KB are passed via memory-mapped
  temp files in the system temp directory (`/tmp/joblib_memmapping_*`). These are
  cleaned up by joblib automatically when the `Parallel` call completes.
- Results are accumulated by joblib into a list in the main process memory.
- **All results are kept in RAM until every segment is done.** There is no
  streaming write to disk.

### Step 3 — Single-segment entry point: `extract_features()`

**`feature_extraction.py:extract_features(idx, segment, resample_rate, update_features, verbose)`** — line 14

This is the function that runs inside each worker process. It:

1. Optionally resamples signals to `resample_rate` Hz (currently broken — see CLI notes).
2. Determines which features to compute: all 27, or only the subset named in `--update-features`.
3. Calls **`vf_features.extract_features(signals, sampling_rate, feature_names)`** (the Cython function in `vf_features.pyx:560`).
4. Attaches `detected_beats` and `amplitude` to the `SegmentInfo` object.
5. Prints `"idx: record_name/begin_time"` to stdout as a progress indicator.
6. Returns `(idx, features, info)`.

### Step 4 — Feature computation inside the worker: `vf_features.extract_features()`

**`vf_features.pyx:extract_features(src_samples, sampling_rate, features_to_extract)`** — line 560

This is the core Cython function. For each segment it:

1. **Preprocessing** (on a copy, raw `src_samples` preserved for amplitude and QRS):
   - Mean subtraction
   - Min-max normalization to [0, 1]
   - 5-order moving average (`signal_processing.moving_average`)
   - 1 Hz high-pass drift suppression (`signal_processing.drift_supression`)
   - 30 Hz Butterworth low-pass filter (`signal_processing.butter_lowpass_filter`)

2. **QRS detection** on the **raw** (unfiltered) signal:
   - Calls `qrs_detect.qrs_detect(src_samples, sampling_rate)`
   - Resamples to 200 Hz internally
   - Feeds the signal twice through the OSEA C detector: first pass to warm up
     the adaptive thresholds (waits for 8 beats), second pass to collect actual
     beat detections
   - Uses a threading lock because the OSEA C library is not reentrant

3. **Feature computation** (in canonical order):
   | # | Feature | Method |
   |---|---------|--------|
   | 0 | TCSC | 3-second Tukey-windowed threshold crossing count |
   | 1 | TCI | Average threshold crossing interval per 1-second window |
   | 2 | STE | Standard exponential crossings |
   | 3 | MEA | Modified exponential crossings (local maxima) |
   | 4 | PSR | Phase space reconstruction on 40×40 grid, 0.5 s delay |
   | 5 | HILB | Hilbert transform PSR, downsampled to 50 Hz |
   | 6 | VF | VF leak (shift by half-period of peak frequency) |
   | 7 | M | First spectral moment (0.5–9 Hz band, ≥5% of peak) |
   | 8 | A2 | Fraction of power in 0.7–1.4× peak frequency |
   | 9 | FM | Central frequency of power spectrum |
   | 10 | LZ | Lempel-Ziv complexity (C function in `vf_features.c`) |
   | 11 | SpEn | Sample entropy on last 1250 samples (last 5 s at 250 Hz) |
   | 12 | MAV | Mean absolute value over 2-second windows |
   | 13 | Count1 | Bandpass filter count, range 0.5×max to max (250 Hz assumed) |
   | 14 | Count2 | Bandpass filter count, range mean to max |
   | 15 | Count3 | Bandpass filter count, range mean±MD |
   | 16 | Amplitude | Max peak-to-peak amplitude on raw signal (mV) |
   | 17–21 | IMF1_LZ…IMF5_LZ | LZ complexity of each EMD intrinsic mode function |
   | 22 | RR | Mean RR interval (seconds) |
   | 23 | RR_Std | Std dev of RR intervals |
   | 24 | RR_CV | RR_Std / RR (coefficient of variation) |
   | 25 | UR | Fraction of unclassified beats (type Q) |
   | 26 | VR | Fraction of ventricular premature beats (type V) |

4. Returns `(feature_array, beats, amplitude)`.

### Step 5 — Write output: `output_results()`

Called once after **all** workers have finished. It:

1. Sorts results by `idx` to restore the original segment order.
2. If `--update-features` was given, loads the existing output file and merges
   the newly computed columns into it, preserving unchanged feature columns.
3. Opens the output file and writes two sequential `pickle.dump()` calls:
   - `pickle.dump(x_data, f)` — list of feature arrays, one per segment
   - `pickle.dump(x_data_info, f)` — list of `SegmentInfo` objects

The file format is a raw binary pickle stream; it is read back by
`vf_features.load_features(path)` using two matching `pickle.load()` calls.

---

## Why Nothing Is Written During a Long Run

**There are no intermediate files and no checkpointing.**

All computation results are accumulated in memory inside the joblib `Parallel`
return value. `output_results()` opens and writes the output file exactly once,
after `Parallel(...)` returns. If the process is killed before that point, the
entire run is lost.

With ~84,000 segments across five databases, each requiring:
- Two full passes of the OSEA QRS detector (serial C code, one segment at a time)
- EMD decomposition into 5 IMFs (the most expensive operation)
- Sample entropy via `pyeeg.samp_entropy` on 1250 samples

...a full run on a single machine takes several hours even with all cores.

The only visible progress during the run is the per-segment `print` in
`extract_features()`:

```
0: mitdb/100/0
1: mitdb/100/2048
2: mitdb/100/4096
...
```

These print out of order (parallel workers), but they confirm work is happening.

### Practical fix: add checkpointing

Until the code is modified to write checkpoints, the safest approach is to run
databases one at a time by temporarily editing `DataSet.get_samples()` or by
splitting the job across machines using the Pyro4 distributed mode (one machine
per database, each slave writing its own partial output file that is later merged).

---

## Temporary Files

| File | Creator | Purpose | Cleanup |
|------|---------|---------|---------|
| `/tmp/joblib_memmapping_*` | joblib | Memory-mapped numpy arrays for large segment data passed between processes when `max_nbytes=2048` is exceeded | Deleted automatically by joblib at the end of `Parallel(...)` |

No other temporary files are created. In particular:
- No partial output files
- No per-record caches
- No lock files

---

## Data Flow Diagram

```
file_lists/mitdb.txt           $WFDB/mitdb/100.hea
file_lists/vfdb.txt            $WFDB/mitdb/100.dat
file_lists/cudb.txt       →    $WFDB/mitdb/100.atr
file_lists/edb.txt             $WFDB/vfdb/422.*
corrections_s8.txt             ...
         │
         ▼
vf_data.DataSet.get_samples()         [main process]
  → Record.load() via wfdb_reader
  → get_artifact_free_rhythms()
  → non-overlapping 8-second Segment objects
  → apply label corrections
         │
         ▼  (generator, lazy)
joblib.Parallel(multiprocessing)
  ┌──────────────────────────┐
  │ worker 0                 │
  │  extract_features()      │  ← feature_extraction.py:14
  │    vf_features.extract_features()  ← vf_features.pyx:560
  │      preprocessing
  │      qrs_detect()
  │      27 feature functions
  │    → (idx, features, info)
  └──────────────────────────┘
  ... × N workers in parallel
         │
         ▼  (full list in RAM, sorted by idx)
output_results()               [main process, called ONCE at the end]
  → pickle.dump(x_data)
  → pickle.dump(x_data_info)
         │
         ▼
features/features_s8.dat       [output — written only when 100% complete]
```

---

## Known Issues

| Issue | Location | Effect |
|-------|----------|--------|
| `info.resample_rate` attribute does not exist | `feature_extraction.py:19` | `-r` / `--resample-rate` raises `AttributeError` in every worker; the correct attribute is `info.sampling_rate` |
| No checkpointing | `main()` | Killing the process loses all work |
| OSEA C library uses a global mutex | `qrs_detect.pyx:24` | The threading lock serializes QRS detection; within a single worker process all segments are processed sequentially by the QRS step. Parallelism comes from having multiple worker processes, not threads. |
| `mghdb` record list is hardcoded | `vf_data.pyx:305` | `file_lists/mghdb.txt` is read only by `get_records()` which is never called for mghdb; the 250-record file list is unused |
