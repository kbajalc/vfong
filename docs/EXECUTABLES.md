# Executables Reference

This document lists every runnable file in the repository, organized by role.
The research context is a 2016 NTU master thesis on AED-oriented arrhythmia
detection using handcrafted ECG features and machine learning (see `THESIS.md`
and `PAPER.md` for full background).

---

## A. Core Pipeline (main thesis workflow)

| File | Purpose |
|------|---------|
| `feature_extraction.py` | **Step 1.** Loads ECG segments from all datasets (mitdb, vfdb, cudb, edb, mghdb), extracts the 27 thesis features from each 8-second segment, and writes a pickled `.dat` file. Supports multicore parallelism (joblib) and distributed computing (Pyro4 master/slave). |
| `vf_tests.py` | **Step 2 — the core experiment.** Loads the precomputed feature file, trains classifiers with grid search + cross-validation, evaluates on held-out test sets, and writes per-iteration performance CSV reports plus optional per-sample error logs. Implements the 100-iteration random 70/30 train/test split described in the thesis. |
| `error_analysis.py` | **Step 3 — post-experiment.** Reads error log CSVs from `vf_tests.py`, filters samples by rhythm type / database / error threshold, prints per-sample feature values with clinical reference comments, and optionally plots ECG signals with interactive label correction. |

Minimal command flow:

```bash
make build                                              # compile Cython extensions
python feature_extraction.py -o features/features_s8.dat -s 8 -c corrections_s8.txt
python vf_tests.py -i features/features_s8.dat -t 100 -m svc_rbf -s f1_macro -o results.csv -e errors.csv
python error_analysis.py -i errors.csv -f features/features_s8.dat -t 0.5 -p
```

---

## B. Library Modules (not standalone — imported by the pipeline)

| File | Purpose |
|------|---------|
| `vf_classify.py` | Defines `VfClassifier`, AHA three-way labeling (shockable / intermediate / non-shockable), amplitude and heart-rate thresholds, and estimator factory for all supported model families. |
| `vf_eval.py` | Binary and multi-class evaluation metrics (sensitivity, specificity, precision, accuracy), custom AHA-oriented scorers, and the scorer registry used by `vf_tests.py`. |

---

## C. Visualization and Inspection Tools

| File | Purpose |
|------|---------|
| `vf_viewer.py` | Interactive ECG viewer. Loads a WFDB record segment, applies drift suppression and moving-average smoothing, and displays the raw signal, peak-to-peak amplitude measurement, FFT, and filtered signal via matplotlib. |
| `qrs_test.py` | QRS detector smoke-test. Loads a record segment, runs the OSEA-based detector, plots detected beat positions, and prints estimated heart rate. Also used by `make check`. |
| `inspect_record.py` | Minimal record inspector. Loads a WFDB record and prints all artifact-free rhythm annotations with begin/end sample indices. |
| `demo_emd.py` | EMD demo. Illustrates the Intrinsic Mode Function (IMF) extraction process on a synthetic multi-frequency signal. Also used by `make check`. |

Example usage:

```bash
python vf_viewer.py -r mitdb/111 -b 0 -d 8
python qrs_test.py -r vfdb/422 -b 385788 -d 8
python inspect_record.py mitdb 111
python demo_emd.py
```

---

## D. Dataset Curation Tools

| File | Purpose |
|------|---------|
| `asystole_check.py` | Scans all dataset segments for amplitude < 0.15 mV (the thesis operational asystole threshold) and optionally writes re-label corrections back to the correction file (`corrections_s8.txt`). |
| `statistics.py` | Prints the rhythm-type distribution of the dataset: segment counts per rhythm, unique patient counts, and human-readable rhythm descriptions. Accepts the correction file and/or a pre-computed feature file. |

Example usage:

```bash
python asystole_check.py -u corrections_s8.txt
python statistics.py -c corrections_s8.txt -s 8
python statistics.py -f features/features_s8.dat
```

---

## E. Results Post-processing Tools

| File | Purpose |
|------|---------|
| `merge_reports.py` | Merges multiple experiment result CSV files (e.g. from parallel runs on different machines) into one file, recalculating overall averages. |
| `merge_error_logs.py` | Merges multiple per-sample error log CSV files, concatenating test iterations and recomputing per-sample error rates and most-frequent predictions. |
| `coef_statistics.py` | Reads a logistic-regression or linear-SVM result CSV, averages feature coefficients across iterations, and ranks features by importance magnitude per AHA class. |

Example usage:

```bash
python merge_reports.py -i aha/svc_rbf*.csv -o aha/svc_rbf_merged.csv
python merge_error_logs.py -i aha/svc_rbf*_errors.csv -o aha/svc_rbf_errors_merged.csv
python coef_statistics.py -i aha/logistic_regression_results.csv
```

---

## F. Experimental / Research Tools

These scripts use deprecated scikit-learn APIs (`cross_validation`, `grid_search`)
and will need minor updates before they can run on current sklearn versions.

| File | Purpose |
|------|---------|
| `univariate_test.py` | Tests each feature individually using Pearson/Spearman correlation and random-forest regression, to assess single-feature discriminative power before multi-feature experiments. |
| `cluster.py` | K-means clustering exploration of the feature space. Useful for visualizing how well the 27 features separate rhythm classes without any classifier. |
| `pan_tompkins.py` | Pure-Python implementation of the Pan-Tompkins QRS detection algorithm, used as a baseline comparison against the OSEA C-based detector used in the thesis. |

---

## G. Shell Script Batch Runners

| File | Purpose |
|------|---------|
| `test_classifiers.sh` | **Primary batch experiment script.** Runs `vf_tests.py` for 100 iterations across all classifiers with multiple scoring functions. Assigns models to hosts by hostname for distributed multi-machine execution. Set `MODELS=svc_rbf` to override. |
| `test_mlp.sh` | MLP-specific batch runner. Tests `mlp1` and `mlp2` across multiple scoring functions and label schemes. References an older `vf_tests.py` argument API — may need updating. |
| `univariate_test.sh` | Calls `vf_tests.py` for each of 18 features individually across 3 classifiers (adaboost, logistic_regression, random_forest) to isolate single-feature performance. |
| `univariate_rf.sh` | Minimal random-forest univariate test variant. |

---

## H. Utility Scripts

| File | Purpose |
|------|---------|
| `docs/xtract.py` | Extracts base64-embedded images from the thesis HTML (`THESIS.html`), saves them to `docs/images/`, and rewrites the corresponding image links in `THESIS.md`. Run once after adding a new thesis HTML export. |
| `dblp/query_dblp.py` | Queries the DBLP academic publications database. Used during thesis research to survey the ECG / VF detection literature. |

---

## I. Cython Modules (compiled native extensions)

Built by `make build` or `python setup.py build_ext --inplace`.
These are not executable directly but are imported by the Python scripts above.

| File | Purpose |
|------|---------|
| `vf_features.pyx` | The 27-feature extraction engine implementing all thesis features: TCSC, TCI, STE, MEA, MAV, Count1–3, Amplitude, RR statistics, VF leak, spectral moments (M, A2, FM), LZ complexity, SpEn, EMD-IMF LZ (IMF1–5), PSR, and HILB. |
| `vf_data.pyx` | Dataset loading, non-overlapping 8-second ECG segmentation, rhythm annotation handling, per-record channel and annotator selection, and label correction machinery. |
| `qrs_detect.pyx` | Cython wrapper around the Hamilton/OSEA QRS detection C library (`osea/`). Used by `feature_extraction.py` and `qrs_test.py`. |
| `signal_processing.pyx` | Preprocessing functions: 5-order moving average, 1 Hz high-pass drift suppression, 30 Hz Butterworth low-pass filter, and peak-to-peak amplitude calculation. |
| `wfdb_reader.pyx` | Low-level WFDB record reader (C-level bindings for reading PhysioNet ECG `.dat`/`.hea` files). |

The C source for QRS detection lives in `osea/` and is linked during the
Cython build step.

---

## Recommended Reading Order

To understand the repository in the same order as the thesis workflow:

1. `README` — brief project overview
2. `docs/THESIS.md` — the full thesis (data, features, experiments, results)
3. `docs/PAPER.md` — annotated mapping of thesis claims to repo code
4. `vf_data.pyx` — dataset loading and segmentation
5. `feature_extraction.py` — feature generation driver
6. `vf_features.pyx` — feature implementations
7. `vf_classify.py` — AHA labeling and classifier setup
8. `vf_tests.py` — experiment and reporting harness
9. `test_classifiers.sh` — batch execution
