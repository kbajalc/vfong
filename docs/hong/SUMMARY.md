# Thesis Summary and Code Mapping

**Thesis:** "Detecting Life-Threatening Arrhythmia with Machine Learning Algorithms"  
**Author:** Jen-Yee Hong, National Taiwan University, July 2016  
**Full text:** `docs/hong/THESIS.md`

---

## 1. What This Is

A research pipeline for AED (automated external defibrillator) arrhythmia detection. The core claim: a multi-class SVM classifier trained on 27 handcrafted ECG features can satisfy all AHA performance requirements for shock/no-shock decision-making, something most prior binary-classification VF studies failed to demonstrate because they did not follow AHA reporting rules.

**Headline results (SVM-RBF, 100-iteration 70/30 split):**

| Metric | Value |
|--------|-------|
| Sensitivity (shockable) | 93.21 % |
| Specificity (non-shockable) | 99.88 % |
| Precision (shockable) | 89.28 % |

The AHA minimum thresholds are: coarse VF sensitivity > 90 %, rapid VT sensitivity > 75 %, NSR specificity > 99 %, all other non-shockable > 95 %. All were met.

---

## 2. AHA Three-Class Label Scheme

| Class | Rhythms | AHA threshold |
|-------|---------|---------------|
| Shockable | Coarse VF (amplitude > 0.2 mV) + rapid VT (HR > 180 BPM) | > 90 % / 75 % sensitivity |
| Intermediate | Fine VF (amplitude ≤ 0.2 mV) + slow VT | Report only |
| Non-shockable | NSR, AF, SB, SVT, AV blocks, IVR, asystole, etc. | > 95–99 % specificity |

Intermediate rhythms are the known hard case: SVM-RBF only reached 41 % sensitivity there, mainly because fine VF and coarse VF differ only in amplitude after normalisation.

Asystole is defined operationally: peak-to-peak amplitude < 0.15 mV. These segments are excluded from ML classification entirely.

---

## 3. Datasets

Five PhysioNet databases, totalling **84,027 non-overlapping 8-second segments** from 296 records:

| DB | Key content | Notes |
|----|-------------|-------|
| `mitdb` | NSR, many arrhythmias | Lead II channel 0, annotator `atr` |
| `vfdb` | VF, VFL; some VT excluded | VT excluded — no beat annotations for HR |
| `cudb` | Coarse VF, AF; NSR excluded | CUDB NSR is unreliable (lumps all non-VF) |
| `edb` | Diverse non-shockable rhythms | Main source of non-shockable variety |
| `mghdb` | Coarse VF, fine VF, rapid VT | Lead II channel 1, annotator `ari`; 7 records hardcoded |

747 labels were manually corrected; 87 samples excluded for severe artifacts. Details in `corrections_s8.txt`.

---

## 4. Preprocessing Pipeline (per 8-second segment)

1. Mean subtraction
2. Normalisation by standard deviation *(skipped for amplitude feature)*
3. 5-order moving average
4. Drift suppression — 1 Hz high-pass filter
5. 30 Hz zero-phase Butterworth low-pass filter

Amplitude (peak-to-peak mV) is computed on the raw signal before normalisation, using `scipy` extrema detection (`argrelmax`/`argrelmin`).

QRS detection (Patrick Hamilton OSEA detector) runs twice: a 5-second warm-up pass (discarded) then the full segment, to overcome OSEA initialisation latency.

---

## 5. The 27 Features

| Index | Name | Category | Key detail |
|-------|------|----------|------------|
| 0 | TCSC | Time domain | Both ±20 % thresholds; Tukey-windowed; 3 s moving window, 1 s step |
| 1 | TCI | Time domain | 20 % threshold; averaged over eight 1-second windows |
| 2 | STE | Time domain | Global-peak exponential, τ = 3 s |
| 3 | MEA | Time domain | Local-peak exponential, τ = 0.2 s |
| 4 | PSR | Phase space | Time-delayed method, 0.5 s delay; proportion of 40×40 grid cells visited |
| 5 | HILB | Phase space | Hilbert transform phase shift; same 40×40 grid |
| 6 | VF | Frequency | VF leak (Kuo & Dillman 1978); Hamming-windowed FFT |
| 7 | M | Frequency | Spectral centroid-like parameter (Barro 1989) |
| 8 | A2 | Frequency | Energy ratio around peak frequency band |
| 9 | FM | Frequency | Central frequency (spectral mass centre, Dzwonczyk 1990) |
| 10 | LZ | Complexity | Lempel-Ziv complexity of binary-thresholded ECG |
| 11 | SpEn | Complexity | Sample entropy; last 1250 pts at 250 Hz (= last 5 s) |
| 12 | MAV | Time domain | Mean absolute value with 2 s sliding window |
| 13 | Count1 | Time domain | Samples in 50–100 % of max amplitude range |
| 14 | Count2 | Time domain | Samples above mean amplitude |
| 15 | Count3 | Time domain | Samples within mean ± mean-deviation band |
| 16 | Amplitude | Time domain | Peak-to-peak mV (raw, pre-normalisation) |
| 17–21 | IMF1_LZ – IMF5_LZ | EMD | LZ complexity of each EMD intrinsic mode function |
| 22 | RR | QRS-derived | Mean RR interval (ms) |
| 23 | RR_Std | QRS-derived | Std of RR intervals |
| 24 | RR_CV | QRS-derived | Coefficient of variation = RR_Std / RR |
| 25 | UR | QRS-derived | Unknown-beat ratio |
| 26 | VR | QRS-derived | VPC-beat ratio |

Count1–3 require 250 Hz; segments are resampled before these features. SpEn also requires 250 Hz resampling.

---

## 6. Classifier and Evaluation

**Primary model:** SVM with RBF kernel (`sklearn.svm.SVC`, libsvm backend).

**Compared against:** linear SVM, logistic regression. Both met all AHA thresholds too, with slightly lower precision — important for AEDs with limited compute.

**Evaluation:** Random 70/30 train/test split, 100 iterations, average reported. Class weighting used during training due to heavily imbalanced dataset (NSR dominates). Grid search (C, γ) with 5-fold cross-validation on training fold to select parameters.

**Limitation:** many samples come from the same patients. AHA requires one sample per patient per rhythm; by that rule, the study had insufficient patient counts for fine VF (10 samples, 4 patients; AHA requires 25).

---

## 7. Thesis → Code Mapping

Source files below are the original thesis implementation, now under `hong/`
(e.g. `hong/vf_features.pyx`). The pure-Python reimplementation is in `algo/`.

| Thesis section | Primary source file |
|----------------|---------------------|
| Datasets, segmentation, labels | `vf_data.pyx` |
| Preprocessing | `signal_processing.pyx` |
| QRS detection | `qrs_detect.pyx` + `osea/` |
| All 27 features | `vf_features.pyx` + `vf_features_native.c` |
| Feature extraction driver | `feature_extraction.py` |
| AHA label logic, estimators | `vf_classify.py` |
| Experiment loop, CSV output | `vf_tests.py` |
| Performance metrics | `vf_eval.py` |

Repo is broader than the thesis alone: it adds ensemble models (random forest, adaboost, gradient boosting), MLP variants, and a batch experiment harness (`test_classifiers.sh`) not described in the thesis.

---

## 8. Key Design Decisions (clinically grounded)

- **8-second segments** — justified by prior studies and preliminary tests; matches AHA sample definition.
- **Segmentation before preprocessing** — simulates real AED scenario where the device only ever sees one segment.
- **Multi-class (3-way) not binary** — required to comply with AHA reporting; most prior work only did VF vs non-VF.
- **Feature mixing** — threshold-crossing features capture rapidity and morphology; complexity/phase-space features capture irregularity; EMD-LZ features help VF vs VT separation.
- **Open dataset + open code** — explicit thesis goal: reproducibility and benchmarking for future AED research.

---

## 9. Known Weak Points

- Intermediate class (fine VF, slow VT) sensitivity is poor (~41 %); both rhythms are inherently ambiguous near their amplitude/rate thresholds.
- Insufficient unique patient counts for fine VF and other VT.
- Artifact-free assumption: AHA goals are for clean segments; real AEDs encounter CPR artifacts.
- CUDB NSR exclusion (all CUDB NSR treated as unreliable) is not handled by many comparison studies, making direct performance comparison difficult.
- Annotation errors in public datasets are partially corrected but not exhaustively.

---

## 10. Suggested Reading Order

1. `docs/hong/SUMMARY.md` — this file: overall context and code mapping
2. `docs/hong/THESIS.md` — full thesis with figures, tables, formulas
3. `vf_data.pyx` — dataset loading, segmentation, label correction
4. `feature_extraction.py` — parallelised feature generation driver
5. `vf_features.pyx` — all 27 feature implementations
6. `vf_classify.py` — AHA labeling and estimator setup
7. `vf_tests.py` — experiment loop and CSV reporting
8. `docs/hong/EXECUTABLES.md` — every runnable file with usage examples
