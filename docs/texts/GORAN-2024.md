# A Method to Detect Ventricular Fibrillation in Electrocardiograms

**Authors:** G. Temelkov, M. Gusev
**Affiliation:** Innovation DOOEL, Skopje; Sts. Cyril and Methodius University, North Macedonia
**Note:** Appears to be a conference/workshop paper (no journal/DOI visible). Filename: "Detection of VF in ECG v.3.pdf"

---

## Abstract

Algorithm for automatic VF detection in ECG records for wearable single-channel sensors. Based on frequency-domain analysis of ECG using FFT. Frequency peaks, energy distribution by band, and band ratios are used as features. F1 score of 0.77 and accuracy of 0.92 on VFDB (4-second segments).

**Keywords:** Electrocardiogram, Fast Fourier Transform, ECG, FFT, Ventricular Fibrillation, Ventricular Tachycardia, Ventricular Flutter

---

## Databases

- **MITDB** (MIT-BIH Arrhythmia Database): 48 records × 30 min, reference for normal rhythms
- **VFDB** (MIT-BIH Malignant Ventricular Arrhythmia Database): 22 half-hour recordings, VT/VFL/VF annotations
- **CUDB** (Creighton University VT Database): 35 × 8-min recordings, VT/VFL/VF annotations

All from PhysioNet. Lead II (first lead) analyzed.

---

## Method

### Sliding Window

Fixed-length window (SF = 4 seconds, overlap = SO steps). Sliding window produces segments covering the full record. Each segment decomposed from time domain to frequency domain via FFT.

**Frequency resolution:** FFT without zero-padding gives 1/T spacing → 0.25 Hz for 4s windows, 0.125 Hz for 8s windows.

### Frequency Bands

Three bands defined based on VF/VT spectral properties:
- **Low-Frequency Band (LFB):** 0.5–3.5 Hz
- **Middle-Frequency Band (MFB):** 3.5–8 Hz
- **High-Frequency Band (HFB):** 8–20 Hz

Note: <0.5 Hz = sinus arrest/DC; >20 Hz = not correlated to classification.

**Band energy density:** LFD, MFD, HFD = sum of all FFT intensities in each band.

**Key spectral observations:**
- VF dominant frequency: 4–6 Hz (in MFB range); can reach up to 500 BPM = 8.33 Hz
- VT/VFL: clear peak in MFB also, but more regular
- NSR: broad spectrum, primarily HFB due to QRS complex energy at 15–25 Hz

### Detection Features

**Ratio R:**
$$R = \frac{MBD}{HBD}$$

Clear separation between NSR (low R, HFB dominant) and VF/VT/VFL (high R, MFB dominant).

**Frequency peak analysis:**
- F = main frequency peak in MFB (1.66–8.33 Hz range for VT/VFL/VF)
- A(F) = peak amplitude
- E(F) = peak energy (sum of intensities in peak width)
- OD = outer energy in 1.5–24 Hz range
- R2 = OB / D(F) (outer-to-peak energy ratio)

### ML Classifier

KNN (K-Nearest Neighbors) outperforms Random Forest and Decision Tree on these features. SMOTE used for class imbalance. Training/test: 1/6 split per record (to prevent overlap). 10-fold cross-validation.

### Labeling

Segment labeled class V (ventricular-based arrhythmia) if completely within annotated VT/VFL/VF episode; class N if completely outside. Mixed-transition segments excluded from training but tested.

---

## Results

### Table I: KNN Performance on VFDB (4s window)

| Class | PPV | SEN | F1 | Support |
|-------|-----|-----|----|---------|
| N (normal) | 95.54 | 95.96 | 95.24 | 75736 |
| V (ventricular) | 80.12 | 74.60 | 77.26 | 16532 |
| **Overall accuracy** | | | | **92.13%** |

### Table II: KNN Performance on CUDB

| Class | PPV | SEN | F1 | Support |
|-------|-----|-----|----|---------|
| N | 89.43 | 91.51 | 90.46 | 13927 |
| V | 65.46 | 59.82 | 62.51 | 3748 |
| **Overall accuracy** | | | | **84.78%** |

### Table III: KNN Performance on MITDB (note: few V episodes)

| Class | PPV | SEN | F1 | Support |
|-------|-----|-----|----|---------|
| N | 99.90 | 99.99 | 99.94 | 21303 |
| V | 83.33 | 40.54 | 54.55 | 37 |
| **Overall accuracy** | | | | **99.88%** |

### Table IV: Comparison of Algorithms (SEN/SPC on MITDB and CUDB)

| Algorithm | MITDB SEN | MITDB SPC | CUDB SEN | CUDB SPC | Overall SEN | Overall SPC |
|-----------|-----------|-----------|----------|----------|-------------|-------------|
| TCI | 100.00 | 56.82 | 90.15 | 55.12 | 90.84 | 56.70 |
| STE | 95.16 | 40.54 | 40.54 | 88.79 | 40.54 | 94.70 |
| MEA | 80.94 | 70.66 | 81.53 | 66.44 | 81.49 | 70.35 |
| CPLX | 25.25 | 87.39 | 52.46 | 86.14 | 81.49 | 87.29 |
| SPEC | 58.99 | 99.83 | 37.18 | 98.82 | 38.69 | 99.76 |
| HILB | 97.84 | 98.52 | 75.96 | 89.68 | 77.47 | 97.88 |
| PSR | 95.32 | 99.04 | 75.35 | 91.46 | 76.73 | 98.49 |
| TCSC | 97.48 | 99.33 | 79.74 | 88.14 | 80.97 | 98.51 |
| **our** | **40.54** | **99.99** | **59.82** | **91.51** | **59.63** | **96.63** |

---

## Conclusions

- Algorithm designed for wearable sensors (high noise tolerance, FFT-based)
- Optimized for PPV (precision) and ACC rather than SEN — appropriate for noise-prone wearable context
- 4-second window is twice as fast to compute as 8-second window with negligible accuracy loss (<0.2%)
- F1 score outperforms comparison algorithms on combined metric
- Sensitivity is lower than conventional defibrillator-oriented algorithms (intentional tradeoff)
- Future work: improved beat detection, noise-robust classification

---

## Notes

- This paper uses a different optimization objective than traditional AED papers (F1 score vs. sensitivity/specificity tradeoff)
- It is referenced in VTA.ipynb notebook as contextual background
- The comparison table uses same HILB/PSR/TCSC numbers as other comparative papers
