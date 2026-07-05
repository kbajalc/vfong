# VFPred: A Fusion of Signal Processing and Machine Learning Techniques in Detecting Ventricular Fibrillation from ECG Signals

**Authors:** Nabil Ibtehaz, M. Saifur Rahman, M. Sohel Rahman
**Preprint:** arXiv:1807.02684v3, November 2018
**Affiliation:** Department of CSE, BUET, Dhaka, Bangladesh

---

## Abstract

VFPred combines Empirical Mode Decomposition (EMD) with Discrete Fourier Transform (DFT) for feature extraction, and a Support Vector Machine (SVM) for classification. Key property exploited: VF class ECG signals lack QRS complexes. Achieves Sensitivity = 99.99%, Specificity = 98.40% on VFDB+CUDB, even from 5-second signals. Outperforms prior works including traditional signal processing algorithms.

**Index terms:** ECG, Empirical Mode Decomposition, Heart Arrhythmia, Support Vector Machine, Ventricular Fibrillation

---

## Key Insight

**QRS complex absence in VF:** VF class ECG signals have no QRS complexes; all other rhythms do. This asymmetry in oscillatory envelope structure is exploited:

- Non-VF signals: QRS complexes create asymmetric upper/lower envelopes → 1st IMF captures these high-frequency oscillations, deviating from original signal (IMF₁_similarity ≈ 0.002; R_similarity ≈ 0.918)
- VF signals: Symmetric envelopes (no QRS) → 1st IMF closely follows original signal (IMF₁_similarity ≈ 0.995; R_similarity ≈ 0.017)

---

## Datasets

- **VFDB** (MIT-BIH Malignant Ventricular Arrhythmia Database): 22 records, 30-min each, 250 Hz, VT/VFL/VF episodes
- **CUDB** (Creighton University VT Database): 35 records, 8-min each, 250 Hz, 12-bit, 10V range, VT/VFL/VF episodes
- Both from PhysioNet; only channel 1 used from VFDB

**Sliding window:** T_e = 2, 5, or 8 seconds, step = 1 second, excluding noise-annotated segments.

**Class imbalance:** ~9% of all segments are VF class. Total: ~5320 VF, ~56,000 Not-VF (for T_e = 5s).

---

## Algorithm Pipeline

### 1. Signal Preprocessing and Filtering
1. Subtract mean (zero-mean signal)
2. Moving average filter, order 5 (removes interspersions/muscle noise)
3. High-pass filter, f_c = 1 Hz (baseline drift suppression)
4. Low-pass Butterworth, order 12, f_c = 20 Hz (removes HF content)

### 2. Empirical Mode Decomposition

Signal decomposed into IMF components (highest to lowest frequency) + Residue R:
$$x(n) = IMF_1(n) + IMF_2(n) + R(n)$$

(Truncated to first 2 IMFs + residue for efficiency.)

**Noise level handling (NLCR criterion):**
- Noise level: V_n = 0.05 × max(x(n))
- n_L = samples of IMF₁ within [−V_n, V_n]
- NLCR = Σ_{n_L} IMF₁²(n) / Σ_{n_L} x²(n)
- If NLCR ≤ β = 0.02 → use IMF = IMF₁ (clean signal)
- Otherwise → IMF = IMF₁ + IMF₂ (IMF₁ is noise, merge with IMF₂)

### 3. DFT Feature Extraction

Instead of cosine similarity in time domain (insufficient), frequency-domain similarity is computed:

$$IMF_{similarity}[i] = \frac{Signal_{DFT}[i] \cdot IMF_{DFT}[i]}{||Signal_{DFT}|| \cdot ||IMF_{DFT}||}$$, for 1 ≤ i ≤ N

$$R_{similarity}[i] = \frac{Signal_{DFT}[i] \cdot R_{DFT}[i]}{||Signal_{DFT}|| \cdot ||R_{DFT}||}$$, for 1 ≤ i ≤ N

Most discriminative frequency components: **1–5 Hz range**.

Feature vector = [IMF_similarity[1..N], R_similarity[1..N]] (dimension: 2N).

### 4. Feature Selection via Random Forest

750 decision trees rank feature importance. Top 24% of features selected (~plateau at 16%, slight improvement to 24%). This reduces dimensionality and improves generalization.

### 5. SVM Classification

- Kernel: Gaussian RBF, K(x,x') = exp(−γ||x−x'||²)
- Optimal parameters for T_e=5s: C=100, γ=45
- Training data: 3000 VF + 5000 non-VF samples (handles imbalance)
- SMOTE oversampling to balance VF class in training

---

## Evaluation

**Metrics:**
- Sensitivity = TP/(TP+FN)
- Specificity = TN/(TN+FP)
- Accuracy = (TP+TN)/(TP+FP+TN+FN)
- **G-Mean Accuracy** = √(Sensitivity × Specificity) — preferred for imbalanced datasets

**10-fold cross-validation** (both standard and stratified):

| Data | Sensitivity (%) | Specificity (%) | Accuracy (%) | G-Mean Acc (%) |
|------|-----------------|-----------------|--------------|----------------|
| Training | 100 | 100 | 100 | 100 |
| Test (10-fold CV) | 99.988 ± 0.016 | 98.401 ± 0.19 | 99.194 ± 0.092 | 99.191 ± 0.095 |
| Test (stratified 10-fold) | 99.992 ± 0.01 | 98.395 ± 0.187 | 99.194 ± 0.092 | 99.190 ± 0.096 |

**Effect of window length T_e:**
- T_e = 2s: G-Mean 88.244%
- T_e = 5s: G-Mean 91.966%
- T_e = 8s: G-Mean 95.101%
- Each 3s increase improves G-Mean by ~4%

---

## Comparison with Other Methods (Table 2)

| Algorithm | Se (%) | Sp (%) | Acc (%) | G-Mean (%) |
|-----------|--------|--------|---------|------------|
| Atienza et al. [14] (SVM) | 74.1 | 94.7 | - | 83.77 |
| Qiao et al. [16] (SVM+GA) | 96.2 | 96.2 | 96.3 | 96.2 |
| Verma et al. [17] (RF, 5s) | 95.04 | 94.78 | 94.79 | 94.91 |
| Mi et al. [18] (SVM+DR) | 92.396 | 99.121 | 99.350 | 95.70 |
| Asl et al. [19] (GDA+SVM) | 95.77 | 99.40 | 99.16 | 97.57 |
| **VFPred** | **99.988** | **98.401** | **99.194** | **99.191** |

Note on signal-processing-only methods tested on entire dataset (not pre-selected):
- HILB: Se=79.73, Sp=98.83, Ac=98.40
- PSR: Se=78.07, Sp=99.01, Ac=98.53

---

## Key Conclusions

1. Combining signal processing (EMD+DFT) with ML (SVM) substantially outperforms either alone
2. The 1–5 Hz frequency band is the most discriminative for VF vs. non-VF
3. Class imbalance must be addressed; SMOTE is preferred over random oversampling
4. Short windows (5s) sufficient for high accuracy — enables faster real-time monitoring
5. False positives often contain brief VF segments within them (pre-VF transitions) — not harmful in monitoring context

---

## Critical Notes

- Uses 10-fold CV only, no independent test set (methodology criticized by authors themselves)
- Pre-selects only channel 1 from VFDB to avoid redundancy
- Excludes noise-annotated segments — reduces difficult cases
- VF class includes only VF annotations; VT/VFL in non-VF class (binary problem)
- Very high performance may reflect favorable dataset properties for this feature approach
