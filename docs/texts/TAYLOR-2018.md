# Detection of Life Threatening Ventricular Arrhythmia Using Digital Taylor Fourier Transform

**Authors:** Rajesh K. Tripathy, Alejandro Zamora-Mendez, José A. de la O Serna, Mario R. Arrieta Paternina, Juan G. Arrieta, Ganesh R. Naik
**Journal:** Frontiers in Physiology, 2018, 9:722
**DOI:** https://doi.org/10.3389/fphys.2018.00722
**Note:** This is the TAYLOR paper in the reference list.

---

## Abstract

A novel method for detection and classification of VT and VF using the Digital Taylor-Fourier Transform (DTFT). ECG signals are decomposed into 10 oscillatory modes; magnitude and phase difference (PD) features are extracted from the Taylor-Fourier coefficients and classified using a Least Squares SVM (LSSVM) with linear and RBF kernels. Tasks: VT vs. VF, Non-VF vs. VF, Non-Shock vs. Shock.

**Keywords:** ventricular arrhythmia, Taylor-Fourier transform, magnitude and phase features, LSSVM, RBF kernel

---

## Databases

- **CUDB** (Creighton University VT Database): 35 recordings, 8-min, 250 Hz, annotations NSR/VF/VT/VFL and other rhythms
- **VFDB** (MIT-BIH Malignant VF Database): 22 recordings, 35-min, 250 Hz, two-lead ECG

Window sizes: 4 s, 5 s, 8 s (rectangular frames).

**Total frames computed:**
- Non-VF vs. VF classification (CUDB+VFDB combined):
  - 4s: 3432 total (1712 VF); 5s: 2744; 8s: 1712
- VT vs. VF classification:
  - 4s: 2593; 5s: 2072; 8s: 1291

Preprocessing: Zero-phase Butterworth bandpass 0.5–45 Hz (from Tripathy et al. 2016).

---

## Method: Digital Taylor-Fourier Transform (DTFT)

The DTFT filter bank **B** decomposes ECG frames into **M = 10 oscillatory modes** with central frequencies f₀–f₉ spaced at 5 Hz: 0.5, 5, 10, 15, 20, 25, 30, 35, 40, 45 Hz.

The synthesis and analysis equations for M frames:
$$[\hat{s}_1, \hat{s}_2, ..., \hat{s}_M] = \mathbf{B}[\hat{\xi}_1, \hat{\xi}_2, ..., \hat{\xi}_M]$$
$$[\hat{\xi}_1, \hat{\xi}_2, ..., \hat{\xi}_M] = \mathbf{B}^\dagger[s_1, s_2, ..., s_M]$$

where **B**† is the pseudoinverse. Filters cover the full spectral range for VF detection.

**Feature extraction (20-dimensional feature vector per frame):**

For each mode j (j = 1,...,10):
- **Magnitude feature:** MF_j = ||â_j||₂ (L²-norm of Taylor-Fourier magnitude vector)
- **Phase difference feature:** PD_j = (1/(K-1)) Σ|d1_j(l) − d2_j(l)|

Final 20-dimensional feature vector: [MF₁...MF₁₀, PD₁...PD₁₀].

**Rationale:** VF and rapid VT have higher temporal variability in ECG → phase features have higher mean values for VF class vs. non-VF class.

---

## Classification: LSSVM

Least Squares SVM (LSSVM) with linear and RBF kernels. Gaussian RBF kernel:
$$K(\mathbf{z}, \mathbf{z_i}) = \exp\left(-\gamma ||\mathbf{z} - \mathbf{z_i}||^2\right)$$

Evaluation: Hold-out (65% training, 35% testing) and 5-fold cross-validation.

Classification tasks:
1. VT vs. VF
2. Non-VF vs. VF
3. Non-Shock vs. Shock (VFL+VT+VF = Shock)

---

## Results

### Table 1 (All databases, RBF kernel):

#### Non-VF vs. VF

| Window | Acc (%) | Sen (%) | Spe (%) |
|--------|---------|---------|---------|
| 4 s    | 83.75   | 85.20   | 82.46   |
| 5 s    | 82.66   | 83.74   | 81.73   |
| 8 s    | 82.84   | 83.82   | 82.05   |

#### VT vs. VF

| Window | Acc (%) | Sen (%) | Spe (%) |
|--------|---------|---------|---------|
| 4 s    | 82.36   | 81.38   | 82.82   |
| 5 s    | 84.30   | 82.02   | 85.26   |
| 8 s    | 83.41   | 82.19   | 83.88   |

### Table 2 (VFDB only, RBF kernel, Non-VF vs. VF):

| Window | Acc (%) | Sen (%) | Spe (%) |
|--------|---------|---------|---------|
| 4 s    | 89.05   | 85.81   | 92.97   |
| 5 s    | 89.44   | 86.58   | 92.81   |
| **8 s** | **89.81** | **86.38** | **93.97** |

### Table 4: Comparison with existing features (8s, VFDB+CUDB):

| Features used | Sen (%) | Spe (%) |
|---------------|---------|---------|
| TCI (Thakor 1990) | 71.00 | 70.50 |
| VF-Filter (Kuo 1978) | 30.00 | 99.50 |
| SPEC (Barro 1989) | 29.00 | 99.30 |
| CPLX (Zhang 1999) | 56.40 | 86.60 |
| PSR (Amann 2007) | 70.20 | 89.30 |
| **DTFT Magnitude+Phase (VFDB)** | **86.38** | **93.97** |
| DTFT Magnitude+Phase (CUDB) | 84.15 | 77.22 |

---

## Key Findings

- DTFT features significantly outperform all classical signal-processing features (TCI, SPEC, VF-filter, CPLX, PSR) for Non-VF vs. VF classification
- 8s window gives best results; 4s and 5s are close
- CUDB has lower performance due to mixed VT/VF annotations (rhythm ambiguity)
- Phase features are more discriminative than magnitude for VF vs. non-VF
- Only 17 out of 20 features statistically significant (p < 0.001) for 8s window

---

## Conclusions

- DTFT provides a physically meaningful decomposition of ECG into frequency modes
- Taylor-Fourier magnitude + phase features capture pathological temporal variability of VF
- Method is computationally efficient (B matrix computed once for all signals)
- Applicable to VT vs. VF, Shock vs. Non-Shock classification
- Performance on CUDB lower than VFDB due to mixed rhythm annotations

---

## Comparison to Classical Algorithms (Perspective for VTA.ipynb)

This paper (Table 4) provides a direct comparison of SPEC, VF-filter (VFLEAK), and PSR performance on CUDB+VFDB using 8-second frames:
- SPEC: Se=29%, Sp=99.3% — confirms low sensitivity pattern from COMP55-2005
- VFLEAK: Se=30%, Sp=99.5% — very high specificity, very low sensitivity without QRS detection
- PSR: Se=70.2%, Sp=89.3% — balanced, best among classical methods here
