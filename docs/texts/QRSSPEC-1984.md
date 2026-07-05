# Estimation of QRS Complex Power Spectra for Design of a QRS Filter

**Authors:** Nitish V. Thakor, John G. Webster, Willis J. Tompkins
**Journal:** IEEE Transactions on Biomedical Engineering, 1984, BME-31(11):702–706
**DOI:** https://doi.org/10.1109/TBME.1984.325393

---

## Abstract

Power spectral analysis of ECG waveforms including isolated QRS complexes and noise/artifact episodes. A bandpass filter maximizing signal (QRS)-to-noise (T-waves, 60 Hz, EMG, etc.) ratio is designed. Coherence function and SNR calculated. Analysis of 3875 QRS complexes from 100 Holter tapes: **optimal bandpass filter has center frequency 17 Hz and Q = 5**.

---

## Theory

### Power Spectral Model

Signal model: Y(f) = H(f)X(f) + N(f)

where X(f) is QRS spectrum, H(f) is system transfer function, N(f) is additive noise.

Cross-power spectrum: G_yx = HG_xx
Output power spectrum: G_yy = |H|²G_xx + G_nn

Coherence function:
$$C^2_{xy}(f) = \frac{|G_{yx}(f)|^2}{G_{xx}(f) G_{yy}(f)} = \frac{|H|^2 G_{xx}}{|H|^2 G_{xx} + G_{nn}}$$

SNR:
$$\text{SNR}(f) = \frac{C^2_{xy}(f)}{1 - C^2_{xy}(f)}$$

### Filter Optimization

Second-order Butterworth bandpass filter: maximize SNR by selecting center frequency f_c and quality factor Q.

Transfer function:
$$|H'|^2 = \frac{(f/f_c)^2}{[1-(f/f_c)^2]^2 + (f/f_c)^2/Q^2}$$

Constraint on Q: Impulse response time constant τ = Q/(√2 π f_c). Transients should die in <200 ms (no interference with next beat) → Q ≤ 5 for f_c = 17 Hz.

---

## Methods

- 512 samples/s ECG, 0.5–40 Hz bandwidth
- 150 ECG cycles per category: normal resting, muscle noise (arm flexion, jogging), motion artifacts, 14077 A-C arrhythmia tapes
- QRS complex: 100 samples around centered R-peak, P and T blanked
- FFT: 512-point window (100-point QRS → zero-filled to 512; 1024-point for ECG)
- 3875 QRS complexes from 100 Holter tapes (randomly selected, 4 per 24h recording)

---

## Key Results

### Power Spectral Observations (Figure 4)

From 150 noise-free ECG cycles:
- **QRS complex:** Broad peak centered around **17 Hz**; extends from ~5 Hz to 30+ Hz
- **P- and T-waves:** Low-frequency content, below 5 Hz
- **ECG (complete):** Composite of all components
- **Muscle noise:** Broad, overlaps QRS spectrum
- **Motion artifact:** Very low frequency (below 5 Hz)

### Optimal Filter (Figure 7)

SNR is maximized at:
- **Center frequency f_c = 17 Hz**
- **Quality factor Q = 5**

Simple op-amp bandpass filter circuit: 330kΩ + 10nF + 2.2MΩ components achieves this response.

---

## Clinical/Algorithmic Relevance

This paper establishes the spectral properties of the QRS complex:
- QRS energy concentrated 5–30 Hz, peak at ~17 Hz
- Optimal SNR for QRS detection: 17 Hz center, Q=5 bandpass filter
- P/T waves and motion artifacts primarily <5 Hz
- Muscle noise is broadband

**Relevance to VF detection:**
- VF lacks QRS complexes → energy concentrated in 3–8 Hz (dominant fibrillation frequency)
- Normal rhythm and VT have strong 15–25 Hz QRS energy
- This spectral difference underlies the VF-filter (VFLEAK) algorithm principle
- JEKOVA-2004 explicitly uses a 14.6 Hz center band-pass filter to isolate non-shockable rhythms (which have QRS complex energy there) from shockable VF/VFL (which do not)

---

## Notes

- Thakor is the same author as the TCI paper [HYPO-1990]
- This 1984 paper predates the TCI algorithm (1990)
- The 17 Hz optimal QRS filter was used in Thakor's later arrhythmia monitor design
- PDF file also contains unrelated paper: "Data Reduction of Body Surface Potential Maps by Means of Orthogonal Expansions" (Uijen, Heringa, van Oosterom, 1984) — different topic, ignore
