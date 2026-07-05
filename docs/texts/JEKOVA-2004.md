# Real Time Detection of Ventricular Fibrillation and Tachycardia

**Authors:** Irena Jekova, Vessela Krasteva
**Journal:** Physiological Measurement, 2004, 25(5):1167–1178
**DOI:** https://doi.org/10.1088/0967-3334/25/5/007
**PMID:** 15535182

---

## Abstract

An algorithm for VF/VT detection based on a band-pass digital filter with integer coefficients, for real-time AED implementation. Sensitivity 95.93% and specificity 94.38% on AHA and MIT databases (99 complete ECG files).

**Keywords:** VF detection, external ECG, sensitivity, specificity

---

## Introduction

VF and VT above 180 bpm are dangerous cardiac disturbances requiring immediate defibrillation. AED algorithms must discriminate shockable from non-shockable rhythms with sensitivity and specificity approaching 100%. Simple solutions suitable for microprocessor embedding preferred.

---

## Material and Method

### ECG Signals

- **99 full-length ECG recording files**, all annotated by cardiologist + biomedical engineer
- 93/99 (96%) contain shockable episodes
- Sources: AHA (A8001–A8010, 30-min 2-channel), MIT vfdb (35-min 2-channel), MIT cudb (8-min single-channel), all at 250 Hz, 12-bit
- **Non-shockable dataset (9726 × 10-s episodes):**
  - NSR: 80 files (20 AHA + 40 vfdb + 20 cudb)
  - Branch blocks: 13 files
  - Bradycardia: 4 AHA
  - Paced beats: 5 files
  - Ectopic beats: 6 files
  - Supraventricular tachycardia: 2 vfdb
  - Bigeminy: 5 files
  - Trigeminy: 1 cudb
  - Low amplitude: 3 vfdb
  - Non-shockable VTs below 180 bpm: 16 files
  - Noise contaminated: 42 files
- **Shockable dataset (2528 × 10-s episodes):**
  - VF: 80 files (20 AHA + 38 vfdb + 35 cudb); includes 2 agonal rhythm files
  - VT above 180 bpm: 11 files (6 vfdb + 5 cudb)

### Algorithm

Analysis on successive 10-second epochs.

**Step 1: Signal preprocessing**
- Two successive 1st-order HP filters (1 Hz) — baseline drift suppression
- 2nd-order 30 Hz Butterworth LP — muscle noise reduction
- Notch filter (50 Hz) — powerline interference
- Effective bandwidth: ~1.4 Hz HP (equivalent high-pass cutoff)

**Step 2: Noise detection**
- Signals with abnormal amplitudes/slopes classified as 'Noise'
- Maximum slew rate limit: 400 μV/ms
- Signals with max amplitude <150 μV → 'Asystoly'

**Step 3: Band-pass digital filter (key innovation)**
Central frequency 14.6 Hz, bandwidth 13–16.5 Hz (−3 dB). Integer coefficient recursive filter designed for 250 Hz:
$$FS_i = \frac{14FS_{i-1} - 7FS_{i-2} + \frac{S_i - S_{i-2}}{2}}{8}$$

Rationale: Non-shockable rhythms (NSR, atrial tachycardia, AFL, AFIB, SVT) have frequencies up to 20 Hz, VT complexes up to 14 Hz. VF/flutter concentrated below 10 Hz. Band 13–17 Hz selected as containing frequencies of non-shockable but NOT shockable rhythms.

**Step 4: Rhythm classification by Count parameters**
Three parameters from absolute filter output (AbsFS) per 10-s interval, computed per 1-s stages:
- **Count1:** Samples with AbsFS in range [0.5×max(AbsFS), max(AbsFS)]
- **Count2:** Samples with AbsFS in range [mean(AbsFS), max(AbsFS)]
- **Count3:** Samples with AbsFS in range [mean(AbsFS)−MD, mean(AbsFS)+MD] (MD = mean deviation)

Decision rules:
- If Count1 < 250 AND Count2 > 950 AND Count1×Count2/Count3 < 210 → **Non-Shockable**
- If 250 ≤ Count1 < 400 AND Count2 < 600 AND Count1×Count2/Count3 < 210 → **Non-Shockable**
- If Count1 ≥ 250 AND Count2 > 950 → **Shockable**
- If Count2 ≥ 1100 → **Shockable**
- Otherwise → **Not Classified**

**Step 5: Wave detection for 'Not Classified' rhythms**
Detected positive/negative peaks with adaptive thresholds (initialized at ±150 μV). Parameter Period calculated. If >87.5% of peaks in range [75%, 125%] of mean amplitude → Period = segment_length / num_waves; otherwise Period uses VF-filter formula (Kuo & Dillman 1978). Heart rate from Period:
- If no waves detected for last 5 s → **Asystoly**
- If Period → heart rate < 180 bpm → **Non-Shockable**
- If Period → heart rate ≥ 180 bpm → **Shockable**

---

## Results

### Table 1: Detection Accuracy

| Database | N (non-shockable) | Correct N | S (shockable) | Correct S | Sp (%) | Se (%) |
|----------|-------------------|-----------|---------------|-----------|--------|--------|
| AHA | 2366 | 2315 | 1013 | 985 | 97.84 | 97.24 |
| MIT vfdb | 6159 | 5736 | 1208 | 1160 | 93.13 | 96.03 |
| MIT cudb | 1201 | 1128 | 307 | 280 | 93.92 | 91.21 |
| **Total** | **9726** | **9179** | **2528** | **2425** | **94.38** | **95.93** |

---

## Discussion

- Lowest accuracy on MIT cudb (shortest files, 8 min — artefacts disproportionately affect)
- Main errors: atrial fibrillation with bizarre ventricular complexes (similar to paced beats) — misclassified as shockable due to wave detection finding >40 waves/period
- VF with peak-like artefacts: Count1/Count2 reduced → Count3 increased → non-shockable misclassification
- Pacemaker pulses in low-amplitude VF: problematic (should be handled by separate hardware)
- Very low frequency VF (~2 Hz): wave detection branch → Period → non-shockable (missed)
- Algorithm operates on full 10-s epoch → real-time decision at end of each epoch

## Comparison with Prior Work (from Discussion)

| Method | Sensitivity | Specificity | Database |
|--------|-------------|-------------|----------|
| Clayton et al. 1993 (TCI) | 77% | 93% | CCU |
| Clayton et al. 1993 (SPEC) | 46% | 72% | CCU |
| Jekova 2000 (5 algorithms, mean) | 84% | 73% | AHA+MIT |
| **This algorithm** | **95.93%** | **94.38%** | AHA+MIT |
