# Detection of Ventricular Tachyarrhythmias #

## Project Statement ##

Ventricular Tachycardia (VT), Flutter (VFL), and Fibrillation (VF) form a continuum of life-threatening ventricular tachyarrhythmias that can rapidly progress to hemodynamic collapse and Sudden Cardiac Death. VT typically retains discrete, repeatable QRS complexes and a relatively regular rhythm. VFL is an extremely fast, regular, sine-wave–like rhythm in which individual beats are no longer clearly separable; it is often transient and can quickly deteriorate into VF. VF is disorganized, with chaotic, low-amplitude activity and no coordinated ventricular contraction; without immediate treatment it is fatal within minutes.

QRS detectors are designed to extract discrete events by emphasizing slope, amplitude, and temporal isolation. During VFL, the ECG becomes a continuous narrow-band oscillation, causing detectors to miss peaks or over-count oscillations. In VF the signal becomes increasingly irregular and may resemble noise, making beat detection inapplicable. 
VFL and VF are primarily rhythm-detection problems that require analysis of short signal segments using measures of spectral organization rather than beat fiducials.

While the literature is extensive, papers lack reference implementations, hindering reproducibility and comparative evaluation. This project will implement core temporal and spectral metrics, identify reference databases, build annotated datasets, and explore whether convolutional neural networks can detect ventricular tachyarrhythmias from time-domain segments and/or spectral representations.

## Structure ##

- Дефинирање на темата и целите на проектот (20 February 2026)
- Фаза 1 - Претходни истражувања, литература
Completion requirements (28 February 2026)
- Фаза 2 - База на податоци и претпроцесирање на податоците (7 March 2026)
- Фаза 3 - Креирање на модели/Употреба на методи (7 March 2026)
- Фаза 4 - Експерименти, резултати и дискусија (3 April 2026)
- Фаза 5 - Документација, изворен код и презентација (3 April 2026)

## Introduction ##

The heart beats because of electrical impulses that travel through it in a precise, coordinated sequence. When this sequence breaks down in the ventricles — the heart's main pumping chambers — the result is a group of dangerous rhythms collectively called **ventricular tachyarrhythmias**. This report focuses on three of them: Ventricular Tachycardia (VT), Ventricular Flutter (VFL), and Ventricular Fibrillation (VF).

#### Clinical Importance

These arrhythmias are a leading cause of Sudden Cardiac Death (SCD). In the United States alone, approximately 325,000 people suffer an out-of-hospital cardiac arrest each year, and VF is the initial rhythm in up to 75% of those cases [MODERN-2024]. Overall, SCD accounts for roughly 50% of all cardiovascular deaths [MODERN-2024]. Survival depends almost entirely on how fast treatment is delivered: for every minute without defibrillation, the chance of survival drops by about 10% [MODERN-2024]. This makes rapid and reliable automatic detection of these rhythms a matter of life and death.

Automated External Defibrillators (AEDs) are designed for exactly this purpose — to recognize a shockable rhythm and deliver a defibrillating shock without requiring a trained clinician [JEKOVA-2004, COMP5-2000]. Their built-in detection algorithms must be both highly sensitive (never miss a true event) and highly specific (never shock unnecessarily). Missing a VF episode is fatal; delivering an inappropriate shock is harmful and potentially dangerous. This dual requirement makes the detection problem genuinely difficult [COMP4-1993].

#### The Three Rhythms and How They Differ

VT, VFL, and VF form a continuum of increasingly disorganized electrical activity. Understanding where each one sits on that continuum explains why they pose very different challenges for automated detection.

**Ventricular Tachycardia (VT)** is a fast but still organized rhythm, typically above 100 bpm and originating in the ventricles rather than the normal conduction pathway. The ECG still shows distinct QRS complexes, though they are wide and abnormal in shape. The rhythm is regular or nearly so. Because individual beats are still identifiable, standard QRS detectors can handle VT, though they must distinguish it from other fast rhythms like supraventricular tachycardia. VT rates above 180 bpm are considered shockable [JEKOVA-2004].

**Ventricular Flutter (VFL)** sits between VT and VF. The ventricular rate is extremely fast (typically 200–350 bpm), and the ECG takes on a continuous, smooth, sine-wave-like appearance. Individual beats can no longer be clearly separated; the signal is a nearly monomorphic oscillation. VFL is usually short-lived and unstable — it tends to either revert to a more organized rhythm or deteriorate into VF. Because the ECG is a narrow-band oscillation rather than a sequence of isolated beats, traditional QRS detectors either over-count oscillation peaks or fail to detect any beats at all.

**Ventricular Fibrillation (VF)** is the most severe form. The ventricles are activated chaotically by many independent wavefronts at once. The ECG shows rapid, irregular, low-amplitude activity with no discernible P waves, QRS complexes, or T waves — just disordered fluctuations. There is no effective pumping, and without immediate defibrillation the patient will die within minutes. On the ECG, VF can look superficially similar to noise or motion artifact, which adds another layer of difficulty.

#### Why Detection Is Hard

Each rhythm type presents a distinct detection challenge [COMP4-1993, COMP5-2000]:

- **VT** is organized enough for beat-by-beat analysis, but must be differentiated from fast normal rhythms, paced rhythms, and bundle branch blocks that also produce wide, abnormal-looking QRS complexes.

- **VFL and VF** cannot be reliably detected by counting beats. Instead, detection depends on characterizing the statistical and spectral properties of the signal as a whole — its dominant frequency, regularity, amplitude distribution, and spectral concentration. The key difficulty is that several other conditions can produce similar-looking signals: muscle artifacts, electrode noise, and atrial flutter or fibrillation all share surface ECG features with VF and can cause false positives [COMP4-1993].

- **Noise and artifact** are a persistent practical challenge. Poor electrode contact produces signals that closely mimic VF, while actual VF can be masked by movement artifact. The tolerance for false alarms is very low in AEDs, since a misclassified shock can cause injury.

- **Data scarcity** complicates algorithm development and evaluation. VF is unpredictable; recordings that capture the full onset and progression of VF are rare, and available databases contain limited numbers of labeled episodes.

#### Approaches to Detection

Decades of research have produced two main families of algorithms, both of which remain active areas of work.

**Time-domain methods** analyze the raw ECG waveform directly. Examples include counting the rate of threshold crossings (TCI), measuring the autocorrelation of the signal to assess periodicity, and computing simple amplitude or energy statistics over short windows [COMP4-1993, COMP5-2000, TCSC-2009]. These methods are computationally light and suitable for real-time embedded devices.

**Frequency-domain methods** transform the signal (usually with the FFT) and look at how energy is distributed across frequencies. VF energy concentrates in a characteristic band roughly between 4 and 9 Hz, while organized rhythms have energy at the fundamental heart rate and its harmonics [SPEC-1989, COMP4-1993]. Metrics such as the dominant frequency, spectral concentration, and normalized power in specific bands form the basis of these detectors.

More recently, **machine learning and deep learning** methods have been applied to this problem. Convolutional neural networks (CNNs), recurrent networks (LSTMs), and hybrid models using wavelet or Fourier preprocessing have demonstrated sensitivities and specificities above 95% on standard benchmark databases [MODERN-2024, DEEP-2023]. These approaches learn discriminative features automatically rather than relying on hand-crafted criteria, but they require large annotated training sets and carry higher computational cost.

Despite this extensive literature, published algorithms are rarely released with reference implementations. Reproducing and comparing methods is therefore difficult, and performance figures from different papers are often not directly comparable because they use different databases, preprocessing pipelines, and evaluation protocols.

#### Project Goals

This project aims to:
1. Implement the core temporal and spectral detection metrics from the classical literature.
2. Identify the standard reference databases (AHA, MIT-BIH cudb/vfdb) and build a consistent annotated dataset.
3. Evaluate and compare classical algorithms on a common test bed.
4. Explore whether convolutional neural networks can reliably detect VT, VFL, and VF from short ECG segments in both time and frequency domains.

The result will be a self-contained, reproducible codebase that makes it straightforward to replicate, compare, and extend published detection methods.


<figcaption><b>Fig. 1a — Ventricular Tachycardia (VT).</b> Fast, wide QRS complexes at a regular rate (~150–200 bpm). The rhythm is organized; individual beats are still identifiable.</figcaption>

<figcaption><b>Fig. 1b — Ventricular Flutter (VFL).</b> Continuous sinusoidal waveform at ~250–350 bpm. QRS and T waves merge; no individual beats can be distinguished.</figcaption>

<figcaption><b>Fig. 1c — Ventricular Fibrillation (VF).</b> Chaotic, irregular, low-amplitude activity. No P waves, QRS complexes, or T waves are present.</figcaption>

<p style="font-size:0.85em; color:#555">Source: ECGpedia / CardioNetworks, via <a href="https://commons.wikimedia.org/wiki/Category:Ventricular_tachycardia">Wikimedia Commons</a> (CC BY-SA 3.0)</p>

## Related Work ##

The algorithms developed for VT/VFL/VF detection span several decades and reflect an evolution from very simple signal counting rules to full spectral analysis. This progression was shaped not only by scientific understanding but also by the hardware constraints of the time: early algorithms had to run on dedicated microprocessors with limited memory and no floating-point units, making simple counting and threshold operations the only practical option. As embedded computing became more capable, more computationally demanding methods — FFT-based spectral analysis, entropy measures, and eventually neural networks — became feasible. Rather than listing algorithms chronologically, this section organizes them by their underlying principle — from the simplest time-domain counting methods through statistical and complexity measures to frequency-domain techniques. This ordering also builds intuition: each category addresses a limitation left by the previous one.

---

### Group A — Time-Domain Threshold Crossing Methods

The first and simplest family of algorithms asks a single question: *how often does the signal cross a threshold?* In normal sinus rhythm, the ECG spends most of its time near the isoelectric baseline; crossings happen briefly once per beat. During VF, the signal is in constant motion, crossing the threshold continuously. Counting crossings or measuring the time between them gives a direct, computationally trivial estimate of the underlying oscillation rate.

#### 1. Zero Crossing Rate (ZCR)

The Zero Crossing Rate is the most elementary feature: count the number of times the signal crosses zero (or a fixed baseline) per unit time. In normal sinus rhythm, the ECG crosses zero a few times per beat (P wave, QRS, T wave), giving a low ZCR. During VF, the signal oscillates continuously at 4–9 Hz, producing many crossings per second. ZCR was among the earliest discriminators proposed [COMP4-1993]. Its main weakness is sensitivity to DC offset and baseline drift — a shifted baseline changes the zero-crossing count dramatically.

#### 2. Threshold Crossing Intervals (TCI)

TCI [COMP4-1993, COMP5-2000] addresses ZCR's sensitivity to baseline by using an *adaptive* threshold set at 20% of the local signal maximum for each 1-second segment. Rather than just counting crossings, it measures the average time *between* consecutive upward threshold crossings. In sinus rhythm the threshold is crossed once per QRS, so the mean TCI equals roughly the R–R interval (≥400 ms). In VF, crossings occur rapidly and the mean TCI drops to ~100–200 ms (corresponding to a dominant frequency of 5–9 Hz). The decision rule is:

- TCI ≥ 400 ms → Sinus Rhythm  
- TCI in VT range (~220 ms) or VF range (~105–160 ms) → shockable rhythm

Sequential hypothesis testing is used to distinguish VT from VF based on the distribution of successive TCIs [COMP4-1993]. Performance on standard databases is good for sensitivity but degrades for specificity when signals have irregular rate or ectopic beats [COMP5-2000].

#### 3. Threshold Crossing Sample Count (TCSC)

TCSC [TCSC-2009] is a direct improvement on TCI. Instead of measuring the *interval* between crossings, it counts the *fraction of samples* whose absolute value exceeds a normalized threshold V₀ = 0.2 (after amplitude normalization to unit peak). The TCSC parameter N is:

$$N = \frac{\text{samples with } |x_i| > V_0}{\text{total samples}} \times 100\%$$

The key insight is that a normal ECG spends most of its time near the isoelectric line (P–R and S–T segments, inter-beat intervals), so N is low (~5%). VF has no isoelectric segments — the signal is continuously active — so N is high (~70–80%). TCSC also fixes several TCI weaknesses: it uses both positive and negative thresholds (not just positive), works on 3-second segments rather than 1-second, and is applied with a cosine taper window to reduce edge effects. On the complete MIT-BIH and CU databases without pre-selection, TCSC yields the best area under the ROC curve of all classical algorithms compared in that study.

---

### Group B — Time-Domain Distribution and Shape Features

Threshold crossing methods capture oscillation *rate*. A second family of time-domain features instead asks about the *distribution* or *shape* of the signal amplitude. In sinus rhythm the amplitude distribution is sharply concentrated near zero with narrow high-amplitude peaks at QRS complexes. In VF the distribution is broader and more uniform. These features can be computed without any threshold, making them more robust to DC offsets.

#### 4. Time in Band (TIB) / Time Outside Band

Related to TCSC but framed differently: TIB measures the fraction of time the signal amplitude stays *within* a defined central band (e.g., within ±20% of peak). In sinus rhythm most samples fall in the central band (high TIB); in VF, samples spread across the amplitude range (low TIB). "Time Outside Band" is the complement. These metrics are closely related to the Standard Exponential (STE) and Modified Exponential (MEA) algorithms [COMP55-2005], which count intersections of the ECG with an exponential envelope function. All three approaches capture the same underlying property: normal rhythms spend time near baseline; VF does not. STE achieves IROC ≈ 67%, while the more sophisticated MEA reaches IROC ≈ 82% on standard databases [COMP55-2005].

#### 5. Amplitude Distribution / Histogram Features

Rather than a single threshold, this approach characterizes the full amplitude histogram of the ECG segment. Sinus rhythm produces a clearly bimodal or sharply peaked distribution (most samples near zero, with rare high-amplitude excursions at QRS peaks). VF produces a flatter, more uniform distribution. Feature vectors derived from histogram moments (mean, variance, skewness, kurtosis) or from explicit bin counts can separate VF from SR. The Signal Comparison Algorithm (SCA) [COMP55-2005] is a more sophisticated variant: it compares the ECG against four reference signals (three SR templates and one VF cosine template) using L1-norm residuals. SCA achieves IROC ≈ 92% — the best among classical algorithms in [COMP55-2005] — though it is computationally more demanding.

---

### Group C — Complexity and Entropy Measures

Time-domain rate and distribution features capture *how fast* the signal oscillates and *where* its energy sits in amplitude. A third family of methods asks a different question: *how predictable is the signal?* Sinus rhythm is quasi-periodic and therefore highly predictable; VF is chaotic and unpredictable. Information-theoretic measures formalize this distinction.

#### 6. Sample Entropy / Approximate Entropy

Approximate Entropy (ApEn) and Sample Entropy (SampEn) measure the conditional probability that a pattern of length m that matches within tolerance r also matches at length m+1. A signal with many repeated patterns (like a periodic QRS complex) has low entropy; a disordered signal (like VF) has high entropy. These measures work directly on the raw time series with no explicit frequency analysis.

The related **Complexity Measure (CPLX)** [COMP5-2000, COMP55-2005], introduced by Zhang et al. (1999), uses Lempel–Ziv complexity: the ECG segment is converted to a binary string by thresholding, and the complexity C(n) is computed as the normalized number of distinct substrings needed to reconstruct the string (using the Lempel–Ziv algorithm). A random string has C(n) → 1; a periodic string has C(n) → 0. VF signals score high, SR scores low, with a threshold of C > 0.173 for VF detection. In comparative evaluations, CPLX achieves IROC ≈ 87% [COMP55-2005], competitive with spectral methods. Its main drawback is sensitivity to noise: random artifacts can mimic the high complexity of VF.

---

### Group D — Frequency-Domain Spectral Methods

VF produces a characteristic narrow-band signal in the 4–9 Hz range, whereas organized rhythms (SR, VT) contain energy at the heart rate fundamental and multiple harmonics spread across a wider bandwidth. Frequency-domain methods exploit this directly by transforming the ECG into the spectral domain and measuring where the energy is concentrated.

#### 7. VF-Filter Algorithm

Proposed by Kuo and Dillman (1978) and evaluated in [COMP4-1993, COMP5-2000], the VF-filter is conceptually elegant. If VF is approximately sinusoidal with mean period T, then adding each sample to the sample half a period away should cancel the signal by destructive interference. The **leakage** after this cancellation measures how sinusoidal the signal is:

$$\text{leakage} = \frac{\sum |V_i + V_{i-T/2}|}{\sum (|V_i| + |V_{i-T/2}|)}$$

A pure sine wave has leakage → 0; a random or wideband signal has leakage → 1. VF is classified when leakage < 0.406 (or 0.625 in the absence of QRS detection). The VF-filter achieves high specificity (nearly 100% on standard databases) because only truly sinusoidal signals pass the test. Its sensitivity depends on VF being sufficiently organized — coarse VF with variable waveform shape can be missed. In COMP55-2005, VF-filter achieves IROC ≈ 87% at 8s window length.


#### 8. Dominant Frequency (DF) and Regularity Index (RI)

The Dominant Frequency is simply the frequency of the peak bin in the power spectrum, restricted to the physiological range of interest (typically 4–15 Hz). For VF, DF is typically in the 4–9 Hz band; for sinus rhythm at 60 bpm, DF is at 1 Hz with harmonics at 2, 3, ... Hz. DF alone is a simple but effective feature.

The **Regularity Index (RI)** extends this by measuring how much of the total spectral power is concentrated at the dominant frequency and its harmonics. An organized rhythm (SR, VT) has energy tightly packed at multiples of the fundamental, giving high RI. VF spreads energy broadly across the spectrum, giving low RI. RI is used in SPEC-derived systems and in modern AED algorithms. The combination DF + RI is robust to amplitude variation because both measures are normalized by total power [JEKOVA-2004].

#### 9. Organization Index (OI)

The Organization Index is a direct measure of spectral concentration. It is defined as the ratio of the power contained within narrow bands centered on the dominant frequency and its first few harmonics to the total spectral power:

$$\text{OI} = \frac{\sum_{k=1}^{K} P(k \cdot f_{dom}, \pm \Delta f)}{P_{total}}$$

where Δf is the half-width of each harmonic band. High OI indicates an organized, quasi-periodic signal (SR or VT); low OI indicates disorganized activity (VF). OI is the spectral analog of the temporal regularity captured by the ACF: both measure how periodic the signal is, but OI is more robust to non-stationarity because it operates on a short windowed segment rather than requiring stationarity over a long epoch. OI appears prominently in descriptions of AED analysis systems and in spectral analysis of VF dynamics [JEKOVA-2004, SPEC-1989].

#### 10. SPEC — Spectral Algorithm

The SPEC algorithm (Barro et al., 1989 [SPEC-1989]) is the most comprehensive classical spectral method. The ECG segment is multiplied by a Hamming window, transformed by FFT, and four parameters are extracted from the amplitude spectrum:

- **FSMN** (First Spectral Moment Normalized): the amplitude-weighted average frequency across the spectrum. Low for VF (energy concentrated at low frequencies), high for SR.
- **A₁**: ratio of energy below F/2 to total energy (F = dominant peak frequency). Captures sub-fundamental noise.  
- **A₂**: ratio of energy in the band [0.7F, 1.4F] to total energy. Measures spectral concentration around the dominant peak. High for VF (narrow-band), low for SR.
- **A₃**: ratio of energy at the 2nd–8th harmonics to total energy. Measures harmonic content. Low for VF (no clear harmonics), high for structured rhythms.

VF is declared if FSMN ≤ 1.55, A1 > 0.19, A2 ≥ 0.45, A3 ≤ 0.09 (thresholds were adjusted in later studies). SPEC achieves IROC ≈ 89% [COMP55-2005] — the best among the five classic algorithms — and nearly 100% specificity, though sensitivity on some databases is modest without threshold tuning [COMP5-2000].


#### 11. Band-Pass Filter Algorithm

Jekova & Krasteva [JEKOVA-2004] observed that normal cardiac complexes — whether supraventricular or ventricular — contain significant energy in the 13–17 Hz range (QRS slopes and fine detail), whereas VF energy sits almost entirely below 10 Hz. This spectral separation motivates a simple but effective approach: apply a narrow band-pass filter centered at 14.6 Hz (−3 dB bandwidth: 13–16.5 Hz) and examine what remains.

After filtering, a normal sinus or tachycardia signal retains appreciable amplitude (high-frequency QRS content passes through), while a VF episode is heavily attenuated. Three simple counts derived from the absolute value of the filtered output are compared against fixed thresholds to produce a shockable / non-shockable decision. The filter uses integer coefficients, requiring only additions and bit-shifts — trivially implementable in real-time on any AED microprocessor without floating-point arithmetic. A subsidiary branch activates beat detection and heart-rate measurement for borderline cases near the VT/VF boundary.

Tested on the combined AHA and MIT databases (9726 non-shockable and 2528 shockable 10-second epochs), the algorithm achieves **95.93% sensitivity and 94.38% specificity** [JEKOVA-2004] — competitive with the more demanding spectral methods and ready for embedded deployment.

---

### Group E — Phase Space Methods

A signal can be visualized not just as amplitude over time, but as a trajectory in a two-dimensional *phase space*: plot each sample against a transformed version of itself. Periodic signals trace closed curves; chaotic signals fill the plane broadly. This geometric view turns VF detection into a shape-recognition problem that needs no explicit frequency analysis and no amplitude threshold.

#### 12. Hilbert Transform Phase Space (HILB)

The Hilbert transform algorithm [HILB-2005] constructs the *analytic signal* from the ECG: for a real signal x(t), the Hilbert transform x_H(t) shifts all frequency components by −90°. Plotting x(t) on the x-axis and x_H(t) on the y-axis yields a phase-space portrait of the signal. Sinus rhythm traces a tight, repeating loop — the same QRS-T morphology each beat. VF traces an irregular path that fills a large area of the plane.

To quantify how much area is filled, the portrait is overlaid on a 40 × 40 grid and the fraction of occupied boxes is measured:

$$d = \frac{\text{visited boxes}}{1600}$$

A threshold of d₀ = 0.15 separates VF (d ≈ 0.18–0.21) from SR (d ≈ 0.05–0.07). Because the computation reduces to one FFT-based convolution plus a box-counting loop, it is fast and simple to implement. Crucially, d depends on the *shape* of the trajectory rather than absolute amplitude, making HILB more robust to amplitude variation than threshold-crossing methods.

On the combined BIH-MIT, CU, and AHA databases (>330 000 decisions), HILB achieves **IROC = 95%** [HILB-2005] — the highest of all algorithms evaluated in that study, outperforming TCI (82%), VF-filter (87%), SPEC (89%), and CPLX (87%).


---

### Group F — Hybrid Signal Processing and Machine Learning

Classical algorithms compute a single hand-crafted scalar feature and compare it to a fixed threshold. Their simplicity is a strength for real-time embedded use, but limits discriminative power on borderline rhythms or poor-quality signals. A natural extension is to compute *multiple* features and let a machine learning classifier learn the optimal decision boundary from data.

#### 13. EMD + DFT + SVM (VFPred)

VFPred [VFPRED-2018] introduces **Empirical Mode Decomposition (EMD)** as the primary feature extraction step, combined with a Support Vector Machine (SVM) classifier.

**What is EMD and why is it better suited than FFT for ECG?**

The Fourier transform decomposes a signal into fixed sinusoids that span the *entire* signal duration. This assumes the signal is *stationary* — its statistical properties do not change over time. The ECG violates this assumption: a QRS complex is a sharp 100 ms transient, while VF is an irregular oscillation whose dominant frequency drifts over seconds. Applying FFT to the whole segment mixes these events into a single spectrum, losing information about *when* each feature occurs.

EMD [EMD-1998] takes a fundamentally different approach. Rather than projecting onto a fixed basis, it extracts oscillations directly from the signal's own local structure via an iterative *sifting* process:

1. Find all local maxima and minima of the signal.
2. Fit smooth envelopes through the maxima and minima respectively.
3. Subtract the running mean of the two envelopes from the signal.
4. Repeat until what remains oscillates symmetrically around zero with equal numbers of extrema and zero-crossings — this is one **Intrinsic Mode Function (IMF)**.
5. Subtract that IMF from the signal and repeat on the residue, extracting progressively slower oscillations.

The output is a set of IMFs ordered from fastest to slowest, plus a low-frequency residue trend:

$$x(t) = \text{IMF}_1(t) + \text{IMF}_2(t) + \cdots + \text{IMF}_n(t) + R(t)$$

Each IMF is a narrow-band, *amplitude- and frequency-modulated* oscillation whose instantaneous frequency can vary over time. This is the key advantage over FFT: EMD adapts to the signal itself, requiring no assumption of stationarity and no pre-defined frequency grid. It is particularly well-suited for non-stationary biomedical signals like the ECG.

**How VFPred applies EMD to VF detection**

The driving observation is that QRS complexes — sharp, high-amplitude transients — dominate the highest-frequency IMF (IMF₁). In normal sinus rhythm, IMF₁ captures the QRS fine structure and is therefore *unlike* the smooth original waveform; the Residue, containing the P–T wave baseline, more closely resembles it. In VF there are no QRS complexes — the signal is a continuous irregular oscillation — so IMF₁ tracks the original ECG directly, and the Residue is nearly flat.

VFPred exploits this asymmetry by:
1. Decomposing each 5-second ECG segment into IMF₁ (or IMF₁ + IMF₂ if high-frequency noise dominates) and Residue R.
2. Computing the DFT of the selected IMF and R.
3. Forming feature vectors from the per-frequency *similarity* between the original signal's DFT and the DFTs of IMF and R (normalized inner products, bin-wise cosine similarity).
4. Using a Random Forest to rank and select the most discriminative frequency bins — the 1–5 Hz band proves most informative.
5. Training an SVM with an RBF kernel on the selected features.

The hybrid approach achieves **sensitivity = 99.99% and specificity = 98.40%** on MIT-BIH and CU databases using only 5-second windows [VFPRED-2018], outperforming all purely hand-crafted classical algorithms on both metrics simultaneously.

### Summary Table

| # | Algorithm | Group | Domain | Key Metric | VF indicated by | Reference |
|---|-----------|-------|--------|------------|-----------------|-----------|
| 1 | ZCR | Threshold Crossing | Time | Crossings/sec | High rate | [COMP4-1993] |
| 2 | TCI | Threshold Crossing | Time | Mean crossing interval (ms) | TCI < 400 ms | [COMP4-1993], [COMP5-2000] |
| 3 | TCSC | Threshold Crossing | Time | % samples outside threshold | N > ~50% | [TCSC-2009] |
| 4 | TIB / STE / MEA | Distribution & Shape | Time | % time inside/outside band | Low TIB, high TOB | [COMP55-2005] |
| 5 | Amplitude Histogram / SCA | Distribution & Shape | Time | Amplitude PDF flatness / L1 residual | Flat histogram, low residual vs VF template | [COMP55-2005] |
| 6 | SampEn / CPLX | Complexity & Entropy | Time | Lempel-Ziv / entropy | High C(n) or entropy | [COMP5-2000], [COMP55-2005] |
| 7 | VF-Filter | Spectral | Frequency | Half-period cancellation leakage | Leakage < 0.4 | [COMP4-1993], [COMP5-2000] |
| 8 | DF + RI | Spectral | Frequency | Dominant frequency + harmonic ratio | DF ∈ 4–9 Hz, low RI | [JEKOVA-2004], [SPEC-1989] |
| 9 | OI | Spectral | Frequency | Spectral concentration at harmonics | OI < threshold | [JEKOVA-2004], [SPEC-1989] |
| 10 | SPEC | Spectral | Frequency | FSMN, A₁, A₂, A₃ | Low FSMN, high A₂, low A₃ | [SPEC-1989], [COMP5-2000] |
| 11 | Band-pass filter | Spectral | Frequency | Counts from 14.6 Hz BPF output | Low amplitude after filtering | [JEKOVA-2004] |
| 12 | Hilbert Phase Space (HILB) | Phase Space | Time+Freq | Box-fill fraction d of phase portrait | d > 0.15 | [HILB-2005] |
| 13 | EMD + DFT + SVM (VFPred) | Hybrid ML | Time+Freq | Per-frequency IMF/Residue similarity | SVM decision on feature vector | [VFPRED-2018] |

The algorithms in Groups A–D are lightweight and well-suited for real-time embedded AED systems, progressing from trivial counting rules to principled spectral analysis. Group E (phase space) achieves the best single-algorithm performance among classical methods without requiring explicit frequency thresholds. Group F bridges into machine learning: EMD provides a signal-adaptive decomposition that respects the non-stationary nature of the ECG, and the SVM learns the optimal decision boundary from data rather than from hand-tuned thresholds. Comparative evaluations [COMP4-1993, COMP5-2000, COMP55-2005, TCSC-2009, HILB-2005, VFPRED-2018] consistently show that no single classical algorithm achieves both high sensitivity and high specificity across all databases without parameter tuning, motivating the hybrid and learning-based approaches explored in later sections.


## Materials and Methods ##

### Databases ###

Evaluation is performed on four publicly available, widely used ECG databases [MITDB, CUDB, VFDB, AHADB]:

**MIT-BIH Malignant Ventricular Arrhythmia Database (VFDB)** contains 22 recordings, each approximately 35 minutes long (two-channel, 250 Hz, 12-bit). Recordings are from patients who experienced sustained ventricular tachycardia (VT), ventricular flutter (VFL), and ventricular fibrillation (VF). Annotations mark rhythm transitions throughout each recording.

**Creighton University Ventricular Tachyarrhythmia Database (CUDB)** contains 35 single-channel recordings, each 8 minutes long (250 Hz, 12-bit, 10 V range), filtered by a second-order Bessel low-pass at 70 Hz. Recordings are from patients who experienced sustained VT, VFL, and VF.

**MIT-BIH Arrhythmia Database (MITDB)** contains 48 two-channel recordings, each approximately 30 minutes long (360 Hz, 11-bit over 10 mV range). It includes a diverse range of rhythms including normal sinus rhythm (NSR), atrial fibrillation (AFIB), atrial flutter (AFL), supraventricular tachycardias, bundle branch blocks, paced rhythms, and various ectopic beats. It serves as the primary source of non-shockable rhythm diversity in this evaluation. MITDB is resampled from its native 360 Hz to 250 Hz prior to processing (see Signal Preprocessing).


**American Heart Association ECG Database (AHADB)** contains 80 two-channel recordings, each 30 minutes long (250 Hz, 12-bit). The database includes recordings from patients with sustained VT, VFL, VF, and a wide variety of non-shockable rhythms. Rhythm annotations and beat labels are provided throughout. Access requires a licence from the Emergency Care Research Institute (ECRI). The AHADB is used in [COMP55-2005] and [JEKOVA-2004] as the largest component of the combined evaluation set.

---

### Signal Preprocessing ###

All ECG signals are preprocessed using the standard pipeline established in Amann et al. [COMP55-2005] and subsequently adopted in [HILB-2005, TIME-2007, TCSC-2009], with all databases unified at 250 Hz:

1. **Resampling to 250 Hz:** MITDB recordings (native 360 Hz) are resampled to 250 Hz using polyphase anti-aliasing resampling (ratio 25:36) before any further processing. VFDB, CUDB, and AHADB are natively at 250 Hz and require no resampling. All subsequent steps operate at a uniform 250 Hz, simplifying algorithm implementation and enabling direct cross-database comparison.
2. **Moving average filter (order 5):** Removes high-frequency interspersions and muscle noise
3. **High-pass filter (1 Hz cutoff):** Suppresses baseline drift and motion artifacts
4. **Low-pass Butterworth filter (30 Hz cutoff):** Attenuates remaining high-frequency content

For multi-channel databases, only one channel is used: the first channel (modified lead II) for MITDB, channel 1 for VFDB and AHADB.

---

### Evaluation Framework ###

#### Sliding Window Segmentation

Each recording is segmented using a **sliding window with a step of 1 second**. Each window position produces one classification decision. No preselection of episodes is performed — the complete recording is analysed from start to finish, assigning the decision to the endpoint of each window interval (simulating real-time operation).

**Primary window length: 4 seconds (1000 samples at 250 Hz).** A 4-second window provides sufficient spectral resolution (frequency bin width 0.25 Hz), covers at least two full VF cycles at the lowest expected dominant frequency (~3 Hz), and yields more decisions per recording than longer windows — improving statistical resolution at episode boundaries and reducing latency in a real-time setting. The 1-second step is retained from the benchmark literature.

The implementation is window-length agnostic. To enable comparison with the published benchmark literature, each algorithm is also evaluated at the window length used in its original paper:

| Algorithm | Original window | Original Fs | Equivalent at 250 Hz |
|-----------|----------------|-------------|----------------------|
| VFLEAK-1978 | 8 s | mixed | 2000 samples |
| SPEC-1989 | ~5.12 s (1024 pt at 200 Hz) | 200 Hz | 1280 samples (~5.12 s), zero-padded to 2048 |
| TCSC-2009 | 8 s | mixed | 2000 samples |

Total decisions at the primary 4-second window, from Amann et al. [COMP55-2005] (scaled):
$$N = 2 \times 48 \times (1805 - 3) + 35 \times (508 - 3) + 2 \times 40 \times (1800 - 3) \approx 333{,}583$$
for MITDB + CUDB + AHADB [COMP55-2005] at 8 s; the 4 s primary window yields approximately four times as many decisions. The exact count for VFDB + CUDB + AHADB will be reported in Results.

#### Window Labeling

Each window is classified as **VF** or **non-VF** based on expert annotations:
- A window is labeled **VF** if its endpoint falls within an annotated VF episode
- A window is labeled **non-VF** otherwise (regardless of which rhythm is present: NSR, VT, VFL, AFIB, AFL, OTHER)

This binary labeling convention follows all benchmark studies. No distinction is made between VT and VF at this stage, because database annotations do not consistently distinguish them [COMP55-2005].

#### Performance Metrics

The following metrics are computed at the optimal operating point:

| Metric | Formula | Notes |
|--------|---------|-------|
| F1 Score | 2·TP / (2·TP + FP + FN) | Primary metric; harmonic mean of PPV and Se |
| Sensitivity (Se) | TP / (TP + FN) | Fraction of VF duration correctly detected |
| Specificity (Sp) | TN / (TN + FP) | Fraction of non-VF duration correctly rejected |
| Positive Predictive Value (PPV) | TP / (TP + FP) | Precision |
| Accuracy (Acc) | (TP + TN) / N | Overall correct fraction |
| G-Mean | √(Se × Sp) | Balanced metric under class imbalance |

This work targets **continuous monitoring** rather than AED operation, so the primary objective is accurate episode delineation with minimal delay rather than minimising inappropriate shocks. **F1 score** is the headline metric, reflecting the tradeoff between detection completeness and false alarm rate. TP, FP, TN, and FN are reported as **total duration in milliseconds** rather than window counts, to reflect actual episode coverage. Given the extreme class imbalance (VF windows constitute approximately 5–10% of all decisions), G-Mean = √(Se × Sp) is also reported [VFPRED-2018]. ROC curves and IROC are referenced where cited in benchmark papers but are not computed in this work.

---

### Benchmark Algorithms ###

Three classical signal-processing algorithms are evaluated as benchmarks. They represent distinct algorithmic families and have well-documented comparative performance on the same databases.

#### VF Leakage Filter — VFLEAK (Kuo & Dillman, 1978) [VFLEAK-1978]

The VF filter is a frequency-domain algorithm. It estimates the **mean signal period** T as:
$$T = 2\pi \sum_{i=1}^{m}|V_i| \left( \sum_{i=1}^{m}|V_i - V_{i-1}| \right)^{-1}$$
then applies a narrow band-stop filter centred at frequency 1/T. The residual energy — the **leakage** — is low for periodic signals (NSR, VT) and high for irregular signals (VF):
$$\ell = \frac{\sum_{i=1}^{m}|V_i + V_{i-T/2}|}{\sum_{i=1}^{m}(|V_i| + |V_{i-T/2}|)}$$

Classification: **VF if ℓ > ℓ₀**. The default operating point is ℓ₀ = 0.625 (without QRS detection) [COMP55-2005]. The algorithm is evaluated here at this default point without QRS detection, consistent with [COMP5-2000, COMP55-2005].

**Reported performance on MITDB + CUDB + AHADB (8 s window, no preselection) [COMP55-2005]:**
Se = 18.8%, Sp = 100%, IROC = 87%. At Sp = 95%: Se = 73.4%; at Sp = 99%: Se = 59.7%.

> **Note:** Sensitivity is low on standardised databases because the dominant VF frequency in these recordings (3–5 Hz) is lower than the range for which the original algorithm was designed [COMP5-2000].

---

#### Spectral Algorithm — SPEC (Barro et al., 1989) [SPEC-1989]

SPEC is a frequency-domain algorithm that characterises the ECG spectrum using four descriptors computed from the FFT with a Hamming window. The original paper used 1024-point segments (~5.12 s at 200 Hz); adapted here to 1280-point segments (~5.12 s at 250 Hz), zero-padded to 2048 for FFT efficiency; at the primary 4 s window, 1000 samples are used (zero-padded to 1024):

- **FSMN** — normalised frequency of the spectral maximum
- **A1** — spectral area below FSMN (relative)
- **A2** — spectral area in a band around FSMN (relative)
- **A3** — spectral area above FSMN up to 2·FSMN (relative)

**Decision rule (original thresholds):**
$$\text{VF if: } \text{FSMN} \leq 1.55 \text{ AND } A_1 < 0.19 \text{ AND } A_2 \geq 0.45 \text{ AND } A_3 \leq 0.09$$

The primary threshold parameter is A₂,₀.

**Reported performance on MITDB + CUDB + AHADB (8 s window, no preselection) [COMP55-2005]:**
Se = 29.1%, Sp = 99.9%, IROC = 89%. At Sp = 95%: Se = 69.8%; at Sp = 99%: Se = 58.9%.

> **Important caveat:** SPEC was trained on CCU recordings where VF had a dominant frequency of 5–9 Hz. On standardised databases, the typical VF dominant frequency is 3–5 Hz, causing severe sensitivity degradation with the original thresholds — Jekova [COMP5-2000] reports only 3% sensitivity. Since CCU recordings are not available for retraining, only the **Jekova adapted thresholds** (FSMN ≤ 2.5, A2 ≥ 0.35, A3 ≤ 0.25) are used in this work. The frequency mismatch and its impact on the original thresholds are noted for reference.

---

#### Threshold Crossing Sample Count — TCSC (Arafat et al., 2009) [TCSC-2009]

TCSC is a time-domain algorithm that counts how many samples cross a threshold in the binarised signal. After the standard preprocessing and normalisation to zero mean and unit standard deviation, the signal is binarised:
$$x_b(t) = \begin{cases} 1 & \text{if } x(t) \geq b \\ 0 & \text{if } x(t) < b \end{cases}$$

The **threshold crossing sample count** N_a is the number of positions where x_b(t) ≠ x_b(t−1) in the 8-second window. This is computed for a range of thresholds b ∈ [−3σ, +3σ]. Classification: **VF if N_a ≥ N_d**, where N_d = 48 was optimised on MITDB + CUDB at mixed sampling rates [TCSC-2009]. At the uniform 250 Hz used here, N_d is re-evaluated on the same databases; the retuned value is reported in Results.

**Reported performance on MITDB + CUDB (8 s window, no preselection) [TCSC-2009]:**
Se = 90.53%, Sp = 93.36%, Acc = 93.15%, AUC = 98.44%.

## References ##

### Databases ###

**[MITDB]** Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. https://physionet.org/content/mitdb/1.0.0/

**[CUDB]** Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. https://physionet.org/content/cudb/1.0.0/

**[VFDB]** Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. https://physionet.org/content/vfdb/1.0.0/

**[AHADB]** Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. https://physionet.org/content/ahadb/1.0.0/

### Comparative studies ###

**[COMP4-1993]** Clayton RH, Murray A, Campbell RW. Comparison of four techniques for recognition of ventricular fibrillation from the surface ECG. Med Biol Eng Comput. 1993 Mar;31(2):111-7. doi: https://doi.org/10.1007/BF02446668  PMID: 8331990.

**[COMP5-2000]** Jekova I. Comparison of five algorithms for the detection of ventricular fibrillation from the surface ECG. Physiol Meas. 2000 Nov;21(4):429-39. doi: https://doi.org/10.1088/0967-3334/21/4/301  PMID: 11110242.

**[COMP55-2005]** Amann, A., Tratnig, R. & Unterkofler, K. Reliability of old and new ventricular fibrillation detection algorithms for automated external defibrillators. BioMed Eng OnLine 4, 60 (2005). https://doi.org/10.1186/1475-925X-4-60

**[MODERN-2024]** Fira, Monica & Costin, Hariton & Liviu, Goras. (2024). Ventricular Fibrillation Prediction and Detection: A Comprehensive Review of Modern Techniques. Applied Sciences. 14. 11167.  https://doi.org/10.3390/app142311167

**[DEEP-2023]** Ansari Y, Mourad O, Qaraqe K, Serpedin E. Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017-2023. Front Physiol. 2023 Sep 15;14:1246746. doi: 10.3389/fphys.2023.1246746. PMID: 37791347; PMCID: PMC10542398.
https://pmc.ncbi.nlm.nih.gov/articles/PMC10542398/

### Algorithm papers ###

**[VFLEAK-1978]** Kuo S and Dillman R 1978 Computer detection of ventricular fibrillation Proc. Computers in Cardiology 1978 (Long
Beach, CA: IEEE Computer Society Press) pp 347—9

**[SPEC-1989]** Barro S, Ruiz R, Cabello D, Mira J. Algorithmic sequential decision-making in the frequency domain for life threatening ventricular arrhythmias and imitative artefacts: a diagnostic system. J Biomed Eng. 1989;11:320–8. doi: https://doi.org/10.1016/0141-5425(89)90067-8

**[EMD-1998]** Huang, Norden & Shen, Zheng & Long, Steven & Wu, Manli & Shih, Hsing & Zheng, Quanan & Yen, Nai-Chyuan & Tung, Chi-Chao & Liu, Henry. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences. 454. 903-995. https://doi.org/10.1098/rspa.1998.0193.

**[EMD-2010]** A. Zeiler, R. Faltermeier, I. R. Keck, A. M. Tomé, C. G. Puntonet and E. W. Lang, "Empirical Mode Decomposition - an introduction," The 2010 International Joint Conference on Neural Networks (IJCNN), Barcelona, Spain, 2010, pp. 1-8, doi: https://doi.org/10.1109/IJCNN.2010.5596829.

**[JEKOVA-2004]** Jekova I, Krasteva V. Real time detection of ventricular fibrillation and tachycardia. Physiol Meas. 2004 Oct;25(5):1167-78. doi: https://doi.org/10.1088/0967-3334/25/5/007 PMID: 15535182.

**[HILB-2005]** Amann, Anton & Tratnig, R. & Unterkofler, Karl. (2005). A new ventricular fibrillation detection algorithm for automated external defibrillators. Computers in Cardiology. 32. 559 - 562. doi: https://doi.org/10.1109/CIC.2005.1588162

**[TIME-2007]** Amann A, Tratnig R, Unterkofler K. Detecting ventricular fibrillation by time-delay methods. IEEE Trans Biomed Eng. 2007 Jan;54(1):174-7. doi: https://doi.org/10.1109/TBME.2006.880909  PMID: 17260872. 

**[TCSC-2009]** Arafat, M.A., Chowdhury, A.W. & Hasan, M.K. A simple time domain algorithm for the detection of ventricular fibrillation in electrocardiogram. SIViP 5, 1–10 (2011). https://doi.org/10.1007/s11760-009-0136-1

**[VFPRED-2018]** A Fusion of Signal Processing and Machine Learning techniques in Detecting Ventricular Fibrillation from ECG Signals
https://ar5iv.labs.arxiv.org/html/1807.02684


## Decisions ##

Design decisions recorded here as they are resolved. Each entry references the question it answers and the feedback session that informed it.

| Q | Topic | Status |
|---|-------|--------|
| Q1 | Database Selection | ✓ VFDB + CUDB + AHADB (primary); + MITDB (extended) |
| Q2 | AHADB Availability | ✓ Licensed, included |
| Q3 | SPEC Thresholds | ✓ Jekova adapted thresholds |
| Q4 | VFL Labeling | ✓ Separate class; run all three configs |
| Q5 | Transition Windows | ✓ 90% purity for training; all windows for evaluation |
| Q6 | Window Lengths | ✓ 4 s primary + each algorithm's original window |
| Q7 | Metrics | ✓ F1 primary; Se, Sp, PPV, Acc, G-Mean; durations in ms; ROC deferred |
| Q8 | TCSC N_d | ✓ N_d=48 for reference + retuned at 250 Hz as primary |
| Q9 | Proposed Method | ⏳ Open |

---



### Q1: Database Selection — MITDB Inclusion ✓ Resolved

Two evaluation sets are defined:

- **Primary set:** VFDB + CUDB + AHADB (selected records). Focuses on databases with rich VF content; AHADB records are selected to exclude recordings lacking relevant arrhythmia episodes.
- **Extended set:** VFDB + CUDB + AHADB + MITDB. Used for secondary evaluations and comparability with [COMP55-2005]. MITDB contributes non-shockable rhythm diversity despite its very few VF episodes.

All benchmark results are reported on both sets where applicable.

---

### Q2: AHADB Availability ✓ Resolved

AHADB licence is available. AHADB is included as the fourth database in the primary evaluation set (VFDB + CUDB + AHADB), matching the COMP55-2005 combination and enabling direct comparison with published benchmark totals (~333k decisions).

---

### Q3: SPEC Threshold Version ✓ Resolved

The SPEC algorithm is severely affected by the dominant VF frequency mismatch between its training data (CCU, 5–9 Hz) and standardised databases (3–5 Hz). Two threshold versions exist:
- **Original thresholds** (Barro 1989): ~3% sensitivity on standardised databases [COMP5-2000]
- **Adapted thresholds** (Jekova 2000): FSMN ≤ 2.5, A2 ≥ 0.35, A3 ≤ 0.25 → ~79% sensitivity

**Question:** Should we evaluate both threshold versions, or only the original? The original is technically correct for "pure benchmark" comparison with [COMP55-2005], but the adapted version is more practically meaningful.

**Decision:** CCU recordings are not available, so the original thresholds cannot be meaningfully validated or retrained. The **Jekova adapted thresholds** (FSMN ≤ 2.5, A2 ≥ 0.35, A3 ≤ 0.25) are used as the sole SPEC operating point. The frequency mismatch and its effect on the original thresholds are documented in the Results discussion.

---

### Q4: Labeling of VFL Episodes ✓ Resolved

Ventricular flutter (VFL) is physiologically intermediate between VT and VF: regular sinusoidal oscillation, 200–350 BPM, haemodynamically unstable. Databases annotate it separately from VF. However:
- Some papers include VFL in the "VF" (shockable) class [TCSC-2009, TAYLOR-2018]
- Others include VFL in the "non-VF" class or exclude it
- COMP55-2005 uses binary VF vs. non-VF based on the native annotation

**Question:** How should VFL be labeled — as VF, as non-VF, or excluded from evaluation? This has a significant impact on reported sensitivity numbers.

**Decision:** VFL is retained as a **separate class** in the dataset. Evaluation is run under three configurations — VFL-as-VF, VFL-as-non-VF, and VFL-excluded — and results are reported for each. This preserves full flexibility and allows direct comparison with papers using any of the three conventions.

---

### Q5: Handling of VF Onset/Offset Transitions ✓ Resolved

Segments spanning the transition between normal rhythm and VF (onset) or VF and post-shock rhythm (offset) contain both morphologies within the 8-second window. The benchmark papers assign the label based on the endpoint annotation, which may be misleading for transition windows.

> **Note — Endpoint labeling:** The label of a window is determined solely by the annotation active at its **last sample**. A window ending inside a VF episode is labeled VF even if VF onset occurs mid-window and the majority of the window is non-VF, and vice versa. This convention simulates real-time AED operation (decision made at end of analysis period) and is used in all benchmark studies, but introduces label noise near episode boundaries.

**Question:** Should transition windows (e.g., those where the onset/offset annotation falls within the window interior) be (a) included with endpoint labeling (standard practice), (b) excluded, or (c) labeled by the dominant rhythm (> 50% of window duration)?

**Decision:** A two-regime approach is used:
- **Training:** only windows where the episode occupies ≥ 90% of the window duration are used. This ensures the model learns from clean, morphologically homogeneous examples.
- **Evaluation:** all windows are included, with endpoint labeling. Transition windows are not excluded, reflecting realistic deployment where the algorithm cannot skip boundary windows. This also maintains comparability with benchmark results.

---

### Q6: Window Length — 8 s vs. Shorter Windows ✓ Resolved

The 8-second standard comes from AED design requirements (decision at end of analysis period). However, shorter windows (5 s, 4 s, even 2 s) have been shown to be viable [TCSC-2009, VFPRED-2018, TAYLOR-2018, DETECVF-2024], and shorter windows give more decisions per recording (better statistics).

**Question:** Should the benchmark evaluation be performed at multiple window lengths (4 s, 5 s, 8 s) to enable comparison with the full published literature, or should we fix to 8 s for simplicity?

**Decision:** Each algorithm is evaluated at (b) the **primary 4 s window** plus its **original paper window length**, enabling direct numerical comparison with published results. Window lengths per algorithm: VFLEAK → 4 s + 8 s; SPEC → 4 s + ~5.12 s (1280 samples); TCSC → 4 s + 8 s.

---

### Q7: Evaluation Metric Priority ✓ Resolved

Different papers optimise for different metrics:
- AED papers optimise specificity (avoid inappropriate shocks)
- Monitoring/prediction papers optimise sensitivity (avoid missed VF)
- [VFPRED-2018] argues for G-Mean Accuracy = √(Se × Sp)
- IROC (AUC) is the most complete summary

**Question:** What is the primary metric for the eventual comparison in this paper? Should IROC be the headline number, or sensitivity at a fixed specificity (e.g., Se at Sp = 99%)?

**Decision:** The primary metric is **F1 score**, supported by Se, Sp, PPV, Accuracy, and G-Mean = √(Se × Sp). TP, FP, TN, FN are reported as **total duration in milliseconds** rather than window counts, to reflect actual episode coverage. ROC curve and IROC are **not computed** in this work — they are referenced only where cited in benchmark papers. *Note: reconsider whether to add ROC/IROC if comparison with published IROC values becomes necessary.*

---

### Q8: Reproducibility — TCSC Implementation ✓ Resolved

The TCSC paper [TCSC-2009] reports N_d = 48 as the optimal threshold. However, the threshold depends on the preprocessing pipeline, sampling rate, and window length. Re-implementing TCSC from scratch may produce slightly different results.

**Question:** Should we implement TCSC strictly from the paper (with verification against reported numbers), or is it acceptable to retune N_d on our dataset and report the retuned value as our reproduction?

**Decision:** Option (c) — both are reported. TCSC is implemented at uniform 250 Hz (MITDB resampled), which requires adjusting any sample-count-dependent parameters (including N_d). Results are reported with (1) N_d = 48 as in the original paper for reference comparability, and (2) N_d retuned on our dataset at 250 Hz as the primary operating point. Any discrepancy from published numbers is attributed to the unified sampling rate and database differences, and documented in Results.

---

### Q9: Future Algorithm Scope

The instructions indicate this paper will eventually compare benchmark algorithms with a proposed new method. The nature of the proposed method is not yet specified.

**Question:** What is the general approach of the proposed method (signal processing, ML features + classifier, deep learning, other)? This affects which features and preprocessing steps should be described in M&M now vs. later sections.

## Feedback ## 

### Gusev(2026-03-06) ###

---

**1. Primary evaluation metric is F1 score** *(→ resolved in Q7)*

F1 score (harmonic mean of precision and recall) is the primary performance measure. Supporting metrics include Sensitivity (Se), Specificity (Sp), Positive Predictive Value (PPV), Accuracy, and G-Mean = √(Se × Sp).

---

**2. TP, FP, TN, FN reported as total duration in milliseconds** *(→ resolved in Q7)*

Rather than counting windows, confusion matrix entries are expressed as cumulative durations in milliseconds. This better reflects actual episode coverage and is more meaningful for clinical interpretation.

---

**3. Clean training samples vs. mixed (all) samples for testing** *(→ resolved in Q5)*

Training should use only "clean" windows where the rhythm episode occupies the full (or near-full) window duration. Testing and evaluation should be performed on all windows, including transition windows at episode onset and offset. This distinction separates the quality requirements of the learning phase from the realism requirements of the evaluation phase.

---

**4. ROC curve is referenced but IROC is not a priority** *(→ resolved in Q7)*

ROC curves and IROC values appear frequently in the benchmark literature and are cited where relevant for comparison. However, computing the full ROC curve is not a goal of this work. The focus is on operating-point metrics (F1, Se, Sp) rather than threshold-sweep summaries.

---

**5. From per-window decisions to episode annotations** *(→ relates to Q5, Q6; post-processing)*

Window-based classification is an intermediate step. The ultimate output required for standard compliance testing is an **episode annotation file**: a sequence of labeled intervals with explicit start and end times, in WFDB-compatible format. These annotation files are consumed by the standard WFDB evaluation tools — `bxb`, `rxr`, and `epicmp` — which compare detector output against reference annotations at the episode level, not the window level.

The bridge from per-window decisions to episode annotations is a **post-processing smoothing step based on majority voting**: each signal sample participates in multiple overlapping windows (up to W seconds worth, where W is the window length). The sample is assigned the majority label across all windows it belongs to. Contiguous runs of the same label are then collapsed into episodes with start and end timestamps. This reduces isolated false positives and false negatives and produces clean episode boundaries.

Because edge samples of each window are down-weighted by the FFT tapering window (see point 6), they contribute less to the spectral decision — making interior samples the most reliable contributors to the majority vote. Where no majority is reached, the sample may be left unlabeled or assigned to the preceding episode, rather than forcing a noisy label.

This system is designed for **continuous monitoring**, not for AED/defibrillator operation. It is not required to be strictly real-time, but should operate with **minimal delay** — a latency of one window length (4 s primary, or up to 8 s for benchmark configurations) is acceptable. The focus is on accurate episode delineation over sustained recordings rather than fastest-possible single-decision latency.

---

**6. FFT Hanning window attenuates samples at the edges** *(technical note, relates to point 5)*

The Hanning (or Hamming) window applied before FFT tapers to near-zero at both ends of the segment. Samples near the start and end of the window therefore contribute little to the spectral estimate. This is an inherent property of windowed FFT: interior samples dominate the decision. In the context of majority voting (point 5), this means that episode boundary regions — where the rhythm is transitioning — are naturally down-weighted, which is desirable. However, it also means that the precise onset and offset timestamps in the output annotation may be slightly offset from the true physiological boundary, depending on how the majority vote resolves near the transition.

---

**7. For ambiguous windows: prefer interior windows or withhold decision** *(→ relates to Q5)*

When a window cannot be confidently labeled (e.g., it straddles an episode boundary and no dominant rhythm is clear), preference is to use windows drawn from the interior of an episode rather than from its edges. If neither a confident label nor a confident decision can be assigned, the window may be excluded rather than forcing a noisy label.