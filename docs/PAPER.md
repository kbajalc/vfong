# Paper Notes for `vf_filter`

This document now uses the converted thesis files `doc/HONG-2016.md` and `doc/HONG-2016.html` in addition to the repository source. The markdown conversion gives us the thesis text and tables; the HTML preserves embedded figures and captions that help explain the intended workflow and failure modes.

The conversion is useful but noisy. It repeats university seals and watermarks, and some OCR or layout recovery artifacts are visible. The sections below therefore distinguish between:

- claims stated directly in the thesis text
- implementation details verified in this repository
- places where the repo has evolved beyond the thesis experiments

## 1. Executive Summary

The thesis is not merely “about VF detection” in the abstract. It is a 2016 NTU master thesis titled “Detecting Life-Threatening Arrhythmia with Machine Learning Algorithms,” and its central claim is much more specific:

- build an AED-oriented arrhythmia classifier that follows AHA reporting rules
- classify 8-second ECG segments into shockable, intermediate, and non-shockable groups
- use handcrafted signal-processing features plus machine learning, with SVM as the primary thesis model

The thesis abstract reports that the proposed AED algorithm achieved:

- 93.21% sensitivity
- 99.88% specificity
- 89.28% precision

for the final shock/no-shock decision, and states that these results satisfied the AHA performance goals.

The repository matches that overall structure very closely, but it is broader than the thesis artifact alone. The repo contains the thesis-era data pipeline and feature engine, while also exposing extra models, experiment scripts, and utility tooling that make it a research platform rather than a minimal thesis snapshot.

## 2. What the Thesis Actually Says

The thesis frames the problem as out-of-hospital cardiac arrest, with ventricular fibrillation and pulseless ventricular tachycardia being critical rhythms for prompt defibrillation. Its motivation is that many earlier VF studies reported good binary-classification numbers, but did not follow the AHA requirements that matter for AED deployment.

The thesis therefore emphasizes three things repeatedly:

1. AHA-compatible class definitions and reporting.
2. A more comprehensive public dataset than many prior studies used.
3. An open implementation intended to be reproducible and extensible.

That is a stronger and more concrete statement than the earlier version of this note, which had to infer intent from the code alone.

## 3. Thesis Workflow

The thesis includes an explicit workflow diagram for ventricular arrhythmia classification. In words, the documented pipeline is:

1. collect datasets
2. segment and label rhythms
3. exclude noisy or artifact-laden segments
4. exclude operationally defined asystole
5. preprocess each 8-second ECG segment
6. compute QRS-based statistics, amplitude, and handcrafted features
7. run a machine-learning classifier
8. output `shockable`, `intermediate`, or `non-shockable`

Two details from the flowchart matter because they are easy to miss when reading only the code:

- the thesis defines asystole operationally as amplitude `< 0.15 mV`
- the thesis flowchart routes surviving segments into a classifier evaluated with 5-fold cross-validation during model development

This matches the repo reasonably well, though the executable code also supports repeated train/test iterations beyond the figure’s simplified summary.

## 4. Data Sources and Inclusion Rules

The thesis and the repo align on the major dataset pool:

- `mitdb`
- `vfdb`
- `cudb`
- `edb`
- `mghdb`

The thesis is more specific about why these were chosen. It argues that the common public VF datasets were not diverse enough, especially for non-shockable rhythms and for robust AED evaluation, so it expanded the dataset using EDB and selected lead-II signals from MGHDB.

Important inclusion rules stated in the thesis and reflected in `vf_data.pyx`:

- MITDB: use lead II from the first channel.
- MGHDB: select channels containing lead II.
- VFDB: exclude most VT segments because rapid-vs-slow VT requires beat annotations for heart-rate estimation.
- CUDB: exclude NSR because CUDB documentation effectively lumps non-VF rhythms into NSR, making them unsafe as true non-shockable labels.

The thesis dataset table reports a total of 84,022 non-overlapping 8-second samples from 293 records after corrections and exclusions. The largest class by far is NSR, mostly from EDB. The core shockable counts reported by the thesis are:

- 746 coarse VF samples
- 75 rapid VT samples

The intermediate class is much smaller:

- 47 fine VF samples
- 10 slow VT samples in Table 3.1

The later limitations section also highlights that patient counts for fine VF and other VT remained inadequate for strict AHA-style expectations.

The repo-side correction machinery also lines up with the thesis text. The thesis says 747 labels were corrected and 87 samples were excluded for severe artifacts, with details stored in `corrections_s8.txt`.

## 5. Segmentation, Preprocessing, and Labeling

### 8-second segmentation

The thesis explicitly justifies 8-second segments based on prior studies and preliminary tests. It also states a methodological choice that is important for AED realism: segmentation is performed before preprocessing, because a real AED does not get to inspect a long future recording before deciding.

That directly matches the repo’s working conventions:

- `feature_extraction.py -s 8`
- `features/features_s8.dat`
- `test_classifiers.sh` expecting the `s8` feature file

### Preprocessing

The thesis specifies this preprocessing pipeline for each segment:

1. mean subtraction
2. normalization
3. five-order moving average
4. drift suppression with 1 Hz cutoff
5. zero-phase Butterworth low-pass filtering at 30 Hz

It also says normalization is skipped when computing the amplitude feature, because amplitude thresholds are in physical units.

### Amplitude and asystole

The thesis uses peak-to-peak amplitude computed from adjacent peaks and valleys located with `scipy` extrema detection. That is a stronger statement than “some amplitude feature exists”; it explains exactly why `Amplitude` is central:

- coarse VF vs fine VF is separated at `0.2 mV`
- operational asystole is defined as `< 0.15 mV`

That aligns with the repo’s AHA-oriented label logic and with helper tooling like `asystole_check.py`.

### AHA-oriented labels

The thesis follows the three-way AHA framing used in the repo:

- shockable: coarse VF and rapid VT
- intermediate: fine VF and slow VT
- non-shockable: everything else, including asystole

The thresholds are exactly the ones used in code:

- rapid VT: heart rate `> 180 BPM`
- coarse VF: amplitude `> 0.2 mV`

So the repo’s label constants are not just plausible choices; they are thesis-grounded design decisions.

## 6. Feature Set and Clinical Intent

The thesis states that 27 features are extracted from each 8-second ECG segment, grouped into several categories. This is the exact feature inventory implemented in `vf_features.pyx`:

- Time-domain: `TCSC`, `TCI`, `STE`, `MEA`, `MAV`, `Count1`, `Count2`, `Count3`, `Amplitude`
- QRS-derived: `RR`, `RR_Std`, `RR_CV`, `UR`, `VR`
- Frequency-domain: `VF`, `M`, `A2`, `FM`
- Complexity: `LZ`, `SpEn`
- EMD-based: `IMF1_LZ` through `IMF5_LZ`
- Phase space: `PSR`, `HILB`

The thesis also explains why these categories were mixed instead of picking a single family. Time-domain and spectral features capture rapidity and morphology, while complexity and phase-space methods capture irregularity. That is exactly how the repo reads: it is a feature aggregation strategy rather than one novel feature plus a thin classifier.

Some thesis details that materially improve understanding of the repo:

- `TCI` is averaged over eight 1-second windows.
- `Count1`/`Count2`/`Count3` require a 250 Hz design assumption and were computed after resampling.
- `SpEn` was computed on the last 1250 samples after resampling to 250 Hz, i.e. the last 5 seconds of an 8-second segment.
- `PSR` uses a 0.5-second delay.
- the QRS detector is the Patrick Hamilton OSEA implementation, which is exactly what the repo wraps in `qrs_detect.pyx` and `osea20-gcc/`.

The thesis also names likely important feature roles rather than listing them mechanically:

- threshold-crossing features proxy rapid, wide-complex rhythms
- RR statistics proxy rate and regularity
- VF leak and spectral moments proxy sine-like ventricular rhythms vs broader NSR spectra
- complexity and phase-space features proxy irregularity
- EMD-based LZ features help separate VF from VT

## 7. Signal Processing and Native Components

The repo includes the same native surfaces implied by the thesis implementation section:

- Cython preprocessing and data loading
- a C implementation of Lempel-Ziv for speed
- Hamilton/OSEA QRS detection code
- distributed feature extraction via joblib and Pyro4

That makes the current build fixes and native-module work from this session directly relevant to recovering the thesis workflow on a modern machine.

The most faithful mapping is:

- `qrs_detect.pyx` and `osea20-gcc/`: thesis QRS detector
- `signal_processing.pyx`: preprocessing helpers
- `vf_features.pyx`: thesis feature extraction engine
- `vf_data.pyx`: segmentation, rhythm handling, and label preparation
- `feature_extraction.py`: large-scale feature generation

That makes the repo a hybrid of:

- classical signal processing
- handcrafted feature engineering
- classical machine learning

## 8. Thesis Model and Reported Results

The thesis model family is narrower than the repo's current menu. The thesis centers on multiclass SVM, especially SVM with an RBF kernel, and compares it against:

- linear SVM
- logistic regression

The thesis evaluation procedure is also explicit:

- random 70/30 train/test split
- 100 repeated iterations
- average reported metrics across those iterations
- cross-validation on the training side for model selection

The headline thesis result for the AHA-style shock/no-shock decision is:

- SVM-RBF: 93.21% sensitivity, 99.88% specificity, 89.28% precision

The detailed tables add important nuance:

- coarse VF sensitivity: 93.33%
- rapid VT sensitivity: 92.14%
- all non-shockable specificity: 99.88%
- average testing F1 for SVM-RBF: 77.38%

The thesis discussion also makes a practical point that matters for this repo: simpler linear SVM and logistic-regression baselines still satisfied the AHA thresholds in the reported experiments, so the best-precision model was not automatically the only operationally reasonable one.

## 9. Where the Thesis Performed Poorly

The thesis is very clear that the intermediate class remained difficult.

Reported intermediate-class results:

- SVM-RBF intermediate sensitivity: 41.24%
- fine VF sensitivity: 46.38%
- slow VT sensitivity: 18.38%

The author states this directly in the discussion: shockable and non-shockable rhythms were recognized well, but intermediate rhythms were often missed. In practice, the hard cases are exactly the ones near the clinically defined thresholds, especially fine-VF vs coarse-VF and slow-VT vs rapid-VT.

## 10. Error Analysis and Figures

The converted HTML is especially useful here because the embedded figures make the thesis's failure analysis much more concrete.

The recurring error causes named by the thesis are:

- low-frequency noise and severe baseline wander
- large amplitude variation within a segment
- high-frequency noise
- borderline coarse-VF vs fine-VF amplitude cases
- labeling errors in public datasets
- non-shockable wide-complex tachycardias that mimic VT

The figure set in the HTML materially improves interpretation of these points:

- Figure 2.1 shows the intended end-to-end classification flow.
- Figure 2.2 shows the peak-to-peak amplitude measurement process.
- Figures 2.3 and 2.4 explain the threshold-crossing family visually.
- Figures 2.5 through 2.7 show how spectral, EMD, and phase-space methods are supposed to separate rhythms.
- Figures 4.1 through 4.4 show concrete failure modes caused by baseline wander, broadband noise, and mislabeled records.
- Figure 4.5 shows a morphology-level clinical confounder: AF with pre-existing LBBB can mimic VT even for human readers.

This also strengthens confidence that helper scripts in the repo such as `asystole_check.py`, `inspect_record.py`, and the correction-file pathway are not incidental utilities. They match documented thesis concerns about amplitude thresholds, artifact handling, and label cleanup.

## 11. Repo Models Beyond the Thesis

`vf_classify.py` supports these model names:

- `logistic_regression`
- `random_forest`
- `adaboost`
- `gradient_boosting`
- `svc_linear`
- `svc_poly`
- `svc_rbf`
- `mlp1`
- `mlp2`

So the repo should be understood as "thesis core plus later comparison harness." The thesis itself is SVM-centered, but the executable code is set up to compare multiple estimators, run grid search, and perform feature-selection experiments.

That broader scope is visible in:

- `vf_classify.py` for estimator creation and parameter grids
- `vf_tests.py` for repeated train/test experiments and exports
- `test_classifiers.sh` for batch execution

## 12. What the Repo Implements Beyond the Thesis Text

Even where the repo matches the thesis closely, it also contains extra engineering surfaces that are only lightly implied by the write-up:

- distributed feature extraction via Pyro4 and joblib
- correction-file support for label overrides
- viewer and inspection tools
- standalone test and batch scripts
- multiple classifier families, not just the thesis comparison set

So the cleanest mental model is:

- the thesis defines the clinical framing, data curation rules, thresholds, and primary evaluation story
- the repo is the working implementation and experiment platform

## 13. Current State of the Implementation

The repo has been modernized enough to run large parts of the original workflow on a current macOS/Python stack, but there are still practical caveats.

### What has been fixed already

- native build definitions in `setup.py`
- `qrs_detect` symbol/link issues
- `vf_features` native helper integration
- SciPy namespace changes
- NumPy/Cython dtype mismatches in active paths
- old scikit-learn imports in `vf_tests.py` and `vf_classify.py`
- Makefile bootstrap target: `make setup`

### What is still externally required

- WFDB-compatible ECG datasets must exist outside the repo
- `setwfdb` or equivalent environment setup is still assumed by shell scripts

### What is still research-code fragile

- some older scripts outside the active path may still assume legacy sklearn APIs
- the project depends on bundled legacy packages (`ptsa`, `pyeeg`, `sknn` for some paths)
- the converted thesis files are much better than the original PDF for documentation, but they still include OCR and layout noise

## 14. Runnable Command Flow

This is the command flow that best matches the repo as it exists now.

### Environment bootstrap

```bash
make setup
```

That now does the following:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install Cython numpy scipy matplotlib joblib Pyro4 scikit-learn
.venv/bin/python -m pip install -e .
.venv/bin/python setup.py build_ext --inplace
```

### Activate the environment

```bash
source .venv/bin/activate
```

### Build extensions again if needed

```bash
make build
```

### Quick functional check for QRS detection

```bash
MPLBACKEND=Agg python qrs_test.py -r vfdb/422 -b 385788 -d 8
```

### Generate feature data

```bash
python feature_extraction.py -o features/features_s8.dat -s 8
```

Optional knobs:

```bash
python feature_extraction.py -o features/features_s8.dat -s 8 -j -1
python feature_extraction.py -o features/features_s8.dat -s 8 -r 250
python feature_extraction.py -o features/features_s8.dat -s 8 -c corrections_s8.txt
```

### Run classifier experiments directly

```bash
python vf_tests.py -i features/features_s8.dat -t 100 -m logistic_regression -s custom -o results.csv -e errors.csv
```

### Run the batch experiment script

```bash
bash test_classifiers.sh
```

That script expects:

- WFDB environment configured
- `features/features_s8.dat` already generated

## 15. Recommended Reading of the Repo

If you want to understand the repository in the same order as the paper/problem statement, read files in this order:

1. `README`
2. `doc/HONG-2016.pdf`
3. `vf_data.pyx`
4. `feature_extraction.py`
5. `vf_features.pyx`
6. `vf_classify.py`
7. `vf_tests.py`
8. `test_classifiers.sh`

That sequence goes from problem framing to data to features to modeling to experiment automation.

## 16. Bottom Line

Even without direct PDF text extraction, the repository clearly implements a paper-style AED arrhythmia detection pipeline:

- curated multi-dataset ECG segment generation
- handcrafted VF/VT-oriented feature extraction
- AHA-informed target labeling
- repeated machine-learning evaluation across multiple classifiers

The strongest code evidence is in:

- `vf_data.pyx` for datasets and segmentation
- `vf_features.pyx` for literature-derived features
- `vf_classify.py` for AED/AHA labeling and model setup
- `vf_tests.py` for the experiment and reporting harness

If the PDF is intended as the documentation for this repo, then this repository should be read as the executable implementation of that research workflow, with a broader experimental scope than a single-paper minimal reproduction.