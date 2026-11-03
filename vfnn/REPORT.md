# vfnn results report: one model under three splits

First real results of `vfnn`, run on 2026-11-03. `DESIGN.md` describes the model and the
protocol, and `NOTES.md` the state of the work. This report covers research question 1 of
the design: how much of the near-perfect accuracy reported for small CNNs on VF detection
survives when random segment folds are replaced by record-wise and leave-one-database-out
splits. Section 7 sets the network against the JEKOVA band-pass detector on the same windows
and folds.

## 1. Setup

The same network and training recipe ran under three splits. Nothing else changed between
the runs.

| Item | Value |
|---|---|
| Data | vfdb, cudb, mitdb, ahadb (all 79 records): 184 records, 81.3 h |
| Windows | 8 s at 250 Hz, 1 s step: 291,532 windows, 12,353 positive (4.2%) |
| Positive class | VF and VFL covering at least 50% of the window. VT is a negative |
| Model | B-lite: 6 residual blocks, 286k parameters, 48.5 MMAC per window, receptive field 6.2 s |
| Training | Adam 1e-3, divided by 10 after 2 stalled dev epochs, stop after 6, at most 30 epochs |
| Batch and epoch | batch 256, a full pass over the training windows per epoch, jitter and polarity augmentation, seed 0 |
| Threshold | the one that maximises F1 on the dev set |
| Hardware | Apple M5 Max (48 GB), PyTorch MPS, one process per fold |

The batch size and the full-pass epochs differ from the starting recipe in `NOTES.md`
(batch 64, 40,000 windows per epoch), which was planned for a slower machine. A batch-64
control has not been run.

The three splits are:

- Random 10-fold: windows are assigned to folds at random, so windows from one record sit in
  both training and test. This is the protocol behind the published 99% figures. Dev is a
  random 15% of the training windows.
- Record-wise 5-fold: whole records go to one fold, balanced on positive and total windows.
  Dev is whole records held out of training.
- Leave-one-database-out: each database in turn is the test set, and the model trains on the
  other three. Dev is whole records.

## 2. Main result

| Split | AUC | Se | Sp | PPV | F1 | Accuracy |
|---|---|---|---|---|---|---|
| Random 10-fold | 0.998 ± 0.001 | 0.954 ± 0.012 | 0.991 ± 0.002 | 0.83 ± 0.03 | 0.89 ± 0.02 | 0.990 |
| Record-wise 5-fold | 0.985 ± 0.010 | 0.91 ± 0.12 | 0.972 ± 0.011 | 0.60 ± 0.13 | 0.72 ± 0.13 | 0.970 |
| Leave-one-database-out | 0.92–1.00 | 0.64–0.95 | 0.87–1.00 | 0.36–0.98 | 0.52–0.96 | 0.972 |

Values are the mean and standard deviation over folds. Accuracy comes from the confusion
counts pooled over folds. Leave-one-database-out is given as the range over its four folds,
because the folds are too uneven to average (mitdb has one VF record, cudb has 33).

Under random folds the model reaches 99.0% accuracy, the same level as the 99% reported by
Nasimi and Yazdchi (LDIAED, 2022) and the 99.2% of Hai (2021), both on random folds. The task
here is harder than LDIAED's, because VT counts as a negative. With record-wise folds, F1
falls from 0.89 to 0.72 and PPV from 0.83 to 0.60. AUC drops less, from 0.998 to 0.985, so
the network still ranks VF windows well on patients it has never seen. Most of the loss is in
the thresholded metrics.

Leakage also shrinks the fold-to-fold spread. The standard deviation of F1 is 0.02 under
random folds and 0.13 under record-wise folds. A random-fold protocol makes the detector look
more stable as well as more accurate.

These numbers are not a head-to-head with LDIAED or Hai: their labels, databases and windows
differ. The comparison that matters is within this table, where only the split changes.

## 3. Per-fold results

Record-wise 5-fold:

| Fold | Epochs | Threshold | AUC | Se | Sp | PPV | F1 |
|---|---|---|---|---|---|---|---|
| g0 | 18 | 0.35 | 0.996 | 0.994 | 0.984 | 0.720 | 0.835 |
| g1 | 14 | 0.39 | 0.983 | 0.928 | 0.971 | 0.590 | 0.721 |
| g2 | 14 | 0.44 | 0.993 | 0.983 | 0.983 | 0.726 | 0.835 |
| g3 | 8 | 0.52 | 0.983 | 0.951 | 0.964 | 0.539 | 0.688 |
| g4 | 10 | 0.93 | 0.971 | 0.703 | 0.959 | 0.424 | 0.529 |

Leave-one-database-out (the fold is named by the held-out database):

| Held out | Epochs | Threshold | AUC | Se | Sp | PPV | F1 | Positive windows |
|---|---|---|---|---|---|---|---|---|
| ahadb | 12 | 0.61 | 1.000 | 0.952 | 0.999 | 0.972 | 0.962 | 5,495 |
| cudb | 9 | 0.11 | 0.921 | 0.807 | 0.870 | 0.627 | 0.706 | 3,745 |
| mitdb | 10 | 0.95 | 1.000 | 0.640 | 1.000 | 0.980 | 0.774 | 150 |
| vfdb | 8 | 0.97 | 0.943 | 0.940 | 0.882 | 0.355 | 0.515 | 2,963 |

The random 10 folds are uniform (AUC 0.997 to 0.999, F1 0.86 to 0.92) and are in
`random8_lite.csv`.

Unseen databases are the hard case. With cudb held out, AUC is 0.92. Most of cudb's negative
windows lie just before or after VF, and some are noisy, which makes them harder to separate
than the sinus rhythm that dominates the other databases. With vfdb held out, the model trains
on almost no VT and its specificity on vfdb drops to 0.88. The ahadb and mitdb folds are
close to perfect, but mitdb rests on a single VF episode and says little.

## 4. Where the errors are

### VT is the main source of false positives

| Split | vfdb share of FP | vfdb alarm s/h | cudb alarm s/h | ahadb alarm s/h | mitdb alarm s/h |
|---|---|---|---|---|---|
| Random 10-fold | 85% | 165 | 66 | 1.1 | 0.1 |
| Record-wise 5-fold | 82% | 498 | 243 | 5.5 | 1.8 |
| Leave-one-database-out | 72% | 396 | 369 | 3.8 | 0.1 |

Alarm seconds per hour count one second per false-positive window, since windows step by
1 s. vfdb holds 87 of the 91 VT minutes in the corpus. In the record-wise run, every one of
the 4,811 false-positive windows in vfdb's VF-free records falls in one of the 14 records
that contain VT, and records with more VT produce more alarms (record 427: 14.7 VT minutes,
1,187 false-positive windows; record 611: 15.1 minutes, 897). Records without VT are quiet:
the 69 ahadb series 1–7 records alarm about 2.3 s per hour and the VF-free mitdb records
about 1.7 s per hour.

Random folds cut the vfdb alarms by two thirds (498 to 165 s/h), because the model has seen
other windows of the same VT episodes during training. This part of the leakage gap is the
one that matters clinically.

### Thresholds do not transfer between records

The F1-optimal threshold is stable under random folds (0.86 ± 0.05), because dev and test
windows come from the same records. With whole records held out it ranges from 0.35 to 0.93
across the record-wise folds and from 0.11 to 0.97 across the database folds. Fold g4 shows
the cost: its threshold of 0.93 cut sensitivity to 0.70, and its misses concentrate in three
records (cudb cu20, 211 of 260 VF windows missed; ahadb 8206, 203 missed; vfdb 426, 163
missed). AUC does not depend on the threshold and is the more reliable figure for these
splits.

### Training length is a leakage signal too

Random folds ran 19 to 29 epochs, because their dev loss kept improving even at learning
rates of 1e-6 and below. Record-wise folds stopped after 8 to 18 epochs and the database
folds after 8 to 12. A dev set that keeps improving that long is close to the training set.

## 5. What this does and does not show

The results answer research question 1. Most of the near-perfect published figure comes
from the protocol: the same network reaches 99% accuracy on random folds and has a PPV of
0.60 when the test patients are new. On strict splits it is a good detector (AUC 0.985 on
unseen records), and its errors concentrate on VT and on databases it was not trained on.

The results do not show that the network is better than a handcrafted detector. Section 7
scores the JEKOVA band-pass detector on the same windows, labels and folds, and the network
does not beat it on any strict split.

Further limits:

- Everything comes from one seed. With 52 VF records, the fold spread is large and a single
  seed does not show the seed spread.
- Batch 256 and full-pass epochs replace the batch-64 recipe in `NOTES.md` without a control
  run.
- The dev-set F1 threshold is a design choice. A fixed-sensitivity operating point may be the
  better reporting rule, given how poorly the threshold transfers.
- The VT-positive replication of LDIAED's task (`--vt-positive`) has not been run on random
  folds, so the closest comparison with its 99% is still missing. Section 6 runs the
  VT-positive task under leave-one-database-out instead.

## 6. VT positive at 85% coverage, leave-one-database-out

A second labelling moves closer to the benchmark convention used in `paper/PAPER.md`. VT, VFL
and VF all count as positive, a window is positive when they cover at least 85% of it, and
windows that a positive episode covers only partly (more than 0% and less than 85%) are
dropped from training, dev and test. `--vt-positive` keeps only the 10 annotated ahadb
8-series records, because series 1–7 have no rhythm labels and series 7 holds wide-complex
tachycardia that would count as negative. The corpus is 115 records and 164,749 windows,
16,688 of them positive (10.1%). Everything else is as in section 1.

| Held out | Epochs | Threshold | AUC | Se | Sp | PPV | F1 | Accuracy | Positive windows |
|---|---|---|---|---|---|---|---|---|---|
| ahadb | 15 | 0.10 | 0.999 | 0.994 | 0.987 | 0.970 | 0.982 | 0.989 | 5,458 |
| cudb | 19 | 0.70 | 0.923 | 0.755 | 0.901 | 0.668 | 0.709 | 0.871 | 3,534 |
| mitdb | 8 | 0.99 | 0.634 | 0.507 | 1.000 | 1.000 | 0.673 | 0.999 | 203 |
| vfdb | 16 | 0.91 | 0.972 | 0.921 | 0.896 | 0.643 | 0.757 | 0.900 | 7,493 |

Pooled over the four folds: accuracy 0.958, Se 0.905, Sp 0.964, PPV 0.739.

The ahadb fold alone matches the published 99%, on a split where no ahadb record was seen in
training. It does not carry over. Accuracy is 0.87 with cudb held out and 0.90 with vfdb held
out, so the ahadb result says more about ahadb than about the detector. The ahadb 8-series
holds no annotated VT, its positives are long VF episodes, and with mixed windows removed its
negatives are clean, which makes it the easiest of the four test sets.

Compared with the VF-only labelling of section 3, held-out AUC is unchanged on cudb (0.921 to
0.923) and rises on vfdb (0.943 to 0.972), because vfdb's VT is now a positive and no longer
the main source of false alarms. The errors that remain concentrate in a few records:

- vfdb: records 418 and 419 hold 2,493 of the 3,836 false-positive windows, with 1,393 of
  418's 1,530 negative windows flagged. Record 611 holds 285 of the 589 missed windows.
- cudb: cu03, cu11 and cu06 hold 970 of the 1,329 false positives, and cu20 and cu30 hold
  436 of the 865 misses.
- mitdb: the 203 positive windows are its few sustained VT and VFL episodes, and the model
  finds half of them (AUC 0.634). Training on vfdb, cudb and ahadb did not prepare it for
  mitdb's ventricular runs. The fold has almost no positives, so its 0.999 accuracy reflects
  the negatives.

Dropping the mixed windows makes this task easier than the one in section 2: a deployed
detector sees every window, episode edges included. These numbers are therefore an upper
bound for this labelling, and even so the strict split stays well below 99% on three of the
four databases.

## 7. JEKOVA baseline

`vfnn/jekova.py` implements the detector of `jekova/JEKOVA5.ipynb`. It needs no training.
Jekova's 14.6 Hz band-pass (Jekova 2004, eq. 1) runs once over each filtered record, which is
the same baseline-removed, 40 ms Lynn low-passed signal the network reads. The output is
rectified and cut into 1 s segments, and in each segment `cnt2` counts the samples at or above
the segment mean. A window sums `cnt2` over its eight segments. The published cascade reduces
to a single rule (`jekova/FINDINGS.md` section 1): shockable when `cnt2` exceeds 950 per 10 s,
which is 760 for an 8 s window. `cnt2` is the score for AUC, and two thresholds are reported:
the published one, and the one that maximises F1 on the same dev records the network used.

The implementation reproduces the JEKOVA5 table. On the notebook's protocol (VF and VFL
positive at 85%, mixed windows scored as negatives, published threshold), it gives Se 0.951,
Sp 0.999 and F1 0.960 on ahadb against the notebook's 95.1%, 99.9% and 96.4%. On cudb it gives
0.773, 0.957 and 0.795 against 76.6%, 95.9% and 79.3%, and on mitdb 0.902, 1.000 and 0.935
against 91.0%, 100.0% and 94.0%. vfdb agrees less well (F1 0.458 against 37.6%); the
notebook's last run left vfdb out of its database list, so its vfdb row may come from an
earlier configuration. The same run with VT positive changes vfdb and mitdb a lot, which
suggests the notebook's label list (`"#VT"`) does not match the VT episodes, so its table is
in effect VF and VFL only. This was not checked further.

### Network against baseline

Means over folds. Accuracy is pooled over folds. "Dev" is the baseline with the dev-F1
threshold.

| Split | Detector | AUC | Se | Sp | PPV | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| Random 10-fold | CNN | 0.998 | 0.954 | 0.991 | 0.827 | 0.886 | 0.990 |
| | JEKOVA published | 0.985 | 0.875 | 0.977 | 0.628 | 0.731 | 0.973 |
| | JEKOVA dev | 0.985 | 0.833 | 0.981 | 0.660 | 0.736 | 0.975 |
| Record-wise 5-fold | CNN | 0.985 | 0.912 | 0.972 | 0.600 | 0.722 | 0.969 |
| | JEKOVA published | 0.985 | 0.874 | 0.977 | 0.636 | 0.734 | 0.973 |
| | JEKOVA dev | 0.985 | 0.851 | 0.978 | 0.648 | 0.730 | 0.973 |
| Leave-one-database-out | CNN | 0.966 | 0.835 | 0.938 | 0.733 | 0.739 | 0.972 |
| | JEKOVA published | 0.970 | 0.824 | 0.957 | 0.785 | 0.762 | 0.973 |
| | JEKOVA dev | 0.970 | 0.867 | 0.946 | 0.740 | 0.759 | 0.970 |
| VT positive, 85%, leave-one-database-out | CNN | 0.882 | 0.794 | 0.946 | 0.820 | 0.780 | 0.958 |
| | JEKOVA published | 0.955 | 0.765 | 0.978 | 0.906 | 0.814 | 0.972 |
| | JEKOVA dev | 0.955 | 0.794 | 0.970 | 0.888 | 0.818 | 0.972 |

Leave-one-database-out by held-out database (AUC, then F1; the baseline at the published
threshold):

| Held out | CNN, VF only | JEKOVA, VF only | CNN, VT positive 85% | JEKOVA, VT positive 85% |
|---|---|---|---|---|
| ahadb | 1.000, 0.962 | 0.999, 0.960 | 0.999, 0.982 | 0.997, 0.966 |
| cudb | 0.921, 0.706 | 0.949, 0.799 | 0.923, 0.709 | 0.953, 0.813 |
| mitdb | 1.000, 0.774 | 0.999, 0.819 | 0.634, 0.673 | 0.895, 0.664 |
| vfdb | 0.943, 0.515 | 0.934, 0.468 | 0.972, 0.757 | 0.977, 0.810 |

### What the comparison shows

The network wins only under random folds. There its AUC is 0.998 against 0.985 and its F1
0.886 against 0.731. On record-wise folds the two have the same AUC (0.985) and the baseline
has a slightly higher F1 (0.734 against 0.722). With whole databases held out the baseline is
ahead on AUC and F1 under both labellings. The clearest gaps are cudb (AUC 0.949 against
0.921 with VF only) and mitdb's ventricular runs (0.895 against 0.634 with VT positive). The
network is ahead only on vfdb with VF-only labels (AUC 0.943 against 0.934), where it has
learned to reject some of vfdb's VT.

The baseline also measures the leakage directly. It has no parameters to fit, so its AUC is
0.985 under random folds and under record-wise folds alike. The network gains 0.013 AUC and
0.16 F1 from random folds. That gain is the leakage: the network has learned the test
records' own windows.

The two cost very different amounts. The baseline is a three-coefficient recursive filter
and a few comparisons per sample, on the order of ten thousand operations per 8 s window.
The network takes 48.5 million multiply-accumulates for the same window.

On this corpus, then, the network does not earn its 286k parameters. Its one strong number,
99% accuracy, comes from the protocol that lets it see its test records. Under every split
that keeps patients or databases apart, a one-feature detector from 2004 does as well or
better, at a small fraction of the cost.

Two checks remain open. Both detectors come from one configuration (one seed, B-lite only),
and the network's per-window predictions were not saved, so the VF-against-VT AUC on those
windows alone, where a learned model should gain the most, could not be computed for the
network.

## 8. Next steps

1. Save the network's per-window predictions and compare it with the baseline on VF and VT
   windows alone, since that split is where a learned model should gain the most.
2. Run a size ablation: a tiny network (about 10k parameters) and B-mid, under record-wise
   folds.
3. Run 3 to 5 seeds of the record-wise experiment.
4. Run the VT-positive replication on random folds, the 4 s windows, and the raw-input
   ablation listed in `DESIGN.md`.
5. Report at a fixed sensitivity as well as at the F1 threshold.

Two fixes come before these runs: stop training once the learning rate falls below about
1e-6 (the random folds spent 10 to 15 epochs below it), and repair the concurrency limit in
`data/vfnn/results/queue.sh`, which ran all 14 jobs at once.

## 9. Runs and files

All results are under `data/vfnn/results/` (git-ignored).

| Run | Folds | Wall time | Files |
|---|---|---|---|
| Record-wise 5-fold | 5 in parallel | about 18 min | `group8_lite.csv`, `group8_lite_records.csv`, `group8_lite/` |
| Random 10-fold and leave-one-database-out | 14 in parallel | about 81 min | `random8_lite*.csv`, `lodo8_lite*.csv`, their folders, `queue.log` |
| VT positive at 85%, leave-one-database-out (section 6) | ahadb alone, then 3 in parallel | about 3 min, then 6 min | `lodo8_vtpos85*.csv`, `lodo8_vtpos85/` |
| JEKOVA baseline, all protocols (section 7) | all folds, no training | seconds per run | `jekova/<split>_<published or dev>.csv`, `jekova/nbcheck.csv` |

Each folder holds one log and one CSV pair per fold. Each run used
`python -m vfnn.train --split <split> --only <k> --dbs vfdb cudb mitdb ahadb --epochs 30 --batch 256`,
with `--folds 5` or `--folds 10` where needed, and with
`--vt-positive --min-frac 0.85 --exclude-mixed` for section 6. `run.sh` and `queue.sh` start the folds, and
`merge.py` combines the per-fold CSVs. The baseline runs with
`python -m vfnn.jekova` and the same data and split options plus `--threshold published` or
`--threshold dev`.
