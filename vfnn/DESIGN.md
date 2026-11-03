# vfnn design

`vfnn` is a small 1-D residual CNN for detecting VF and ventricular flutter (VFL) in 8 s and
4 s windows at 250 Hz. It is a self-contained package: it does not import `vfta/` or
`vftx/`, and it is not compared against the JEKOVA, TCSC or HILB detectors. Its job is to
test the LDIAED claim (99% accuracy on random segment folds) and to measure what a small
CNN achieves on VF+VFL under a leakage-free split.

Sources are in `hannun/`: Hannun 2019 and its preprint (Rajpurkar 2017), the LDIAED paper
(Nasimi and Yazdchi 2022), and Hai 2021. The reference code in `hannun/ref/awni-ecg` is
GPL-3.0, so `vfnn` is written from the papers' descriptions and does not copy it.

## 1. Questions the work answers

1. How much of the LDIAED result survives when random segment folds are replaced by
   record-wise folds and leave-one-database-out?
2. What accuracy does a small CNN reach on VF+VFL against everything else, including VT,
   under those strict splits?
3. What does the network cost in parameters and multiply-accumulates?

## 2. Background from the papers

Hannun's network takes 30 s of single-lead ECG at 200 Hz and outputs a label every 256
samples (1.28 s) over 12 rhythm classes. It has a stem convolution, 16 residual blocks of
two convolutions each, and a time-distributed softmax. Filters are 16 wide, with 32·2^k
channels (k grows by 1 every fourth block). Every second block halves the time resolution
(a stride-2 convolution, with a max-pool shortcut), so the total stride is 2^8 = 256.
Blocks are pre-activation (batch norm, ReLU, then convolution) with dropout 0.2 between the
two convolutions. Training uses He initialization, Adam, a learning rate of 10^-3 divided by
10 when the development loss stalls for two epochs, and a patient-disjoint split. The
network has no VF class. Its receptive field is about 19,000 samples (77 s), longer than a
record.

LDIAED keeps Hannun's settings, cuts the depth to 6 blocks, and uses a binary
shockable/non-shockable label on VFDB, CUDB and MIT-BIH (negatives only). Its hyperparameters
match Hannun's example config (`examples/cinc17/config.json`). Its evaluation uses random
10-fold splits of segments, its reported F1 and accuracy disagree with its own confusion
matrix, and its window length (256 samples, "0.9 s or 1.4 s") does not match the sampling
rates it names. Its reported model size (1.551 MB) implies about 390k parameters, twice the
count for the architecture as described (197k), so the true channel plan is unknown.

## 3. Architecture (B-lite)

```
input   1 x 2000      8 s at 250 Hz (1 x 1000 for 4 s; the network is fully convolutional)
stem    conv k=15, 16 filters, stride 1, BN, ReLU
block1  BN-ReLU-conv(k=15, 16, stride 2)  BN-ReLU-dropout-conv(k=15, 16)   shortcut: maxpool 2
block2  same, 16 channels, stride 2
block3  stride 2, 32 channels  (shortcut: maxpool 2, zero-padded channels)
block4  stride 2, 32 channels
block5  stride 1, 64 channels  (zero-padded shortcut)
block6  stride 1, 64 channels
BN, ReLU, concat(global average pool, global max pool) over time, dense to 1 logit
```

Size, as measured by `vfnn.model` (`n_params`, `macs`, `receptive_field`): total stride 16, receptive field 1,541
samples (6.2 s), 286k parameters (counting batch norm and the head), 48 million multiply-accumulates per 8 s window (24 million
per 4 s).

### Choices and reasons

- Depth is 6 blocks. Hannun found residual connections useful beyond about 8 layers, and a
  VF texture does not need 77 s of context.
- The receptive field targets 4 to 8 s, so an output sees several fibrillation cycles
  (3–8 Hz gives 0.12–0.33 s per cycle). Design B reaches 6.2 s. Alternating strides of 2 and
  1 (3 strided blocks) give only 2.8 s, and LDIAED's 4/1 pattern gives 16 s.
- Strides are 2, 2, 2, 2, 1, 1. Early striding cuts compute, and the signal is band-limited
  to 30 Hz, so a 250 to 15.6 Hz effective rate loses little.
- Kernels are 15 samples (60 ms) instead of Hannun's 16, so every layer can use symmetric
  `same` padding. Width is a tuning knob.
- Channels run 16 to 64 because the corpus is small.
- The head is global average plus max pooling and one logit. Hannun's per-step softmax
  fits multi-rhythm 30 s records. This task needs one decision per window, and the max pool
  catches short bursts.

## 4. Data and labels

- Records come from `pxg.cbor` (`CborRecord` from the `pxg` package, which lives outside
  this repo in `exg-rad`). `vfnn` imports `pxg`, and nothing from `vfta/` or `vftx/`. The
  databases are the ones in the cbor store (vfdb, cudb, mitdb, ahadb and the others
  `vfta` uses), with vfdb, cudb and mitdb first because they are the LDIAED set.
- Sampling rate is 250 Hz everywhere. `CborRecord(..., freq=250)` resamples on read, so
  `vfnn` does no resampling of its own. Channel 0 (the default) lets cbor pick the best lead
  (lead II where present).
- Filtering is done once per record, before windowing, with a `vfnn` implementation of the
  `vfta` front end. Baseline removal has two steps: a moving median of 596 ms (149 samples)
  gives a first baseline estimate, and a moving mean of the same length smooths it. In
  `vfta` that mean is the Lynn low-pass filter run with the median window, which is a
  triangular average (two cascaded moving averages of 74 samples each). The smoothed
  estimate is subtracted from the signal. The Lynn low-pass with a 40 ms window (10 samples
  at 250 Hz, two cascaded 5-sample moving averages, -3 dB near 16 Hz) follows. The baseline
  subtraction is the high-pass side, so the two together form the band-pass. After filtering
  each window is normalized to zero mean and unit variance. A no-filter arm (LDIAED style,
  raw samples) is an ablation.
- Positive class is VF and VFL. Everything else is negative, VT included. A second labeling
  with VT positive reproduces LDIAED's task and is used only for the replication arm.
- Where the VF comes from differs by database. vfdb has only rhythm notes (`(VF`, `(VFL`,
  `(VFIB`) and no `[` or `]` markers. cudb, mitdb and ahadb use `[` and `]` for VF onset and
  offset (cudb and mitdb also carry `(VF` and `(VFL` notes). A `[` with no matching `]` runs
  to the end of the record, which is the rule `pxg` applies. It affects cudb (8 records) and
  7 of the 10 ahadb 8-series records, and about 31% of the ahadb positive windows depend on
  it.
- ahadb has 79 records. Only the 10 in the 8-series carry rhythm annotations. The other 69
  (series 1 to 7, 34.5 hours) have beat labels only, so every window in them is negative.
  They stay in by default as extra negatives for false-positive checks, and `--ahadb-8-only`
  drops them. The VT-positive labeling forces the 8-series only, because there an
  unannotated VT would count as negative.
- Windows are 8 s and 4 s with a 1 s step. The label comes from the rhythm episode
  annotations (`.Time` and `.End` in samples). A window is positive when the VF and VFL
  episodes together cover at least 50% of it, and negative otherwise. The "(VF", "(VFL" and
  "(VFIB" notes count as positive, and so do the "[" VF-onset markers that `vfta` also
  treats as VF. The 50% threshold is applied identically to every arm and every split.

## 5. Training

- He-normal initialization, Adam (default betas), learning rate 10^-3 divided by 10 when
  the development loss stalls for two epochs, keep the best development-loss model.
- Binary cross-entropy with `pos_weight` or balanced sampling to handle class imbalance.
- Dropout 0.2 between the two convolutions of each block.
- Batch size 64 to start (Hannun 128, LDIAED 32); tunable.
- Augmentation (neither paper used any): random time shifts, amplitude scale, polarity flip,
  mild baseline wander.
- Development data are whole records held out of training, never windows from a training
  record.

## 6. Evaluation

Splits, in order of strictness:

1. Random segment folds, 10-fold. Replicates the LDIAED protocol and shows what leakage
   buys.
2. Record-wise grouped folds.
3. Leave-one-database-out.

Metrics: AUC, sensitivity and specificity at a fixed operating point, and F1, each with the
spread over folds. The results table reports the same model under the three splits, and sets
the numbers next to the figures published in LDIAED and Hai 2021 (labeled as different
protocols, not as a head-to-head). Cost is reported as parameters and MACs per window. The
run also prints a false-positive table over every test record that holds no positive window
(the ahadb series 1 to 7 records, and the negative-only records of the other databases):
windows, false-positive windows, their rate, and the alarmed seconds per hour of record.
`--detail-out` writes the per-record rows.

Ablations:

- VFDB+CUDB with and without MIT-BIH negatives, to test whether sampling rate is a shortcut
  (in LDIAED all MIT-BIH data are 360 Hz negatives and all positives are 250 Hz).
- Band-passed versus raw input.
- B-lite versus a wider B-mid (channels 32, 32, 32, 64, 64, 64; 433k parameters, 97 million
  MACs per 8 s).

## 7. Known uncertainties

- LDIAED's real channel plan, stride placement, and the way its per-position outputs become
  one decision are not specified, so this is a close approximation of LDIAED, not a copy.
- Hannun's per-step labels have no analogue here, so the pooled-logit head is our design,
  not theirs.
- The Lynn window is 40 ms, as in the JEKOVA notebooks. `vfta` uses 48 ms, so features built
  by `vfta` are not on the identical signal. The 596 ms median window is the `vfta` value.
  The two window lengths are parameters of `filters.py`.
- Few records exist (CUDB has 35, VFDB 22), so fold variance will be large. Report spread
  over folds, not only means.
- Without a handcrafted baseline in the same table, the strict-split numbers have no local
  reference point. If that becomes a problem, a single trivial baseline can be added inside
  `vfnn` later without importing the other packages.

## 8. Layout and status

```
vfnn/
  DESIGN.md      this file
  filters.py     baseline removal (moving median, then moving mean), Lynn low-pass     done
  model.py       the network (block count, strides, channels as parameters)             done
  data.py        pxg.cbor reading, 50% labels, windows, splits                          done
  train.py       training loop, fold runner, command line                               done
  eval.py        metrics, threshold choice, per-fold table                              done
tests/test_vfnn_*.py   20 tests (filters match vfta, design sizes, labels, splits, a fit)
```

Run one protocol with, for example:

```
python -m vfnn.train --split group --seconds 8 --dbs vfdb cudb mitdb
python -m vfnn.train --split lodo --dbs vfdb cudb mitdb ahadb
python -m vfnn.train --split random --folds 10 --vt-positive    # LDIAED-style replication
```

A run prints a header (records, windows, device, settings), one line per fold with its train,
dev and test window counts, one line per epoch (train loss, dev loss, `*` when the dev loss is
a new best, learning rate, seconds, epochs since the last best), and one result line per fold.
For a long run, send the output to a log and follow it with `tail -f`.

`--only N` runs a single fold, `--per-epoch N` trains on a random subset of windows each epoch,
and `--out file.csv` writes the per-fold rows. Filtered records are cached under `data/vfnn`.

Training runs on the GPU where there is one (`mps` on Apple silicon, then `cuda`, else CPU). All
filtered signals are held in one device tensor (`data.Store`), and each batch is cut,
normalized and augmented on the device, so the loop has no per-window Python work. Measured on
a MacBook Air M4: B-lite trains at about 1,900 windows/s (batch 64) and scores about 8,700
windows/s; B-mid trains at about 1,300 windows/s and scores about 5,300. Windows step by 1 s
and are 8 s long, so neighbours overlap by 87.5%: an epoch is a random subset
(`--per-epoch`, for example 40,000) and not a full pass over the roughly 200,000 training
windows of a fold.

The decision threshold for sensitivity, specificity and F1 is the one that maximises F1 on the
development set (whole records held out of training for the record-wise and database splits,
random windows for the random split). AUC does not use a threshold.

Code follows the repo's Python style: each compound statement closes with a `pass #<keyword>`
sentinel at the header's indentation (`python docs/tools/close_blocks.py vfnn`). Dependencies
are `numpy`, `scipy`, `scikit-learn`, `pxg` (from `exg-rad`) and `torch`.
