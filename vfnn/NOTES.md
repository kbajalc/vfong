# vfnn handoff notes

Status as of 2026-11-03 (evening), after the first experiments on an Apple M5 Max. `DESIGN.md`
is the design and the reasons for it, `REPORT.md` holds the results. This file says where the
work stands and what to do next.

## 1. Where we are

The main experiments are done and written up in `REPORT.md`. The conclusion: on these four
databases, under splits that keep patients and databases apart, the CNN does not improve on
the JEKOVA band-pass count (record-wise AUC 0.985 for both, F1 0.722 against 0.734). Its only
clear win is under random segment folds, and that win is leakage.

| Piece | State |
|---|---|
| `filters.py` baseline removal (moving median, then moving mean) and 40 ms Lynn low-pass | done, matches `vfta.filters.signal_filter` (tested) |
| `model.py` B-lite and B-mid residual CNN, pooled binary head | done, sizes match the design |
| `data.py` pxg.cbor reading, cache, window labels (`min_frac`, `vt_positive`, `exclude_mixed`), `Store` for GPU batching, `make_folds` for the three split types | done |
| `train.py` training loop, fold runner, verbose progress, command line (`--min-frac`, `--exclude-mixed`) | done |
| `eval.py` metrics, F1-optimal threshold on dev, per-fold and false-positive tables | done |
| `jekova.py` JEKOVA5 band-pass baseline on the same windows and folds, published or dev threshold | done, reproduces the JEKOVA5 table within about a point on ahadb, cudb, mitdb |
| `tests/test_vfnn_*.py` | 31 tests, all pass |
| Data preparation run (`python -m vfnn.data --rebuild`) | done, 184 records cached, no bad data |
| Experiments: record-wise, random 10-fold, leave-one-database-out, VT-positive 85% leave-one-database-out, baseline on all of them | done, see `REPORT.md` |

## 2. Decisions that are settled

- Standalone package. It imports `pxg` (from `exg-rad`) and nothing from `vfta/` or `vftx/`, and
  it is not compared against JEKOVA, TCSC or HILB.
- 250 Hz everywhere. `pxg.cbor` resamples on read, including mitdb (360 Hz).
- Filter: 596 ms moving median, then a moving mean of the same length (the Lynn low-pass at
  that window), subtracted from the signal; then the Lynn low-pass with a 40 ms window. No
  separate moving-mean term. Each window is then normalized to zero mean and unit variance.
- Positive class is VF and VFL. Notes `(VF`, `(VFL`, `(VFIB` and `[` onset markers count. VT is
  a negative. `--vt-positive` gives the LDIAED-style task for the replication arm only.
- vfdb has no `[` or `]` markers, only rhythm notes. cudb, mitdb and ahadb use brackets. An
  unpaired `[` runs to the end of the record (confirmed, and what `pxg` does).
- A window is positive when VF and VFL cover at least 50% of it. 8 s and 4 s windows, 1 s step.
- ahadb: all 79 records are loaded. The 10 in the 8-series are annotated. The other 69
  (series 1 to 7, 34.5 h) have beat labels only and act as extra negatives for false-positive
  checks. I checked that series 7 contains no VF (no beat-free stretches, no `[` markers, only
  wide-complex tachycardia). `--ahadb-8-only` drops them, and `--vt-positive` forces that.
- Architecture B-lite: stem 16, six pre-activation residual blocks, channels 16, 16, 32, 32, 64,
  64, strides 2, 2, 2, 2, 1, 1, kernel 15, dropout 0.2, global average plus max pooling, one
  logit. 286k parameters, 48.5 MMAC per 8 s window, receptive field 6.2 s.
- Training recipe from Hannun: He-normal init, Adam 1e-3, divide by 10 when the dev loss stalls
  for two epochs, best dev-loss weights kept, `pos_weight` capped at 20, jitter and polarity
  augmentation. Batch 64 to start.
- The reported threshold (for Se, Sp, PPV, F1) is the one that maximizes F1 on the dev set. Dev
  is whole records for the record-wise and database splits, random windows for the random split.

## 3. Data (after the preparation run)

| db | records | hours | VF min | VT min | records with VF | 8 s windows | positive |
|---|---|---|---|---|---|---|---|
| vfdb | 22 | 12.8 | 50.2 | 87.3 | 8 | 46,046 | 2,963 |
| cudb | 35 | 4.9 | 63.5 | 0.5 | 33 | 17,535 | 3,745 |
| mitdb | 48 | 24.1 | 2.4 | 3.6 | 1 | 86,304 | 150 |
| ahadb | 79 | 39.5 | 92.0 | 0.0 | 10 | 141,647 | 5,495 |
| total | 184 | 81.3 | 208.1 | 91.4 | 52 | 291,532 | 12,353 |

Only 52 of 184 records contain VF, so fold variance will be large. mitdb has one positive
record, so a leave-one-database-out fold on mitdb has almost nothing to score. vfdb holds
nearly all the VT, so it is the database that tests VT false positives.

## 4. What the paper review found (short version)

Sources are in `hannun/` (see its README). Details are in `DESIGN.md` section 2.

- LDIAED reuses Hannun's example hyperparameters with fewer blocks. Its 99% comes from random
  10-fold segment splits, its F1 and accuracy disagree with its own confusion matrix, its window
  length does not match its sampling rates, and its model size implies about twice the parameters
  its text describes.
- Hannun's network was never trained on VF. Hai 2021 (99.2%) also uses random folds.
- The question this work answers is how much of those numbers survives record-wise and
  leave-one-database-out splits.

## 5. Setting up a machine

1. `pxg` comes from the separate `exg-rad` repo. Install it as an editable package, for
   example `pip install -e ../exg-rad`.
2. The signals come from the cbor store. In the repo, `work` is a symlink to `../exg-work`, and
   the store is `../exg-work/cbor` (8.1 GB, not in git). Copy `exg-work` next to the repo, or
   recreate the symlink.
3. Python environment: the conda env `dev` (Python 3.14) with torch 2.14.1, numpy, scipy,
   scikit-learn, joblib, cbor2 and pytest. Install packages with `mamba install -n dev`.
4. Check: `python -m pytest tests/test_vfnn_*.py -q` (31 pass in about 10 s), then
   `python -m vfnn.data --rebuild` (about 10 s on the M5 Max, builds the 262 MB cache under
   `data/vfnn`, which is git-ignored), then the smoke run in section 9.
5. Run everything from the repo root, because the cache path and the cbor path are relative.

Speed on the M5 Max (48 GB, MPS): B-lite trains at about 4,500 windows/s at batch 64,
10,400 at batch 256 and 12,600 at batch 512 in one process. Running 5 folds as parallel
processes (`--only k`) gives about 18,000 windows/s in total, and 14 at once about 13,000. A
record-wise 5-fold run at batch 256 with full-pass epochs takes about 18 minutes. The
JEKOVA baseline needs no training and runs in seconds.

## 6. Next steps, in order

The main question is answered (see `REPORT.md` section 8). What is left would only narrow
the claim, not change it:

1. Save the network's per-window predictions and compare it with the baseline on VF and VT
   windows alone. This is the one place a learned model could still show an advantage.
2. Run 3 to 5 seeds of the record-wise experiment, so the spread covers seeds as well as folds.
3. Size ablation under record-wise folds: a tiny network (about 10k parameters) and B-mid.
4. The VT-positive replication on random folds, the 4 s windows, and the raw-input ablation.
5. A fixed-sensitivity operating point next to the dev-F1 threshold.

Before more training runs: stop once the learning rate falls below about 1e-6 (the random
folds spent 10 to 15 epochs below it), and fix the concurrency limit in
`data/vfnn/results/queue.sh` (`jobs -r` inside `$(...)` sees no jobs, so all 14 jobs started
at once).

## 7. Open items and things not yet verified

- The runs used batch 256 and a full pass per epoch instead of batch 64 and 40,000 windows.
  No batch-64 control was run.
- `jekova/JEKOVA5.ipynb` labels VT as `"#VT"`. Its table matches the vfnn baseline only with
  VT as negative, which suggests that name does not match the VT episodes. Not checked.
- vfdb record 418 has no VT, yet the network flags 1,393 of its 1,530 negative windows in the
  VT-positive run. The signal has not been looked at.
- The raw-input ablation (no filter, LDIAED style) is listed in `DESIGN.md` but not built. The
  cache stores the filtered signal only. It needs a flag in `data.py` and a second cache.
- The MIT-BIH ablation is just a run without `mitdb` in `--dbs`. Not run.
- The window grid starts at sample 0 with a 1 s step and keeps full windows only. It was not
  compared against the `vfta` TSV grid. The `vfta` label table also lacks a `(VFIB` entry, so
  its labels differ from `vfnn` on vfdb.
- Series 1 to 7 of ahadb are assumed free of VF. The beat-annotation evidence supports that,
  but it was not checked against the AHA documentation.
- LDIAED's real channel plan, stride placement and how its outputs become one decision are
  unknown, so the replication is an approximation. Section 7 of `DESIGN.md` says so.
- The loop prints "fold 1/1" for `--only`, and the run header prints the whole window count
  where it means the training windows per epoch. Both are cosmetic.
- `--min-frac` compares with `>=`, so "85%" means at least 85%, not more than 85%.

## 8. Git state

The code, tests and `REPORT.md` are committed on the `vfnn` branch. `hannun/` holds the
`awni-ecg` snapshot, which is GPL-3.0, so keep it as reference only. `data/` (cache, results,
the launcher scripts `run.sh`, `queue.sh` and `merge.py`) is git-ignored and does not travel:
copy `data/vfnn/results` by hand to keep the run logs and per-record tables.

## 9. Useful commands

```
python -m pytest tests/test_vfnn_*.py -q
python -m vfnn.data --rebuild --records        # prepare, cache and summarise
python -m vfnn.train --split group --dbs vfdb cudb mitdb ahadb --folds 5 --only 0 \
    --epochs 3 --per-epoch 3000                # smoke test
python -m vfnn.train --split group --dbs vfdb cudb mitdb ahadb --folds 5 --only 0 \
    --epochs 30 --batch 256 --out data/vfnn/results/group8_lite/fold0.csv \
    --detail-out data/vfnn/results/group8_lite/fold0_records.csv   # one fold of the main run
python -m vfnn.jekova --split group --dbs vfdb cudb mitdb ahadb --threshold published
python docs/tools/close_blocks.py --check vfnn tests/test_vfnn_*.py   # house style check
```
