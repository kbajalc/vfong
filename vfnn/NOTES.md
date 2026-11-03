# vfnn handoff notes

Status as of 2026-11-03, written on a MacBook Air M4 before moving to a bigger machine.
`DESIGN.md` is the design and the reasons for it. This file says where the work stands and
what to do next.

## 1. Where we are

The package is written, tested and working end to end. No real experiment has been run yet.
The only runs so far are small smoke tests (3 epochs of 3,000 windows on one fold), so none of
their numbers should be quoted.

| Piece | State |
|---|---|
| `filters.py` baseline removal (moving median, then moving mean) and 40 ms Lynn low-pass | done, matches `vfta.filters.signal_filter` (tested) |
| `model.py` B-lite and B-mid residual CNN, pooled binary head | done, sizes match the design |
| `data.py` pxg.cbor reading, cache, 50% window labels, `Store` for GPU batching, three split types | done |
| `train.py` training loop, fold runner, verbose progress, command line | done |
| `eval.py` metrics, F1-optimal threshold on dev, per-fold and false-positive tables | done |
| `tests/test_vfnn_*.py` | 25 tests, all pass |
| Data preparation run (`python -m vfnn.data --rebuild`) | done, 184 records cached, no bad data |
| Full experiments | not started |

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

## 5. Setting up on the new machine

1. Clone or copy the repo. The `vfnn/`, `hannun/` and `tests/test_vfnn_*.py` files are not
   committed yet (see section 8). Commit and push them first, or copy the folders.
2. `pxg` comes from the separate `exg-rad` repo (`pip show pxg` here reports version 0.1.0).
   Install it in the new environment, for example `pip install -e ../exg-rad`.
3. The signals come from the cbor store. In the repo, `work` is a symlink to `../exg-work`, and
   the store is `../exg-work/cbor` (8.1 GB, not in git). Copy `exg-work` next to the repo, or
   recreate the symlink.
4. Python environment: here it is the conda env `dev` (Python 3.14.8) with torch 2.14.1, numpy,
   scipy, scikit-learn, joblib, cbor2 and pytest.
5. Check: `python -m pytest tests/test_vfnn_*.py -q` (25 pass in about 10 s), then
   `python -m vfnn.data --rebuild` (about 20 s here, builds the 262 MB cache under `data/vfnn`,
   which is git-ignored), then the 30 s smoke run in `DESIGN.md` section 8.
6. Run everything from the repo root, because the cache path and the cbor path are relative.

## 6. Next steps, in order

1. Run the main experiment, 5-fold record-wise at 8 s with B-lite. This is the first real number.
   On the Air M4 the estimate was about an hour. Use the bigger machine to run it with the
   settings below.
2. Run the same model under random 10-fold segment splits (`--split random --folds 10`) to
   measure the leakage gap, and leave-one-database-out (`--split lodo`). Together with step 1
   these answer research question 1.
3. Repeat step 1 with `--seconds 4` and with `--net mid`.
4. Run the VT-positive replication (`--split random --folds 10 --vt-positive`) to compare with
   LDIAED's reported 99% on its own kind of protocol.
5. Check variance. Run 3 to 5 seeds of the record-wise run (`--seed`) and report mean and spread
   over folds and seeds. With 52 positive records, one seed is not enough.
6. Write the results table (the same model under three splits, set next to the LDIAED and Hai
   figures and labelled as different protocols), plus parameters and MACs per window.

Suggested starting settings for a much stronger GPU (all untested, tune after the first run):
batch 256 instead of 64, a larger `--per-epoch` (a full pass over about 200,000 training
windows is feasible, though neighbouring windows overlap by 87.5%), and two runs in parallel
if memory allows. On the Air M4, B-lite trained at about 1,900 windows per second and scored
about 8,700. Re-measure on the new machine before planning run times.

## 7. Open items and things not yet verified

- The raw-input ablation (no filter, LDIAED style) is listed in `DESIGN.md` but not built. The
  cache stores the filtered signal only. It needs a flag in `data.py` and a second cache.
- The MIT-BIH ablation is just a run without `mitdb` in `--dbs`. Because `pxg` resamples
  360 Hz to 250 Hz, the sampling-rate shortcut that LDIAED may have is probably weaker here, but
  this has not been tested.
- The window grid starts at sample 0 with a 1 s step and keeps full windows only. It was not
  compared against the `vfta` TSV grid because `data/s8` does not exist on the Air. The
  `vfta` label table also lacks a `(VFIB` entry, so its labels differ from `vfnn` on vfdb.
- Series 1 to 7 of ahadb are assumed free of VF. The beat-annotation evidence supports that,
  but it was not checked against the AHA documentation, and the annotators' conventions are
  assumed to be the same across series.
- LDIAED's real channel plan, stride placement and how its outputs become one decision are
  unknown, so the replication is an approximation. Section 7 of `DESIGN.md` says so.
- The F1-optimal dev threshold is my choice. A fixed sensitivity or specificity operating point
  may suit the paper better.
- The loop prints "fold 1/1" for `--only`, which is cosmetic.
- The smoke runs gave AUC 0.97 to 0.98 and large false-positive counts in vfdb records 421, 423,
  607 and 612. With 3 tiny epochs that says nothing about the real result.

## 8. Git state

Nothing from this work is committed. `hannun/` is staged (including the `awni-ecg` snapshot,
which is GPL-3.0, so keep it as reference only). `vfnn/` and `tests/test_vfnn_*.py` are
untracked. The repo also has unrelated staged changes in `jekova/` and `PAPER.ipynb` from
earlier. Decide what to commit before switching machines. `data/` (cache and results) is
git-ignored and does not travel, so copy `data/vfnn/results` by hand if you want the smoke
outputs (not needed).

## 9. Useful commands

```
python -m pytest tests/test_vfnn_*.py -q
python -m vfnn.data --rebuild --records        # prepare, cache and summarise
python -m vfnn.train --split group --dbs vfdb cudb mitdb ahadb --folds 5 --only 0 \
    --epochs 3 --per-epoch 3000                # 30 s smoke test
python -m vfnn.train --split group --dbs vfdb cudb mitdb ahadb --folds 5 \
    --epochs 30 --per-epoch 40000 --out data/vfnn/results/group8_lite.csv \
    --detail-out data/vfnn/results/group8_lite_records.csv \
    2>&1 | tee data/vfnn/results/group8_lite.log   # the main run
python docs/tools/close_blocks.py --check vfnn tests/test_vfnn_*.py   # house style check
```
