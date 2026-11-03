"""JEKOVA baseline for vfnn: the band-pass count detector of ``jekova/JEKOVA5.ipynb``.

No training. The detector runs on the same filtered records, windows, labels and folds as
the network, so the two can sit in one table.

1. The 14.6 Hz band-pass of Jekova (2004, eq. 1) runs once over the whole filtered record
   (vfnn's baseline removal and 40 ms Lynn low-pass, the input JEKOVA5 uses).
2. The output is rectified and cut into 1 s segments. In each segment ``cnt1`` counts the
   samples at or above half the segment maximum, ``cnt2`` those at or above the segment
   mean, and ``cnt3`` those within one mean absolute deviation of the mean.
3. A window sums the counts of the 1 s segments it covers.
4. The published cascade reduces to one rule (``jekova/FINDINGS.md`` section 1): shockable
   when ``cnt2 > 950`` per 10 s, scaled to the window length (760 for 8 s).

``cnt2`` is the score for AUC. The threshold is either the published one or the one that
maximises F1 on the same dev records the network uses.

    python -m vfnn.jekova --split lodo --dbs vfdb cudb mitdb ahadb --vt-positive \\
        --min-frac 0.85 --exclude-mixed
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
from scipy import signal

from vfnn import data
from vfnn.data import Windows
from vfnn.eval import best_f1_threshold, false_positive_table, per_record, scores, table
from vfnn.filters import FS

# Jekova eq. 1 scaled by 16 to integer coefficients:
#     16*y[i] = 28*y[i-1] - 14*y[i-2] + (x[i] - x[i-2])
JEK_B = [16.0, 0.0, -16.0]
JEK_A = [16.0, -28.0, 14.0]
JEK_GAIN = 4.0           # as in JEKOVA5; the counts are scale-free, so the gain does not matter
EPOCH_S = 10             # the paper's epoch length
C2_HI = 950              # published Count2 threshold per 10 s epoch


def band_pass(x: np.ndarray) -> np.ndarray:
    """Jekova's 14.6 Hz band-pass over a whole record."""
    return signal.lfilter(JEK_B, JEK_A, np.asarray(x, dtype=float)) / JEK_GAIN
pass #def


def second_counts(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """``(seconds, 3)`` int array of cnt1, cnt2, cnt3 per 1 s segment of the band-passed record."""
    a = np.abs(band_pass(x))
    n = len(a) // fs
    seg = a[:n * fs].reshape(n, fs)
    fmax = seg.max(axis=1, keepdims=True)
    favg = seg.mean(axis=1, keepdims=True)
    mdev = np.abs(seg - favg).mean(axis=1, keepdims=True)
    cnt1 = (seg >= fmax / 2).sum(axis=1)
    cnt2 = (seg >= favg).sum(axis=1)
    cnt3 = ((seg >= favg - mdev) & (seg <= favg + mdev)).sum(axis=1)
    return np.stack([cnt1, cnt2, cnt3], axis=1).astype(np.int64)
pass #def


def window_counts(win: Windows, fs: int = FS) -> np.ndarray:
    """``(len(win), 3)`` summed counts per window. Windows must start and end on whole seconds."""
    if win.size % fs or np.any(win.start % fs):
        raise ValueError("JEKOVA counts need windows aligned to whole seconds")
    pass #if
    k = win.size // fs
    out = np.zeros((len(win), 3), dtype=np.int64)
    for r in np.unique(win.rec):
        c = second_counts(win.recs[r].x, fs)
        cs = np.vstack([np.zeros((1, 3), dtype=np.int64), np.cumsum(c, axis=0)])
        m = win.rec == r
        s = win.start[m] // fs
        out[m] = cs[s + k] - cs[s]
    pass #for
    return out
pass #def


def published_threshold(seconds: float) -> float:
    """Score threshold for ``cnt2 > C2_HI`` scaled to the window (scores are >= thr positive)."""
    return float(np.floor(C2_HI * seconds / EPOCH_S) + 1)
pass #def


def run_split(win: Windows, cnt2: np.ndarray, folds, labels, mode: str, by_record: bool, seed: int):
    """Score each fold's test windows. ``mode`` is ``published`` or ``dev`` (F1 on the dev set)."""
    from vfnn.train import dev_split      # same dev records as the network
    rows, detail = [], []
    seconds = win.size / FS
    for (tr, te), lab in zip(folds, labels):
        if mode == "dev":
            _, dev_idx = dev_split(tr, win, 0.15, by_record, seed)
            thr = best_f1_threshold(win.y[dev_idx], cnt2[dev_idx].astype(float))
        else:
            thr = published_threshold(seconds)
        pass #if
        test = win.subset(te)
        p = cnt2[te].astype(float)
        r = scores(test.y, p, thr)
        r["thr"] = thr
        rows.append(r)
        for d in per_record(test.groups(), test.databases(), test.y, p, thr):
            detail.append({"fold": lab, **d})
        pass #for
        print(f"[{lab}] thr {thr:.0f}  AUC {r['auc']:.3f}  Se {r['se']:.3f}  Sp {r['sp']:.3f}  "
              f"PPV {r['ppv']:.3f}  F1 {r['f1']:.3f}", flush=True)
    pass #for
    return rows, detail
pass #def


def _write(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pass #with
pass #def


def _main() -> None:
    ap = argparse.ArgumentParser(description="JEKOVA band-pass baseline under one vfnn split protocol.")
    ap.add_argument("--split", choices=["random", "group", "lodo"], default="group")
    ap.add_argument("--dbs", nargs="+", default=["vfdb", "cudb", "mitdb"])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--vt-positive", action="store_true")
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--exclude-mixed", action="store_true")
    ap.add_argument("--ahadb-8-only", action="store_true")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", choices=["published", "dev"], default="published")
    ap.add_argument("--out", default=None, help="CSV path for the per-fold rows")
    ap.add_argument("--detail-out", default=None, help="CSV path for the per-record rows")
    args = ap.parse_args()

    t0 = time.time()
    recs = data.load_records(tuple(args.dbs), ahadb_8_only=args.ahadb_8_only or args.vt_positive)
    win = data.make_windows(recs, args.seconds, min_frac=args.min_frac, vt_positive=args.vt_positive,
                            exclude_mixed=args.exclude_mixed)
    cnt2 = window_counts(win)[:, 1]
    print(f"{len(recs)} records, {len(win)} windows, {int(win.y.sum())} positive; labels "
          f"{'VT+VF+VFL' if args.vt_positive else 'VF+VFL'} at >= {args.min_frac:g}, mixed "
          f"{'excluded' if args.exclude_mixed else 'negative'}; threshold {args.threshold}; "
          f"counts took {time.time() - t0:.1f}s", flush=True)

    folds, labels = data.make_folds(win, args.split, args.folds, args.seed)
    if args.only is not None:
        folds, labels = [folds[args.only]], [labels[args.only]]
    pass #if
    rows, detail = run_split(win, cnt2, folds, labels, args.threshold, args.split != "random", args.seed)
    print()
    print(table(f"jekova {args.threshold} / {args.split} / {args.seconds:g} s", rows, labels))
    print("\nfalse positives on records without positive windows")
    print(false_positive_table(detail))
    if args.out:
        _write(args.out, [{"fold": lab, **r} for lab, r in zip(labels, rows)])
    pass #if
    if args.detail_out and detail:
        _write(args.detail_out, detail)
    pass #if
pass #def


if __name__ == "__main__":
    _main()
pass #if
