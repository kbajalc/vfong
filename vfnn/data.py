"""Records, window labels and splits for vfnn.

Records are read through ``pxg.cbor`` at 250 Hz (cbor resamples on read) and filtered once
with :func:`vfnn.filters.filter_record`. Rhythm episodes become per-sample masks, and a
window is positive when VF and VFL together cover at least ``min_frac`` (50%) of it. The
alternative labeling ``vt_positive=True`` also counts VT, for the LDIAED replication arm.
With ``exclude_mixed=True`` the windows that a positive episode only partly covers
(``0 < frac < min_frac``) are dropped, as the paper's MIX class is.

Filtered records are cached as ``.npz`` under ``data/vfnn`` (git-ignored).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from vfnn.filters import FS, filter_record, normalize_window

CACHE = "data/vfnn"
DATABASES = ("vfdb", "cudb", "mitdb", "ahadb")
# vfdb writes fibrillation as "(VFIB" (and some "(VF"), cudb and ahadb use "[" markers.
VF_NOTES = ("(VF", "(VFL", "(VFIB")
VT_NOTES = ("(VT",)
# Stored in each cache file, so a change to the note lists rebuilds the masks.
TAG = "VF:" + ",".join(VF_NOTES) + ";VT:" + ",".join(VT_NOTES)


@dataclass
class Rec:
    """One filtered record with its per-sample rhythm masks."""
    db: str
    rid: str
    x: np.ndarray      # float32, filtered, FS Hz
    vf: np.ndarray     # bool, VF or VFL (and VF-onset "[" markers)
    vt: np.ndarray     # bool, VT

    @property
    def key(self) -> str:
        return f"{self.db}/{self.rid}"
    pass #def
pass #class


def record_ids(db: str, ahadb_8_only: bool = False) -> list[str]:
    """Record ids of a database.

    ahadb has 79 records. Only the 8-series carries rhythm annotations (the `[` `]` VF
    brackets); the other 69 have beat labels only, so every window in them is negative. They
    are kept by default as extra negatives for false-positive checks. ``ahadb_8_only`` drops
    them, which the VT-positive labeling needs: there an unannotated VT would count as negative.
    """
    from pxg.cbor import CborDatabase
    ids = list(CborDatabase(db, freq=FS, chn=0).Records)
    if db == "ahadb" and ahadb_8_only:
        ids = [r for r in ids if r.startswith("8")]
    pass #if
    return ids
pass #def


def episode_masks(anns, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample VF/VFL/VFIB and VT masks from the rhythm-episode annotations."""
    vf = np.zeros(n, dtype=bool)
    vt = np.zeros(n, dtype=bool)
    for a in anns:
        if a.Type == "[":
            kind = vf
        elif a.Type == "+" and a.Note[:1] == "(":
            kind = vf if a.Note in VF_NOTES else vt if a.Note in VT_NOTES else None
        else:
            kind = None
        pass #if
        if kind is None:
            continue
        pass #if
        lo, hi = max(a.Time, 0), min(a.End, n)
        if hi > lo:
            kind[lo:hi] = True
        pass #if
    pass #for
    return vf, vt
pass #def


def load_record(db: str, rid: str, cache: str | None = CACHE, chn: int = 0) -> Rec:
    """Load, filter and label one record, using the npz cache when present."""
    path = os.path.join(cache, db, f"{rid}.npz") if cache else None
    if path and os.path.exists(path):
        z = np.load(path)
        if "tag" in z.files and str(z["tag"]) == TAG:
            return Rec(db, rid, z["x"], z["vf"], z["vt"])
        pass #if
    pass #if

    from pxg.cbor import CborRecord
    rec = CborRecord(db, rid, atr="atr", chn=chn, freq=FS)
    x = filter_record(rec.Signal.astype(float)).astype(np.float32)
    vf, vt = episode_masks(rec.Annotations, len(x))
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, x=x, vf=vf, vt=vt, tag=np.array(TAG))
    pass #if
    return Rec(db, rid, x, vf, vt)
pass #def


def load_records(
    dbs=("vfdb", "cudb", "mitdb"), cache: str | None = CACHE, ahadb_8_only: bool = False
) -> list[Rec]:
    """Load every record of the given databases."""
    return [load_record(db, rid, cache) for db in dbs for rid in record_ids(db, ahadb_8_only)]
pass #def


@dataclass
class Windows:
    """Window table over a list of records: which record, where it starts, its label."""
    recs: list[Rec]
    rec: np.ndarray     # int32 index into recs
    start: np.ndarray   # int64 first sample
    frac: np.ndarray    # float32 fraction of the window covered by positive rhythm
    y: np.ndarray       # uint8 label
    size: int           # samples per window

    def __len__(self) -> int:
        return len(self.y)
    pass #def

    def groups(self) -> np.ndarray:
        """Record key of every window (for record-wise splits)."""
        keys = np.array([r.key for r in self.recs])
        return keys[self.rec]
    pass #def

    def databases(self) -> np.ndarray:
        dbs = np.array([r.db for r in self.recs])
        return dbs[self.rec]
    pass #def

    def subset(self, idx: np.ndarray) -> "Windows":
        return Windows(self.recs, self.rec[idx], self.start[idx], self.frac[idx], self.y[idx], self.size)
    pass #def
pass #class


def make_windows(
    recs: list[Rec],
    seconds: float = 8.0,
    step_s: float = 1.0,
    min_frac: float = 0.5,
    vt_positive: bool = False,
    exclude_mixed: bool = False,
) -> Windows:
    """Slide windows over every record. Positive when positive rhythm covers >= min_frac.

    With ``exclude_mixed`` only windows with no positive rhythm or at least ``min_frac`` of
    it are kept.
    """
    n = int(seconds * FS)
    step = int(step_s * FS)
    ri, st, fr = [], [], []
    for i, r in enumerate(recs):
        mask = r.vf | r.vt if vt_positive else r.vf
        cs = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
        starts = np.arange(0, len(r.x) - n + 1, step, dtype=np.int64)
        ri.append(np.full(len(starts), i, dtype=np.int32))
        st.append(starts)
        fr.append(((cs[starts + n] - cs[starts]) / n).astype(np.float32))
    pass #for
    frac = np.concatenate(fr)
    ri, st = np.concatenate(ri), np.concatenate(st)
    if exclude_mixed:
        keep = (frac == 0) | (frac >= min_frac)
        ri, st, frac = ri[keep], st[keep], frac[keep]
    pass #if
    return Windows(recs, ri, st, frac, (frac >= min_frac).astype(np.uint8), n)
pass #def


class WindowSet(Dataset):
    """Torch dataset over a :class:`Windows` table; yields ``(1, n)`` float and a label.

    With ``augment`` it jitters the start by up to half a second and flips the polarity at
    random (the window is renormalised afterwards, so amplitude scaling is a no-op).
    """

    def __init__(self, win: Windows, augment: bool = False, seed: int = 0):
        self.win = win
        self.augment = augment
        self.rng = np.random.default_rng(seed)
    pass #def

    def __len__(self) -> int:
        return len(self.win)
    pass #def

    def __getitem__(self, i: int):
        w = self.win
        x = w.recs[w.rec[i]].x
        s = int(w.start[i])
        if self.augment:
            s = int(np.clip(s + self.rng.integers(-FS // 2, FS // 2 + 1), 0, len(x) - w.size))
        pass #if
        seg = normalize_window(x[s:s + w.size])
        if self.augment and self.rng.random() < 0.5:
            seg = -seg
        pass #if
        return torch.from_numpy(np.ascontiguousarray(seg))[None, :], torch.tensor(float(w.y[i]))
    pass #def
pass #class


class Store:
    """Every record's filtered signal in one device tensor, so batches are cut on the device.

    On Apple silicon the tensor sits in unified memory, so this removes the per-window Python
    work from the training loop. Windows are addressed through a :class:`Windows` table whose
    ``recs`` list is the one the store was built from.
    """

    def __init__(self, recs: list[Rec], device):
        self.device = device
        self.offset = np.zeros(len(recs), dtype=np.int64)
        self.length = np.array([len(r.x) for r in recs], dtype=np.int64)
        self.offset[1:] = np.cumsum(self.length)[:-1]
        self.x = torch.from_numpy(np.concatenate([r.x for r in recs])).to(device)
        self._arange = {}
    pass #def

    def batch(self, win: "Windows", idx: np.ndarray, augment: bool = False):
        """Normalised ``(len(idx), 1, size)`` windows and their labels, gathered on the device."""
        n = win.size
        ar = self._arange.get(n)
        if ar is None:
            ar = self._arange[n] = torch.arange(n, device=self.device)
        pass #if
        rec = win.rec[idx]
        lo = torch.from_numpy(self.offset[rec]).to(self.device)
        hi = lo + torch.from_numpy(self.length[rec] - n).to(self.device)
        s = lo + torch.from_numpy(win.start[idx]).to(self.device)
        if augment:
            jitter = torch.randint(-FS // 2, FS // 2 + 1, s.shape, device=self.device)
            s = torch.minimum(torch.maximum(s + jitter, lo), hi)
        pass #if
        x = self.x[s[:, None] + ar[None, :]]
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, unbiased=False, keepdim=True) + 1e-6)
        if augment:
            sign = (torch.rand(x.shape[0], 1, device=self.device) < 0.5).float() * 2 - 1
            x = x * sign
        pass #if
        y = torch.from_numpy(win.y[idx].astype(np.float32)).to(self.device)
        return x[:, None, :], y
    pass #def
pass #class


def random_folds(n: int, k: int = 10, seed: int = 0):
    """k random folds over window indices (the LDIAED-style protocol, with leakage)."""
    perm = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(perm, k)
    return [(np.concatenate(parts[:i] + parts[i + 1:]), parts[i]) for i in range(k)]
pass #def


def group_folds(groups: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0):
    """Record-wise folds: whole records go to one fold, balanced on positives and on windows.

    Records are placed largest first (positive windows, then all windows). Each goes to the
    fold where the summed share of positives and of windows ends up smallest, so folds get
    similar numbers of both and the records with no positives spread out too.
    """
    keys, inv = np.unique(groups, return_inverse=True)
    pos = np.bincount(inv, weights=y, minlength=len(keys))
    cnt = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(keys)), -cnt, -pos))
    lp = np.zeros(k)
    ln = np.zeros(k)
    fold_of = np.zeros(len(keys), dtype=int)
    tp, tn = max(pos.sum(), 1.0), cnt.sum()
    for g in order:
        f = int(np.argmin((lp + pos[g]) / tp + (ln + cnt[g]) / tn))
        fold_of[g] = f
        lp[f] += pos[g]
        ln[f] += cnt[g]
    pass #for
    fold = fold_of[inv]
    idx = np.arange(len(groups))
    return [(idx[fold != f], idx[fold == f]) for f in range(k)]
pass #def


def leave_db_out(dbs: np.ndarray):
    """One fold per database: train on the others, test on it."""
    idx = np.arange(len(dbs))
    return [(idx[dbs != d], idx[dbs == d]) for d in np.unique(dbs)]
pass #def


def make_folds(win: "Windows", split: str, k: int | None = None, seed: int = 0):
    """Folds and their labels for one split protocol: ``random``, ``group`` or ``lodo``."""
    if split == "random":
        folds = random_folds(len(win), k or 10, seed)
        return folds, [f"r{i}" for i in range(len(folds))]
    pass #if
    if split == "group":
        folds = group_folds(win.groups(), win.y, k or 5, seed)
        return folds, [f"g{i}" for i in range(len(folds))]
    pass #if
    return leave_db_out(win.databases()), sorted(set(win.databases()))
pass #def


def flat_windows(rec: Rec, seconds: float = 8.0, step_s: float = 1.0, tol: float = 1e-6) -> int:
    """Number of windows whose standard deviation is below ``tol`` (a dead lead or a gap)."""
    n = int(seconds * FS)
    step = int(step_s * FS)
    x = rec.x.astype(np.float64)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    st = np.arange(0, len(x) - n + 1, step)
    mean = (c1[st + n] - c1[st]) / n
    var = np.maximum((c2[st + n] - c2[st]) / n - mean * mean, 0.0)
    return int((np.sqrt(var) < tol).sum())
pass #def


def prepare_record(db: str, rid: str, rebuild: bool = False) -> tuple[str, str, int]:
    """Filter and cache one record. Returns ``(db, rid, samples)``."""
    if rebuild:
        path = os.path.join(CACHE, db, f"{rid}.npz")
        if os.path.exists(path):
            os.remove(path)
        pass #if
    pass #if
    r = load_record(db, rid)
    return db, rid, len(r.x)
pass #def


def summary(recs: list[Rec], windows: tuple[float, ...] = (8.0, 4.0)) -> str:
    """Per-database table: records, hours, VF and VT minutes, windows and positive windows."""
    dbs = sorted({r.db for r in recs}, key=lambda d: [r.db for r in recs].index(d))
    cols = "".join(f"{f'win {w:g}s':>10}{f'pos {w:g}s':>9}" for w in windows)
    lines = [f"{'db':<8}{'records':>8}{'hours':>8}{'VF min':>8}{'VT min':>8}{'VF recs':>9}{cols}{'flat 8s':>9}{'bad':>5}"]
    for db in dbs:
        rs = [r for r in recs if r.db == db]
        cells = ""
        for w in windows:
            win = make_windows(rs, w)
            cells += f"{len(win):>10}{int(win.y.sum()):>9}"
        pass #for
        flat = sum(flat_windows(r) for r in rs)
        bad = sum(int((~np.isfinite(r.x)).sum()) for r in rs)
        lines.append(
            f"{db:<8}{len(rs):>8}{sum(len(r.x) for r in rs) / FS / 3600:>8.1f}"
            f"{sum(int(r.vf.sum()) for r in rs) / FS / 60:>8.1f}{sum(int(r.vt.sum()) for r in rs) / FS / 60:>8.1f}"
            f"{sum(int(r.vf.any()) for r in rs):>9}{cells}{flat:>9}{bad:>5}"
        )
    pass #for
    return "\n".join(lines)
pass #def


def _main() -> None:
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser(description="Filter, label and cache the records, then summarise them.")
    ap.add_argument("--dbs", nargs="+", default=list(DATABASES))
    ap.add_argument("--ahadb-8-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="ignore and replace cached records")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--records", action="store_true", help="also print the per-record table")
    args = ap.parse_args()

    todo = [(db, rid) for db in args.dbs for rid in record_ids(db, args.ahadb_8_only)]
    Parallel(n_jobs=args.jobs)(delayed(prepare_record)(db, rid, args.rebuild) for db, rid in todo)
    recs = [load_record(db, rid) for db, rid in todo]
    print(f"{len(recs)} records cached under {CACHE}, tag {TAG}\n")
    print(summary(recs))
    if args.records:
        print(f"\n{'record':<14}{'min':>7}{'VF min':>8}{'VT min':>8}{'pos 8s':>8}")
        for r in recs:
            w = make_windows([r], 8.0)
            print(f"{r.key:<14}{len(r.x) / FS / 60:>7.1f}{r.vf.sum() / FS / 60:>8.1f}{r.vt.sum() / FS / 60:>8.1f}{int(w.y.sum()):>8}")
        pass #for
    pass #if
pass #def


if __name__ == "__main__":
    _main()
pass #if
