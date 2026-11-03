"""vfnn.data: window labels (the 50% rule) and the three split types, on synthetic records."""

import numpy as np

from vfnn import data
from vfnn.filters import FS


def synthetic(db, rid, seconds=60, vf=(), vt=()):
    n = seconds * FS
    mv, mt = np.zeros(n, bool), np.zeros(n, bool)
    for a, b in vf:
        mv[a * FS:b * FS] = True
    pass #for
    for a, b in vt:
        mt[a * FS:b * FS] = True
    pass #for
    return data.Rec(db, rid, np.zeros(n, np.float32), mv, mt)
pass #def


def test_fifty_percent_rule():
    r = synthetic("d", "r", 30, vf=[(10, 14)])        # 4 s of VF
    w = data.make_windows([r], seconds=8.0, step_s=1.0)
    by_start = dict(zip(w.start // FS, zip(w.frac, w.y)))
    assert by_start[10][0] == 0.5 and by_start[10][1] == 1      # exactly 50%: positive
    assert by_start[11][0] < 0.5 and by_start[11][1] == 0       # 3 s of 8: negative
    assert by_start[6][1] == 1                                   # window 6..14 holds the 4 s
    assert by_start[0][1] == 0
pass #def


def test_vt_only_counts_when_asked():
    r = synthetic("d", "r", 30, vt=[(0, 30)])
    assert data.make_windows([r]).y.sum() == 0
    assert data.make_windows([r], vt_positive=True).y.all()
pass #def


def test_window_count():
    r = synthetic("d", "r", 30)
    assert len(data.make_windows([r], seconds=8.0, step_s=1.0)) == 23
    assert len(data.make_windows([r], seconds=4.0, step_s=1.0)) == 27
pass #def


def test_random_folds_partition():
    folds = data.random_folds(103, k=10, seed=1)
    tests = np.concatenate([t for _, t in folds])
    assert sorted(tests) == list(range(103))
    for tr, te in folds:
        assert not set(tr) & set(te)
    pass #for
pass #def


def test_group_folds_keep_records_together():
    recs = [synthetic("d", f"r{i}", 40, vf=[(10, 30)] if i % 2 else []) for i in range(8)]
    w = data.make_windows(recs)
    folds = data.group_folds(w.groups(), w.y, k=4, seed=0)
    g = w.groups()
    for tr, te in folds:
        assert not set(g[tr]) & set(g[te])
    pass #for
    assert sorted(np.concatenate([te for _, te in folds])) == list(range(len(w)))
    pos_per_fold = [int(w.y[te].sum()) for _, te in folds]
    assert min(pos_per_fold) > 0
pass #def


def test_leave_db_out():
    recs = [synthetic("a", "1"), synthetic("b", "2"), synthetic("c", "3")]
    w = data.make_windows(recs)
    folds = data.leave_db_out(w.databases())
    assert len(folds) == 3
    for tr, te in folds:
        assert not set(w.databases()[tr]) & set(w.databases()[te])
    pass #for
pass #def


def test_dataset_item_is_normalised():
    r = synthetic("d", "r", 20)
    r.x[:] = np.random.default_rng(0).normal(3, 2, r.x.size)
    x, y = data.WindowSet(data.make_windows([r]))[0]
    assert x.shape == (1, 2000)
    assert abs(x.mean().item()) < 1e-4 and abs(x.std().item() - 1) < 1e-3
pass #def


class Ann:
    def __init__(self, typ, note, t, e):
        self.Type, self.Note, self.Time, self.End = typ, note, t, e
    pass #def
pass #class


def test_episode_notes():
    anns = [Ann("+", "(VFIB", 0, 100), Ann("+", "(VF", 100, 200), Ann("+", "(VFL", 200, 300),
            Ann("+", "(VT", 300, 400), Ann("+", "(N", 400, 500), Ann("+", "(NOISE", 500, 600),
            Ann("+", "(VER", 600, 700), Ann("[", "", 700, 800), Ann("N", "", 800, 801)]
    vf, vt = data.episode_masks(anns, 1000)
    assert vf[:300].all() and vf[700:800].all()
    assert not vf[300:700].any() and not vf[800:].any()
    assert vt[300:400].all() and vt.sum() == 100
pass #def


def test_flat_windows_counts_dead_stretches():
    r = synthetic("d", "r", 30)
    r.x[:] = np.random.default_rng(0).normal(size=r.x.size).astype(np.float32)
    r.x[10 * FS:20 * FS] = 0.0                       # a 10 s dead stretch
    assert data.flat_windows(r, seconds=8.0, step_s=1.0) == 3     # starts at 10, 11, 12 s
pass #def


def test_group_folds_balance_windows_and_positives():
    rng = np.random.default_rng(3)
    recs = []
    for i in range(40):
        vf = [(10, 10 + int(rng.integers(5, 40)))] if i % 4 == 0 else []
        recs.append(synthetic("d", f"r{i:02d}", int(rng.integers(40, 120)), vf=vf))
    pass #for
    w = data.make_windows(recs)
    folds = data.group_folds(w.groups(), w.y, k=5, seed=0)
    n = np.array([len(te) for _, te in folds], float)
    pos = np.array([w.y[te].sum() for _, te in folds], float)
    assert n.min() > 0.7 * n.mean() and n.max() < 1.3 * n.mean()
    assert pos.min() > 0.5 * pos.mean() and pos.max() < 1.5 * pos.mean()
pass #def


def test_store_batch_matches_windowset():
    import torch
    rng = np.random.default_rng(0)
    recs = []
    for i in range(3):
        r = synthetic("d", f"r{i}", 30, vf=[(5, 15)] if i == 1 else [])
        r.x[:] = rng.normal(i, 1 + i, r.x.size).astype(np.float32)
        recs.append(r)
    pass #for
    win = data.make_windows(recs, seconds=4.0, step_s=3.0)
    store = data.Store(recs, torch.device("cpu"))
    idx = np.array([0, 5, len(win) // 2, len(win) - 1])
    x, y = store.batch(win, idx)
    ref = data.WindowSet(win)
    for j, i in enumerate(idx):
        xr, yr = ref[int(i)]
        assert np.allclose(x[j].numpy(), xr.numpy(), atol=1e-4)
        assert float(y[j]) == float(yr)
    pass #for
    xa, _ = store.batch(win, idx, augment=True)          # stays in range and normalised
    assert xa.shape == x.shape and abs(xa.mean().item()) < 1e-3
pass #def


def test_exclude_mixed_drops_partial_windows():
    r = synthetic("d", "r", 30, vf=[(10, 20)])       # 10 s of VF, 8 s windows, 1 s step
    w = data.make_windows([r], seconds=8.0, step_s=1.0, min_frac=0.85, exclude_mixed=True)
    by_start = dict(zip((w.start // FS).tolist(), w.y.tolist()))
    assert {k for k, y in by_start.items() if y == 1} == set(range(9, 14))     # >= 7 of 8 s
    assert {k for k, y in by_start.items() if y == 0} == {0, 1, 2, 20, 21, 22}  # no VF at all
    full = data.make_windows([r], seconds=8.0, step_s=1.0, min_frac=0.85)
    assert full.y.sum() == 5 and len(full) == 23 and len(w) == 11             # 12 mixed dropped
pass #def
