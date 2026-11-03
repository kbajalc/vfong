"""vfnn.eval: scores, threshold choice, and a tiny end-to-end fit on separable data."""

import numpy as np
import torch

from vfnn import data
from vfnn.eval import best_f1_threshold, scores
from vfnn.filters import FS
from vfnn.model import Config
from vfnn.train import Fit, fit, predict


def test_scores_known_counts():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.6, 0.1, 0.0])
    s = scores(y, p, 0.5)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (2, 1, 1, 4)
    assert abs(s["se"] - 2 / 3) < 1e-9 and abs(s["sp"] - 4 / 5) < 1e-9
    assert abs(s["f1"] - 2 / 3) < 1e-9
pass #def


def test_best_f1_threshold_separates():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = best_f1_threshold(y, p)
    assert 0.3 < thr <= 0.7
    assert scores(y, p, thr)["f1"] == 1.0
pass #def


def test_fit_learns_separable_windows():
    # "VF": a fast sine; negative: a slow sine. A tiny net must separate them.
    rng = np.random.default_rng(0)
    n = 60 * FS
    t = np.arange(n) / FS
    recs = []
    for i in range(6):
        f = 5.0 if i % 2 == 0 else 1.2
        x = (np.sin(2 * np.pi * f * t) + 0.2 * rng.standard_normal(n)).astype(np.float32)
        vf = np.full(n, f == 5.0)
        recs.append(data.Rec("d", f"r{i}", x, vf, np.zeros(n, bool)))
    pass #for
    win = data.make_windows(recs, seconds=4.0, step_s=2.0)
    tr = win.subset(np.where(win.rec < 4)[0])
    dev = win.subset(np.where(win.rec >= 4)[0])
    cfg = Config(channels=(8, 8), strides=(2, 2), stem=8)
    model = fit(tr, dev, cfg, Fit(epochs=6, batch=16, augment=False, patience=6), torch.device("cpu"))
    p = predict(model, dev, torch.device("cpu"))
    assert scores(dev.y, p, 0.5)["auc"] > 0.95
pass #def


def test_per_record_and_false_positive_table():
    from vfnn.eval import false_positive_table, per_record
    groups = np.array(["a/1"] * 4 + ["a/2"] * 4 + ["b/3"] * 2)
    dbs = np.array(["a"] * 8 + ["b"] * 2)
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9])
    d = {r["record"]: r for r in per_record(groups, dbs, y, p, 0.5)}
    assert (d["a/1"]["tp"], d["a/1"]["fp"], d["a/1"]["pos"]) == (1, 1, 2)
    assert (d["a/2"]["fp"], d["a/2"]["pos"]) == (2, 0)
    tab = false_positive_table(list(d.values()))
    assert "a " in tab and "b " in tab        # only the two records without positives
    assert "a/1" not in tab
pass #def
