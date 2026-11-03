"""Metrics, threshold choice and result tables for vfnn."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y = y.astype(bool)
    pred = pred.astype(bool)
    return int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum()), int((~y & ~pred).sum())
pass #def


def scores(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """AUC plus sensitivity, specificity, PPV and F1 at ``thr`` (p >= thr is positive)."""
    tp, fp, fn, tn = confusion(y, p >= thr)
    se = tp / (tp + fn) if tp + fn else float("nan")
    sp = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    both = 0 < y.sum() < len(y)
    auc = float(roc_auc_score(y, p)) if both else float("nan")
    return {"n": len(y), "pos": int(y.sum()), "auc": auc, "se": se, "sp": sp, "ppv": ppv, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
pass #def


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Probability threshold that maximises F1 on ``(y, p)``; 0.5 if there is no positive."""
    if y.sum() == 0:
        return 0.5
    pass #if
    order = np.argsort(-p)
    ys = y[order].astype(float)
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    f1 = 2 * tp / (tp + fp + y.sum())                   # 2TP / (2TP + FP + FN)
    i = int(np.argmax(f1))
    return float(p[order][i])
pass #def


def per_record(groups: np.ndarray, dbs: np.ndarray, y: np.ndarray, p: np.ndarray, thr: float) -> list[dict]:
    """Per-record counts of windows, positives and predicted positives at ``thr``."""
    pred = p >= thr
    out = []
    for g in np.unique(groups):
        m = groups == g
        out.append({
            "record": str(g), "db": str(dbs[m][0]), "n": int(m.sum()), "pos": int(y[m].sum()),
            "pred": int(pred[m].sum()), "fp": int((pred[m] & (y[m] == 0)).sum()),
            "tp": int((pred[m] & (y[m] == 1)).sum()),
        })
    pass #for
    return out
pass #def


def false_positive_table(detail: list[dict], step_s: float = 1.0) -> str:
    """False positives on records that hold no positive window at all, per database.

    The rate is false-positive windows over windows. Windows step by ``step_s`` seconds, so
    ``fp * step_s`` is the alarmed time and the last column is that per hour of record time.
    """
    neg = [r for r in detail if r["pos"] == 0]
    if not neg:
        return "no records without positives in the test folds"
    pass #if
    lines = [f"{'db':<8}{'records':>8}{'windows':>9}{'FP win':>8}{'FP rate':>9}{'alarm s/h':>11}   worst records"]
    for db in sorted({r["db"] for r in neg}):
        rs = [r for r in neg if r["db"] == db]
        n = sum(r["n"] for r in rs)
        fp = sum(r["fp"] for r in rs)
        per_hour = fp * step_s / (n * step_s / 3600) if n else float("nan")
        worst = sorted(rs, key=lambda r: -r["fp"])[:3]
        w = " ".join(f"{r['record'].split('/')[-1]}:{r['fp']}" for r in worst if r["fp"])
        lines.append(f"{db:<8}{len(rs):>8}{n:>9}{fp:>8}{fp / n:>9.4f}{per_hour:>11.1f}   {w}")
    pass #for
    return "\n".join(lines)
pass #def


def summarize(rows: list[dict]) -> dict:
    """Mean and standard deviation over folds, plus the pooled counts."""
    out = {}
    for k in ("auc", "se", "sp", "ppv", "f1"):
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(np.nanmean(v)), float(np.nanstd(v)))
    pass #for
    tp, fp, fn, tn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "tn"))
    out["pooled"] = scores_from_counts(tp, fp, fn, tn)
    return out
pass #def


def scores_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    return {
        "se": tp / (tp + fn) if tp + fn else float("nan"),
        "sp": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"),
    }
pass #def


def table(name: str, rows: list[dict], labels: list[str]) -> str:
    """Plain-text per-fold table followed by the mean and spread."""
    head = f"{name}\n{'fold':<10}{'n':>8}{'pos':>7}{'AUC':>7}{'Se':>7}{'Sp':>7}{'PPV':>7}{'F1':>7}"
    lines = [head]
    for lab, r in zip(labels, rows):
        lines.append(f"{lab:<10}{r['n']:>8}{r['pos']:>7}{r['auc']:>7.3f}{r['se']:>7.3f}"
                     f"{r['sp']:>7.3f}{r['ppv']:>7.3f}{r['f1']:>7.3f}")
    pass #for
    s = summarize(rows)
    lines.append(f"{'mean':<10}{'':>15}" + "".join(f"{s[k][0]:>7.3f}" for k in ("auc", "se", "sp", "ppv", "f1")))
    lines.append(f"{'sd':<10}{'':>15}" + "".join(f"{s[k][1]:>7.3f}" for k in ("auc", "se", "sp", "ppv", "f1")))
    return "\n".join(lines)
pass #def
