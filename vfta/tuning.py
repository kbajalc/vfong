"""Phase 4 candidate tuning: TCSC threshold sweep and JEKOVA cascade grid search.

Two detectors are tuned on the clean windows (shockable vs non-shockable, MIX
dropped): TCSC by a single-threshold ROC sweep as in COMP55-2005, and JEKOVA by a
grid search over its cascade thresholds (see :mod:`vfta.jekova`). Both are scored
by the same operating-point metrics, under the three VFL configurations (flutter
shockable, non-shockable, or excluded) and at both window lengths.

Everything works on the per-record TSVs already built, never on the raw ECG.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from vfta import jekova

# VF coverage = VT + VF-onset bracket + VF; VFL is handled per configuration.
_VF_COLS = ["VTH", "VFN", "VFB"]


def confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) for boolean-like arrays."""
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    tn = int((~yt & ~yp).sum())
    fn = int((yt & ~yp).sum())
    return tp, fp, tn, fn
pass #def


def metrics(y_true, y_pred) -> dict:
    """Operating-point metrics: Se, Sp, PPV, Acc, F1, G-Mean, and the confusion counts."""
    tp, fp, tn, fn = confusion(y_true, y_pred)
    se = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    f1 = 2 * ppv * se / (ppv + se) if (ppv + se) else 0.0
    gmean = (se * sp) ** 0.5
    return {"se": se, "sp": sp, "ppv": ppv, "acc": acc, "f1": f1, "gmean": gmean,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}
pass #def


def binary_set(df: pd.DataFrame, window_sec: float = 8.0, fs: int = 250,
               purity: float = 0.9, vfl: str = "shock") -> pd.DataFrame:
    """Clean shockable/non-shockable set for one VFL configuration.

    ``vfl`` selects how flutter is treated: ``"shock"`` counts VFL as shockable
    (the default benchmark target), ``"non"`` counts it as non-shockable, and
    ``"excluded"`` drops windows that contain any VFL. Returns the clean rows
    (SHOCK or NON, MIX dropped) with an added integer ``y`` column (1 = shockable).
    """
    thr = purity * window_sec * fs
    d = df[df["VFL"] == 0] if vfl == "excluded" else df
    vf = d[_VF_COLS].sum(axis=1)
    shock_cov = vf + d["VFL"] if vfl == "shock" else vf

    lab = pd.Series("MIX", index=d.index)
    lab = lab.mask(shock_cov >= thr, "SHOCK").mask(shock_cov == 0, "NON")
    clean = d[lab != "MIX"].copy()
    clean["y"] = (lab[lab != "MIX"] == "SHOCK").astype(int).to_numpy()
    return clean
pass #def


def _orient(y, score) -> np.ndarray:
    """Return the score flipped if needed so higher means more shockable."""
    s = np.asarray(score, dtype=float)
    return -s if roc_auc_score(y, s) < 0.5 else s
pass #def


def roc(y, score) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Oriented ROC curve: (fpr, tpr, thresholds, AUC)."""
    s = _orient(y, score)
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr, roc_auc_score(y, s)
pass #def


def sweep_threshold(y, score, objective: str = "f1", n: int = 300) -> tuple[float, dict]:
    """Best single threshold on an oriented score. Returns (threshold, metrics).

    Thresholds are the quantiles of the score; the one maximising ``objective``
    (F1 by default) is returned with its full operating-point metrics.
    """
    s = _orient(y, score)
    cuts = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n)))
    best_t, best_m, best_key = cuts[0], None, -1.0
    for t in cuts:
        m = metrics(y, s >= t)
        if m[objective] > best_key:
            best_key, best_t, best_m = m[objective], float(t), m
        pass #if
    pass #for
    return best_t, best_m
pass #def


def tcsc_tune(clean: pd.DataFrame, feature: str = "tcsc",
              objective: str = "f1") -> tuple[dict, tuple]:
    """Tune a single-feature detector by threshold sweep. Returns (metrics, roc arrays)."""
    y = clean["y"].to_numpy()
    _, m = sweep_threshold(y, clean[feature].to_numpy(dtype=float), objective)
    r = roc(y, clean[feature].to_numpy(dtype=float))
    return m, r
pass #def


def _default_grid() -> dict[str, list[float]]:
    """Coarse grid around the published JEKOVA constants (normalised fractions)."""
    return {
        "c1_lo": [0.06, 0.10, 0.16],
        "c2_hi": [0.30, 0.40, 0.50],
        "c2_top": [0.35, 0.45, 0.55, 0.65],
        "c3_fallback": [0.55, 0.65, 0.75, 0.85],
        "ratio": [0.08, 0.20, 0.50],
    }
pass #def


def jekova_tune(clean: pd.DataFrame, window_sec: float = 8.0, fs: int = 250,
                grid: dict[str, list[float]] | None = None,
                objective: str = "f1") -> tuple[dict, dict, pd.DataFrame]:
    """Grid-search the JEKOVA cascade thresholds. Returns (best_params, best_metrics, points).

    ``points`` is one row per grid combination (its Se, Sp, F1) so the achievable
    operating envelope can be plotted. The counts are normalised to fractions of
    the window sample count before the cascade, so one parameter set is comparable
    across window lengths (see :func:`vfta.jekova.to_fractions`).
    """
    grid = grid or _default_grid()
    n = int(window_sec * fs)
    y = clean["y"].to_numpy()
    f1, f2, f3 = jekova.to_fractions(clean["jc1"], clean["jc2"], clean["jc3"], n)

    fixed = {"c1_hi": jekova.DEFAULT_PARAMS["c1_hi"], "c2_lo": jekova.DEFAULT_PARAMS["c2_lo"]}
    keys = list(grid)
    rows, best = [], None
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {**fixed, **dict(zip(keys, combo))}
        # keep c1_hi above c1_lo so the mid-count rule stays well formed
        params["c1_hi"] = max(params["c1_hi"], params["c1_lo"] + 0.04)
        m = metrics(y, jekova.decide(f1, f2, f3, params))
        rows.append({**{k: params[k] for k in keys}, "se": m["se"], "sp": m["sp"], "f1": m["f1"]})
        if best is None or m[objective] > best[1][objective]:
            best = (params, m)
        pass #if
    pass #for
    return best[0], best[1], pd.DataFrame(rows)
pass #def


def jekova_published(clean: pd.DataFrame, window_sec: float = 8.0, fs: int = 250) -> dict:
    """Metrics of the published-constant cascade (DEFAULT_PARAMS), as a baseline."""
    n = int(window_sec * fs)
    y = clean["y"].to_numpy()
    f1, f2, f3 = jekova.to_fractions(clean["jc1"], clean["jc2"], clean["jc3"], n)
    return metrics(y, jekova.decide(f1, f2, f3, jekova.DEFAULT_PARAMS))
pass #def
