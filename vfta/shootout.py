"""Candidate shootout for Phase 3.

Five deterministic detectors, each a feature (or small feature group) plus a
threshold decision, compared on shockable vs non-shockable by discrimination and
by compute cost per window. See paper/PLAN.md "Candidate detectors and winner
selection".

Discrimination side: each detector is scored by its strongest constituent
feature (oriented single-feature AUC, mutual information, and F1 at the best
swept threshold). A multi-feature detector (SPEC, JEKOVA) is represented by its
best sub-feature here; the full multi-threshold tuning is Phase 4's job, so the
shootout stays a transparent single-threshold comparison.

Cost side: the wall-clock time to compute the detector's decision feature on one
window, timed on representative signal windows through the same vftx path the
build uses (preprocess once, then the feature). Preprocessing is shared by all
detectors and reported once as a baseline. Cost is charged at the primary
feature, matching the single-threshold discrimination. That also avoids a
measurement artifact: vftx computes count1/count2/count3 (and the SPEC
descriptors) independently, each redoing the shared heavy step (the 14.6 Hz
band-pass, the power spectrum), so summing them would triple-count work a real
detector does once. The primary feature carries that heavy step exactly once.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_recall_curve, roc_auc_score

from vfta import jekova
from vfta.features import _CHEAP, default_config
from vftx.preprocessing import preprocess

# Detector -> its feature column(s). Order matches paper/PLAN.md. JEKOVA uses the
# absolute-output counts jc1/jc2/jc3 (paper-faithful), not vftx's signed count1/2/3.
CANDIDATES: dict[str, list[str]] = {
    "TCSC": ["tcsc"],
    "VFLEAK": ["vf_leak"],
    "SPEC": ["m", "a2", "fm"],
    "HILB": ["hilb"],
    "JEKOVA": ["jc1", "jc2", "jc3"],
}

# Rough cost tier for each detector (for the discussion; measured cost is exact).
COST_TIER: dict[str, str] = {
    "TCSC": "cheap",
    "VFLEAK": "cheap",
    "SPEC": "FFT",
    "HILB": "FFT",
    "JEKOVA": "cheap",
}

_FEATURE_FN = dict(_CHEAP)


def _best_f1(y: np.ndarray, score: np.ndarray) -> float:
    """Best F1 over all thresholds for a monotone-increasing score of SHOCK."""
    prec, rec, _ = precision_recall_curve(y, score)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    return float(f1.max())
pass #def


def sweep_f1(y: np.ndarray, x: np.ndarray) -> float:
    """Best F1 at any single threshold on x, either orientation."""
    return max(_best_f1(y, x), _best_f1(y, -x))
pass #def


def discrimination(df: pd.DataFrame, candidates: dict[str, list[str]] = CANDIDATES,
                   mi_sample: int = 30000, seed: int = 0) -> pd.DataFrame:
    """Score each detector by its strongest feature on the clean set.

    Returns one row per detector: the primary (best) feature, its oriented AUC,
    mutual information, and best swept-threshold F1. ``df`` is the clean screen
    set (SHOCK vs NON, MIX dropped).
    """
    y = (df["Shock"] == "SHOCK").astype(int).to_numpy()

    samp = df.sample(min(mi_sample, len(df)), random_state=seed)
    ys = (samp["Shock"] == "SHOCK").astype(int).to_numpy()

    rows = []
    for name, feats in candidates.items():
        # primary feature = best oriented AUC among the detector's features
        aucs = {f: roc_auc_score(y, df[f].to_numpy(dtype=float)) for f in feats}
        primary = max(feats, key=lambda f: max(aucs[f], 1.0 - aucs[f]))
        auc = max(aucs[primary], 1.0 - aucs[primary])
        x = df[primary].to_numpy(dtype=float)
        f1 = sweep_f1(y, x)
        mi = float(mutual_info_classif(
            samp[[primary]].to_numpy(dtype=float), ys, random_state=seed)[0])
        rows.append((name, "+".join(feats), primary, COST_TIER[name], auc, mi, f1))
    pass #for

    res = pd.DataFrame(rows, columns=[
        "detector", "features", "primary", "tier", "auc", "mutual_info", "f1"])
    return res.sort_values("auc", ascending=False).reset_index(drop=True)
pass #def


def _sample_windows(win_samples: int, n: int, seed: int) -> np.ndarray:
    """Synthetic mV windows for timing: mixed oscillation plus noise.

    Compute cost depends on window length and algorithm structure, not on the
    exact signal, so a deterministic synthetic ensemble gives a stable per-window
    cost without needing the raw ECG at analysis time.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(win_samples) / 250.0
    out = np.empty((n, win_samples))
    for i in range(n):
        f = rng.uniform(1.0, 8.0)
        out[i] = np.sin(2 * np.pi * f * t) + 0.3 * rng.standard_normal(win_samples)
    pass #for
    return out
pass #def


def cost_per_window(window_sec: float = 8.0, fs: int = 250, n: int = 200,
                    repeats: int = 3, seed: int = 0) -> pd.Series:
    """Median ms/window for preprocessing and for each cheap feature.

    Times the same vftx path the build uses: preprocess once per window, then
    each feature on the shared preprocessed signal. Index includes ``preprocess``
    plus every feature name.
    """
    cfg = default_config(fs)
    wins = _sample_windows(int(window_sec * fs), n, seed)
    pp = [preprocess(w, cfg) for w in wins]

    times = {"preprocess": _median_ms(lambda w: preprocess(w, cfg), wins, repeats)}
    for name, fn in _CHEAP:
        times[name] = _median_ms(lambda w, fn=fn: fn(w, cfg), pp, repeats)
    pass #for
    # JEKOVA's three absolute counts come from one band-pass pass; charge them one cost.
    jc = _median_ms(lambda w: jekova.abs_counts(w, cfg), pp, repeats)
    times["jc1"] = times["jc2"] = times["jc3"] = jc
    return pd.Series(times)
pass #def


def _median_ms(fn, wins, repeats: int = 3) -> float:
    """Median (best-of-repeats) ms/call for ``fn`` over the given windows."""
    n = len(wins)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        for w in wins:
            fn(w)
        pass #for
        best = min(best, (time.perf_counter() - t0) / n * 1e3)
    pass #for
    return best
pass #def


def candidate_cost(disc: pd.DataFrame, timings: pd.Series,
                   with_preprocess: bool = True) -> pd.Series:
    """ms/window for each detector: shared preprocess plus its primary feature.

    Cost is charged at the primary feature (the one discrimination uses), which
    carries each detector's shared heavy step exactly once: for JEKOVA the
    primary ``count3`` is a single 14.6 Hz band-pass, and for SPEC the primary
    descriptor is a single power spectrum. Summing sub-features would redo that
    step, overstating the real detector.
    """
    base = float(timings["preprocess"]) if with_preprocess else 0.0
    return pd.Series({
        row.detector: base + float(timings[row.primary])
        for row in disc.itertuples()
    })
pass #def


def shootout_table(df: pd.DataFrame, window_sec: float = 8.0, fs: int = 250,
                   candidates: dict[str, list[str]] = CANDIDATES) -> pd.DataFrame:
    """Full discrimination-vs-cost table: discrimination merged with cost.

    Cost is reported as ms per 1000 windows (``ms_per_kwin``), a more readable
    scale than the sub-millisecond per-window figure. It is a pure-Python,
    single-thread measurement on this machine, meant only for relative
    comparison between detectors, not as an absolute or portable timing.
    """
    disc = discrimination(df, candidates)
    timings = cost_per_window(window_sec, fs)
    cost = candidate_cost(disc, timings)
    disc["ms_per_kwin"] = disc["detector"].map(cost) * 1000.0
    return disc
pass #def
