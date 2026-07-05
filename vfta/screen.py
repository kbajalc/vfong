"""Feature screen for Phase 3.

Rank the per-window features by how well they separate shockable from
non-shockable windows, on the clean set (SHOCK vs NON, MIX dropped), by three
measures: point-biserial correlation (signed), single-feature AUC (oriented so
1.0 is perfect either way), and mutual information. A feature-feature
correlation matrix flags redundancy. See paper/PLAN.md Phase 3.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score


def clean_set(df: pd.DataFrame) -> pd.DataFrame:
    """Screen set: shockable vs non-shockable, MIX transition windows dropped."""
    return df[df["Shock"] != "MIX"].reset_index(drop=True)
pass #def


def screen(df: pd.DataFrame, features: list[str], mi_sample: int = 30000, seed: int = 0) -> pd.DataFrame:
    """Rank features against the shockable label. Returns a table sorted by AUC.

    Columns: point-biserial correlation (signed direction), raw AUC, oriented
    AUC = max(auc, 1-auc), and mutual information (estimated on a subsample for
    speed).
    """
    y = (df["Shock"] == "SHOCK").astype(int).to_numpy()
    rows = []
    for f in features:
        x = df[f].to_numpy(dtype=float)
        r = stats.pointbiserialr(y, x).statistic
        auc = roc_auc_score(y, x)
        rows.append((f, r, auc, max(auc, 1.0 - auc)))
    pass #for
    res = pd.DataFrame(rows, columns=["feature", "pointbiserial", "auc_raw", "auc"])

    samp = df.sample(min(mi_sample, len(df)), random_state=seed)
    ys = (samp["Shock"] == "SHOCK").astype(int).to_numpy()
    res["mutual_info"] = mutual_info_classif(samp[features].to_numpy(dtype=float), ys, random_state=seed)

    return res.sort_values("auc", ascending=False).reset_index(drop=True)
pass #def


def redundancy(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Feature-feature Pearson correlation matrix (for a redundancy heatmap)."""
    return df[features].corr()
pass #def
