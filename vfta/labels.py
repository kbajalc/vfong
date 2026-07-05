"""Assign detection targets to the built windows.

Analysis side, works on the per-record TSVs produced by :mod:`vfta.build`,
never on the raw ECG. Each window gets a ``Rhythm`` class from its dominant
clean episode and a binary ``Shock`` label. A window whose dominant rhythm does
not cover at least ``purity`` of the window is marked ``MIX`` (used for
evaluation but excluded from the clean training set, per ``paper/PLAN.md``).

The shockable class is VT / VFL / VF vs everything else, matching the published
benchmark target. Rapid-vs-slow VT is not split here: the annotations do not
carry rate, so that refinement waits on the RR / rate feature.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

# Rhythm classes and which ones count as shockable.
SHOCKABLE = {"VT", "VFL", "VF"}

# Episode-label column -> rhythm class it contributes to.
_RHYTHM_COVERAGE = {
    "VT": ["VTH"],
    "VFL": ["VFL"],
    "VF": ["VFN", "VFB"],
    "NSR": ["NSR"],
    "OTHER": ["BGM", "TGM", "AFL", "AFB", "EPX"],
}


def default_outdir(window_sec: float) -> str:
    return f"data/s{int(window_sec)}"
pass #def


def load_database(db: str, window_sec: float = 8.0, outdir: str | None = None) -> pd.DataFrame:
    """Concatenate every per-record TSV for a database into one DataFrame."""
    outdir = outdir or default_outdir(window_sec)
    files = sorted(glob.glob(os.path.join(outdir, db, "*.tsv")))
    if not files:
        raise FileNotFoundError(f"no TSVs under {os.path.join(outdir, db)}")
    pass #if
    frames = [pd.read_csv(f, sep="\t", dtype={"RID": str}) for f in files]
    return pd.concat(frames, ignore_index=True)
pass #def


def assign_targets(df: pd.DataFrame, window_sec: float = 8.0, fs: int = 250, purity: float = 0.9) -> pd.DataFrame:
    """Add ``Rhythm`` and ``Shock`` columns using the purity threshold."""
    thr = purity * window_sec * fs

    cov = pd.DataFrame(index=df.index)
    for rhythm, cols in _RHYTHM_COVERAGE.items():
        cov[rhythm] = df[cols].sum(axis=1)
    pass #for

    dominant = cov.idxmax(axis=1)
    strong = cov.max(axis=1) >= thr

    out = df.copy()
    out["Rhythm"] = dominant.where(strong, "MIX")
    out["Shock"] = out["Rhythm"].map(
        lambda r: "MIX" if r == "MIX" else ("SHOCK" if r in SHOCKABLE else "NON")
    )
    return out
pass #def


def load_dataset(dbs, window_sec: float = 8.0, fs: int = 250, purity: float = 0.9, outdir: str | None = None) -> pd.DataFrame:
    """Load and label several databases, returning one concatenated DataFrame."""
    if isinstance(dbs, str):
        dbs = [dbs]
    pass #if
    frames = []
    for db in dbs:
        df = load_database(db, window_sec, outdir)
        frames.append(assign_targets(df, window_sec, fs, purity))
    pass #for
    return pd.concat(frames, ignore_index=True)
pass #def
