"""Build per-record TSV dataset files from the cbor ECG databases.

Milestone 1: cbor record -> record-level filtering -> sliding window ->
per-record TSV of episode-duration labels and beat-type counts. Records are
processed in parallel, each writing its own file, so there is no shared state.

Output layout: ``data/s<seconds>/<db>/<rid>.tsv`` (for example
``data/s8/vfdb/418.tsv`` for the 8 s window). ``data/`` is git-ignored.

Run from the repo root so the default work directory resolves:

    python -m vfta.build vfdb --window 8
    python -m vfta.build mitdb --window 4 --jobs -1
"""

from __future__ import annotations

import argparse
import os

from joblib import Parallel, delayed
from pxg import env
from pxg.cbor import CborDatabase, CborRecord

from vfta.features import default_config, feature_names, window_features
from vfta.filters import LYN_WIND, MED_WIND, signal_filter
from vfta.segment import Segment, header


def default_outdir(window_sec: float) -> str:
    """Output directory for a given window length (git-ignored ``data/``)."""
    return f"data/s{int(window_sec)}"
pass #def


def slide_segments(rec: CborRecord, window: int, step: int):
    """Yield one :class:`Segment` per window position over a loaded record.

    ``window`` and ``step`` are in samples. Episodes and beats are tracked in
    rolling lists and trimmed to the window as it advances, matching the VFML
    prototype. Partial windows at the start (before a full window fits) are
    skipped.
    """
    anns = rec.Annotations
    size = len(rec.Signal)

    aix = 0
    epis: list = []
    beats: list = []
    for t in range(step, size, step):
        while aix < len(anns) and anns[aix].Time < t:
            a = anns[aix]
            if a.Type == "[" or (a.Type == "+" and a.Note[:1] == "("):
                epis.append(a)
            else:
                beats.append(a)
            pass #if
            aix += 1
        pass #while

        c = t - window

        r = 0
        while r < len(beats) and beats[r].Time < c:
            r += 1
        pass #while
        beats = beats[r:]

        r = 0
        while r < len(epis) and epis[r].End <= c:
            r += 1
        pass #while
        epis = epis[r:]

        if c < 0:
            continue
        pass #if

        yield Segment(rec.DB, rec.RID, c, t, epis, beats)
    pass #for
pass #def


def build_record(
    db: str,
    rid: str,
    window_sec: float = 8.0,
    step_sec: float = 1.0,
    outdir: str | None = None,
    fs: int = 250,
    chn: int = 0,
    lyn: int = LYN_WIND,
    med: int = MED_WIND,
    hi: int = 0,
    spen: bool = False,
) -> tuple[str, int]:
    """Filter one record and write its sliding-window TSV. Returns (path, rows).

    Each row carries the labels, beat counts, and the vftx feature columns
    (``spen`` adds sample entropy; QRS and EMD features are excluded, see
    ``vfta.features``).
    """
    if outdir is None:
        outdir = default_outdir(window_sec)
    pass #if

    # chn=0 (the default) tells cbor to pick the best channel: lead II (MLII/ML2/II)
    # where present, otherwise the nearest lead. It is not channel index 0.
    rec = CborRecord(db, rid, atr="atr", chn=chn, freq=fs)
    rec.Signal = signal_filter(rec.Signal.astype(float), w=lyn, m=med, h=hi)

    window = int(window_sec * fs)
    step = int(step_sec * fs)
    cfg = default_config(fs)
    names = feature_names(spen=spen, jekova=True)

    dbdir = os.path.join(outdir, db)
    os.makedirs(dbdir, exist_ok=True)
    path = os.path.join(dbdir, f"{rid}.tsv")

    rows = 0
    with open(path, "w") as out:
        out.write(header(names) + "\n")
        for seg in slide_segments(rec, window, step):
            feats = window_features(rec.Signal[seg.start:seg.end], cfg, spen=spen, jekova=True)
            out.write(seg.line(feats))
            out.write("\n")
            rows += 1
        pass #for
    pass #with
    return path, rows
pass #def


def build_database(
    db: str,
    window_sec: float = 8.0,
    step_sec: float = 1.0,
    jobs: int = -1,
    fs: int = 250,
    chn: int = 0,
    outdir: str | None = None,
    records: list[str] | None = None,
    spen: bool = False,
) -> list[tuple[str, int]]:
    """Build every record of a database in parallel. Returns [(path, rows), ...]."""
    if outdir is None:
        outdir = default_outdir(window_sec)
    pass #if

    cdb = CborDatabase(db, freq=fs, chn=chn)
    rids = records if records is not None else cdb.Records

    return Parallel(n_jobs=jobs)(
        delayed(build_record)(db, rid, window_sec, step_sec, outdir, fs, chn, spen=spen)
        for rid in rids
    )
pass #def


def _main() -> None:
    ap = argparse.ArgumentParser(description="Build per-record TSV dataset files.")
    ap.add_argument("db", help="database name (mitdb, cudb, vfdb, ahadb)")
    ap.add_argument("--window", type=float, default=8.0, help="window length in seconds")
    ap.add_argument("--step", type=float, default=1.0, help="window step in seconds")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    ap.add_argument("--outdir", default=None, help="output directory (default data/s<sec>)")
    ap.add_argument("--spen", action="store_true", help="also compute sample entropy (~55 ms/window)")
    args = ap.parse_args()

    results = build_database(
        args.db, window_sec=args.window, step_sec=args.step, jobs=args.jobs,
        outdir=args.outdir, spen=args.spen,
    )
    total = sum(rows for _, rows in results)
    print(f"{args.db}: {len(results)} records, {total} windows -> {default_outdir(args.window) if not args.outdir else args.outdir}")
pass #def


if __name__ == "__main__":
    _main()
pass #if
