"""One TSV row per window: keys, episode-duration labels, beat-type counts.

The row layout follows ``ideas/VFML.ipynb`` so the built files stay compatible
with the existing analysis notebooks. Episode labels are durations in samples
(how much of the window each rhythm covers); beat entries are counts of each
annotation type in the window. The vftx feature columns are appended in a
later step; keep new feature columns after the ones defined here so existing
readers are not disturbed.
"""

from __future__ import annotations

# Episode-duration labels (samples of the window covered by each rhythm).
LABELS = ["NSR", "BGM", "TGM", "VTH", "VFL", "VFN", "VFB", "AFL", "AFB", "EPX"]

# Beat-type counts (annotation marks in the window), one bucket per column.
BEATS = ["N", "L", "R", "B", "A", "S", "C", "V", "W", "F", "Q", "P", "O", "Z"]

META = ["DB", "RID", "Start", "End"]

HEADER = "\t".join(META + LABELS + BEATS)

# Episode note (WFDB rhythm string) -> label bucket. Type "[" maps to VFN.
_EPISODE_MAP = {
    "(N": "NSR",
    "(B": "BGM",
    "(T": "TGM",
    "(VT": "VTH",
    "(VFL": "VFL",
    "(VF": "VFB",
    "(AFL": "AFL",
    "(AFIB": "AFB",
}

# Annotation beat type -> beat bucket. Anything unlisted falls into "Z".
_BEAT_MAP = {
    "N": "N", "L": "L", "R": "R", "B": "B", "A": "A", "S": "S",
    "J": "C", "a": "C", "j": "C", "e": "C", "n": "C",
    "V": "V", "r": "W", "E": "W", "F": "F", "Q": "Q", "f": "Q", "/": "P",
    "~": "O", "!": "O", "[": "O", "]": "O",
}


def _episode_bucket(ann) -> str:
    if ann.Type == "[":
        return "VFN"
    pass #if
    return _EPISODE_MAP.get(ann.Note, "EPX")
pass #def


class Segment:
    """Labels and beat counts for one window ``[start, end)`` of a record.

    Parameters
    ----------
    db, rid:
        Database and record identifiers.
    start, end:
        Window bounds in samples.
    epis:
        Episode annotations overlapping the window ("+" rhythm markers and
        "[" VF-onset markers), each carrying ``.Time`` and ``.End``.
    beats:
        Beat annotations inside the window, each carrying ``.Type``.
    """

    def __init__(self, db: str, rid: str, start: int, end: int, epis: list, beats: list):
        self.db = db
        self.rid = rid
        self.start = start
        self.end = end
        self.labels = {k: 0 for k in LABELS}
        self.counts = {k: 0 for k in BEATS}

        for e in epis:
            a = max(start, e.Time)
            b = min(end, e.End)
            k = b - a
            if k > 0:
                self.labels[_episode_bucket(e)] += k
            pass #if
        pass #for

        for ann in beats:
            self.counts[_BEAT_MAP.get(ann.Type, "Z")] += 1
        pass #for
    pass #def

    def row(self) -> list:
        return (
            [self.db, self.rid, self.start, self.end]
            + [self.labels[k] for k in LABELS]
            + [self.counts[k] for k in BEATS]
        )
    pass #def

    def line(self) -> str:
        return "\t".join(str(x) for x in self.row())
    pass #def
pass #class
