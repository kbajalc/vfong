"""vfta -- Ventricular Tachyarrhythmias dataset construction.

Builds per-record TSV feature files from the cbor ECG databases, for the
paper's feature-discrimination study (see ``paper/PLAN.md``, Phase 2).

The pipeline reuses the skeleton prototyped in ``ideas/VFML.ipynb``:

    cbor record -> record-level filtering -> sliding window -> per-record TSV

Milestone 1 (this scaffold) writes episode-duration labels and beat-type
counts per window. The 27 ``vftx`` feature columns are wired in a later step.

Modules
-------
``filters``   record-level Lynn / baseline filters (lifted from VFML)
``segment``   one TSV row: window keys, episode-duration labels, beat counts
``build``     sliding window, per-record build, parallel per-database build
``labels``    shockable / VT / VFL / VF target assignment from a built dataset
"""

__all__ = ["filters", "segment", "build", "labels"]
