"""Phase 4 diagnostic: side-by-side reference (Cython) vs vftx/ feature vectors.

Prerequisites (see vftx/PLAN.md "Phase 4"):
    conda activate dev
    pip install Cython setuptools wfdb            # one-time
    python setup_ref.py build_ext --inplace       # builds signal_processing + vf_features

Then:
    python tests/compare_reference.py

Downloads each segment from PhysioNet, feeds the SAME raw-mV array to both the
reference Cython ``vf_features.extract_features`` and ``vftx``, and prints an
absolute/relative-error table per feature. QRS features (idx 22-26) are 0 in the
reference (qrs_detect is stubbed) and excluded from the DIFF flag.
"""
import os
import sys

import numpy as np
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "_refstub"))  # qrs_detect stub wins over .pyx
sys.path.insert(0, PROJ)                            # vf_features .so + vftx/

import vf_features  # reference Cython extension (built by setup_ref.py)
from vftx.extract import extract_features as vftx_extract
from vftx.types import Features, SegmentConfig, SignalConfig

NAMES = Features.NAMES

# Small diverse set: NSR + VF across both 250/360 Hz, plus a fine-VF mghdb segment.
# (db_name, record, pn_dir, fs, sampfrom, channel)
SEGMENTS = [
    ("NSR",     "100",    "mitdb", 360, 0,      0),
    ("VF",      "418",    "vfdb",  250, 0,      0),
    ("VF",      "cu01",   "cudb",  250, 0,      0),
    ("edb",     "e0103",  "edb",   250, 0,      0),
    ("mghFine", "mgh040", "mghdb", 360, 0,      1),  # mghdb uses channel 1
]


def fetch(record, pn_dir, fs, sampfrom, channel):
    n = int(fs * 8)
    rec = wfdb.rdrecord(record, pn_dir=pn_dir, sampfrom=sampfrom,
                        sampto=sampfrom + n, channels=[channel])
    return rec.p_signal[:, 0].astype(np.float64) # type: ignore


def compare(label, record, pn_dir, fs, sampfrom, channel):
    try:
        sig = fetch(record, pn_dir, fs, sampfrom, channel)
    except Exception as exc:  # noqa: BLE001 — diagnostic tool, keep going
        print(f"\n===== {label} ({pn_dir}/{record}) SKIPPED: {exc}")
        return
    ref = np.asarray(vf_features.extract_features(sig, int(fs))[0], dtype=np.float64)
    cfg = SegmentConfig(signal=SignalConfig(sampling_rate=float(fs)))
    vftx = vftx_extract(sig, cfg, qrs_detector=None).to_array()
    print(f"\n===== {label}  ({pn_dir}/{record} @{sampfrom}, fs={fs}) =====")
    print(f"{'idx':>3} {'name':<10} {'reference':>16} {'vftx':>16} {'absdiff':>12} {'reldiff':>10}")
    for i in range(27):
        d = abs(ref[i] - vftx[i])
        rel = d / (abs(ref[i]) + 1e-12)
        flag = "  <-- DIFF" if (i < 22 and rel > 1e-3 and d > 1e-6) else ""
        print(f"{i:>3} {NAMES[i]:<10} {ref[i]:>16.8f} {vftx[i]:>16.8f} {d:>12.2e} {rel:>10.2e}{flag}")


if __name__ == "__main__":
    for spec in SEGMENTS:
        compare(*spec)
    print("\nDONE")
