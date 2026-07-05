"""Generate hermetic test fixtures for the vftx/ agreement suite.

Run ONCE per change to the reference or the segment set. Requires the reference
build + network (PhysioNet); the resulting cache lets test_agreement.py run with
no network, no libwfdb, and no Cython.

    conda activate dev
    python setup_ref.py build_ext --inplace     # if not already built
    python tests/gen_fixtures.py

Writes:
    tests/data/fixtures.npz   — per-segment: <id>__sig (raw mV), <id>__ref (27-vec),
                                <id>__fs (1-elem)
    tests/data/fixtures.json  — ordered list of {id, label, record, pn_dir, fs, sampfrom}

Reference QRS features (idx 22-26) are 0 here because qrs_detect is stubbed; the
vftx side is run with qrs_detector=None so those indices match trivially.
"""
import json
import os
import sys

import numpy as np
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(HERE, "_refstub"))  # qrs_detect stub wins over .pyx
sys.path.insert(0, PROJ)

import vf_features  # reference Cython extension (built by setup_ref.py)

# Small diverse set (agreed): NSR + VF across 250/360 Hz + a fine-VF mghdb segment.
# (label, record, pn_dir, fs, sampfrom, channel)
SEGMENTS = [
    ("NSR",     "100",    "mitdb", 360, 0, 0),
    ("VF",      "418",    "vfdb",  250, 0, 0),
    ("VF",      "cu01",   "cudb",  250, 0, 0),
    ("edb",     "e0103",  "edb",   250, 0, 0),
    ("mghFine", "mgh040", "mghdb", 360, 0, 1),  # mghdb uses channel 1
]


def main():
    os.makedirs(DATA, exist_ok=True)
    arrays = {}
    meta = []
    for label, record, pn_dir, fs, sampfrom, channel in SEGMENTS:
        seg_id = f"{pn_dir}_{record}_{sampfrom}"
        n = int(fs * 8)
        print(f"fetching {pn_dir}/{record} @{sampfrom} (fs={fs}, ch={channel}) ...", flush=True)
        rec = wfdb.rdrecord(record, pn_dir=pn_dir, sampfrom=sampfrom,
                            sampto=sampfrom + n, channels=[channel])
        sig = rec.p_signal[:, 0].astype(np.float64)
        ref = np.asarray(vf_features.extract_features(sig, int(fs))[0], dtype=np.float64)
        arrays[f"{seg_id}__sig"] = sig
        arrays[f"{seg_id}__ref"] = ref
        arrays[f"{seg_id}__fs"] = np.array([float(fs)])
        meta.append({"id": seg_id, "label": label, "record": record,
                     "pn_dir": pn_dir, "fs": fs, "sampfrom": sampfrom})

    np.savez_compressed(os.path.join(DATA, "fixtures.npz"), **arrays)
    with open(os.path.join(DATA, "fixtures.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {len(meta)} segments to {DATA}/fixtures.npz + fixtures.json")


if __name__ == "__main__":
    main()
