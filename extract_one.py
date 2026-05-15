#!/usr/bin/env python3
"""
Single-segment feature extraction — debugger-friendly, no parallelism.

Usage examples:
  # List all segments in a record
  python extract_one.py -r mitdb/100

  # Extract segment by index (0-based)
  python extract_one.py -r mitdb/100 -n 0

  # Extract segment by begin sample number
  python extract_one.py -r mitdb/100 -b 0

  # Extract with label correction file
  python extract_one.py -r vfdb/422 -n 0 -c corrections_s8.txt

  # Extract only specific features
  python extract_one.py -r mitdb/100 -n 0 -f TCSC TCI LZ

Set a breakpoint on the line marked BREAKPOINT below to step into the
feature computation in your debugger.
"""
import pyximport; pyximport.install()
import argparse
import numpy as np
from datetime import timedelta

import vf_data
import vf_features
from vf_features import feature_names


def load_record(db_name, record_name, channel, annotator):
    record = vf_data.Record()
    record.load(db_name, record_name, channel=channel, annotator=annotator)
    return record


def collect_segments(record, db_name, segment_duration, corrections):
    """Return a list of all Segment objects for a single record (same logic as DataSet.get_samples)."""
    segments = []
    segment_size = int(np.round(segment_duration * record.sampling_rate))
    for rhythm in record.get_artifact_free_rhythms():
        for begin in range(rhythm.begin_time, rhythm.end_time - segment_size, segment_size):
            end = begin + segment_size
            signals = record.signals[begin:end]
            info = vf_data.SegmentInfo(record, rhythm, begin, end)
            correction = corrections.get((info.record_name, begin), None)
            if correction and correction != "C":
                info.rhythm = correction
            segment = vf_data.Segment(info, signals)
            segments.append(segment)
    return segments


def load_corrections(correction_file):
    corrections = {}
    if not correction_file:
        return corrections
    try:
        with open(correction_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    rec, begin_time, label = parts
                    corrections[(rec, int(begin_time))] = label
    except IOError as e:
        print(f"Warning: could not read correction file: {e}")
    return corrections


def print_segment_list(segments, sampling_rate):
    print(f"{'idx':>5}  {'begin_sample':>12}  {'time':>10}  {'duration':>8}  rhythm")
    print("-" * 60)
    for i, seg in enumerate(segments):
        info = seg.info
        t = timedelta(seconds=info.begin_time / sampling_rate)
        dur = info.get_duration()
        print(f"{i:>5}  {info.begin_time:>12}  {str(t):>10}  {dur:>7.1f}s  {info.rhythm}")


def run_extraction(segment, features_to_extract):
    info = segment.info
    signals = np.array(segment.signals, dtype="float64")

    print(f"\nRecord   : {info.record_name}")
    print(f"Begin    : {info.begin_time}  ({timedelta(seconds=info.begin_time / info.sampling_rate)})")
    print(f"End      : {info.end_time}")
    print(f"Duration : {info.get_duration():.2f} s")
    print(f"Rhythm   : {info.rhythm}")
    print(f"Rate     : {info.sampling_rate} Hz")
    print(f"Samples  : {len(signals)}")
    print(f"Features : {sorted(features_to_extract)}")
    print()

    # ------------------------------------------------------------------ BREAKPOINT
    # Set your debugger breakpoint here, then step into vf_features.extract_features()
    # to trace the full preprocessing + feature computation pipeline.
    features, qrs_beats, amplitude = vf_features.extract_features(
        signals, info.sampling_rate, features_to_extract
    )
    # ------------------------------------------------------------------ end

    print(f"QRS beats detected: {len(qrs_beats)}")
    if qrs_beats:
        hr = (len(qrs_beats) / info.get_duration()) * 60
        print(f"Estimated HR      : {hr:.1f} BPM")

    print(f"\n{'Feature':<12}  {'Value':>18}")
    print("-" * 32)
    for name, value in zip(feature_names, features):
        marker = " *" if name in features_to_extract else "  (skipped)"
        print(f"{name:<12}  {float(value):>18.6f}{marker}")

    return features, qrs_beats, amplitude


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from a single ECG segment without parallelism.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-r", "--record", type=str, required=True,
                        help="Record path in db/name format, e.g. mitdb/100 or vfdb/422")
    parser.add_argument("-n", "--segment-index", type=int, default=None,
                        help="0-based index of the segment within the record. "
                             "Omit to list all segments.")
    parser.add_argument("-b", "--begin", type=int, default=None,
                        help="Begin sample number of the segment (alternative to -n). "
                             "The segment whose begin_time matches this value is selected.")
    parser.add_argument("-s", "--segment-duration", type=int, default=8,
                        help="Segment length in seconds (default: 8)")
    parser.add_argument("-c", "--correction-file", type=str, default=None,
                        help="Label correction file (e.g. corrections_s8.txt)")
    parser.add_argument("-f", "--features", type=str, nargs="+",
                        choices=feature_names, default=None,
                        help="Compute only these features. Default: all 27.")
    args = parser.parse_args()

    # Parse db/record
    parts = args.record.split("/", maxsplit=1)
    if len(parts) != 2:
        parser.error("--record must be in db/name format, e.g. mitdb/100")
    db_name, record_name = parts

    # Select channel and annotator by database convention
    if db_name == "mghdb":
        channel = 1       # lead II is channel 1 in mghdb
        annotator = "ari"
    else:
        channel = 0
        annotator = "atr"

    print(f"Loading {db_name}/{record_name}  (channel={channel}, annotator={annotator}) ...")
    record = load_record(db_name, record_name, channel, annotator)
    print(f"  sampling_rate={record.sampling_rate} Hz, "
          f"total_samples={len(record.signals)}, "
          f"duration={record.get_total_time():.1f} s")

    corrections = load_corrections(args.correction_file)
    segments = collect_segments(record, db_name, args.segment_duration, corrections)
    print(f"  {len(segments)} segments of {args.segment_duration} s found\n")

    # Determine which features to compute
    features_to_extract = set(args.features) if args.features else vf_features.feature_names_set

    # Select the target segment
    if args.segment_index is None and args.begin is None:
        # No segment specified — just list them
        print_segment_list(segments, record.sampling_rate)
        print("\nRe-run with -n <index> or -b <begin_sample> to extract a segment.")
        return

    if args.begin is not None:
        # Find by begin sample
        matches = [s for s in segments if s.info.begin_time == args.begin]
        if not matches:
            print(f"No segment with begin_time={args.begin} found.")
            print_segment_list(segments, record.sampling_rate)
            return
        segment = matches[0]
    else:
        # Find by index
        if args.segment_index < 0 or args.segment_index >= len(segments):
            print(f"Index {args.segment_index} out of range (0–{len(segments)-1}).")
            print_segment_list(segments, record.sampling_rate)
            return
        segment = segments[args.segment_index]

    run_extraction(segment, features_to_extract)


if __name__ == "__main__":
    main()
