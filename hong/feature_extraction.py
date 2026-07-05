#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_data
import vf_features
import numpy as np
from joblib import Parallel, delayed
import scipy.signal as signal
import pickle
import argparse
import sys
import threading
import os
import glob
from itertools import groupby


def extract_features(idx, segment, resample_rate, update_features, verbose):
    info = segment.info
    segment_duration = info.get_duration()
    signals = segment.signals
    # resample to DEFAULT_SAMPLING_RATE as needed
    if resample_rate and info.sampling_rate != resample_rate:
        signals = signal.resample(signals, int(resample_rate * segment_duration))
        sampling_rate = resample_rate
    else:
        sampling_rate = info.sampling_rate

    if update_features:
        feature_names = set(update_features)
    else:
        feature_names = vf_features.feature_names_set
    features, qrs_beats, amplitude = vf_features.extract_features(signals, sampling_rate, feature_names)

    # store all of the detected beats as part of the info
    info.detected_beats = qrs_beats

    # store amplitude info,too
    info.amplitude = amplitude

    print("{0}: {1}/{2}".format(idx, info.record_name, info.begin_time))
    if verbose:
        print("\t", features, qrs_beats)
    return idx, features, info


# this is a generator function
def load_all_segments(args):
    idx = 0
    dataset = vf_data.DataSet()
    if args.correction_file:
        dataset.load_correction(args.correction_file)
    for segment in dataset.get_samples(args.segment_duration):
        yield idx, segment
        idx += 1


# --- checkpointing helpers ---

def _ckpt_dir(output_file):
    return output_file + ".ckpt"


def _ckpt_path(ckpt_dir, record_name):
    safe = record_name.replace("/", "__")
    return os.path.join(ckpt_dir, safe + ".pkl")


def load_done_records(ckpt_dir):
    """Return set of record_names that already have a checkpoint."""
    if not os.path.isdir(ckpt_dir):
        return set()
    done = set()
    for path in glob.glob(os.path.join(ckpt_dir, "*.pkl")):
        safe = os.path.basename(path)[:-4]  # strip .pkl
        done.add(safe.replace("__", "/"))
    return done


def save_record_checkpoint(ckpt_dir, record_name, results):
    """Atomically write one record's results to a checkpoint file."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = _ckpt_path(ckpt_dir, record_name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp, path)


def load_all_checkpoints(ckpt_dir, done_records):
    """Load and concatenate results from all checkpoint files (preserving order)."""
    all_results = []
    for record_name in sorted(done_records):
        path = _ckpt_path(ckpt_dir, record_name)
        with open(path, "rb") as f:
            all_results.extend(pickle.load(f))
    return all_results


def parallel_extract_features_per_record(args, all_segments_iter):
    """Extract features with per-record checkpointing."""
    ckpt_dir = _ckpt_dir(args.output) if args.output else None
    use_checkpoint = ckpt_dir and not getattr(args, "no_checkpoint", False)
    done_records = load_done_records(ckpt_dir) if use_checkpoint else set()

    if done_records:
        print(f"Resuming: {len(done_records)} record(s) already done, loading from checkpoints.")

    all_results = load_all_checkpoints(ckpt_dir, done_records) if use_checkpoint else []

    parallel = Parallel(n_jobs=args.jobs, verbose=0, backend="multiprocessing", max_nbytes=2048)

    # Group the flat segment stream by record_name (DataSet.get_samples yields all segments
    # of one record before moving to the next, so groupby works without sorting).
    for record_name, group in groupby(all_segments_iter, key=lambda t: t[1].info.record_name):
        if record_name in done_records:
            # Skip — already checkpointed; consume the group to keep idx in sync.
            for _ in group:
                pass
            continue

        group_list = list(group)
        print(f"Processing record {record_name} ({len(group_list)} segments) ...")
        record_results = parallel(
            delayed(extract_features)(idx, seg, args.resample_rate, args.update_features, args.verbose)
            for idx, seg in group_list
        )

        if use_checkpoint:
            save_record_checkpoint(ckpt_dir, record_name, record_results)
            print(f"  Checkpointed {record_name}")

        all_results.extend(record_results)

    return all_results


def output_results(output_file, update_features, results):
    x_data_info = []
    x_data = []
    max_tci = 0
    # sort the results from multiple jobs according to the order they are emitted
    results.sort(key=lambda result: result[0])
    for idx, features, segment_info in results:
        x_data_info.append(segment_info)
        x_data.append(features)
        if features[1] > max_tci:
            max_tci = features[1]

    # try to update existing feature file
    if update_features:
        # load the old file
        try:
            with open(output_file, "rb") as f:
                old_x_data = pickle.load(f)
                old_x_data_info = pickle.load(f)
                assert len(old_x_data) == len(x_data)
                # preserve features in the old file if it's not updated
                preserve_idx = [i for (i, name) in enumerate(vf_features.feature_names) if name not in update_features]
                # copy values we want to preserve from the old data to the new ones
                for new_x, new_info, old_x, old_info in zip(x_data, x_data_info, old_x_data, old_x_data_info):
                    assert (new_info.record_name == old_info.record_name and new_info.begin_time == old_info.begin_time)
                    for i in preserve_idx:
                        new_x[i] = old_x[i]
        except Exception:
            print("Unable to load", output_file, sys.exc_info())

    # write to output files
    with open(output_file, "wb") as f:
        pickle.dump(x_data, f)
        pickle.dump(x_data_info, f)


# distributed computing server
class Server:
    def __init__(self, args):
        self.segment_iter = iter(load_all_segments(args))
        self.n_all_segments = 0
        self.finish_read = False
        self.results = []
        self.verbose = args.verbose
        self.lock = threading.Lock()
        self.args = args

    def get_args(self):
        return self.args

    # called by slaves to get next job to compute
    def next_segment(self):
        idx = -1
        segment = None
        if not self.finish_read:
            self.lock.acquire()  # prevent concurrent calls to the underlying iterator/generator
            try:
                idx, segment = next(self.segment_iter)
                self.n_all_segments += 1
                if self.verbose:
                    info = segment.info
                    print(idx, info.record_name, info.begin_time)
            except StopIteration:  # no more data to compute
                self.finish_read = True  # all data are fetched
            self.lock.release()
        return idx, segment

    def add_results(self, results):  # receive results from slaves
        self.results.extend(results)
        print("got results", self.finish_read, len(self.results), self.n_all_segments)
        if self.finish_read and len(self.results) == self.n_all_segments:
            # all computation are done, stop the server
            self._pyroDaemon.shutdown()


# this is a generator function
def load_all_segments_from_server(master_proxy):
    while True:
        # get next segment to compute from the server
        idx, segment = master_proxy.next_segment()
        if idx != -1 and segment:
            yield idx, segment
        else:  # no more task to do
            break


def parallel_extract_features(args, data_generator):
    parallel = Parallel(n_jobs=args.jobs, verbose=0, backend="multiprocessing", max_nbytes=2048)
    results = parallel(delayed(extract_features)(idx, segment, args.resample_rate, args.update_features, args.verbose) for idx, segment in data_generator)
    return results


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-s", "--segment-duration", type=int, default=8)
    parser.add_argument("-r", "--resample-rate", type=int, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-u", "--update-features", type=str, nargs="+", choices=vf_features.feature_names, default=None)
    parser.add_argument("-c", "--correction-file", type=str, help="Override the incorrect labels of the original dataset.")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable per-record checkpointing (default: enabled when -o is given).")

    # for distributed computing using Pyro4
    parser.add_argument("-l", "--listen-port", type=int, help="Launch a distributed computing server to collect the computed features on the port number.")
    parser.add_argument("-m", "--master-uri", type=str, help="Launch a distributed computing slave pulling jobs from the server and send the results back.")
    args = parser.parse_args()

    # perform feature extraction task in parellel (optionally using master/slave distributed computing)
    if args.master_uri:  # if we're a slave, connect to the master to fetch data to compute
        import Pyro4
        Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
        Pyro4.config.SERIALIZER = "pickle"
        master_proxy = Pyro4.Proxy(args.master_uri)  # connect to the remote server object
        server_args = master_proxy.get_args()
        results = parallel_extract_features(server_args, load_all_segments_from_server(master_proxy))
        master_proxy.add_results(results)  # send the results to the master
    elif args.listen_port:  # if we're launching a master, create the server object
        import Pyro4
        Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
        Pyro4.config.SERIALIZER = "pickle"
        import socket
        host_ip = socket.gethostbyname(socket.gethostname())
        pyro_daemon = Pyro4.Daemon(host=host_ip, port=args.listen_port)
        server = Server(args)
        uri = pyro_daemon.register(server)
        print("Launch server at", uri)
        pyro_daemon.requestLoop()  # blocked until all data are processed
        results = server.results
    else:  # do not use distributed computing, parallel on this machine with multi-core only
        results = parallel_extract_features_per_record(args, load_all_segments(args))

    if not args.master_uri and args.output:  # if we're not a computing slave, save the results to disks
        output_results(args.output, args.update_features, results)


if __name__ == "__main__":
    main()
