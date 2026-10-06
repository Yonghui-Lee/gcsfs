import os
import sys
import time
import json
import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["cpp", "gcsfs_c4", "gcsfs_c1"], required=True)
parser.add_argument("--mode", choices=["count", "materialize"], default="count")
parser.add_argument("--ray-concurrency", type=int, default=8)
parser.add_argument("--num-files", type=int, default=32)
parser.add_argument("--warmup-runs", type=int, default=2)
parser.add_argument("--measured-runs", type=int, default=5)
args = parser.parse_args()

# Setup environment variables before imports
if args.backend.startswith("gcsfs"):
    os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
    conc = "4" if args.backend == "gcsfs_c4" else "1"
    os.environ['DEFAULT_GCSFS_CONCURRENCY'] = conc

import ray
import pyarrow.fs as pafs
import gcsfs

runtime_env = {}
if args.backend.startswith("gcsfs"):
    conc = "4" if args.backend == "gcsfs_c4" else "1"
    runtime_env = {
        "env_vars": {
            "GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT": "true",
            "DEFAULT_GCSFS_CONCURRENCY": conc,
        }
    }

ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

bucket = 'hf-pile-deduplicated-us-central1-b-gcsfs'
fs_gcs = gcsfs.GCSFileSystem()
all_files = sorted(fs_gcs.glob(f'{bucket}/*.parquet'))
selected_files = all_files[:args.num_files]
total_bytes = sum(fs_gcs.info(f)['size'] for f in selected_files)
total_mb = total_bytes / (1024 * 1024)

if args.backend == "cpp":
    cpp_fs = pafs.GcsFileSystem()
    target_paths = selected_files  # PyArrow GcsFileSystem takes bucket/path
    fs_arg = cpp_fs
else:
    target_paths = [f"gs://{f}" for f in selected_files]
    fs_arg = None  # Ray Data handles gs:// via fsspec / gcsfs

def run_workload(paths, fs):
    if fs is not None:
        ds = ray.data.read_parquet(paths, filesystem=fs, concurrency=args.ray_concurrency)
    else:
        ds = ray.data.read_parquet(paths, concurrency=args.ray_concurrency)
    
    if args.mode == "count":
        res = ds.count()
    elif args.mode == "materialize":
        res = ds.materialize()
        _ = res.count()
    return res

# 1. Warmup runs (discarded)
warmup_paths = target_paths[:args.ray_concurrency]
for w in range(args.warmup_runs):
    _ = run_workload(warmup_paths, fs_arg)

# 2. Measured runs
timings = []
throughputs = []
for m in range(args.measured_runs):
    t0 = time.perf_counter()
    _ = run_workload(target_paths, fs_arg)
    dur = time.perf_counter() - t0
    tput = total_mb / dur
    timings.append(dur)
    throughputs.append(tput)

ray.shutdown()

results = {
    "backend": args.backend,
    "mode": args.mode,
    "ray_concurrency": args.ray_concurrency,
    "num_files": args.num_files,
    "total_mb": total_mb,
    "runs": [
        {"duration_s": dur, "throughput_mib_s": tp}
        for dur, tp in zip(timings, throughputs)
    ],
    "mean_throughput_mib_s": statistics.mean(throughputs),
    "median_throughput_mib_s": statistics.median(throughputs),
    "min_throughput_mib_s": min(throughputs),
    "max_throughput_mib_s": max(throughputs),
    "stdev_throughput_mib_s": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
    "mean_duration_s": statistics.mean(timings),
    "per_worker_mib_s": statistics.mean(throughputs) / args.ray_concurrency
}

print("RESULT_JSON:" + json.dumps(results))
