import os
import sys
import time
import json
import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--bucket-type", choices=["zonal", "regional"], required=True)
parser.add_argument("--backend", choices=["cpp", "gcsfs", "gcsfs_no_prefetch", "gcsfs_no_cache"], required=True)
parser.add_argument("--dataset", choices=["40mb", "200mb", "pile"], default="40mb")
parser.add_argument("--num-files", type=int, default=None)
parser.add_argument("--ray-concurrency", type=int, default=8)
parser.add_argument("--batch-format", choices=["numpy", "pyarrow"], default="numpy")
parser.add_argument("--warmup-runs", type=int, default=1)
parser.add_argument("--measured-runs", type=int, default=3)
args = parser.parse_args()

# Setup environment variables before imports
if args.backend.startswith("gcsfs"):
    os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
    if args.backend in ("gcsfs_no_prefetch", "gcsfs_no_cache"):
        os.environ['USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING'] = 'false'
    if args.backend == "gcsfs_no_cache":
        os.environ['GCSFS_DEFAULT_CACHE_TYPE'] = 'none'

import ray
import pyarrow.fs as pafs
import gcsfs

runtime_env = {
    "env_vars": {
        "RAY_DATA_PARQUET_READER_IO_THREAD_COUNT": "128",
        "RAY_DATA_PARQUET_READER_CPU_COUNT": "32",
        "RAY_DATA_PARQUET_FRAGMENT_BUFFER_SIZE": str(8 * 1024 * 1024),
        "RAY_DATA_READ_FILES_NUM_THREADS": "4",
    }
}
if args.backend.startswith("gcsfs"):
    runtime_env["env_vars"]["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "true"
    if args.backend in ("gcsfs_no_prefetch", "gcsfs_no_cache"):
        runtime_env["env_vars"]["USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING"] = "false"
    if args.backend == "gcsfs_no_cache":
        runtime_env["env_vars"]["GCSFS_DEFAULT_CACHE_TYPE"] = "none"

ray.init(runtime_env=runtime_env, ignore_reinit_error=True)
ctx = ray.data.DataContext.get_current()
ctx.execution_options.preserve_order = False
ctx.target_max_block_size = 64 * 1024 * 1024
ctx.override_object_store_memory_limit_fraction = 0.95
from ray.data._internal.execution.backpressure_policy import ConcurrencyCapBackpressurePolicy
ctx.set_config("backpressure_policies.enabled", [ConcurrencyCapBackpressurePolicy])

if args.dataset == "pile":
    bucket_prefix = "hf-pile-deduplicated-us-central1-b-gcsfs" if args.bucket_type == "zonal" else "hf-pile-deduplicated-us-central1-b-gcsfs-standard"
elif args.dataset == "200mb":
    bucket_prefix = f"yonghui-us-central1-b/ray_data_200mb" if args.bucket_type == "zonal" else f"yonghui-gcsfs-regional-us/ray_data_200mb"
else:
    bucket_prefix = f"yonghui-us-central1-b/ray_data_large_dataset" if args.bucket_type == "zonal" else f"yonghui-gcsfs-regional-us/ray_data_large_dataset"

fs_gcs = gcsfs.GCSFileSystem()
all_files = sorted(fs_gcs.glob(f'{bucket_prefix}/*.parquet'))
if args.num_files is not None:
    all_files = all_files[:args.num_files]
total_bytes = sum(fs_gcs.info(f)['size'] for f in all_files)
total_mb = total_bytes / (1024 * 1024)

target_paths = all_files
if args.backend == "cpp":
    fs_arg = pafs.GcsFileSystem()
else:
    fs_arg = pafs.PyFileSystem(pafs.FSSpecHandler(gcsfs.GCSFileSystem()))

def run_workload(paths, fs):
    # Canonical streaming data-loading pipeline (as used in ML training and ETL)
    ds = ray.data.read_parquet(paths, filesystem=fs, concurrency=args.ray_concurrency)
    batch_format_arg = None if args.batch_format == "numpy" else "pyarrow"
    for batch in ds.iter_batches(batch_size=None, prefetch_batches=2, batch_format=batch_format_arg):
        pass

# 1. Warmup
warmup_paths = target_paths[:args.ray_concurrency]
for _ in range(args.warmup_runs):
    run_workload(warmup_paths, fs_arg)

# 2. Measured runs
timings = []
throughputs = []
for _ in range(args.measured_runs):
    t0 = time.perf_counter()
    run_workload(target_paths, fs_arg)
    dur = time.perf_counter() - t0
    tput = total_mb / dur
    timings.append(dur)
    throughputs.append(tput)

ray.shutdown()

results = {
    "bucket_type": args.bucket_type,
    "backend": args.backend,
    "batch_format": args.batch_format,
    "mode": "streaming",
    "ray_concurrency": args.ray_concurrency,
    "num_files": len(all_files),
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
