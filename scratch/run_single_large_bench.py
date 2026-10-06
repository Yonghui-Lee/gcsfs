#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--backend", required=True, choices=["gcsfs_no_cache", "gcsfs_prefetch_on", "gcsfs_readahead", "pyarrow_cpp"])
parser.add_argument("--bucket-type", choices=["zonal", "regional"], default="zonal")
parser.add_argument("--concurrency", type=int, default=8)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()

runtime_env = {"env_vars": {}}
if args.backend == "gcsfs_no_cache":
    os.environ["GCSFS_DEFAULT_CACHE_TYPE"] = "none"
    os.environ["USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING"] = "false"
    runtime_env["env_vars"]["GCSFS_DEFAULT_CACHE_TYPE"] = "none"
    runtime_env["env_vars"]["USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING"] = "false"
elif args.backend == "gcsfs_prefetch_on":
    os.environ["GCSFS_DEFAULT_CACHE_TYPE"] = "readahead"
    os.environ["USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING"] = "true"
    runtime_env["env_vars"]["GCSFS_DEFAULT_CACHE_TYPE"] = "readahead"
    runtime_env["env_vars"]["USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING"] = "true"
elif args.backend == "gcsfs_readahead":
    os.environ["GCSFS_DEFAULT_CACHE_TYPE"] = "readahead"
    os.environ["USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING"] = "false"
    runtime_env["env_vars"]["GCSFS_DEFAULT_CACHE_TYPE"] = "readahead"
    runtime_env["env_vars"]["USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING"] = "false"

runtime_env["env_vars"]["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "true"

import ray
import pyarrow.fs as pafs
import gcsfs

fs_gcs = gcsfs.GCSFileSystem()
bucket_name = "hf-pile-deduplicated-us-central1-b-gcsfs" if args.bucket_type == "zonal" else "hf-pile-deduplicated-us-central1-b-gcsfs-standard"
files = sorted(fs_gcs.glob(f"{bucket_name}/bench_large_files/file_1gb_*.parquet"))
assert len(files) == 8, f"Expected 8 files, found {len(files)}: {files}"

sizes = [fs_gcs.info(f)["size"] for f in files]
total_bytes = sum(sizes)
total_mib = total_bytes / (1024 * 1024)

# Create filesystem object
if args.backend == "pyarrow_cpp":
    fs_arg = pafs.GcsFileSystem()
    target_paths = [f"{bucket_name}/bench_large_files/{os.path.basename(f)}" for f in files]
else:
    fs_arg = pafs.PyFileSystem(pafs.FSSpecHandler(gcsfs.GCSFileSystem()))
    target_paths = files

ray.init(
    runtime_env=runtime_env,
    ignore_reinit_error=True,
    logging_level="warning"
)
ctx = ray.data.DataContext.get_current()
ctx.execution_options.preserve_order = False
ctx.target_max_block_size = 64 * 1024 * 1024
ctx.override_object_store_memory_limit_fraction = 0.95
from ray.data._internal.execution.backpressure_policy import ConcurrencyCapBackpressurePolicy
ctx.set_config("backpressure_policies.enabled", [ConcurrencyCapBackpressurePolicy])

def run_workload(paths, fs):
    ds = ray.data.read_parquet(paths, filesystem=fs, concurrency=args.concurrency)
    for batch in ds.iter_batches(batch_size=None, prefetch_batches=2, batch_format="pyarrow"):
        pass

# Warmup
for _ in range(args.warmup):
    run_workload(target_paths, fs_arg)

# Measured runs
timings = []
throughputs = []
for _ in range(args.runs):
    t0 = time.perf_counter()
    run_workload(target_paths, fs_arg)
    dur = time.perf_counter() - t0
    tput = total_mib / dur
    timings.append(dur)
    throughputs.append(tput)

ray.shutdown()

results = {
    "backend": args.backend,
    "bucket_type": args.bucket_type,
    "concurrency": args.concurrency,
    "num_files": len(files),
    "total_mib": total_mib,
    "runs": [
        {"duration_s": dur, "throughput_mib_s": tp}
        for dur, tp in zip(timings, throughputs)
    ],
    "mean_throughput_mib_s": statistics.mean(throughputs),
    "median_throughput_mib_s": statistics.median(throughputs),
    "min_throughput_mib_s": min(throughputs),
    "max_throughput_mib_s": max(throughputs),
    "stdev_throughput_mib_s": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
}

print(f"RESULT_JSON: {json.dumps(results)}")
