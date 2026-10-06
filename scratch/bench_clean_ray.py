import os
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gcsfs-concurrency", type=int, default=4)
parser.add_argument("--ray-concurrency", type=int, default=8)
args = parser.parse_args()

os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
os.environ['DEFAULT_GCSFS_CONCURRENCY'] = str(args.gcsfs_concurrency)

import ray
import gcsfs

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

fs = gcsfs.GCSFileSystem()
bucket = 'hf-pile-deduplicated-us-central1-b-gcsfs'
files = fs.glob(f'{bucket}/*.parquet')[:16]
paths = [f'gs://{f}' for f in files]
total_mb = sum(fs.info(f)['size'] for f in files) / (1024 * 1024)

# Warmup run on 2 files
warmup_paths = paths[:2]
ds_warmup = ray.data.read_parquet(warmup_paths, concurrency=2, columns=['text'])
_ = ds_warmup.count()

print(f"=== Benchmarking GCSFS_CONC={args.gcsfs_concurrency} | RAY_CONC={args.ray_concurrency} ({total_mb:.2f} MiB / 16 files) ===")

runs = []
for i in range(3):
    t0 = time.perf_counter()
    ds = ray.data.read_parquet(paths, concurrency=args.ray_concurrency, columns=['text'])
    cnt = ds.count()
    dur = time.perf_counter() - t0
    tput = total_mb / dur
    runs.append(tput)
    print(f"  Run {i+1}: {dur:.3f} s -> {tput:.2f} MiB/s ({tput/1024:.2f} GiB/s)")

avg_tput = sum(runs) / len(runs)
print(f"==> Steady-state Avg Throughput: {avg_tput:.2f} MiB/s ({avg_tput/1024:.2f} GiB/s) | Per-worker: {avg_tput/args.ray_concurrency:.2f} MiB/s\n")
