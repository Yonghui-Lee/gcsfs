import os, sys, time, argparse
import pyarrow.fs as pafs
import ray

parser = argparse.ArgumentParser()
parser.add_argument("--ray-concurrency", type=int, default=8)
args = parser.parse_args()

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

gcs = pafs.GcsFileSystem()
bucket = 'hf-pile-deduplicated-us-central1-b-gcsfs'
file_infos = gcs.get_file_info(pafs.FileSelector(bucket, recursive=False))
parquet_files = [f.path for f in file_infos if f.path.endswith('.parquet')][:16]
total_mb = sum(gcs.get_file_info(f).size for f in parquet_files) / (1024 * 1024)

# Warmup run on 2 files
warmup_files = parquet_files[:2]
ds_warmup = ray.data.read_parquet(warmup_files, filesystem=gcs, concurrency=2, columns=['text'])
_ = ds_warmup.count()

print(f"=== Benchmarking C++ SDK | RAY_CONC={args.ray_concurrency} ({total_mb:.2f} MiB / 16 files) ===")

runs = []
for i in range(3):
    t0 = time.perf_counter()
    ds = ray.data.read_parquet(parquet_files, filesystem=gcs, concurrency=args.ray_concurrency, columns=['text'])
    cnt = ds.count()
    dur = time.perf_counter() - t0
    tput = total_mb / dur
    runs.append(tput)
    print(f"  Run {i+1}: {dur:.3f} s -> {tput:.2f} MiB/s ({tput/1024:.2f} GiB/s)")

avg_tput = sum(runs) / len(runs)
print(f"==> Steady-state Avg Throughput: {avg_tput:.2f} MiB/s ({avg_tput/1024:.2f} GiB/s) | Per-worker: {avg_tput/args.ray_concurrency:.2f} MiB/s\n")
