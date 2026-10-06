import os, sys, time, cProfile, pstats
import pyarrow.parquet as pq
import gcsfs

os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
os.environ['DEFAULT_GCSFS_CONCURRENCY'] = '1'

bucket_type = sys.argv[1] if len(sys.argv) > 1 else "zonal"
if bucket_type == "zonal":
    prefix = "yonghui-us-central1-b/ray_data_large_dataset"
else:
    prefix = "yonghui-gcsfs-regional-us/ray_data_large_dataset"

fs = gcsfs.GCSFileSystem()
files = sorted(fs.glob(f"{prefix}/*.parquet"))[:16]
total_bytes = sum(fs.info(f)["size"] for f in files)
print(f"Profiling {bucket_type.upper()} read of {len(files)} files ({total_bytes/(1024*1024):.2f} MB)...")

def read_all():
    tables = []
    for f in files:
        with fs.open(f, "rb") as fp:
            tables.append(pq.read_table(fp))
    return len(tables)

# Warmup 1 file
with fs.open(files[0], "rb") as fp:
    _ = pq.read_table(fp)

profiler = cProfile.Profile()
t0 = time.perf_counter()
profiler.enable()
n = read_all()
profiler.disable()
dt = time.perf_counter() - t0

mb = total_bytes / (1024 * 1024)
print(f"Completed {n} files in {dt:.3f} s -> {mb/dt:.2f} MB/s")

stats = pstats.Stats(profiler).sort_stats("cumulative")
print("\n--- TOP 25 BY CUMULATIVE TIME ---")
stats.print_stats(25)

stats_self = pstats.Stats(profiler).sort_stats("time")
print("\n--- TOP 25 BY SELF (INTERNAL) TIME ---")
stats_self.print_stats(25)
