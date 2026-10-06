import os
import time
import cProfile
import pstats
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs as pafs
import gcsfs

os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
os.environ['DEFAULT_GCSFS_CONCURRENCY'] = '1'

fs = gcsfs.GCSFileSystem()
arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
bucket_prefix = "yonghui-us-central1-b/ray_data_large_dataset"
files = sorted(fs.glob(f'{bucket_prefix}/*.parquet'))

print(f"Total files: {len(files)}")
sample_file = files[0]
print(f"Profiling file: {sample_file} (size={fs.info(sample_file)['size']} bytes)")

# 1. Profile with cProfile
profiler = cProfile.Profile()

# Configure Arrow thread counts matching Ray Master
pa.set_io_thread_count(128)
pa.set_cpu_count(32)

print("\n--- PROFILING PARQUET READ (1 FILE) ---")
profiler.enable()
t0 = time.perf_counter()

dataset = pds.dataset(sample_file, filesystem=arrow_fs, format="parquet")
scanner = dataset.scanner(
    batch_size=10000,
    batch_readahead=8,
    fragment_scan_options=pds.ParquetFragmentScanOptions(
        use_buffered_stream=True,
        buffer_size=8 * 1024 * 1024
    )
)
batches = list(scanner.to_batches())
total_rows = sum(b.num_rows for b in batches)

t1 = time.perf_counter()
profiler.disable()

file_mb = fs.info(sample_file)['size'] / (1024 * 1024)
duration = t1 - t0
throughput = file_mb / duration
print(f"Read {total_rows} rows in {duration:.3f}s ({throughput:.2f} MiB/s)")

stats = pstats.Stats(profiler).strip_dirs()
print("\n=== TOP 25 BY TOTTIME (CPU in-function) ===")
stats.sort_stats("tottime").print_stats(25)

print("\n=== TOP 25 BY CUMTIME (Cumulative Wall/Call Time) ===")
stats.sort_stats("cumtime").print_stats(25)
