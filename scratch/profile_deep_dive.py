import os
import time
import cProfile
import pstats
import io
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
sample_file = files[0]
file_size_bytes = fs.info(sample_file)['size']

pa.set_io_thread_count(128)
pa.set_cpu_count(32)

print(f"Profiling file: {sample_file} ({file_size_bytes / (1024*1024):.2f} MiB)")

profiler = cProfile.Profile()
profiler.enable()

# Read 4 files sequentially with PyArrow Scanner
for f in files[:4]:
    ds = pds.dataset(f, filesystem=arrow_fs, format="parquet")
    scanner = ds.scanner(
        columns=['tokens', 'label'],
        batch_size=10000,
        batch_readahead=8,
        fragment_scan_options=pds.ParquetFragmentScanOptions(
            use_buffered_stream=True,
            buffer_size=8 * 1024 * 1024
        )
    )
    batches = list(scanner.to_batches())

profiler.disable()

stats = pstats.Stats(profiler).strip_dirs()
print("\n=== TOP 30 BY TOTTIME (CPU in-function) ===")
stats.sort_stats("tottime").print_stats(30)

print("\n=== TOP 30 BY CUMTIME (Cumulative Time) ===")
stats.sort_stats("cumtime").print_stats(30)

print("\n=== TOP 30 BY NCALLS (Call Count) ===")
stats.sort_stats("ncalls").print_stats(30)

# Save pstats file for deep analysis
stats.dump_stats("scratch/deep_dive.prof")
print("Dumped stats to scratch/deep_dive.prof")
