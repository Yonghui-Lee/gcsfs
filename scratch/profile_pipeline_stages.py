import os
import io
import time
import statistics
import pyarrow as pa
import pyarrow.parquet as pq
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
file_size_mb = file_size_bytes / (1024 * 1024)

print(f"=== DETAILED PIPELINE STAGE PROFILING ===")
print(f"Sample file: {sample_file} ({file_size_mb:.2f} MiB)")

# Stage 1: Pure Network Download into memory (using gcsfs read)
t0 = time.perf_counter()
with fs.open(sample_file, 'rb') as f:
    raw_data = f.read()
t1 = time.perf_counter()
net_dur = t1 - t0
net_tput = file_size_mb / net_dur
print(f"Stage 1 [Pure Network Download]: {net_dur:.3f} s -> {net_tput:.2f} MiB/s")

# Stage 2: Pure Decompression & Arrow Table conversion in RAM
pa.set_io_thread_count(128)
pa.set_cpu_count(32)

t0 = time.perf_counter()
buf_reader = pa.BufferReader(raw_data)
table = pq.read_table(buf_reader, columns=['tokens', 'label'], use_threads=True)
t1 = time.perf_counter()
decom_dur = t1 - t0
decom_tput = file_size_mb / decom_dur
print(f"Stage 2 [Pure Decompression from RAM (32 CPU threads)]: {decom_dur:.3f} s -> {decom_tput:.2f} MiB/s ({table.num_rows} rows)")

# Stage 2b: Pure Decompression single thread vs 32 threads
pa.set_cpu_count(1)
t0 = time.perf_counter()
buf_reader = pa.BufferReader(raw_data)
table_single = pq.read_table(buf_reader, columns=['tokens', 'label'], use_threads=False)
t1 = time.perf_counter()
decom_dur_1 = t1 - t0
print(f"Stage 2b [Pure Decompression single thread]: {decom_dur_1:.3f} s -> {file_size_mb/decom_dur_1:.2f} MiB/s")
pa.set_cpu_count(32)

# Stage 3: Combined PyArrow Dataset Streaming Scan (Arrow Scanner over gcsfs)
t0 = time.perf_counter()
ds = pds.dataset(sample_file, filesystem=arrow_fs, format="parquet")
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
t1 = time.perf_counter()
stream_dur = t1 - t0
stream_tput = file_size_mb / stream_dur
print(f"Stage 3 [PyArrow Scanner over gcsfs (uncached)]: {stream_dur:.3f} s -> {stream_tput:.2f} MiB/s")

# Stage 4: Compare with 4 and 8 files concurrently in a single process
from concurrent.futures import ThreadPoolExecutor

def read_one_file(fpath):
    t_start = time.perf_counter()
    ds = pds.dataset(fpath, filesystem=arrow_fs, format="parquet")
    scanner = ds.scanner(
        columns=['tokens', 'label'],
        batch_size=10000,
        batch_readahead=8,
        fragment_scan_options=pds.ParquetFragmentScanOptions(
            use_buffered_stream=True,
            buffer_size=8 * 1024 * 1024
        )
    )
    b = list(scanner.to_batches())
    return time.perf_counter() - t_start

for n_threads in [2, 4, 8]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        list(executor.map(read_one_file, files[:n_threads]))
    total_time = time.perf_counter() - t0
    batch_mb = file_size_mb * n_threads
    tput = batch_mb / total_time
    print(f"Stage 4 [Concurrent {n_threads} files in 1 process]: {total_time:.3f} s -> {tput:.2f} MiB/s ({tput/n_threads:.2f} MiB/s/thread)")
