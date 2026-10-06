import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs

os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
os.environ['DEFAULT_GCSFS_CONCURRENCY'] = '1'

fs = gcsfs.GCSFileSystem()

src_bucket = "yonghui-us-central1-b/ray_data_large_dataset"
dst_bucket = "yonghui-us-central1-b/ray_data_200mb"

shards = sorted(fs.glob(f"{src_bucket}/*.parquet"))
print(f"Found {len(shards)} source shards.")

# Read first 5 shards into memory once as a template to concatenate into ~200MB table
print("Reading template shards...")
tables = [pq.read_table(fs.open(s)) for s in shards[:5]]
template_table = pa.concat_tables(tables)
print(f"Template table: {template_table.num_rows} rows, memory size: {template_table.nbytes/(1024*1024):.2f} MB")

# Serialize to buffer
buf = pa.BufferOutputStream()
pq.write_table(template_table, buf, compression=None)
file_bytes = buf.getvalue().to_pybytes()
file_size_mb = len(file_bytes) / (1024 * 1024)
print(f"Serialized parquet file size: {file_size_mb:.2f} MB")

num_target_files = 32

def upload_file(idx):
    dst_path = f"{dst_bucket}/shard_{idx:05d}.parquet"
    t0 = time.perf_counter()
    with fs.open(dst_path, "wb") as f:
        f.write(file_bytes)
    dt = time.perf_counter() - t0
    return idx, dt

print(f"Uploading {num_target_files} files of {file_size_mb:.2f} MB to {dst_bucket}...")
t_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(upload_file, range(num_target_files)))
total_time = time.perf_counter() - t_start

total_uploaded_mb = num_target_files * file_size_mb
print(f"Successfully uploaded {num_target_files} files ({total_uploaded_mb:.2f} MB / {total_uploaded_mb/1024:.2f} GB) in {total_time:.2f} s ({total_uploaded_mb/total_time:.2f} MB/s)")
