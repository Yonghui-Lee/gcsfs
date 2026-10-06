import os, sys, time
os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'

import ray
from ray.data._internal.execution.backpressure_policy import ConcurrencyCapBackpressurePolicy

print("=== Testing Ray configuration to avoid memory backpressure on 256 files (10.7 GB) ===")

runtime_env = {
    "env_vars": {
        "RAY_DATA_PARQUET_READER_IO_THREAD_COUNT": "128",
        "RAY_DATA_PARQUET_READER_CPU_COUNT": "32",
        "RAY_DATA_PARQUET_FRAGMENT_BUFFER_SIZE": str(8 * 1024 * 1024),
        "RAY_DATA_READ_FILES_NUM_THREADS": "4",
        "GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT": "true",
    }
}

ray.init(
    runtime_env=runtime_env,
    ignore_reinit_error=True
)

ctx = ray.data.DataContext.get_current()
ctx.execution_options.preserve_order = False
ctx.target_max_block_size = 64 * 1024 * 1024

# Set backpressure policies to avoid ResourceBudget throttling
ctx.set_config("backpressure_policies.enabled", [ConcurrencyCapBackpressurePolicy])
ctx.override_object_store_memory_limit_fraction = 0.95

import gcsfs, pyarrow.fs as pafs
fs_gcs = gcsfs.GCSFileSystem()
files = sorted(fs_gcs.glob('yonghui-us-central1-b/ray_data_large_dataset/*.parquet'))
total_bytes = sum(fs_gcs.info(f)['size'] for f in files)
total_mb = total_bytes / (1024 * 1024)
print(f"Dataset has {len(files)} files, {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")

fs_arg = pafs.PyFileSystem(pafs.FSSpecHandler(gcsfs.GCSFileSystem()))

print("\n--- Running materialize() with backpressure policy disabled on 32 workers ---")
t0 = time.perf_counter()
ds = ray.data.read_parquet(files, filesystem=fs_arg, concurrency=32)
res = ds.materialize()
dur = time.perf_counter() - t0
tput = total_mb / dur
print(f"SUCCESS: Materialized {len(files)} files ({total_mb:.1f} MB) in {dur:.2f} s -> {tput:.2f} MiB/s ({tput/1024:.2f} GiB/s)")

ray.shutdown()
