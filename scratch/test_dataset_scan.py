import os, time, gcsfs, pyarrow.fs as pafs, pyarrow.dataset as pds

os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
os.environ['DEFAULT_GCSFS_CONCURRENCY'] = '1'

fs = gcsfs.GCSFileSystem()
arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
files = sorted(fs.glob('yonghui-us-central1-b/ray_data_large_dataset/*.parquet'))[:8]

# 1. Baseline
durations_base = []
for _ in range(3):
    fs.dircache.clear()
    t0 = time.perf_counter()
    for f in files:
        ds = pds.dataset(f, filesystem=arrow_fs, format='parquet')
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
    durations_base.append(time.perf_counter() - t0)

# 2. With object cache
orig_info = fs._info
obj_cache = {}
async def cached_info(path, generation=None, **kwargs):
    key = (path, generation)
    if key in obj_cache:
        return obj_cache[key]
    res = await orig_info(path, generation=generation, **kwargs)
    obj_cache[key] = res
    return res

fs._info = cached_info

durations_cached = []
for _ in range(3):
    obj_cache.clear() # clear between rounds so each round measures cold first hit + warm second hit per file
    t0 = time.perf_counter()
    for f in files:
        ds = pds.dataset(f, filesystem=arrow_fs, format='parquet')
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
    durations_cached.append(time.perf_counter() - t0)

min_base = min(durations_base)
min_cached = min(durations_cached)
total_mb = sum(fs.info(f)['size'] for f in files) / (1024*1024)

print(f"Baseline scan (8 shards): {min_base:.4f} s -> {total_mb/min_base:.2f} MiB/s")
print(f"Cached scan (8 shards):   {min_cached:.4f} s -> {total_mb/min_cached:.2f} MiB/s")
print(f"Throughput improvement:   +{(total_mb/min_cached - total_mb/min_base):.2f} MiB/s ({(min_base/min_cached - 1)*100:.1f}% speedup)")
