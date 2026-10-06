import os, time, gcsfs, pyarrow.fs as pafs

os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
os.environ['DEFAULT_GCSFS_CONCURRENCY'] = '1'

fs = gcsfs.GCSFileSystem()
arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
files = sorted(fs.glob('yonghui-us-central1-b/ray_data_large_dataset/*.parquet'))[:16]

# Test 1: Baseline open_input_file
timings_base = []
for f in files:
    t0 = time.perf_counter()
    arrow_f = arrow_fs.open_input_file(f)
    timings_base.append(time.perf_counter() - t0)
    arrow_f.close()

# Now simulate _object_cache on fs
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

timings_cached = []
for f in files:
    t0 = time.perf_counter()
    arrow_f = arrow_fs.open_input_file(f)
    timings_cached.append(time.perf_counter() - t0)
    arrow_f.close()

print(f"Total time to open 16 files baseline: {sum(timings_base)*1000:.2f} ms (avg {sum(timings_base)/16*1000:.2f} ms/file)")
print(f"Total time to open 16 files cached:   {sum(timings_cached)*1000:.2f} ms (avg {sum(timings_cached)/16*1000:.2f} ms/file)")
print(f"Speedup: {sum(timings_base)/sum(timings_cached):.2f}x faster file opening!")
