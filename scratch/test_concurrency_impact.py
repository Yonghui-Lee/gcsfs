import os, sys, time
import pyarrow.parquet as pq

# Test different concurrencies on Zonal read
results = {}
for conc in [1, 2, 4, 8]:
    os.environ['DEFAULT_GCSFS_CONCURRENCY'] = str(conc)
    # Reload gcsfs modules to pick up new DEFAULT_CONCURRENCY
    import importlib
    import gcsfs, gcsfs.zb_hns_utils, gcsfs.core, gcsfs.zonal_file, gcsfs.prefetcher
    importlib.reload(gcsfs.zb_hns_utils)
    importlib.reload(gcsfs.core)
    importlib.reload(gcsfs.zonal_file)
    importlib.reload(gcsfs.prefetcher)
    importlib.reload(gcsfs)

    fs = gcsfs.GCSFileSystem()
    files = sorted(fs.glob("yonghui-us-central1-b/ray_data_large_dataset/*.parquet"))[:8]
    total_bytes = sum(fs.info(f)["size"] for f in files)
    mb = total_bytes / (1024 * 1024)

    # Warmup
    with fs.open(files[0], "rb") as fp:
        _ = pq.read_table(fp)

    t0 = time.perf_counter()
    tables = []
    for f in files:
        with fs.open(f, "rb", concurrency=conc) as fp:
            tables.append(pq.read_table(fp))
    dt = time.perf_counter() - t0
    tput = mb / dt
    print(f"Concurrency = {conc}: {dt:.3f} s -> {tput:.2f} MB/s (8 files, {mb:.1f} MB)")
    results[conc] = (dt, tput)

print("\nSummary:", results)
