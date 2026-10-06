import time, gcsfs, ray
import pyarrow.fs as pafs

fs = gcsfs.GCSFileSystem()
arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
files = sorted(fs.glob('yonghui-us-central1-b/ray_data_large_dataset/*.parquet'))
print(f'Original files: {len(files)}')

# 256 files (files * 4) = 10.7 GB
files_256 = files * 4
total_mb = sum(fs.info(f)['size'] for f in files) * 4 / (1024*1024)
print(f'Total data to read: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)')

# Test streaming consumption (iter_batches) vs count
for mode in ['count', 'stream']:
    for conc in [8, 16, 32]:
        t0 = time.perf_counter()
        ds = ray.data.read_parquet(files_256, filesystem=arrow_fs, concurrency=conc)
        if mode == 'count':
            n = ds.count()
        else:
            n = 0
            for batch in ds.iter_batches(batch_size=10000):
                n += len(batch['label'])
        dt = time.perf_counter() - t0
        print(f'Mode: {mode}, Conc {conc}: {dt:.2f} s -> {total_mb/dt:.2f} MiB/s ({total_mb/dt/conc:.2f} MiB/s/worker), rows={n}')
