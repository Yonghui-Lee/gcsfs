import asyncio
import os
import gc

os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
from gcsfs.prefetcher import BackgroundPrefetcher

async def dummy_fetch(start, length, **kwargs):
    return b'0' * length

async def run():
    prefetcher = BackgroundPrefetcher(dummy_fetch, 1024*1024, max_prefetch_size=64*1024, concurrency=1)
    data = await prefetcher.afetch(0, 1024)
    print(f"Read {len(data)} bytes")
    # Deliberately drop prefetcher without close
    del prefetcher
    gc.collect()

asyncio.run(run())
print("Test completed.")
