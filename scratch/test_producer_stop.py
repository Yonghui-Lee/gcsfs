import asyncio
import os
import gc

os.environ['GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT'] = 'true'
from gcsfs.prefetcher import PrefetchProducer, BackgroundPrefetcher

async def run():
    class DummyOrch:
        def set_error(self, e): pass
    orch = DummyOrch()
    producer = PrefetchProducer(orch, None, 1000, 1)
    producer.start()
    await asyncio.sleep(0.01)
    await producer.stop()
    del producer
    gc.collect()

asyncio.run(run())
print("Test completed.")
