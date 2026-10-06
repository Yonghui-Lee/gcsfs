import pstats
stats = pstats.Stats('scratch/deep_dive.prof')
stats.strip_dirs()
print("=== CALLEES OF prefetcher.py:_async_fetch ===")
stats.print_callees('prefetcher.py:756')
