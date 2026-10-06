import pstats
stats = pstats.Stats('scratch/deep_dive.prof')
stats.strip_dirs()
print("=== CALLEES OF core.py:1196 (_info) ===")
stats.print_callees('core.py:1196')
