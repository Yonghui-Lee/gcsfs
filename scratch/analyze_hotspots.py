import pstats

stats = pstats.Stats('scratch/deep_dive.prof')
stats.strip_dirs()

print("=== CALLERS OF asyn.py:sync ===")
stats.print_callers('asyn.py:66')

print("\n=== CALLERS OF spec.py:open ===")
stats.print_callers('spec.py:1359')

print("\n=== CALLERS OF zonal_file.py:__init__ ===")
stats.print_callers('zonal_file.py:30')

print("\n=== CALLEES OF zonal_file.py:__init__ ===")
stats.print_callees('zonal_file.py:30')

print("\n=== CALLEES OF prefetcher.py:fetch ===")
stats.print_callees('prefetcher.py:866')
