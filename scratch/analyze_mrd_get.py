import pstats
stats = pstats.Stats('scratch/deep_dive.prof')
stats.strip_dirs()
print("=== CALLEES OF zb_hns_utils.py:get (MRDPoolCache.get) ===")
stats.print_callees('zb_hns_utils.py:880')
