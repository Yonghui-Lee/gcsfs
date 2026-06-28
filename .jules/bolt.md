## 2024-06-20 - Hot Path Micro-Optimizations in gcsfs

**Learning:** Python's standard library `urllib.parse.urlsplit` and `parse_qs` introduce significant parsing overhead, making them unsuitable for critical hot paths like `_split_path` that get executed for every file operation. Similarly, creating throwaway instances like `set()` in deeply nested loops or widely used helpers like `_coalesce_generation` causes measurable overhead due to object allocation and garbage collection.

**Action:** In high-throughput parsing logic, replace heavy standard library functions with native string operations (`find`, `split`, slicing) when the expected input format is strictly controlled or predictable. Avoid unnecessary object allocations (like `set()`) for small argument lists in hot paths by replacing them with simple iteration.
