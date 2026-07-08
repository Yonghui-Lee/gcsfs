## 2024-07-08 - Fast path identity checks
**Learning:** Functions invoked for every path operation (like `_coalesce_generation` in `gcsfs/core.py`) are extremely sensitive to object creation overhead. Avoid `set()` instantiations and mutations (`.remove()`) for small parameter lists in hot paths; instead use simple loops with identity checks for better performance.
**Action:** When replacing sets used in error messages, construct the set only within the error handling block to preserve original string formatting without performance penalties.
