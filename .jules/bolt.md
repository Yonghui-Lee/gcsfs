## 2026-06-23 - Avoiding set() Instantiations in Hot Paths
**Learning:** In functions invoked frequently during path operations (like `_coalesce_generation`), Python's `set()` instantiation adds noticeable overhead.
**Action:** Replace `set()` conversions and mutations with simple iteration and equality checks for small parameter lists in critical code paths to improve execution time.
