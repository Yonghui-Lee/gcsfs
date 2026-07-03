## 2024-07-03 - Avoid object creation in Python hot paths
**Learning:** Functions invoked frequently on path operations are sensitive to object creation overhead. Constructing a `set()` natively costs significant execution time. We can achieve up to a 2.4x speedup by replacing `set()` checks in `_coalesce_generation` with an optimized loop without affecting logic or error propagation.
**Action:** Always scan hot path methods for temporary collections (like sets or lists) that are instantiated unnecessarily for quick lookups and logic conditions, and replace them with loops.
