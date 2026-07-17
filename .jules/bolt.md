
## 2024-05-18 - Avoid Set Instantiation in Hot Paths
**Learning:** Functions invoked for every path operation (like `_coalesce_generation`) are sensitive to object creation overhead. `set()` instantiation and operations like `.remove()` add measurable latency compared to simple iterative checks.
**Action:** Use simple `for` loops and identity checks (`is not None`) instead of sets for deduplication/validation on small parameter lists in hot paths. When reconstructing sets for error messaging compatibility, place the set construction strictly inside the exception-handling block.
