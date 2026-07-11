## 2024-06-12 - Object Creation Overhead in Path Processing
**Learning:** Functions in hot paths (like `_coalesce_generation`) are sensitive to object creation and mutation overhead. Creating and mutating small sets for parameters can be up to 50% slower than using explicit identity checks in a loop.
**Action:** Always consider replacing collection initializations (like `set()`) with simple loops and state variables in frequently executed utility functions. Keep any required structure construction (for exact error messages) localized strictly within the error handling branches.
