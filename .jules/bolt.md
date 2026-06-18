## 2024-06-09 - Cache processing short circuit overhead

**Learning:** When building tree/cache representations of deep directories iteratively by calling objects `.split()` and `.parent()`, using simple short circuits (like breaking if the dir is already computed) and inlining string operations drastically reduces cpu overhead.
**Action:** Always consider using fast local string manipulation inside heavy loops instead of calling general-purpose parsing functions.
