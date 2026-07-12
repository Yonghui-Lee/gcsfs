
## 2024-07-12 - GCSFS _split_path Hot Path Optimization
**Learning:** In highly trafficked path parsing functions (like `_split_path` inside `gcsfs`), Python's standard library `urllib.parse.urlsplit` and `parse_qs` can add significant overhead due to object creation and regex matching.
**Action:** Replaced `urlsplit` and `parse_qs` with manual `.split()` and native string operations, boosting performance by ~3x for URLs with queries/fragments, while carefully mirroring the fallback logic used in the standard library.
