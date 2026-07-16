## 2024-05-24 - Optimize hot path string parsing
**Learning:** Functions like `urllib.parse.urlsplit` and `parse_qs` introduce significant overhead when called frequently (e.g., in critical paths like `_split_path` for parsing object generations in GCS paths).
**Action:** Replace standard library parsing methods with native string slicing (`find`, `split`) in frequently executed path-processing methods to improve performance while carefully preserving the exact original logic and precedence.
