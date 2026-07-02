## 2024-05-24 - Short-circuiting path normalization
**Learning:** Adding short-circuit fast paths to string parsing functions (like `_strip_protocol`) requires extreme care. Returning early `if ":" not in path` successfully skips protocol matching overhead, but it inadvertently skipped the subsequent native `path.lstrip("/")` step, breaking path normalization for root paths and buckets.
**Action:** When adding early returns to existing functions, ensure *all* subsequent data mutations that would have occurred on the regular path are still applied to the short-circuited return value.
