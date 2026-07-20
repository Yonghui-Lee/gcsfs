## 2024-05-24 - [Avoid `urllib.parse` overhead in hot loops]
**Learning:** Functions like `urlsplit` and `parse_qs` from `urllib.parse` are expensive in hot path operations because they execute numerous standard-library checks and regex overhead. Custom native string splitting (`find`, `split`) is much faster for simple URLs parsing like path generation fetching in `gcsfs` file operations.
**Action:** Replace `urlsplit` and `parse_qs` with direct native string checking and manual splitting where parsing is constrained, heavily boosting performance.
