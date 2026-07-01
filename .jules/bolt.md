## 2024-05-24 - [Optimize path processing functions]
**Learning:** Checking for protocol dynamically with a fast exit based on `:` character check improves performance over standard checks and string starts without breaking standard URL structures. When dealing with `urlsplit`, custom python string parsing outperforms standard lib `urlsplit` inside hot loops.
**Action:** Replace `urlsplit` and `parse_qs` with direct native operations when paths have predictable simple schemas on hot paths.
