
## 2024-05-23 - Avoid urllib.parse in hot paths
**Learning:** `urllib.parse.urlsplit` and `parse_qs` are surprisingly slow when called repeatedly on hot paths (like path splitting and protocol stripping in gcsfs). Replacing them with native python string `.find()` and `.split()` methods for targeted use-cases reduced `_split_path` time by ~30-40% while preserving exact edge-case behaviour.
**Action:** Always verify if complex standard library string manipulation utilities (`urllib.parse`, `re`) can be replaced with simpler native string methods (`in`, `find`, `split`) in functions that execute for every item or operation (like routing and path resolving).
