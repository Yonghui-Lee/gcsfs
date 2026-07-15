## 2026-07-15 - Fast Path Parsing

**Learning:** URL parsing in hot paths can become a noticeable bottleneck due to standard library overhead (`urllib.parse`). Even `urlsplit` is surprisingly slow in tight loops.

**Action:** For string operations called repeatedly, use basic string methods (`find`, `split`) instead of fully-featured libraries when precision requirements permit.
