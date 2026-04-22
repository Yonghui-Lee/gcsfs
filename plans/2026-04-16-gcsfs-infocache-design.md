# Low-Level Design (LLD): Metadata InfoCache

## 1. Objective and Motivation
The objective is to introduce a dedicated single-object metadata cache (`InfoCache`) for repeated metadata lookups such as `info()`, `exists()`, `modified()`, and `size()`.

**The problem**

Today, `gcsfs` gets most of its metadata reuse from `dircache`, which is populated by listing operations. If a caller already listed a directory, later `info()` calls for entries in that directory can often be answered from the cached listing. But direct point-lookups on known object paths still go to the object metadata API path, which is expensive in AI/ML workloads that probe the same files repeatedly.

Those workloads are common in `DataLoader`s, dataset manifests, sharded training jobs, and distributed schedulers. They often issue many repeated single-path metadata calls without ever listing the parent directory first.

**The solution**

Introduce a dedicated `InfoCache` for successful single-object metadata fetches. The cache is:

- opt-in by default, so existing freshness semantics do not silently change
- bounded by entry count and pruned consistently across all bookkeeping structures
- keyed by structured, backend-defined keys rather than path-concatenated strings
- invalidated through indexed path-aware helpers rather than full-cache scans

The cache is populated by successful single-object metadata resolutions, including the case where `_info()` resolves a point-lookup from a parent listing already in `dircache` (one entry promoted, not bulk import). It is not populated by bulk traversal in `ls()` or `find()` themselves — the cost-control reason is that a single `ls()` of a large directory could otherwise inflate the cache by tens of thousands of entries from a single call.

---

## 2. Key Design Decisions and Trade-offs

### 2.1 Code Location: `gcsfs` vs. `fsspec`
*Decision:* **Implement in `fsspec` first [Selected]**

The cache should live in `fsspec`, with backend-specific hooks for key derivation and invalidation semantics.

*   **Option A: Implement in `gcsfs` first**
    *   *Pros:*
        1. Matches the immediate hook point in `gcsfs._info()`.
        2. Easier to land quickly for GCS-specific workloads.
        3. Lets GCS-specific invalidation evolve without affecting other backends.
    *   *Cons:*
        1. Duplicates metadata-cache primitives that other backends will likely need.
        2. Pushes cache API design later, after one backend has already baked in assumptions.

*   **Option B: Implement in `fsspec` first [Selected]**
    *   *Pros:*
        1. Provides one metadata-cache abstraction for all backends.
        2. Keeps cache sizing, TTL, and opt-in behavior consistent across the ecosystem.
        3. Allows `gcsfs`, `s3fs`, and other backends to share the same tested bounded-cache implementation while still customizing keys and invalidation helpers.
    *   *Cons:*
        1. The base abstraction must expose enough hooks for backend-specific versioning and path normalization.
        2. Each backend still needs a full mutation-surface audit.

### 2.2 Implementation Structure
*Decision:* **Add `fsspec.infocache.InfoCache` with explicit indexing and backend hooks [Selected]**

`InfoCache` should be a dedicated metadata cache rather than an adaptation of byte-range caching. It should share high-level policy concepts with `DirCache` (optional use, TTL, bounded size), but not copy its internal mechanics blindly. Metadata caching needs exact-key invalidation, path-prefix invalidation, and structured keys.

### 2.3 Cache Policy Defaults
*Decision:* **`use_info_cache=False`, `max_paths=100000`, `info_expiry_time=60` [Selected]**

*   **Rationale:**
    1. Disabling the cache by default preserves current freshness behavior for direct point-lookups.
    2. `max_paths` bounds the number of live cached metadata entries; actual memory usage still depends on metadata payload size and path length, so the design should not claim a fixed byte estimate.
    3. When users opt in, a 60s default TTL keeps the feature safe in mildly mutable environments while still absorbing tight repeated-probe patterns. Users on immutable datasets can pass `info_expiry_time=None` explicitly for unbounded reuse; users on actively mutating datasets can shorten it.

### 2.3.1 Instance Sharing via `_Cached`

`fsspec.spec._Cached` caches `AbstractFileSystem` instances per storage-option tuple, so two callers with the same options share one `InfoCache`. This is intentional — it is how a worker process accumulates benefit from its own previous probes — but the design must account for it:

- The first constructor call wins: if one caller passes `use_info_cache=True` and another passes `use_info_cache=False` with otherwise identical options, they share the instance configured by the first call.
- The three new options (`use_info_cache`, `info_expiry_time`, `max_info_paths`) must participate in `_Cached`'s instance key so that differing configurations yield distinct instances. The implementation will rely on `fsspec`'s existing `storage_options` hashing — these keys flow through `**storage_options` and are thus already part of the identity tuple — but the test plan (§7.1) must cover this explicitly.

### 2.3.2 Cross-Process Consistency

`InfoCache` is per-process. Multi-worker `DataLoader`s, sharded training jobs, and any setup spawning subprocesses each get an independent cache, so a write in one process is invisible to cached reads in another. This is explicitly in scope for the target workloads (read-mostly manifests and dataset shards) but out of scope for mutating distributed pipelines. The default 60s TTL plus opt-in gating is the design's only safety net here; users running multi-writer workloads should either disable the cache or use a short TTL.

### 2.4 Negative Caching
*Decision:* **Do not cache `FileNotFoundError` [Selected]**

*   **Rationale:** This remains the safest behavior in distributed and multi-writer environments. Negative caching can easily outlive the absence it observed.
*   **Acknowledged trade-off:** ML manifest-probing workloads often issue bursty `exists()` checks against shards that do not yet exist (e.g. waiting for a producer). Without negative caching, every probe pays full RPC cost. We accept this for v1 — a short-TTL negative cache (e.g. `negative_info_expiry_time` defaulting to 0/disabled) is a clean follow-up once we have hit-rate data and a concrete user complaint to size it against.

### 2.5 Metadata Fields
*Decision:* **Cache the full metadata dictionary [Selected]**

*   **Rationale:** A cache hit should be semantically equivalent to a successful direct metadata resolution. This includes less common fields such as user metadata, content settings, generations, and backend-specific attributes.

### 2.6 Cache Key Shape
*Decision:* **Use structured hashable keys, not concatenated strings [Selected]**

*   **Rationale:** Keys such as `(normalized_path, generation)` are collision-safe and keep invalidation logic separate from string parsing. The base API should allow each backend to define its own structured key as long as it is hashable.

### 2.7 Invalidation Strategy
*Decision:* **Use indexed invalidation, not full-cache scans [Selected]**

*   **Rationale:** Exact-path invalidation should be O(number of cached versions for that path), and recursive invalidation should scale with indexed descendants instead of every cache entry. This requires path-aware auxiliary indexes that are pruned on every eviction and deletion.

### 2.8 Concurrency Model
*Decision:* **Document the same single-threaded expectation `dircache` already has; do not add locks [Selected]**

*   **Rationale:** `fsspec.dircache.DirCache` does not guarantee thread safety today, and `InfoCache` matches that contract. Two concurrent `_info()` coroutines will each miss, each fetch, and each populate — the last writer wins. This is a missed optimization, not a correctness bug: the values are equivalent successful-metadata fetches.
*   **Non-goal:** request coalescing via an in-flight futures map. This is a worthwhile follow-up for high-fanout `DataLoader` cold-start, but it is orthogonal to the caching mechanism and should land separately once we have production hit-rate data to justify the extra state.

### 2.9 Cost Model for Prefix Index
`_prefix_index` stores one entry per ancestor of each cached path. For a path of depth `d`, an insert touches `d+1` sets, and eviction touches the same set. With `max_paths=100000` and depth-10 paths, the index holds up to ~1M prefix→path edges. This is bounded but non-trivial; the design deliberately pays this cost to keep `pop_prefix()` and `refresh=True` invalidation sublinear in cache size. Users setting very large `max_paths` on deeply nested datasets should size RAM accordingly.

### 2.10 Storage-Option Naming
The three new options are `use_info_cache`, `info_expiry_time`, `max_info_paths`. The first two mirror `DirCache`'s `use_listings_cache`/`listings_expiry_time` conventions. The third is intentionally **not** named `max_paths`: that name is already taken by `DirCache` on the same `AbstractFileSystem` instance, and reusing it would force the two caches to share a sizing knob even though their per-entry costs differ by orders of magnitude (a `DirCache` entry holds an entire listing, an `InfoCache` entry holds one object's metadata). Keeping the names distinct is a small ergonomic cost paid for clearer mental model and independent tuning.

---

## 3. Component Design: `fsspec.infocache`

We will add a new `InfoCache` implementation in `fsspec/infocache.py`.

### 3.1 Design Goals

The implementation must satisfy all of the following:

1. Preserve opt-in semantics.
2. Support structured backend-defined keys.
3. Keep entry count and all bookkeeping bounded to live entries only.
4. Provide efficient exact-path invalidation.
5. Provide efficient subtree invalidation for refresh-style operations.

### 3.2 Data Model

Each live cache entry stores:

- `key`: a structured hashable key, such as `(path, generation)`
- `path`: the normalized logical path used for invalidation
- `value`: the metadata dictionary
- `created`: insertion timestamp for TTL checks

The cache maintains the following internal structures:

- `_entries: OrderedDict[key, CacheEntry]`
  - Keeps LRU order and stores the canonical live entries.
- `_path_index: dict[path, set[key]]`
  - Maps one logical path to all cached keys for that path, including multiple generations.
- `_prefix_index: dict[prefix, set[path]]`
  - Maps a directory prefix to descendant cached paths for efficient subtree invalidation.

All three structures only contain live entries. Whenever an entry expires, is evicted, or is explicitly invalidated, all index references are removed in the same operation.

### 3.3 Code Sketch (`fsspec/infocache.py`)

```python
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    path: str
    value: dict
    created: float


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0


class InfoCache:
    def __init__(
        self,
        use_info_cache=False,
        info_expiry_time=60,
        max_paths=100000,
    ):
        self.use_info_cache = use_info_cache
        self.info_expiry_time = info_expiry_time
        self.max_paths = max_paths
        self._entries = OrderedDict()
        self._path_index = {}
        self._prefix_index = {}
        self.stats = CacheStats()

    def _prefixes(self, path):
        # Normalize: strip leading/trailing slashes so we never yield an
        # empty-string prefix (which would alias unrelated subtrees) or
        # synthesize a leading-slash prefix that no insert path uses.
        path = path.strip("/")
        if not path:
            return
        parts = path.split("/")
        for i in range(1, len(parts)):
            yield "/".join(parts[:i])
        yield path

    def _is_expired(self, entry):
        if self.info_expiry_time is None:
            return False
        return (time.time() - entry.created) > self.info_expiry_time

    def _unlink(self, key, entry):
        self._entries.pop(key, None)

        keys = self._path_index.get(entry.path)
        if keys is not None:
            keys.discard(key)
            if not keys:
                self._path_index.pop(entry.path, None)

        # Only prune prefix references when no other generation for this path
        # is still live; multiple keys can share one path entry.
        if entry.path in self._path_index:
            return
        for prefix in self._prefixes(entry.path):
            paths = self._prefix_index.get(prefix)
            if paths is None:
                continue
            paths.discard(entry.path)
            if not paths:
                self._prefix_index.pop(prefix, None)

    def __getitem__(self, key):
        """Look up a cached entry. Raises KeyError on miss or expiration.

        Gated on use_info_cache: if the flag is flipped off after entries were
        populated (e.g. via direct attribute assignment for testing or runtime
        opt-out), every lookup misses. set() is also gated, so a freshly
        disabled cache cannot grow."""
        if not self.use_info_cache:
            self.stats.misses += 1
            raise KeyError(key)
        try:
            entry = self._entries[key]
        except KeyError:
            self.stats.misses += 1
            raise
        if self._is_expired(entry):
            self._unlink(key, entry)
            self.stats.expirations += 1
            self.stats.misses += 1
            raise KeyError(key)
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return entry.value

    def get(self, key, default=None):
        """Dict-style non-raising lookup."""
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key, path, value):
        if not self.use_info_cache:
            return

        now = time.time()
        old = self._entries.get(key)
        if old is not None:
            self._unlink(key, old)

        self._entries[key] = CacheEntry(path=path, value=value, created=now)
        self._entries.move_to_end(key)

        self._path_index.setdefault(path, set()).add(key)
        for prefix in self._prefixes(path):
            self._prefix_index.setdefault(prefix, set()).add(path)

        while self.max_paths and len(self._entries) > self.max_paths:
            old_key, old_entry = self._entries.popitem(last=False)
            self._unlink(old_key, old_entry)
            self.stats.evictions += 1

    def pop_key(self, key):
        entry = self._entries.get(key)
        if entry is not None:
            self._unlink(key, entry)

    def pop_path(self, path):
        for key in list(self._path_index.get(path, ())):
            self.pop_key(key)

    def pop_prefix(self, prefix):
        for path in list(self._prefix_index.get(prefix, ())):
            self.pop_path(path)

    def clear(self):
        self._entries.clear()
        self._path_index.clear()
        self._prefix_index.clear()
```

Notes on the API shape:

- `__getitem__` is the authoritative lookup because it surfaces miss vs expiration vs hit distinctly through the counters. `get()` is a thin dict-style wrapper for callers that prefer a non-raising probe.
- `CacheStats` gives operators a single attribute (`fs.infocache.stats`) to observe hit rate in production. No logging or metric-system coupling is built in; downstream instrumentation is free to sample `stats` on whatever cadence it wants. This is the minimum needed to validate the feature works under real workloads (§7.4).

### 3.4 Memory-Bound Guarantees

The important guarantee is that there is no stale bookkeeping:

- evicted entries are removed from `_entries`, `_path_index`, and `_prefix_index`
- expired entries are removed from all structures on first access
- overwritten entries unlink old state before inserting new state

This keeps total bookkeeping proportional to live cached entries rather than historical entries.

---

## 4. Integrating with `fsspec` and `GCSFileSystem`

### 4.1 Base `AbstractFileSystem` Update

`AbstractFileSystem` provisions `InfoCache` and exposes backend hooks for both key derivation and invalidation path normalization.

Because this feature is implemented in `fsspec`, the design must account for the inherited method surface in `fsspec.spec.AbstractFileSystem` and `fsspec.asyn.AsyncFileSystem`, not just backend overrides:

- `AbstractFileSystem.exists()` delegates to `info()`
- `AbstractFileSystem.size()` delegates to `info()`
- `AbstractFileSystem.isdir()` delegates to `info()`
- `AbstractFileSystem.isfile()` delegates to `info()`
- `AbstractFileSystem.checksum()` delegates to `info()`
- `AsyncFileSystem._exists()` delegates to `_info()`
- `AsyncFileSystem._size()` delegates to `_info()`

That means:

1. Backends that override `_info()` or `info()` can pick up `InfoCache` benefits through those existing inherited methods.
2. The base class contract must define where cache lookup/population belongs so backends do not reimplement it inconsistently.
3. `invalidate_cache()` changes must preserve the existing transaction bookkeeping in `AbstractFileSystem.invalidate_cache()`.

The snippet below shows **additions** to the existing `AbstractFileSystem` class, not a replacement. The real `__init__` already sets `_intrans`, `_transaction`, `_invalidated_caches_in_transaction`, and `self.dircache = DirCache(**storage_options)` (see [fsspec/spec.py:156](.venv/lib/python3.13/site-packages/fsspec/spec.py:156)); the new lines are inserted alongside that existing setup. Three new `storage_options` keys are popped (not `get`) so they do not leak into backend-forwarded kwargs.

```python
# In fsspec/spec.py — additions to the existing AbstractFileSystem class
from fsspec.infocache import InfoCache


class AbstractFileSystem:
    def __init__(self, *args, **storage_options):
        # ... existing init body (DirCache, _intrans, _transaction, ...) ...
        use_info_cache = storage_options.pop("use_info_cache", False)
        info_expiry_time = storage_options.pop("info_expiry_time", 60)
        max_info_paths = storage_options.pop("max_info_paths", 100000)
        self.infocache = InfoCache(
            use_info_cache=use_info_cache,
            info_expiry_time=info_expiry_time,
            max_paths=max_info_paths,
        )

    def _get_info_cache_key(self, path, **kwargs):
        path = self._strip_protocol(path).rstrip("/")
        return (path, None)

    def _get_info_cache_path(self, path, **kwargs):
        return self._strip_protocol(path).rstrip("/")

    def invalidate_cache(self, path=None):
        # During a transaction, defer ALL cache work to end_transaction so
        # mid-transaction reads can still see the pre-mutation state. The
        # existing base impl records the path and returns; replay at commit
        # time runs invalidate_cache(path) again with _intrans=False, which
        # then takes the eager branch below.
        if self._intrans:
            self._invalidated_caches_in_transaction.append(path)
            return
        if path is None:
            self.infocache.clear()
        else:
            self.infocache.pop_prefix(self._get_info_cache_path(path))
```

**Ownership contract.** The base `invalidate_cache` is the single source of truth for `infocache` eviction *and* for the transaction-deferral bookkeeping. Subclasses that override `invalidate_cache` MUST call `super().invalidate_cache(path)` first, then do their own `dircache`/backend-specific work. They MUST gate any backend work on `not self._intrans` themselves (the base call returns early without raising during a transaction, but it does not signal "I deferred this" to the caller). They MUST NOT call `self.infocache.clear()` or `pop_prefix()` themselves — otherwise we get double-eviction and divergent behavior across backends. The `gcsfs` override in §6.2 follows this rule.

### 4.2 Cache Key Formulation in `gcsfs`

`gcsfs` uses generation-aware structured keys. It inherits `_get_info_cache_path` unchanged from the base class (the base already returns `self._strip_protocol(path).rstrip("/")`, which is what gcsfs wants) and only overrides the key derivation.

```python
# In gcsfs/core.py

class GCSFileSystem(asyn.AsyncFileSystem):
    def _get_info_cache_key(self, path, generation=None, **kwargs):
        path = self._get_info_cache_path(path)
        _, _, path_generation = self.split_path(path)
        resolved_generation = _coalesce_generation(generation, path_generation)
        return (path, resolved_generation)
```

This keeps the key collision-safe and allows exact-path invalidation to remove all generations without guessing from a string prefix.

#### 4.2.1 Generation Aliasing

A caller may first issue `info("gs://bucket/file")` with no generation and later issue `info("gs://bucket/file", generation="123")` with the generation that the first call resolved. Naively these produce different keys — `(path, None)` and `(path, "123")` — so the second call misses and refetches even though the server would return identical bytes.

To close this hole, `_info()` populates **both** keys on a successful single-object fetch whenever the response carries a concrete generation:

- The unqualified key `(path, None)` — satisfies future unqualified probes.
- The qualified key `(path, "123")` — satisfies future probes that pin the generation.

Both entries point at the same dict (no copy), both are tracked in `_path_index` and `_prefix_index`, and invalidation by path (via `pop_path`) removes both together. If the response has no generation field — which should not happen for GCS objects but is handled defensively — only the unqualified key is populated.

---

## 5. Modifying `_info()` Execution Flow in `gcsfs`

`gcsfs.core.GCSFileSystem._info()` will consult `InfoCache` before falling back to the current parent-listing path and network fetch path.

```python
async def _info(self, path, generation=None, **kwargs):
    path = self._strip_protocol(path).rstrip("/")

    # ... existing bucket-root logic ...

    bucket, key, path_generation = self.split_path(path)
    resolved_generation = _coalesce_generation(generation, path_generation)

    cache_path = self._get_info_cache_path(path, generation=generation)
    cache_key = self._get_info_cache_key(path, generation=generation)

    try:
        return self.infocache[cache_key]
    except KeyError:
        pass

    parent_path = self._parent(path)
    parent_cache = self._ls_from_cache(parent_path)
    if parent_cache:
        name = "/".join((bucket, key))
        for entry in parent_cache:
            if entry["name"].rstrip("/") == name and (
                not resolved_generation or entry.get("generation") == resolved_generation
            ):
                # Promote the matched entry into infocache so the *next* point
                # lookup skips even the parent_cache scan. We promote one entry
                # (the one the caller asked about), not the whole listing —
                # see §1 for the bulk-import rationale.
                self._populate_info_cache(cache_path, cache_key, entry)
                return entry

    if self._ls_from_cache(path):
        return {
            "bucket": bucket,
            "name": path,
            "size": 0,
            "storageClass": "DIRECTORY",
            "type": "directory",
        }

    async with parallel_tasks_first_completed(
        [
            self._get_object(path),
            self._get_directory_info(path, bucket, key, generation),
        ]
    ) as (tasks, done, pending):
        get_object_task, get_directory_info_task = tasks

        try:
            result = await get_object_task
            if not _is_directory_marker(result):
                self._populate_info_cache(cache_path, cache_key, result)
                return result
        except FileNotFoundError:
            pass

        return await get_directory_info_task


def _populate_info_cache(self, cache_path, cache_key, result):
    """Insert under both the requested key and the generation-qualified key
    so later probes that pin the generation still hit. See §4.2.1."""
    self.infocache.set(cache_key, cache_path, result)
    actual_generation = result.get("generation")
    if actual_generation is not None and cache_key[1] != actual_generation:
        aliased_key = (cache_path, actual_generation)
        self.infocache.set(aliased_key, cache_path, result)
```

Notes:

- The cache is only populated from successful single-object file metadata results.
- Directory placeholders and listing-derived entries are not inserted into `InfoCache`.
- Opt-in default means this path is inert unless the user explicitly enables it.
- `_is_directory_marker(result)` is an existing helper in `gcsfs/core.py` that detects zero-byte trailing-slash "folder" objects; reusing it avoids caching synthesized directory entries as if they were regular files.
- `info()` returns the cached dict by reference. Callers that mutate the dict would corrupt shared state; the existing `_info()` contract already treats these dicts as read-only, and `checksum(path) = tokenize(info(path))` stays deterministic because identical references tokenize identically.

### 5.1 Required `fsspec` Method-Surface Updates

The `fsspec` design should explicitly document which methods change semantically or operationally once `InfoCache` exists.

#### Synchronous base methods in `fsspec.spec.AbstractFileSystem`

- `info(path, **kwargs)`
  - No mandatory generic cache insertion is required for listing-only backends, but the contract should document that a backend override may consult and populate `InfoCache` if it has a true single-object metadata fast path.
- `exists(path, **kwargs)`
  - Benefits automatically whenever `info()` benefits.
- `size(path)`
  - Benefits automatically whenever `info()` benefits.
- `isdir(path)`
  - Benefits automatically whenever `info()` benefits.
- `isfile(path)`
  - Benefits automatically whenever `info()` benefits.
- `checksum(path)`
  - Benefits automatically whenever `info()` benefits, because it tokenizes `info(path)`.

#### Async base methods in `fsspec.asyn.AsyncFileSystem`

- `_exists(path, **kwargs)`
  - Benefits automatically whenever `_info()` benefits.
- `_size(path)`
  - Benefits automatically whenever `_info()` benefits.

#### Base invalidation contract

- `AbstractFileSystem.invalidate_cache(path=None)` must remain part of the design because it records deferred invalidations during transactions.
- Any backend override of `invalidate_cache()` must call the parent implementation before or alongside backend-specific cache eviction so transaction semantics remain intact.

---

## 6. Cache Consistency and Invalidation Surface

`InfoCache` adds a second metadata state surface beside `dircache`, so every metadata-mutating path must either invalidate exact object entries or update them intentionally. This design chooses invalidation, not write-through updates.

### 6.1 `invalidate_info` Helper

```python
def invalidate_info(self, path, generation=None):
    cache_path = self._get_info_cache_path(path, generation=generation)
    if generation is not None:
        cache_key = self._get_info_cache_key(path, generation=generation)
        self.infocache.pop_key(cache_key)
    else:
        self.infocache.pop_path(cache_path)
```

This gives exact-object invalidation without scanning the entire cache.

### 6.2 `invalidate_cache` Integration

`invalidate_cache(path)` delegates info-cache eviction to the base class per the ownership contract in §4.1, and only does `dircache` work locally.

```python
def invalidate_cache(self, path=None):
    super().invalidate_cache(path)  # handles infocache + transaction bookkeeping

    # super() deferred the work during a transaction; do the same here so
    # dircache and infocache stay in lockstep. end_transaction replays with
    # _intrans=False, which falls through to the eager branch below.
    if self._intrans:
        return

    if path is None:
        self.dircache.clear()
        return

    path = self._get_info_cache_path(path)
    while path:
        self.dircache.pop(path, None)
        parent = self._parent(path)
        if parent == path:  # root-of-protocol fixed point; avoid infinite loop
            break
        path = parent
```

This keeps `refresh=True` behavior aligned with the metadata cache without a full-cache scan, while preserving `fsspec` transaction bookkeeping. The fixed-point guard on `_parent` is defensive — some fsspec-derived backends return the same string for the root of their protocol rather than an empty string, and without the guard the loop would spin forever.

### 6.3 Required Updates in `gcsfs/core.py`

The following methods must call `invalidate_info(...)` in addition to existing listing-cache invalidation. Line numbers are against the tip of `main` as of this plan and are a checklist for review — they will drift once edits land:

- `_rm_file(path, ...)` — [gcsfs/core.py:1425](gcsfs/core.py:1425)
- `_rm_files(paths, ...)` — [gcsfs/core.py:1435](gcsfs/core.py:1435)
- `_mv_file_cache_update(path1, path2, ...)` — [gcsfs/core.py:1386](gcsfs/core.py:1386)
- `_cp_file(path1, path2, ...)` — [gcsfs/core.py:1352](gcsfs/core.py:1352)
- `_put_file(lpath, rpath, ...)` — [gcsfs/core.py:1627](gcsfs/core.py:1627)
- `_pipe_file(path, ...)` — [gcsfs/core.py:1569](gcsfs/core.py:1569)
- `_merge(path, ...)` — [gcsfs/core.py:1321](gcsfs/core.py:1321)
- `_setxattrs(path, ...)` — [gcsfs/core.py:1258](gcsfs/core.py:1258)

**Invalidation ordering.** `invalidate_info(path)` runs **after** the mutating RPC returns successfully, for the same reason the existing `dircache` invalidation does: invalidating before the RPC would drop a valid entry even if the mutation fails and nothing actually changed. If the RPC raises, the call site does not invalidate — the cached metadata still reflects reality. For `_cp_file` and `_mv_file_cache_update`, invalidate destination after success; for `_mv_file_cache_update` also invalidate source after success. This is the same contract used by `dircache` updates today and does not introduce a new ordering surface.

### 6.4 Required Updates in `extended_gcsfs.py`

This repository also contains mutating overrides and cache-specialized code paths in `extended_gcsfs.py`. The design must explicitly update them too, otherwise HNS and zonal paths will drift from the base cache semantics.

Required coverage includes (line numbers against the tip of `main`):

- `ExtendedGCSFileSystem._mv_file_cache_update(path1, path2, response=None)` — [gcsfs/extended_gcsfs.py:417](gcsfs/extended_gcsfs.py:417)
  - Invalidate info cache for both source and destination, even when `dircache` is updated in place.
- `ExtendedGCSFileSystem._put_file(...)` — [gcsfs/extended_gcsfs.py:1123](gcsfs/extended_gcsfs.py:1123)
  - Invalidate destination info entry after successful zonal upload finalization.
- `ExtendedGCSFileSystem._pipe_file(...)` — [gcsfs/extended_gcsfs.py:1209](gcsfs/extended_gcsfs.py:1209)
  - Invalidate destination info entry after successful zonal byte upload finalization.
- `ExtendedGCSFileSystem._cp_file(...)` — [gcsfs/extended_gcsfs.py:1384](gcsfs/extended_gcsfs.py:1384)
  - Invalidate destination info entry after successful copy.
- `ExtendedGCSFileSystem._merge(...)` — [gcsfs/extended_gcsfs.py:1416](gcsfs/extended_gcsfs.py:1416)
  - Invalidate destination info entry after successful compose/merge.
- Any future mutating override added in `extended_gcsfs.py`
  - Must be included in the metadata-cache audit as part of code review.

### 6.5 Required Updates in `fsspec`

Since the implementation is `fsspec`-first, the design must explicitly cover these base-library changes too:

- `fsspec.spec.AbstractFileSystem.__init__`
  - Provision `self.infocache`.
- `fsspec.spec.AbstractFileSystem`
  - Add `_get_info_cache_key(...)`.
- `fsspec.spec.AbstractFileSystem`
  - Add `_get_info_cache_path(...)`.
- `fsspec.spec.AbstractFileSystem.invalidate_cache(...)`
  - Preserve transaction bookkeeping and define how subclasses should integrate info-cache eviction.
- `fsspec.asyn.AsyncFileSystem`
  - No direct behavioral rewrite is required for `_exists()` and `_size()`, but the design should call out that they automatically benefit through `_info()`.

Backends that only implement `ls()` and inherit the default `info()` are not required to adopt `InfoCache` immediately. The primary adoption path is for backends with a true object-level metadata hook (`_info()` or specialized `info()`).

### 6.6 Rollout Sequence

Because the abstraction lives in `fsspec` but the consumer is `gcsfs`, landing this feature requires a coordinated sequence:

1. **fsspec PR.** Land `fsspec/infocache.py`, the base-class hooks (`_get_info_cache_key`, `_get_info_cache_path`, `infocache` attribute), and the `invalidate_cache` integration. Include unit tests from §7.1.
2. **fsspec release.** Wait for an `fsspec` release that contains the changes, or pin a specific `fsspec` commit in the gcsfs dev environment for CI while waiting.
3. **gcsfs version bump.** Raise the `fsspec` floor in `gcsfs/pyproject.toml` (and `setup.py` if present) to the release from step 2.
4. **gcsfs integration PR.** Add `_get_info_cache_key` override, update `_info()` for cache lookup and generation-aliasing population, add `invalidate_info` helper, update the mutation surface (§6.3 and §6.4). Include integration tests from §7.2 and §7.3.
5. **Opt-in rollout.** Feature is off by default after step 4. Internal canary workloads flip `use_info_cache=True` and surface `fs.infocache.stats` for hit-rate validation. Default flip is a separate decision gated on observed hit rate and zero regression reports.

**Fallback if upstream review stalls.** If the fsspec PR is blocked or reshaped in ways that delay step 2 beyond the target quarter, implement the cache inside `gcsfs` under `gcsfs/_infocache.py` with the same API, and wire the base-class hooks as local overrides on `GCSFileSystem`. The internal API stays identical, so migrating to upstream `fsspec.infocache` later is a module rename plus removing the local copy.

---

## 7. Comprehensive Test Plan

Testing should validate both cache mechanics and repository-specific integration points.

### 7.1 Unit Tests (`fsspec/tests/test_infocache.py`)

1. **Entry-count eviction:** Create `InfoCache(max_paths=3)`, insert 4 live entries, assert only 3 remain and the least recently used entry is gone.
2. **TTL expiration:** Create `InfoCache(info_expiry_time=1)`, insert one item, sleep 1.1s, assert lookup raises `KeyError` and all indexes are cleaned.
3. **Structured keys:** Insert two entries for the same path with keys like `("bucket/file", "111")` and `("bucket/file", "222")`, assert both coexist.
4. **Exact-path invalidation:** Call `pop_path("bucket/file")`, assert all generations for that path are removed and unrelated paths remain.
5. **Prefix invalidation:** Cache entries under `bucket/a/...` and `bucket/b/...`, call `pop_prefix("bucket/a")`, assert only `bucket/a` descendants are removed.
6. **Bounded bookkeeping:** Repeatedly insert and evict entries, then assert `_entries`, `_path_index`, and `_prefix_index` contain no references to dead entries.
7. **CacheStats counters:** Insert one entry, assert a hit increments `stats.hits`, a miss on a different key increments `stats.misses`, an evicted insert increments `stats.evictions`, and a lookup of an expired entry increments `stats.expirations`.
8. **`get` non-raising wrapper:** Assert `cache.get(missing_key)` returns `None` and `cache.get(missing_key, sentinel)` returns `sentinel`, while `cache[missing_key]` raises `KeyError`.
9. **`_Cached` option participation:** Instantiate `AbstractFileSystem` twice with different `use_info_cache`/`info_expiry_time`/`max_info_paths` tuples through `fsspec.filesystem(...)` and assert two distinct instances are returned (storage-option hashing must route them separately).
10. **`use_info_cache` runtime gate:** Insert an entry while enabled, flip `cache.use_info_cache = False`, assert subsequent `cache[key]` raises `KeyError` and `stats.misses` increments — covers the defensive gate in `__getitem__` (§3.3).
11. **Path-normalization edge cases for `_prefixes`:** Insert entries with paths `"a"`, `"/a/b"`, `"a/b/"`, and `""` (the last is a no-op). Assert `pop_prefix("")` is a no-op (does not nuke unrelated entries) and `pop_prefix("a")` removes both `"a"` and `"a/b"` regardless of leading/trailing slashes — guards against the empty-string-prefix aliasing bug.

### 7.2 Integration Tests (`gcsfs/tests/test_core.py`)

1. **Default-off behavior:** Construct `GCSFileSystem()` without `use_info_cache=True` and assert repeated direct `info()` calls still hit the network path.
2. **Opt-in cache hit:** Construct with `use_info_cache=True`, call `info(path)` twice, assert the second call bypasses the object metadata API.
3. **Listing does not populate info cache:** Call `ls(dir_path)` and confirm `infocache` stays empty.
4. **Generation separation:** Cache `info(path, generation="111")` and `info(path, generation="222")`, assert separate cache entries exist.
5. **Generation aliasing:** Call `info(path)` first with no generation, assert the returned metadata carries generation `G`, then call `info(path, generation=G)` and assert it is a cache hit (no new metadata RPC). Covers §4.2.1.
6. **Mutation invalidation:** Verify `_rm_file`, `_rm_files`, `_mv_file_cache_update`, `_cp_file`, `_put_file`, `_pipe_file`, `_merge`, and `_setxattrs` invalidate the expected info entries.
7. **Mutation-failure ordering:** Force an RPC failure in `_rm_file` (e.g. 412 precondition-failed) and assert the pre-existing cached info entry is still present — invalidation runs only on success (§6.3).
8. **Refresh invalidation:** Cache entries under a subtree, call `ls(path, refresh=True)` or equivalent invalidation path, assert subtree info entries are evicted.
9. **Ownership contract:** Call `invalidate_cache(path)` on a `GCSFileSystem` with cached info entries at and under `path`; assert they are evicted exactly once (verified by `stats.evictions` delta) — the subclass override must not double-evict.
10. **Parent_cache promotion:** Pre-populate `dircache` for `parent` via `ls(parent)`, then call `info(parent/child)` — assert `infocache` now contains an entry for `parent/child`, and a second `info(parent/child)` call is served from `infocache` (no parent_cache scan, verified by `stats.hits` delta). Covers §1 / §5 single-entry promotion.
11. **Bulk `ls()` does not populate infocache:** Call `ls(parent)` returning N entries, assert `len(infocache._entries) == 0` immediately after — only point lookups via `info()` populate.
12. **`find()` / `walk()` do not populate infocache:** Run `find(parent)` and `list(walk(parent))`, assert `infocache` remains empty regardless of how many entries those traversals enumerate.
13. **Non-FileNotFoundError propagation:** Stub `_get_object` to raise `OSError("500 Internal")` on the first call; assert `_info()` propagates the error (does **not** swallow), `infocache` is unchanged, and the existing parent-fallback logic is unaffected. Distinguishes "absent" from "transient backend failure" — only the former is silently bypassed by the cache miss path.
14. **Transaction-deferred invalidation:** Open `with fs.transaction:` and pre-populate an info entry; mid-transaction call `_rm_file(path)`; assert the cached entry is **still present** during the transaction and is evicted exactly once at commit (verified by `stats.evictions` delta of 1). Covers the deferred-eviction contract in §4.1 / §6.2.

### 7.4 Hit-Rate Benchmark (`gcsfs/tests/benchmarks/test_infocache_hitrate.py`)

A deterministic microbenchmark, runnable locally and in CI in `gcsfs` mock mode, validates that the feature delivers the intended reduction in metadata RPCs:

1. Construct a `GCSFileSystem` with `use_info_cache=False` against a recorded in-process mock; issue a workload of N unique paths probed K times each in round-robin; count metadata RPCs.
2. Construct with `use_info_cache=True`; issue the same workload; count metadata RPCs and assert count ≈ N (first-touch only) and `fs.infocache.stats.hits == N * (K - 1)`.
3. Record the speedup and counter deltas in the benchmark output so regressions are visible in PR diffs. This is the primary signal for "the feature is doing what we said it would."

### 7.3 Integration Tests (`gcsfs/tests/test_extended_gcsfs*.py`)

1. **HNS move path:** Exercise `extended_gcsfs._mv_file_cache_update` and assert source and destination info entries are invalidated even when listing cache is updated in place.
2. **Zonal put path:** Exercise `extended_gcsfs._put_file` and assert destination info cache is invalidated after upload.
3. **Zonal pipe path:** Exercise `extended_gcsfs._pipe_file` and assert destination info cache is invalidated after upload.

---

## 8. Summary

The revised design keeps the implementation in `fsspec`, but avoids the main correctness and maintainability traps:

- cache disabled by default, safe TTL (60s) when opt-in
- structured keys with generation aliasing so unqualified and qualified probes share entries
- single-entry promotion from `dircache` parent_cache hits (no bulk import from `ls`/`find`/`walk`)
- live-entry-bounded bookkeeping with documented prefix-index cost model and normalized prefix derivation
- indexed invalidation instead of full-cache scans, with fixed-point parent-walk termination
- single-ownership contract for `infocache` eviction (base class owns it; subclasses only touch `dircache`)
- transaction-deferred invalidation: both `infocache` and `dircache` are evicted only at `end_transaction`, matching existing fsspec semantics
- invalidation runs only after mutating RPCs succeed, matching existing `dircache` ordering
- explicit mutation-surface coverage in both `core.py` and `extended_gcsfs.py` with line-anchored audit list
- built-in `CacheStats` counters for hit-rate validation in production
- concurrency stance matches `dircache` (no internal locks); request coalescing is a follow-up
- negative-caching deferred to v2 with explicit acknowledgement of the manifest-probe trade-off
- concrete fsspec-first rollout sequence with a gcsfs-local fallback path

This keeps the abstraction reusable across `fsspec` backends without weakening `gcsfs` correctness.
