# Low-Level Design (LLD): GCSFS Custom InfoCache

## 1. Objective and Motivation
The primary objective of this design is to introduce a dedicated, single-object metadata cache (`InfoCache`) into `gcsfs.GCSFileSystem`. 

**The Problem:**
Currently, GCSFS relies heavily on `dircache` (a directory-level cache) to reduce expensive network calls. When a directory is listed, its contents are cached. Subsequent `info()` calls for files *within* that directory can be resolved locally in $O(N)$ time. However, when users perform direct, point-lookups on specific files without first listing the directory, GCSFS makes direct API calls (e.g., `storage.objects.get`).

In AI/ML workloads (such as PyTorch `DataLoader`s, Dask array operations, or Ray datasets), systems frequently probe files by calling `exists()`, `info()`, `modified()`, or `size()` on individual, known paths. Without a cache, this results in massive redundant network overhead, increasing latency and Google Cloud API costs (Class A operations), particularly in high-throughput distributed training environments.

**The Solution:**
The new `InfoCache` will specifically store the results of single-file `info()` metadata calls. It will mirror the semantic behavior of the existing `fsspec.dircache.DirCache`—providing Time-To-Live (TTL) expiration and bounded LRU (Least Recently Used) eviction—but tailored exclusively for single-path lookups rather than hierarchical directory traversals.

---

## 2. Key Design Decisions and Trade-offs

### 2.1 Code Location: GCSFS vs. `fsspec`
*Decision:* Where should the `InfoCache` logic live?
*   **Option A: Upstream to `fsspec` (Base class `fsspec.caching.InfoCache`)**
    *   *Pros:* Available to all filesystem implementations (S3, Azure, local); reduces code duplication across the ecosystem.
    *   *Cons:* `fsspec` caching is historically focused on byte-buffer chunking (readahead, block cache). Introducing a metadata dictionary cache changes the scope of `fsspec.caching`. Different filesystems have vastly different metadata structures and invalidation semantics, making a generic `InfoCache` difficult to generalize safely.
*   **Option B: Implement locally in `gcsfs` [Selected]**
    *   *Pros:* Allows us to tailor the cache specifically to GCS's versioning (`generation`), directory marker behavior, and invalidation rules without waiting for upstream consensus.
    *   *Cons:* Cannot be reused directly by `s3fs` or `adlfs` without duplication.

### 2.2 Implementation Structure: New File vs. Existing File
*Decision:* Should the `InfoCache` class be placed in a new `caching.py` file or kept in `core.py`?
*   **Option A: Inline within `gcsfs/core.py`**
    *   *Pros:* Keeps all filesystem logic together; avoids circular import issues between `core.py` and a potential `caching.py`.
    *   *Cons:* `core.py` is already over 2,500 lines long. Adding more classes bloats the file and violates separation of concerns.
*   **Option B: New or existing `gcsfs/caching.py` [Selected]**
    *   *Pros:* Strong separation of concerns. The caching module handles state, TTL, and LRU eviction, while `core.py` handles GCS API logic. Makes unit testing the cache easier.
    *   *Cons:* Might require careful import management to avoid circular dependencies if cache classes need to reference GCS-specific types.

### 2.3 Cache Policy Defaults
*Decision:* What should the default capacity and TTL be?
*   **Max Capacity (`max_paths`): 100,000 [Selected]**
    *   *Pros:* Capping at 100k limits the memory footprint to ~100MB (assuming ~1KB per dict). This safely protects against OOM errors during massive data lake scans while still providing a high hit rate for typical ML training datasets.
    *   *Cons:* In extremely large directories, entries might be evicted prematurely.
*   **Time-To-Live (`info_expiry_time`): `None` (Infinite) [Selected]**
    *   *Pros:* Aligns with existing `dircache` defaults. Maximizes cache hit rate for read-heavy workloads (like ML training) where data is immutable.
    *   *Cons:* Increases the risk of serving stale metadata if external processes are mutating the bucket. Users with mutable buckets must explicitly configure a TTL.

### 2.4 Negative Caching
*Decision:* Should we cache `FileNotFoundError` (404) responses for objects that do not exist?
*   **Option A: Enable Negative Caching**
    *   *Pros:* Saves API calls when repeatedly querying missing files (e.g., polling for a file to appear).
    *   *Cons:* Highly susceptible to "cache poisoning" in distributed systems. If Node A queries a file just before Node B creates it, Node A will permanently think the file is missing until the TTL expires or the cache is cleared.
*   **Option B: Disable Negative Caching [Selected]**
    *   *Pros:* Safest for distributed, multi-writer environments. If a file isn't found, we assume it might be created momentarily.
    *   *Cons:* Workloads that repeatedly poll non-existent files will not see performance improvements and will incur API costs.

### 2.5 Metadata Fields
*Decision:* Should we cache the entire metadata dictionary returned by the GCS API, or only a subset of critical fields?
*   **Option A: Cache Subset (e.g., `name`, `size`, `updated`, `generation`)**
    *   *Pros:* Reduces memory overhead per cache entry, potentially allowing a higher `max_paths` limit.
    *   *Cons:* GCSFS and downstream libraries (like `pandas` or `xarray`) sometimes rely on obscure or extended metadata fields (e.g., `contentType`, `customTime`, ACLs). Stripping fields might cause unexpected `KeyError`s or behavioral changes.
*   **Option B: Cache Entire Dictionary [Selected]**
    *   *Pros:* Guarantees semantic equivalence between a cache hit and a direct API call. Safest approach.
    *   *Cons:* Higher memory footprint per entry. (Mitigated by the strict `max_paths` limit).

---

## 3. Component Design: `InfoCache` Implementation

We will implement a new class, `InfoCache`, within `gcsfs/caching.py` (or directly within `gcsfs/core.py` to minimize import cyclic dependencies, depending on the current structure).

### 3.1 Class Structure and Mechanics
The class will inherit from `collections.abc.MutableMapping` to ensure it implements the standard Python dictionary interface (`__getitem__`, `__setitem__`, `__delitem__`, `__iter__`, `__len__`). This allows it to be used intuitively throughout the GCSFS codebase.

#### LRU Eviction Mechanism
To prevent Out-Of-Memory (OOM) crashes when scanning data lakes containing millions of files, the cache must be bounded. Instead of importing heavy external dependencies like `cachetools`, we will utilize an elegant and efficient technique using Python's standard library `functools.lru_cache`. By wrapping a dummy function that pops the oldest item from the internal dictionary, we can automatically trigger LRU eviction when the size limit is reached.

#### TTL (Time-To-Live) Mechanism
We will maintain a secondary dictionary, `self._times`, that records the `time.time()` of insertion for each key. During retrieval (`__getitem__`), the system will check if the difference between the current time and the insertion time exceeds the configured TTL. If so, the item is silently deleted and a `KeyError` is raised, forcing a fresh network fetch.

### 3.2 Code Implementation

```python
import time
from collections.abc import MutableMapping
from functools import lru_cache

class InfoCache(MutableMapping):
    """
    Caching of single-object metadata (e.g., from info() calls).
    
    Structure:
        {"path#generation": {"name": "path", "size": 123, "type": "file", ...}, ...}
        
    Parameters
    ----------
    use_info_cache : bool
        If False, this cache never stores or returns items, acting as a pass-through.
    info_expiry_time : int or float, optional
        Time in seconds that a metadata entry is considered valid. If None,
        entries do not expire based on time (infinite TTL).
    max_paths : int, optional
        The maximum number of metadata entries to retain in memory. When exceeded,
        the least recently accessed items are evicted.
    """
    def __init__(
        self,
        use_info_cache=True,
        info_expiry_time=None,
        max_paths=100000,
    ):
        self._cache = {}
        self._times = {}
        self.use_info_cache = use_info_cache
        self.info_expiry_time = info_expiry_time
        self.max_paths = max_paths
        
        # We leverage Python's built-in lru_cache to handle the LRU eviction queue.
        # This wrapper function simply removes the specified key from the underlying dict.
        # The lru_cache decorator manages the queue size and call history.
        if self.max_paths:
            self._q = lru_cache(self.max_paths + 1)(lambda key: self._cache.pop(key, None))

    def __getitem__(self, item):
        # 1. Check for TTL expiration
        if self.info_expiry_time is not None:
            if self._times.get(item, 0) - time.time() < -self.info_expiry_time:
                del self._cache[item]
                # We intentionally do not clean up self._times here to save CPU cycles;
                # the stale entry in _times will be overwritten on the next set operation.
                raise KeyError(item)
                
        # 2. Update LRU tracking
        if self.max_paths:
            self._q(item)  # Accessing via _q updates its position to "most recently used"
            
        # 3. Return cached data
        return self._cache[item]

    def __setitem__(self, key, value):
        if not self.use_info_cache:
            return
            
        # 1. Register or update the key in the LRU queue
        if self.max_paths:
            self._q(key) 
            
        # 2. Store the metadata value
        self._cache[key] = value
        
        # 3. Record the insertion timestamp for TTL validation
        if self.info_expiry_time is not None:
            self._times[key] = time.time()

    def __delitem__(self, key):
        del self._cache[key]
        self._times.pop(key, None)

    def __contains__(self, item):
        try:
            self[item]
            return True
        except KeyError:
            return False

    def clear(self):
        """Flushes the entire cache and resets LRU tracking."""
        self._cache.clear()
        self._times.clear()
        if self.max_paths:
            self._q.cache_clear()

    def __len__(self):
        return len(self._cache)

    def __iter__(self):
        entries = list(self._cache)
        return (k for k in entries if k in self)
```

---

## 4. Integrating with `GCSFileSystem`

### 4.1 Initializer and Configuration
To provide users with granular control over this new caching layer without disrupting existing workflows, we will add three new configuration arguments to the `GCSFileSystem.__init__` method.

These new arguments will be passed along alongside the existing `use_listings_cache` parameters.

```python
class GCSFileSystem(asyn.AsyncFileSystem):
    def __init__(
        self,
        # ... existing parameters ...
        use_info_cache=True,
        info_cache_expiry_time=None,
        info_cache_max_paths=100000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Instantiate the custom InfoCache
        # A default max_paths of 100,000 limits memory overhead to roughly 100MB
        # per process (assuming ~1KB per metadata dictionary), protecting against OOMs.
        self.infocache = InfoCache(
            use_info_cache=use_info_cache,
            info_expiry_time=info_cache_expiry_time,
            max_paths=info_cache_max_paths,
        )
```

### 4.2 Cache Key Formulation
GCS heavily relies on object versioning (Generations). Returning cached metadata for a path without verifying the object's generation could lead to serving stale data representing the wrong version of a file.

Therefore, the cache key must be a composite string.
*   **Key Format:** `f"{path}#{generation}"` if `generation` is explicitly requested, otherwise it simply falls back to `path`.

---

## 5. Modifying the `_info()` Execution Flow

The `_info()` method will be updated to intercept calls immediately after basic path normalization and root-bucket checks. 

**Execution Sequence:**
1. **InfoCache Lookup $O(1)$:** Check if the requested path (and generation) exists in `self.infocache`. If it does, and hasn't expired, return it immediately.
2. **DirCache Fallback $O(N)$:** If not in `infocache`, check if the parent directory exists in the legacy `dircache`. If so, iterate through the parent's children looking for a match.
3. **Network Call:** If both local caches miss, execute the standard parallel API calls to Google Cloud Storage.
4. **Cache Population:** If the network call succeeds and returns valid object metadata, store it in `self.infocache` before returning.

```python
    async def _info(self, path, generation=None, **kwargs):
        """File information about this path."""
        path = self._strip_protocol(path).rstrip("/")
        
        # ... (Existing logic for root bucket checks) ...

        bucket, key, path_generation = self.split_path(path)
        resolved_generation = _coalesce_generation(generation, path_generation)
        
        # -------------------------------------------------------------
        # 1. Custom InfoCache Lookup (O(1) dictionary hit)
        # -------------------------------------------------------------
        cache_key = f"{path}#{resolved_generation}" if resolved_generation else path
        if cache_key in self.infocache:
            return self.infocache[cache_key]

        # -------------------------------------------------------------
        # 2. Existing DirCache Lookup for parent dir (O(N) fallback)
        # -------------------------------------------------------------
        parent_path = self._parent(path)
        parent_cache = self._ls_from_cache(parent_path)
        if parent_cache:
            name = "/".join((bucket, key))
            for o in parent_cache:
                if o["name"].rstrip("/") == name and (
                    not resolved_generation or o.get("generation") == resolved_generation
                ):
                    return o
                    
        # ... (Existing logic dealing with explicit directory placeholders) ...

        # -------------------------------------------------------------
        # 3. Parallel Network API execution
        # -------------------------------------------------------------
        async with parallel_tasks_first_completed(
            [
                self._get_object(path),
                self._get_directory_info(path, bucket, key, generation),
            ]
        ) as (tasks, done, pending):
            get_object_task, get_directory_info_task = tasks

            try:
                get_object_res = await get_object_task
                if not _is_directory_marker(get_object_res):
                    
                    # -------------------------------------------------------------
                    # 4. Populate Custom InfoCache on Success
                    # -------------------------------------------------------------
                    self.infocache[cache_key] = get_object_res
                    return get_object_res
                    
            except FileNotFoundError:
                # Negative Caching Consideration:
                # We explicitly do NOT cache FileNotFoundError scenarios.
                # If an object doesn't exist, it might be actively being written 
                # by another node in a distributed cluster. Caching "not found" 
                # could lead to severe race conditions and inconsistencies.
                pass
                
            return await get_directory_info_task
```

---

## 6. Cache Consistency: Targeted Invalidation

To maintain read-after-write consistency, any mutative operation initiated by the client (e.g., `_rm_file`, `_mv_file`, `_put_file`, `_rmdir`) must explicitly invalidate affected entries in both caches.

The `invalidate_cache(self, path=None)` method will be enhanced to handle `infocache` eviction.

### Eviction Logic Considerations
If a user deletes a directory (e.g., `rm -r /bucket/data/`), we must theoretically evict *all* single-file entries within `infocache` that reside under that directory path. 

Because `InfoCache` operates as a flat dictionary, we can achieve this efficiently using a string prefix match (`k.startswith(f"{path}/")`). For a dictionary capped at 100k items, Python can execute this list comprehension check in milliseconds, making it perfectly acceptable for infrequent cache invalidation events.

```python
    def invalidate_cache(self, path=None):
        """
        Invalidate the directory listing cache AND the single-file info cache 
        for a given path and its descendants.
        """
        if path is None:
            logger.debug("invalidate_cache clearing all caches")
            self.dircache.clear()
            self.infocache.clear()
        else:
            path = self._strip_protocol(path).rstrip("/")

            # -------------------------------------------------------------
            # 1. Clear infocache for exact matches and descendant children
            # -------------------------------------------------------------
            # We must clear:
            # - Exact file paths (e.g., "bucket/file.txt")
            # - Paths with generations (e.g., "bucket/file.txt#123456")
            # - Descendant files if path is a directory (e.g., "bucket/folder/child.txt")
            keys_to_delete = [
                k for k in self.infocache 
                if k == path or k.startswith(f"{path}#") or k.startswith(f"{path}/")
            ]
            for k in keys_to_delete:
                self.infocache.pop(k, None)

            # -------------------------------------------------------------
            # 2. Clear dircache parent directories (Existing Logic)
            # -------------------------------------------------------------
            # Bubble up the directory tree, clearing directory listings
            while path:
                self.dircache.pop(path, None)
                path = self._parent(path)
```

---

## 7. Comprehensive Test Plan

Testing must rigorously validate both the mechanical constraints of the cache class itself and the network-saving behavior within the filesystem integration.

### 7.1 Unit Tests (`test_caching.py` or `test_core.py`)
1. **LRU Eviction Constraint:**
   * Create an `InfoCache` with `max_paths=3`.
   * Insert items `A`, `B`, `C`, `D`.
   * Assert that `len(cache) == 3`.
   * Assert that item `A` was evicted (raises `KeyError`).
   * Access item `B` (moving it to MRU), insert item `E`. Assert `C` is evicted.
2. **Time-To-Live (TTL) Expiration:**
   * Create an `InfoCache` with `info_expiry_time=1`.
   * Insert item `A`. Assert it can be retrieved.
   * `time.sleep(1.1)` (or mock `time.time()`).
   * Assert accessing `A` raises `KeyError` and cleans up the cache entry.

### 7.2 Integration Tests (`test_core.py`)
1. **Network Bypass (Cache Hit):**
   * Execute `fs.info(test_file_path)` (mock network is hit 1 time).
   * Execute `fs.info(test_file_path)` again. 
   * Assert the mock network was *not* hit the second time, and the returned dictionary is identical.
2. **Invalidation upon Deletion:**
   * Execute `fs.info(test_file)`. (Cached).
   * Execute `fs.rm(test_file)`. (Triggers `invalidate_cache`).
   * Execute `fs.info(test_file)`.
   * Assert a fresh network request is made resulting in a `FileNotFoundError`.
3. **Invalidation upon Directory Deletion:**
   * Execute `fs.info(dir/file1.txt)`. (Cached).
   * Execute `fs.rm(dir/, recursive=True)`.
   * Assert `fs.infocache` no longer contains the entry for `dir/file1.txt`.
4. **Generation Separation:**
   * Execute `fs.info(test_file, generation="111")`.
   * Execute `fs.info(test_file, generation="222")`.
   * Assert two distinct API calls occurred and two separate entries exist in `infocache`.