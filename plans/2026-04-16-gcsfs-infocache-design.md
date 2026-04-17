# Low-Level Design (LLD): Metadata InfoCache

## 1. Objective and Motivation
The primary objective of this design is to introduce a dedicated, single-object metadata cache (`InfoCache`) to optimize workloads that make repetitive metadata API calls (`info()`, `exists()`, `modified()`, `size()`).

**The Problem:**
Currently, filesystems like GCSFS rely heavily on `dircache` (a directory-level cache) to reduce expensive network calls. When a directory is listed, its contents are cached. Subsequent `info()` calls for files *within* that directory can be resolved locally in $O(N)$ time. However, when users perform direct, point-lookups on specific files without first listing the directory, the filesystem makes direct API calls (e.g., `storage.objects.get`).

In AI/ML workloads (such as PyTorch `DataLoader`s, Dask array operations, or Ray datasets), systems frequently probe files by calling `exists()`, `info()`, `modified()`, or `size()` on individual, known paths. Without a cache, this results in massive redundant network overhead, increasing latency and Cloud API costs, particularly in high-throughput distributed training environments.

**The Solution:**
The new `InfoCache` will specifically store the results of single-file `info()` metadata calls. It will mirror the semantic behavior of the existing `fsspec.dircache.DirCache`—providing Time-To-Live (TTL) expiration and bounded LRU (Least Recently Used) eviction—but tailored exclusively for single-path lookups rather than hierarchical directory traversals. Crucially, the info cache will *only* be populated by `_info()` successes and *not* by directory listing operations like `ls()` or `find()`.

---

## 2. Key Design Decisions and Trade-offs

### 2.1 Code Location: GCSFS vs. `fsspec`
*Decision:* **Option B: Implement in `fsspec` [Selected]**

The design question is whether the cache should live inside `gcsfs` or in `fsspec` (in a form that async and sync backends can both adopt).

*   **Option A: Implement in `gcsfs`**
    *   *Pros:* 
        1. **Matches the real hook point.** `gcsfs` already owns the async `_info()` implementation containing GCS-specific race logic. Caching here is straightforward.
        2. **Fastest path to user value.** Can land in one repo and be benchmarked immediately.
        3. **Sharper invalidation model.** Allows exact-object invalidation alongside existing parent-listing invalidation without retrofitting across all `fsspec` backends.
    *   *Cons:*
        1. **No immediate ecosystem benefit.** Other backends won't gain info caching automatically.
        2. **Some duplication.** Needs its own cache class or a local adaptation of `DirCache` semantics.

*   **Option B: Implement in `fsspec` [Selected]**
    *   *Pros:*
        1. **Ecosystem Win.** Multiple backends (S3, Azure, local, GCS) can converge on one cache API and implementation.
        2. **Shared Primitives.** A generic TTL/LRU helper serves both listings and info caching, minimizing code duplication.
        3. **Consistent User-facing Knobs.** Aligns `use_info_cache`, `info_expiry_time`, and `max_info_paths` across all implementations.
    *   *Cons:*
        1. **Generation safety.** GCS uses version generation, meaning the path alone is not a sufficient cache key. 
        *   *Mitigation:* This is easily addressed by implementing the `InfoCache` generically in `fsspec`, while allowing specific filesystem implementations (like `GCSFileSystem`) to override a helper method (e.g., `_get_info_cache_key(path)`) to inject `generation` or other backend-specific identifiers into the cache key.
        2. **Backend-specific invalidation.** Each backend still needs to explicitly evict object entries on writes. This requires discipline when integrating the cache into various `fsspec` backends.

### 2.2 Implementation Structure
*Decision:* **Create a new `fsspec.infocache.py` sharing code with `dircache.py` [Selected]**
*   **Rationale:** Keeps the `InfoCache` class cleanly separated from `fsspec.caching` (which is historically focused on byte-buffer chunking for file content). `InfoCache` handles metadata state, TTL, and LRU eviction similarly to `DirCache`.

### 2.3 Cache Policy Defaults
*Decision:* **`max_paths=100000`, `info_expiry_time=None` (Infinite).**
*   **Rationale:** A 100k limit bounds memory usage to ~100MB, preventing OOMs. Infinite TTL aligns with existing `dircache` ergonomics and maximizes read-heavy workload performance. Users with mutable workloads must explicitly configure a TTL.

### 2.4 Negative Caching
*Decision:* **Disable Negative Caching (Do not cache `FileNotFoundError`).**
*   **Rationale:** Safest for distributed, multi-writer environments. Prevents cache poisoning where one node temporarily failing to find an object permanently caches the absence, even after another node creates it.

### 2.5 Metadata Fields
*Decision:* **Cache the entire metadata dictionary.**
*   **Rationale:** Guarantees semantic equivalence between a cache hit and a direct API call. Downstream libraries may rely on obscure metadata fields (e.g., `customTime`, ACLs).

---

## 3. Component Design: `fsspec.infocache` Implementation

We will implement a new class, `InfoCache`, within a new `fsspec/infocache.py` file. It will share design patterns with `DirCache`.

### 3.1 Code Implementation (`fsspec/infocache.py`)

```python
import time
from collections.abc import MutableMapping
from functools import lru_cache

class InfoCache(MutableMapping):
    """
    Caching of single-object metadata (e.g., from info() calls).
    
    Structure:
        {"key": {"name": "path", "size": 123, "type": "file", ...}, ...}
        
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
        
        if self.max_paths:
            self._q = lru_cache(self.max_paths + 1)(lambda key: self._cache.pop(key, None))

    def __getitem__(self, item):
        if self.info_expiry_time is not None:
            if self._times.get(item, 0) - time.time() < -self.info_expiry_time:
                del self._cache[item]
                raise KeyError(item)
                
        if self.max_paths:
            self._q(item)
            
        return self._cache[item]

    def __setitem__(self, key, value):
        if not self.use_info_cache:
            return
            
        if self.max_paths:
            self._q(key) 
            
        self._cache[key] = value
        
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

## 4. Integrating with `fsspec` and `GCSFileSystem`

### 4.1 Base `AbstractFileSystem` Update
In `fsspec.spec.AbstractFileSystem`, we will provision the `InfoCache` identically to how `DirCache` is provisioned. We will also add a helper method to generate the cache key, which subclasses can override.

```python
# In fsspec/spec.py
from fsspec.infocache import InfoCache
from fsspec.dircache import DirCache

class AbstractFileSystem:
    def __init__(self, *args, **storage_options):
        # ... existing init ...
        
        self.dircache = DirCache(**storage_options)
        
        # New instantiation
        self.infocache = InfoCache(
            use_info_cache=storage_options.get("use_info_cache", True),
            info_expiry_time=storage_options.get("info_expiry_time", None),
            max_paths=storage_options.get("info_max_paths", 100000),
        )

    def _get_info_cache_key(self, path, **kwargs):
        """
        Derive the key for the info cache. Subclasses can override this
        to incorporate versioning (e.g., generation) into the key.
        """
        return path
```

### 4.2 Cache Key Formulation in `gcsfs`
GCS heavily relies on object versioning (Generations). Returning cached metadata for a path without verifying the object's generation could lead to serving stale data. `GCSFileSystem` will override the key generation helper to handle this.

```python
# In gcsfs/core.py

class GCSFileSystem(asyn.AsyncFileSystem):
    # ...

    def _get_info_cache_key(self, path, generation=None, **kwargs):
        """
        Override the base cache key generator to include GCS generation.
        """
        path = self._strip_protocol(path).rstrip("/")
        bucket, key, path_generation = self.split_path(path)
        resolved_generation = _coalesce_generation(generation, path_generation)
        
        if resolved_generation:
            return f"{path}#{resolved_generation}"
        return path
```

---

## 5. Modifying the `_info()` Execution Flow in `gcsfs`

The `_info()` method in `gcsfs.core.GCSFileSystem` will be updated to intercept calls immediately after basic path normalization.

```python
    async def _info(self, path, generation=None, **kwargs):
        """File information about this path."""
        path = self._strip_protocol(path).rstrip("/")
        
        # ... (Existing logic for root bucket checks) ...

        bucket, key, path_generation = self.split_path(path)
        resolved_generation = _coalesce_generation(generation, path_generation)
        
        # -------------------------------------------------------------
        # 1. InfoCache Lookup (O(1) dictionary hit)
        # -------------------------------------------------------------
        cache_key = self._get_info_cache_key(path, generation=generation)
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
                    # 4. Populate InfoCache on Success
                    # -------------------------------------------------------------
                    self.infocache[cache_key] = get_object_res
                    return get_object_res
                    
            except FileNotFoundError:
                # Negative Caching Consideration: Do not cache missing files
                pass
                
            return await get_directory_info_task
```

---

## 6. Cache Consistency and Invalidation Surface

`GCSFileSystem` currently uses `invalidate_cache(path)` to clear directory listings. Mutative operations must now explicitly invalidate the exact object from the `infocache` in addition to invalidating the parent directory listing.

### 6.1 `invalidate_info` Helper
We introduce a targeted method for single-file metadata eviction.

```python
    def invalidate_info(self, path, generation=None):
        """
        Invalidate exact object entries in the info cache.
        If generation is None, evicts all generations for the path.
        """
        path = self._strip_protocol(path).rstrip("/")
        if generation is not None:
            self.infocache.pop(self._get_info_cache_key(path, generation=generation), None)
        else:
            # Evict all generations for this path
            keys_to_delete = [k for k in self.infocache if k == path or k.startswith(f"{path}#")]
            for k in keys_to_delete:
                self.infocache.pop(k, None)
```

### 6.2 Broad Mutation Surface Updates
We must insert calls to `self.invalidate_info(path)` in all metadata-mutating and object-creating/deleting methods in `gcsfs/core.py`.

*   **`_rm_file(self, path, ...)`**: Add `self.invalidate_info(path)`.
*   **`_rm_files(self, paths, ...)`**: Add `[self.invalidate_info(p) for p in paths]`.
*   **`_mv_file_cache_update(self, path1, path2, ...)`**: Add `self.invalidate_info(path1)` and `self.invalidate_info(path2)`.
*   **`_cp_file(self, path1, path2, ...)`**: Add `self.invalidate_info(path2)`.
*   **`_put_file(self, lpath, rpath, ...)`**: Add `self.invalidate_info(rpath)`.
*   **`_pipe_file(self, path, ...)`**: Add `self.invalidate_info(path)`.
*   **`_merge(self, path, ...)`**: Add `self.invalidate_info(path)`.
*   **`_setxattrs(self, path, ...)`**: Add `self.invalidate_info(path)` (since `info()` returns these attributes).

*Note: The existing calls to `self.invalidate_cache(self._parent(path))` inside these methods will remain untouched to ensure `dircache` listing invalidation and `fsspec` transaction-delayed invalidation stay correct.*

### 6.3 Handling `refresh=True`
Calls like `ls(refresh=True)` or `info(refresh=True)` invoke `self.invalidate_cache(path)`. To keep behavior consistent, `invalidate_cache` should completely wipe `infocache` entries falling under that directory hierarchy.

```python
    def invalidate_cache(self, path=None):
        if path is None:
            self.dircache.clear()
            self.infocache.clear()
        else:
            path = self._strip_protocol(path).rstrip("/")
            # Wipe exact info matches or descendants
            keys_to_delete = [
                k for k in self.infocache 
                if k == path or k.startswith(f"{path}#") or k.startswith(f"{path}/")
            ]
            for k in keys_to_delete:
                self.infocache.pop(k, None)

            # ... existing dircache while loop ...
            # IMPORTANT: Call super().invalidate_cache(path) if fsspec requires 
            # transaction registration (gcsfs currently handles transactions via 
            # `self._invalidated_caches_in_transaction.append(path)`).
```

---

## 7. Comprehensive Test Plan

Testing must rigorously validate both the mechanical constraints of the cache class itself and the network-saving behavior within the filesystem integration.

### 7.1 Unit Tests (`fsspec/tests/test_infocache.py`)
1. **LRU Eviction Constraint:** Create an `InfoCache` with `max_paths=3`. Insert 4 items. Assert size is 3 and oldest is evicted.
2. **TTL Expiration:** Create `InfoCache` with `info_expiry_time=1`. Insert item. Sleep 1.1s. Assert `KeyError` on retrieval.

### 7.2 Integration Tests (`gcsfs/tests/test_core.py`)
1. **Network Bypass (Cache Hit):** 
   * `fs.info(path)` -> API hit. 
   * `fs.info(path)` -> No API hit.
2. **`ls()` Does Not Populate InfoCache:**
   * `fs.ls(dir_path)` -> Populates `dircache`.
   * Assert `fs.infocache` is empty.
3. **Exact-Entry Invalidation across Mutation Surface:**
   * **Write Ops:** Call `_pipe_file()`, `_put_file()`, `_cp_file()`, `_merge()`. Assert `invalidate_info` is triggered for the destination, forcing the next `info()` to hit the API.
   * **Delete Ops:** Call `_rm_file()`, `_rm_files()`. Assert `invalidate_info` is triggered for the targets.
   * **Move Ops:** Call `_mv_file()`. Assert `invalidate_info` is triggered for both source and destination.
   * **Metadata Ops:** Call `_setxattrs()`. Assert `invalidate_info` is triggered so subsequent `info()` returns the updated metadata.
4. **Generation Separation:**
   * `fs.info(path, generation="111")`
   * `fs.info(path, generation="222")`
   * Assert two distinct API calls occurred and two separate entries exist in `infocache`.