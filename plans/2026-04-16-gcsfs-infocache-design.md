# Low-Level Design (LLD): Metadata InfoCache

## 1. Objective and Motivation
The objective is to introduce a dedicated single-object metadata cache (`InfoCache`) for repeated metadata lookups such as `info()`, `exists()`, `modified()`, and `size()`.

**The problem**

Today, `gcsfs` gets most of its metadata reuse from `dircache`, which is populated by listing operations. If a caller already listed a directory, later `info()` calls for entries in that directory can often be answered from the cached listing. But direct point-lookups on known object paths still go to the object metadata API path, which is expensive in AI/ML workloads that probe the same files repeatedly.

Those workloads are common in `DataLoader`s, dataset manifests, sharded training jobs, and distributed schedulers. They often issue many repeated single-path metadata calls without ever listing the parent directory first.

**The solution**

Introduce a dedicated `InfoCache` for successful single-object metadata fetches. The revised design intentionally stays simple:

- opt-in by default, so freshness does not silently change
- primary cache keyed by normalized path only
- generation stored inside the cached value, not in the key
- no always-on prefix index
- exact-path invalidation on the hot mutation path
- subtree invalidation only for explicit refresh/manual invalidation paths

The cache is populated by successful point lookups. It is not bulk-populated by `ls()` or `find()`.

### 1.1 Concrete Workload Evidence

This section documents the actual call chains in real ML frameworks that produce the repeated `_info()` calls the cache is designed to absorb. All line numbers were verified against the repositories as of April 2026.

#### Why repeated calls happen: the fsspec method surface

Every high-level fsspec method that answers a metadata question delegates independently to `_info()`. None of them share a result with the others:

| Caller | fsspec location | Leads to |
|---|---|---|
| `fs.exists(path)` | `spec.py:668` | `self.info(path)` → `_info()` RPC |
| `fs.size(path)` | `spec.py:728` | `self.info(path)` → `_info()` RPC |
| `fs.isfile(path)` | `spec.py:744` | `self.info(path)` → `_info()` RPC |
| `fs.isdir(path)` | `spec.py:737` | `self.info(path)` → `_info()` RPC |
| `fs.modified(path)` | `core.py:1039` | `self.info(path)` → `_info()` RPC |
| `fs.open(path, 'rb')` | `spec.py:1926` | `self.details["size"]` → `GCSFile.details` → `_info()` RPC |
| `fs.cat(path)` | `core.py:1203` | `await self._info(path)` directly |

`_open()` ([core.py:1902](gcsfs/core.py:1902)) never forwards an already-fetched metadata dict to `GCSFile`, so `GCSFile.details` ([core.py:2152](gcsfs/core.py:2152)) always fires a fresh `_info()` on first access regardless of what the caller fetched moments before.

#### HuggingFace `datasets` — `DatasetDict.load_from_disk`

**Repository:** https://github.com/huggingface/datasets/blob/main/src/datasets/dataset_dict.py  
**Lines:** 1418–1432

```python
fs, dataset_dict_path = url_to_fs(dataset_dict_path, **(storage_options or {}))

dataset_dict_json_path  = posixpath.join(dataset_dict_path, config.DATASETDICT_JSON_FILENAME)
dataset_state_json_path = posixpath.join(dataset_dict_path, config.DATASET_STATE_JSON_FILENAME)
dataset_info_path       = posixpath.join(dataset_dict_path, config.DATASET_INFO_FILENAME)

if not fs.isfile(dataset_dict_json_path):          # → _info() RPC #1
    if fs.isfile(dataset_info_path) and \          # → _info() RPC (different path)
       fs.isfile(dataset_state_json_path):         # → _info() RPC (different path)
        raise FileNotFoundError(...)
    raise FileNotFoundError(...)

with fs.open(dataset_dict_json_path, "r") as f:    # → GCSFile.details → _info() RPC #2
    splits = json.load(f)["splits"]
```

`dataset_dict_json_path` is probed at line 1423 by `isfile()` and again at line 1432 when `open()` initialises `GCSFile.details`. Same exact GCS path, two independent `_info()` calls. With cache: 2 RPCs → 1.

#### HuggingFace `datasets` — `Dataset.load_from_disk` (3-probe burst)

**Repository:** https://github.com/huggingface/datasets/blob/main/src/datasets/arrow_dataset.py  
**Lines:** 2021–2030

```python
fs, dataset_path = url_to_fs(dataset_path, **(storage_options or {}))

dataset_dict_json_path  = posixpath.join(dest_dataset_path, config.DATASETDICT_JSON_FILENAME)
dataset_state_json_path = posixpath.join(dest_dataset_path, config.DATASET_STATE_JSON_FILENAME)
dataset_info_path       = posixpath.join(dest_dataset_path, config.DATASET_INFO_FILENAME)

dataset_dict_is_file  = fs.isfile(dataset_dict_json_path)   # → _info() RPC on path A
dataset_info_is_file  = fs.isfile(dataset_info_path)        # → _info() RPC on path B
dataset_state_is_file = fs.isfile(dataset_state_json_path)  # → _info() RPC on path C
```

Three `_info()` RPCs fire unconditionally on every call. These are three distinct paths, so a single call does not collapse them. However, when a training loop calls `load_from_disk` repeatedly (e.g., one evaluation step per 500 training steps), paths A, B, and C are re-probed each time. After the first call warms the cache, every subsequent call hits for all three paths: 3 RPCs per call → 0.

After line 2057 (`if is_remote_filesystem(fs):`), the code downloads to a local temp directory and switches to native `open()`, so the actual file reads do not go through GCS again. The `isfile` burst is the only GCS metadata cost from this function.

#### Compounding case: HuggingFace `datasets` loaded in a training loop

The most common way a GCS-backed dataset produces repeated metadata RPCs is the standard training loop pattern: a fixed dataset is loaded once per evaluation step, because the caller passes the GCS path directly rather than caching the in-memory `Dataset` object.

**Concrete training loop (common pattern):**

```python
EVAL_DATASET_PATH = "gs://bucket/data/eval/"   # 2-split DatasetDict (train / validation)

for step, batch in enumerate(train_loader):
    model.train_step(batch)

    if step % EVAL_EVERY == 0:
        # Re-loaded from GCS on every evaluation step.
        # Typical in frameworks that rebuild the eval dataset from config
        # each time (e.g. HuggingFace Trainer with evaluate() called per step,
        # or custom loops that load from path rather than keeping a handle).
        eval_ds = load_from_disk(EVAL_DATASET_PATH)
        metrics = evaluate(model, eval_ds)
        log(step, metrics)
```

**Call chain per `load_from_disk` call** (`DatasetDict.load_from_disk` → `Dataset.load_from_disk` × 2 splits):

```
load_from_disk("gs://bucket/data/eval/")
│
├─ DatasetDict.load_from_disk  [dataset_dict.py:1418–1432]
│   fs.isfile("gs://bucket/data/eval/dataset_dict.json")          ← isfile → _info() #1
│   fs.open ("gs://bucket/data/eval/dataset_dict.json")           ← open   → _info() #2
│
├─ Dataset.load_from_disk("gs://bucket/data/eval/train/")  [arrow_dataset.py:2021–2030]
│   fs.isfile("gs://bucket/data/eval/train/dataset_dict.json")    ← _info() #3
│   fs.isfile("gs://bucket/data/eval/train/dataset_info.json")    ← _info() #4
│   fs.isfile("gs://bucket/data/eval/train/state.json")           ← _info() #5
│
└─ Dataset.load_from_disk("gs://bucket/data/eval/validation/")  [arrow_dataset.py:2021–2030]
    fs.isfile("gs://bucket/data/eval/validation/dataset_dict.json") ← _info() #6
    fs.isfile("gs://bucket/data/eval/validation/dataset_info.json") ← _info() #7
    fs.isfile("gs://bucket/data/eval/validation/state.json")        ← _info() #8
```

**Per-step RPC count across a training run:**

| Eval step | Without cache | With cache (60s TTL) | Notes |
|---|---|---|---|
| Step 0 (first) | 8 RPCs | 8 RPCs | Cold — all misses, populates cache |
| Steps 1–N (within TTL) | 8 RPCs each | 1 RPC each | #1 refetches; #2 hits #1; #3–#8 all hit |
| First step after TTL expires | 8 RPCs | 8 RPCs | Full re-fetch, warms cache again |

For a training run with `eval_every=500` steps over 10 000 steps, that is 20 evaluation steps. Without cache: 160 RPCs. With cache and a 60s TTL longer than the eval interval: 8 + 19 × 1 = **27 RPCs total**.

**Why RPC #1 still fires on each warm step:** `dataset_dict.json` is the first call in `DatasetDict.load_from_disk`. It fires `isfile()` on that path (RPC #1), then immediately opens the same path (RPC #2 → cache hit). RPCs #3–#8 are all hits because the six per-split paths were cached from the previous eval step. The single surviving RPC is unavoidable without a higher-level dataset-identity cache outside of `gcsfs`.

**Why `is_remote_filesystem` does not help here:** `Dataset.load_from_disk` checks `is_remote_filesystem(fs)` (line 2057) and downloads the dataset files to a local temp directory before reading them with native `open()`. So the actual Arrow/Parquet file reads are already local. The 8 metadata RPCs are *pure existence checks* — no data is transferred through them. Eliminating them with the cache removes latency that has no data-transfer justification.

#### Training: checkpoint resume (general pattern)

This pattern appears across frameworks (PyTorch Lightning, HuggingFace Trainer, custom training loops). The typical resume-from-checkpoint sequence:

```python
ckpt = "gs://bucket/run-42/checkpoints/step_10000.pt"

# 1. Existence guard (framework or user code)
if not fs.exists(ckpt):               # spec.py:668 → _info() RPC #1
    start_fresh()

# 2. Log metadata before loading
size  = fs.size(ckpt)                 # spec.py:728 → _info() RPC #2
mtime = fs.modified(ckpt)             # core.py:1039 → _info() RPC #3

# 3. Open and load
with fs.open(ckpt, 'rb') as f:        # spec.py:1926 → GCSFile.details → _info() RPC #4
    checkpoint = torch.load(f)
```

4 RPCs on the same path within milliseconds. With cache: 1 RPC, 3 hits. The gap between step 2 and step 3 is always sub-second (no I/O between them), so all four fall inside the 60s default TTL.

#### Inference: model cold start (general pattern)

A serving container (TorchServe, vLLM, Triton with GCS backend) loading weights on startup:

```python
model_uri = "gs://bucket/models/llama-3-8b/model.safetensors"

if not fs.isfile(model_uri):          # spec.py:744 → _info() RPC #1
    raise RuntimeError("Model artifact missing")

model_size = fs.size(model_uri)       # spec.py:728 → _info() RPC #2

with fs.open(model_uri, 'rb') as f:   # spec.py:1926 → GCSFile.details → _info() RPC #3
    model = safetensors.torch.load_stream(f)
```

3 RPCs → 1 with cache. If multiple threads initialise simultaneously (e.g., a serving framework spinning up replica workers in the same process), each thread fires the same sequence before any result is cached, yielding up to 3N RPCs on the same path. The cache reduces steady-state cost to near-zero but does not coalesce the cold-start stampede (§2.10).

#### What does NOT benefit from the cache

For completeness, patterns that look similar but do not produce repeated calls on the same path:

- **HuggingFace Transformers `Trainer._load_from_checkpoint`**: Uses `os.path.isfile()` (local filesystem), not `fs.isfile()`. The Trainer downloads checkpoint files to local storage first; all subsequent operations are local-disk I/O.
- **PyTorch Lightning version-counter loop**: Calls `fs.exists(filepath_with_version_N)` in a while loop, but each iteration increments the version suffix, producing a distinct path. No path is repeated.
- **Ray Train `_delete_fs_path`**: Uses PyArrow's filesystem API (`fs.get_file_info()`), not fsspec. Different subsystem.

---

## 2. Key Design Decisions and Trade-offs

### 2.1 Code Location: `gcsfs` vs. `fsspec`
*Decision:* **Implement in `fsspec` first [Selected]**

The cache should live in `fsspec`, with backend-specific hooks for path normalization and backend-specific `_info()` integration.

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
        3. Allows `gcsfs`, `s3fs`, and other backends to share the same tested bounded-cache implementation while still customizing lookup semantics.
    *   *Cons:*
        1. The base abstraction must stay small enough that listing-only backends are not forced into unnecessary machinery.
        2. Each backend still needs a full mutation-surface audit.

### 2.2 Implementation Structure
*Decision:* **Add `fsspec.infocache.InfoCache` with path-keyed primary storage [Selected]**

`InfoCache` should be a dedicated metadata cache rather than an adaptation of byte-range caching. It should share high-level policy concepts with `DirCache` (optional use, TTL, bounded size), but it should not copy `DirCache` internals or introduce more indexing than the problem needs.

### 2.3 Cache Policy Defaults
*Decision:* **`use_info_cache=False`, `max_info_paths=100000`, `info_expiry_time=60` [Selected]**

*   **Rationale:**
    1. Disabling the cache by default preserves current freshness behavior for direct point-lookups.
    2. `max_info_paths` bounds the number of live cached metadata entries.
    3. When users opt in, a 60-second default TTL keeps the feature safe in mildly mutable environments while still absorbing repeated-probe patterns. Immutable datasets can set `info_expiry_time=None` explicitly.

### 2.3.1 Instance Sharing via `_Cached`

`fsspec.spec._Cached` caches `AbstractFileSystem` instances per storage-option tuple, so two callers with the same options share one `InfoCache`. This is intentional — it is how a worker process accumulates benefit from its own previous probes — but the design must account for it:

- The first constructor call wins: if one caller passes `use_info_cache=True` and another passes `use_info_cache=False` with otherwise identical options, they share the instance configured by the first call.
- The three new options (`use_info_cache`, `info_expiry_time`, `max_info_paths`) must participate in `_Cached`'s instance key so differing configurations yield distinct instances. They flow through `**storage_options` and are thus already part of the identity tuple — the test plan (§7.1) covers this explicitly.

### 2.3.2 Cross-Process Consistency

`InfoCache` is per-process. Multi-worker `DataLoader`s, sharded training jobs, and any setup spawning subprocesses each get an independent cache, so a write in one process is invisible to cached reads in another. This is explicitly in scope for the target workloads (read-mostly manifests and dataset shards) but out of scope for mutating distributed pipelines. The default 60s TTL plus opt-in gating is the design's only safety net; users running multi-writer workloads should either disable the cache or use a short TTL.

### 2.4 Negative Caching
*Decision:* **Do not cache `FileNotFoundError` [Selected]**

*   **Rationale:** This remains the safest behavior in distributed and multi-writer environments. Negative caching can easily outlive the absence it observed.
*   **Acknowledged trade-off:** ML manifest-probing workloads often issue bursty `exists()` checks against shards that do not yet exist (e.g. waiting for a producer). Without negative caching, every probe pays full RPC cost. We accept this for v1 — a short-TTL negative cache is a clean follow-up once we have hit-rate data and a concrete user complaint to size it against.

### 2.5 Metadata Fields
*Decision:* **Cache the full metadata dictionary [Selected]**

*   **Rationale:** A cache hit should be semantically equivalent to a successful direct metadata resolution. This includes user metadata, content settings, generations, and backend-specific attributes.

### 2.6 Key Shape
*Decision:* **Primary cache keyed by normalized path only [Selected]**

*   **Rationale:** For the common case, the cache question is “what metadata do I currently know about this path?”, not “which `(path, generation)` tuple do I have?”. Keying by path alone removes compound-key aliasing and keeps one primary slot per object.

### 2.7 Generation Semantics
*Decision:* **Store generation in the cached value and compare on pinned probes [Selected]**

Pinned-generation lookups work like this:

- if the cached entry for `path` has `generation == requested_generation`, it is a hit
- otherwise it is a miss and the backend refetches

This is the correct semantic for “I asked for version X; you only know version Y”.

### 2.8 Simultaneous Multi-Generation Caching
*Decision:* **Do not support concurrent caching of multiple generations of the same path in v1 [Selected]**

*   **Rationale:** This is the main trade-off of the simpler design. A workload alternating between `file@v1` and `file@v2` will miss unless the currently cached entry happens to match. That is acceptable for v1 because the dominant target workload is unpinned point-lookups. If pinned-generation reuse becomes important, a sparse `_versioned[(path, generation)]` overlay can be added later without changing the primary cache shape.

### 2.9 Invalidation Strategy
*Decision:* **Exact-path invalidation on the hot path, subtree scan only on explicit refresh/manual invalidation [Selected]**

*   **Rationale:** The write path should not pay for always-on ancestor indexing. Exact invalidation is O(1). Subtree invalidation is O(n) over the primary cache, but that cost is acceptable for explicit `refresh=True` and manual subtree invalidation paths.

### 2.10 Concurrency Model
*Decision:* **Match `dircache`'s current single-process, no-locks contract [Selected]**

*   **Rationale:** `InfoCache` does not add internal locks or request coalescing. Two concurrent misses may fetch and populate the same path twice; the last writer wins. This is a missed optimization, not a correctness bug.

### 2.11 Comparison: Path-Keyed vs. Compound-Key + Dual-Index

An earlier iteration used compound keys `(path, generation)` plus two secondary indexes (`_path_index: path → set[key]`, `_prefix_index: ancestor → set[paths]`). The switch to a single path-keyed `OrderedDict` is a deliberate simplification; this section records what was gained and what was given up so future reviewers can decide whether to re-expand.

| Dimension | Path-keyed (selected) | Compound-key + dual-index |
|---|---|---|
| Primary structure | 1 `OrderedDict[path, entry]` | `OrderedDict[(path, gen), entry]` + `_path_index` + `_prefix_index` |
| Entries per cached object (common case) | 1 | 2 (unqualified + generation-qualified alias) |
| Effective capacity at `max_paths=100000` | 100k distinct objects | ~50k distinct objects (halved by alias) |
| Exact-path invalidation | O(1) dict pop | O(versions) via `_path_index` |
| Subtree invalidation | O(n) scan over live entries | O(matching descendants) via `_prefix_index` |
| Bookkeeping cost per insert | O(1) | O(depth): one set update per ancestor |
| Bookkeeping cost per eviction | O(1) | O(depth): one set update per ancestor |
| Invariants to preserve | just dict membership | `_entries` ↔ `_path_index` ↔ `_prefix_index` must all agree |
| Backend hook surface | `_get_info_cache_path` only | `_get_info_cache_path` + `_get_info_cache_key` |
| Multi-generation concurrent caching | one version per path | two or more versions per path |
| Generation-aliasing dual-insert | not needed | needed, must be cleaned up together |
| Approximate lines of core cache code | ~60 | ~150 |

**Why path-keyed wins for this workload.**

1. **Hot-path simplicity.** `_info()` runs millions of times per DataLoader epoch. An O(1) lookup plus O(1) eviction step beats updating three structures per write. The ancestor-index maintenance the old design paid on every insert is pure overhead in the common case, and it scales with path depth.
2. **Effective capacity.** The compound-key design populated both `(path, None)` and `(path, actual_gen)` on every fetch so a later pinned probe would hit. That was two slots per object. The path-keyed design covers the same semantic with one slot plus a generation-field check at lookup — so `max_paths=100000` actually holds 100k objects.
3. **Fewer invariants, fewer bugs.** The old `_unlink` had to prune `_entries`, then `_path_index`, then (conditionally, only if no other generation for the same path was still live) each ancestor in `_prefix_index`, in the right order, under eviction and expiration and explicit invalidation. The new design has one structure; `_entries.pop(path, None)` cannot leave the cache inconsistent.
4. **Simpler backend contract.** Backends override only `_get_info_cache_path`. The `_get_info_cache_key` hook is deleted entirely.

**What we give up.**

1. **Two live generations of the same path.** The compound-key design could cache `file@gen1` and `file@gen2` simultaneously. Path-keyed holds one. A workload that alternates pinned reads between two generations of the same object would see both reads miss repeatedly. Acknowledged in §2.8; addressed by the sparse overlay sketched in §3.5 if that workload ever materializes.
2. **Sublinear subtree invalidation.** The old `pop_prefix` was O(matching descendants); the new `invalidate_subtree` is O(n) over live entries. §2.9 quantifies this (~5ms at 100k entries) and notes it runs only on explicit refresh, not on steady-state mutation.

**If we ever need to revert.** The migration back to a prefix index is local: add a `_prefix_index` dict, update `set()` and invalidation to maintain it, change `invalidate_subtree` to index-driven iteration. No API changes; no caller-side changes.

---

## 3. Component Design: `fsspec.infocache`

We will add a new `InfoCache` implementation in `fsspec/infocache.py`.

### 3.1 Design Goals

The implementation must satisfy all of the following:

1. Preserve opt-in semantics.
2. Keep the primary cache keyed by normalized path only.
3. Keep entry count bounded to live cached entries only.
4. Make exact invalidation trivial.
5. Keep subtree invalidation off the hot mutation path.

### 3.2 Data Model

Each live cache entry stores:

- `path`: normalized logical path
- `value`: metadata dictionary returned by `_info()`
- `created`: insertion timestamp for TTL checks

The primary structure is:

- `_entries: OrderedDict[path, CacheEntry]`
  - The single source of truth for live cached entries and LRU order.

There is no `_path_index` and no `_prefix_index`.

### 3.3 Code Sketch (`fsspec/infocache.py`)

```python
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheEntry:
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
        self.stats = CacheStats()

    def _is_expired(self, entry):
        if self.info_expiry_time is None:
            return False
        return (time.time() - entry.created) > self.info_expiry_time

    def get(self, path):
        if not self.use_info_cache:
            self.stats.misses += 1
            raise KeyError(path)

        try:
            entry = self._entries[path]
        except KeyError:
            self.stats.misses += 1
            raise
        if self._is_expired(entry):
            self._entries.pop(path, None)
            self.stats.expirations += 1
            self.stats.misses += 1
            raise KeyError(path)

        self._entries.move_to_end(path)
        self.stats.hits += 1
        return entry.value

    def set(self, path, value):
        if not self.use_info_cache:
            return

        self._entries[path] = CacheEntry(value=value, created=time.time())
        self._entries.move_to_end(path)

        while self.max_paths and len(self._entries) > self.max_paths:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def invalidate(self, path):
        self._entries.pop(path, None)

    def invalidate_subtree(self, prefix):
        if not prefix:
            # Empty prefix would otherwise match every path; require clear()
            # for full wipes so "nuke everything" is always an explicit call.
            return
        prefix = prefix.rstrip("/")
        if not prefix:
            return
        child_prefix = prefix + "/"
        to_delete = [
            path
            for path in self._entries
            if path == prefix or path.startswith(child_prefix)
        ]
        for path in to_delete:
            self._entries.pop(path, None)

    def clear(self):
        self._entries.clear()
```

### 3.4 Memory-Bound Guarantees

The memory-bound guarantee is simple:

- one live primary cache entry per cached path
- no side indexes that can outlive evictions
- expired entries are removed from `_entries` on first access

This keeps total bookkeeping proportional to live cached entries only.

### 3.5 Future Extension: Sparse Version Overlay

If pinned-generation workloads prove important, a sparse `_versioned[(path, generation)]` overlay can be added later. It would be populated only when a caller pins a generation different from the current primary entry. This is explicitly deferred from v1 to keep the implementation small.

---

## 4. Integrating with `fsspec` and `GCSFileSystem`

### 4.1 Base `AbstractFileSystem` Update

Because this feature is implemented in `fsspec`, the design must account for the inherited method surface in `fsspec.spec.AbstractFileSystem` and `fsspec.asyn.AsyncFileSystem`, not just backend overrides:

- `AbstractFileSystem.exists()` delegates to `info()`
- `AbstractFileSystem.size()` delegates to `info()`
- `AbstractFileSystem.isdir()` delegates to `info()`
- `AbstractFileSystem.isfile()` delegates to `info()`
- `AbstractFileSystem.checksum()` delegates to `info()`
- `AsyncFileSystem._exists()` delegates to `_info()`
- `AsyncFileSystem._size()` delegates to `_info()`

That means:

1. Backends that override `_info()` or `info()` can pick up `InfoCache` benefits through those inherited methods.
2. The base class contract must define path normalization and explicit info-cache invalidation helpers.
3. `invalidate_cache()` must keep its existing transaction bookkeeping role.

### 4.2 Base Hooks and Helpers

`AbstractFileSystem` provisions `InfoCache` and exposes path-level helpers.

```python
# In fsspec/spec.py — additions to the existing AbstractFileSystem class
from fsspec.infocache import InfoCache


class AbstractFileSystem:
    def __init__(self, *args, **storage_options):
        # ... existing init body ...
        use_info_cache = storage_options.pop("use_info_cache", False)
        info_expiry_time = storage_options.pop("info_expiry_time", 60)
        max_info_paths = storage_options.pop("max_info_paths", 100000)
        self.infocache = InfoCache(
            use_info_cache=use_info_cache,
            info_expiry_time=info_expiry_time,
            max_paths=max_info_paths,
        )

    def _get_info_cache_path(self, path, **kwargs):
        return self._strip_protocol(path).rstrip("/")

    def invalidate_info(self, path, **kwargs):
        self.infocache.invalidate(self._get_info_cache_path(path, **kwargs))

    def invalidate_info_subtree(self, path, **kwargs):
        self.infocache.invalidate_subtree(self._get_info_cache_path(path, **kwargs))

    def clear_info_cache(self):
        self.infocache.clear()

    def invalidate_cache(self, path=None):
        # ... existing transaction bookkeeping and dircache handling ...
        if path is None:
            # Full wipe: the no-arg form is semantically "clear all cached
            # state" — pair dircache clear with infocache clear so callers
            # don't have to know about two buttons.
            self.infocache.clear()
        # Note: on a path argument, do NOT touch infocache — see §4.3.
```

### 4.3 Base `invalidate_cache()` Contract

`AbstractFileSystem.invalidate_cache(path=None)` keeps its existing transaction behavior and listing-cache role. It intentionally does **not** auto-evict `InfoCache` on a `path` argument, because many backends already call `invalidate_cache(parent)` on the mutation path for `dircache`, and auto-evicting would turn every write into an O(n) subtree scan.

The specific semantics are:

- `invalidate_cache(path)` — unchanged. Listing-cache invalidation and transaction bookkeeping only. Does not touch `InfoCache`.
- `invalidate_cache(None)` — full wipe. Callers invoking the no-arg form expect a "nuke everything" button, so the base implementation ALSO calls `self.infocache.clear()` here. This is a one-shot O(n) that users already accept when they ask for a global invalidation.
- `invalidate_info(path)` — the hot-path exact invalidation primitive for mutations.
- `invalidate_info_subtree(path)` — used only for explicit refresh/manual subtree invalidation.

The asymmetry between `invalidate_cache(path)` (leaves infocache alone) and `invalidate_cache(None)` (clears infocache) is deliberate: the no-arg form is rare and semantically "clear all state"; the path form is frequent and would be prohibitively expensive if it scanned. Callers doing a targeted refresh should pair `invalidate_cache(path)` with an explicit `invalidate_info_subtree(path)` — see §6.3.

### 4.4 Cache Key Formulation in `gcsfs`

`gcsfs` inherits `_get_info_cache_path()` unchanged. No generation-aware key helper is needed in v1 because the primary cache is path-keyed.

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

    try:
        cached = self.infocache.get(cache_path)
        if resolved_generation is None:
            return cached
        if cached.get("generation") == resolved_generation:
            return cached
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
                if resolved_generation is None and not _is_directory_marker(entry):
                    self.infocache.set(cache_path, entry)
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
                if resolved_generation is None:
                    self.infocache.set(cache_path, result)
                return result
        except FileNotFoundError:
            pass

        return await get_directory_info_task
```

Notes:

- unpinned lookups can populate and hit the primary cache
- pinned-generation lookups can hit the primary cache only if the cached value's `generation` matches
- pinned-generation misses refetch, but do not overwrite the primary path slot in v1
- directory placeholders are not inserted into `InfoCache`

### 5.1 Required `fsspec` Method-Surface Updates

The `fsspec` design should explicitly document which methods benefit automatically once a backend adopts `InfoCache`:

#### Synchronous base methods in `fsspec.spec.AbstractFileSystem`

- `info(path, **kwargs)`
  - No mandatory generic cache insertion is required for listing-only backends, but a backend override may consult and populate `InfoCache` if it has a true single-object metadata fast path.
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

---

## 6. Cache Consistency and Invalidation Surface

`InfoCache` adds a second metadata state surface beside `dircache`, so every metadata-mutating path must explicitly invalidate exact object entries. This design chooses invalidation, not write-through updates.

### 6.1 Exact Invalidations on the Mutation Path

Use `invalidate_info(path)` for the following mutators. Line numbers are against the tip of `main` as of this plan and are a checklist for review — they will drift once edits land:

- `_rm_file(path, ...)` — [gcsfs/core.py:1425](gcsfs/core.py:1425)
- `_rm_files(paths, ...)` — [gcsfs/core.py:1435](gcsfs/core.py:1435)
- `_mv_file_cache_update(path1, path2, ...)` — [gcsfs/core.py:1386](gcsfs/core.py:1386) (invalidate both source and destination)
- `_cp_file(path1, path2, ...)` — [gcsfs/core.py:1352](gcsfs/core.py:1352) (invalidate destination)
- `_put_file(lpath, rpath, ...)` — [gcsfs/core.py:1627](gcsfs/core.py:1627)
- `_pipe_file(path, ...)` — [gcsfs/core.py:1569](gcsfs/core.py:1569)
- `_merge(path, ...)` — [gcsfs/core.py:1321](gcsfs/core.py:1321)
- `_setxattrs(path, ...)` — [gcsfs/core.py:1258](gcsfs/core.py:1258)

**Invalidation ordering.** `invalidate_info(path)` runs only **after** the mutating RPC returns successfully, matching the current `dircache` invalidation ordering. If the RPC raises, the call site does not invalidate — the cached metadata still reflects reality, and pre-RPC invalidation would drop a valid entry even when nothing actually changed.

### 6.2 Subtree Invalidations on Explicit Refresh Paths

Use `invalidate_info_subtree(path)` only when the caller explicitly asks for a subtree refresh, for example:

- `ls(path, refresh=True)`
- `info(path, refresh=True)` if supported by the backend API surface
- manual refresh APIs that are semantically recursive

This keeps O(n) subtree scans off the hot write path.

### 6.3 `gcsfs.invalidate_cache()` Integration

`gcsfs.invalidate_cache(path)` should continue to handle `dircache` the way it does today. It should not automatically evict `InfoCache`.

Instead:

- write and metadata mutation paths call `invalidate_info(path)`
- refresh/manual subtree paths call both `invalidate_cache(path)` and `invalidate_info_subtree(path)`
- global clear paths call both `invalidate_cache(None)` and `clear_info_cache()`

This preserves the current `dircache` semantics while keeping info-cache subtree scans rare.

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
- Any future mutating override added in `extended_gcsfs.py` must be included in the metadata-cache audit as part of code review.

### 6.5 Required Updates in `fsspec`

Since the implementation is `fsspec`-first, the design must explicitly cover these base-library changes:

- `fsspec.spec.AbstractFileSystem.__init__`
  - Provision `self.infocache`.
- `fsspec.spec.AbstractFileSystem`
  - Add `_get_info_cache_path(...)`.
- `fsspec.spec.AbstractFileSystem`
  - Add `invalidate_info(...)`.
- `fsspec.spec.AbstractFileSystem`
  - Add `invalidate_info_subtree(...)`.
- `fsspec.spec.AbstractFileSystem`
  - Add `clear_info_cache()`.
- `fsspec.spec.AbstractFileSystem.invalidate_cache(...)`
  - Keep transaction bookkeeping unchanged; do not add automatic info-cache subtree eviction.

Backends that only implement `ls()` and inherit the default `info()` are not required to adopt `InfoCache` immediately. The primary adoption path is for backends with a true object-level metadata hook (`_info()` or specialized `info()`).

### 6.6 Rollout Sequence

Because the abstraction lives in `fsspec` but the consumer is `gcsfs`, landing this feature requires a coordinated sequence:

1. **fsspec PR.** Land `fsspec/infocache.py`, the base-class hook and helpers (`_get_info_cache_path`, `invalidate_info`, `invalidate_info_subtree`, `clear_info_cache`, `self.infocache`), and the updated `invalidate_cache(None)` behavior. Include unit tests from §7.1.
2. **fsspec release.** Wait for an `fsspec` release that contains the changes, or pin a specific `fsspec` commit in the gcsfs dev environment for CI while waiting.
3. **gcsfs version bump.** Raise the `fsspec` floor in `gcsfs/pyproject.toml` to the release from step 2.
4. **gcsfs integration PR.** Update `_info()` for cache lookup and pinned-probe semantics; wire `invalidate_info` into the mutation surface (§6.1 and §6.4). Include integration tests from §7.2 and §7.3.
5. **Opt-in rollout.** Feature is off by default after step 4. Internal canary workloads flip `use_info_cache=True` and surface `fs.infocache.stats` for hit-rate validation. Default flip is a separate decision gated on observed hit rate and zero regression reports.

**Fallback if upstream review stalls.** If the fsspec PR is blocked or reshaped in ways that delay step 2, implement the cache inside `gcsfs` under `gcsfs/_infocache.py` with the same API, and provide the helpers as local overrides on `GCSFileSystem`. The internal API stays identical, so migrating to upstream `fsspec.infocache` later is a module rename plus removing the local copy.

---

## 7. Comprehensive Test Plan

Testing should validate both cache mechanics and repository-specific integration points.

### 7.1 Unit Tests (`fsspec/tests/test_infocache.py`)

1. **Entry-count eviction:** Create `InfoCache(max_paths=3)`, insert 4 live entries, assert only 3 remain and the least recently used entry is gone.
2. **TTL expiration:** Create `InfoCache(info_expiry_time=1)`, insert one item, sleep 1.1s, assert lookup raises `KeyError` and the entry is removed.
3. **Path-keyed overwrite:** Insert two values for the same path, assert the second replaces the first and cache length stays 1.
4. **Exact invalidation:** Cache entries for `a/x` and `a/y`, call `invalidate("a/x")`, assert only `a/x` is removed.
5. **Subtree invalidation:** Cache entries under `a/...` and `b/...`, call `invalidate_subtree("a")`, assert only `a` descendants are removed.
6. **Bounded bookkeeping:** Repeatedly insert and evict entries, then assert only live entries remain in `_entries`.
7. **Stats counters:** Assert hits, misses, evictions, and expirations update as expected.

### 7.2 Integration Tests (`gcsfs/tests/test_core.py`)

1. **Default-off behavior:** Construct `GCSFileSystem()` without `use_info_cache=True` and assert repeated direct `info()` calls still hit the network path.
2. **Opt-in cache hit:** Construct with `use_info_cache=True`, call unpinned `info(path)` twice, assert the second call bypasses the object metadata API.
3. **Listing does not bulk-populate info cache:** Call `ls(dir_path)` and confirm `infocache` stays empty.
4. **Parent-cache promotion:** Pre-populate `dircache` via `ls(parent)`, then call unpinned `info(parent/child)` and assert one entry is promoted into `infocache`.
5. **Pinned-generation primary hit:** Cache unpinned `info(path)` returning generation `G`, then call `info(path, generation=G)` and assert it hits the primary cache.
6. **Pinned-generation mismatch miss:** Cache unpinned `info(path)` returning generation `G1`, then call `info(path, generation=G2)` and assert it misses and refetches.
7. **Mutation invalidation:** Verify `_rm_file`, `_rm_files`, `_mv_file_cache_update`, `_cp_file`, `_put_file`, `_pipe_file`, `_merge`, and `_setxattrs` invalidate the expected exact info entries.
8. **Refresh invalidation:** Cache entries under a subtree, run an explicit refresh path, and assert `invalidate_info_subtree()` removes the expected entries.

### 7.3 Integration Tests (`gcsfs/tests/test_extended_gcsfs*.py`)

1. **HNS move path:** Exercise `extended_gcsfs._mv_file_cache_update` and assert source and destination info entries are invalidated even when listing cache is updated in place.
2. **Zonal put path:** Exercise `extended_gcsfs._put_file` and assert destination info cache is invalidated after upload.
3. **Zonal pipe path:** Exercise `extended_gcsfs._pipe_file` and assert destination info cache is invalidated after upload.
4. **Extended copy/merge paths:** Assert `_cp_file` and `_merge` invalidate destination info entries on success.

### 7.4 Hit-Rate Benchmark (`gcsfs/tests/benchmarks/test_infocache_hitrate.py`)

A deterministic microbenchmark, runnable locally and in CI in `gcsfs` mock mode, validates that the feature delivers the intended reduction in metadata RPCs:

1. Construct a `GCSFileSystem` with `use_info_cache=False` against a recorded in-process mock; issue a workload of N unique paths probed K times each in round-robin; count metadata RPCs.
2. Construct with `use_info_cache=True`; issue the same workload; count metadata RPCs and assert count ≈ N (first-touch only) and `fs.infocache.stats.hits == N * (K - 1)`.
3. Record the speedup and counter deltas in the benchmark output so regressions are visible in PR diffs. This is the primary signal for "the feature is doing what we said it would."

### 7.5 Additional Unit-Test Cases

Extend §7.1 with:

8. **`_Cached` option participation:** Instantiate `AbstractFileSystem` twice through `fsspec.filesystem(...)` with differing `use_info_cache`/`info_expiry_time`/`max_info_paths` tuples and assert two distinct instances are returned (storage-option hashing must route them separately).
9. **`invalidate_subtree` empty-prefix guard:** Call `invalidate_subtree("")` and `invalidate_subtree("/")` on a non-empty cache; assert both are no-ops (only `clear()` wipes everything).
10. **`invalidate_subtree` sibling isolation:** Cache `bucket/a` and `bucket/abc/file`; call `invalidate_subtree("bucket/a")`; assert `bucket/abc/file` is NOT evicted. Guards against a bare-`startswith` aliasing bug.
11. **`invalidate_cache(None)` clears infocache:** Populate infocache; call `fs.invalidate_cache()` with no argument; assert `len(fs.infocache._entries) == 0`. Covers the §4.3 asymmetry.
12. **`invalidate_cache(path)` does NOT touch infocache:** Populate infocache with entries under `path`; call `fs.invalidate_cache(path)`; assert those infocache entries are still present. Covers the §4.3 hot-path guarantee.

---

## 8. Summary

The revised design keeps the implementation in `fsspec` and materially simplifies the cache:

- cache disabled by default, safe 60s TTL when opt-in
- one primary entry per path — no compound key, no alias dual-insert
- generation compared in the value on pinned probes, never encoded in the key
- single `OrderedDict` — no `_path_index`, no `_prefix_index`, no cross-structure invariants
- O(1) exact-path invalidation on the mutation hot path
- O(n) subtree invalidation only on explicit refresh / manual invalidation
- `invalidate_cache(None)` is the "clear all" button and wipes infocache too; `invalidate_cache(path)` intentionally leaves infocache alone (§4.3)
- explicit mutation-surface coverage in both `core.py` and `extended_gcsfs.py` with line-anchored audit list
- built-in `CacheStats` counters for hit-rate validation in production
- concurrency stance matches `dircache` (no locks); request coalescing is a follow-up
- `_Cached` instance sharing and cross-process consistency documented explicitly
- concrete fsspec-first rollout sequence with a gcsfs-local fallback path
- explicit comparison table (§2.11) against the earlier compound-key + dual-index design, with a documented escape hatch if the simpler design ever needs to be reverted

This keeps the abstraction reusable across `fsspec` backends while staying closer to the actual workload shape the cache is meant to serve.
