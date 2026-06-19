# Findings

## Adjacent opportunities

1. `_fetch_range_split` still creates one asyncio task per requested chunk. The MRD pool bounds resources, but task objects and coroutine scheduling remain proportional to `len(chunk_lengths)`. Use a fixed worker pool or rolling window; a semaphore alone would not bound task creation.
2. Inventory-backed listing creates one coroutine per estimated 5,000-object page and gathers all of them. Large inventories can therefore launch hundreds or thousands of list requests at once. Add a bounded listing concurrency.
3. Generation handling remains incomplete beyond #921: zonal `_cat_file` and `_fetch_range_split` choose MRD pools using only path-embedded generation; standard concurrent size lookup and negative-range metadata lookup do not receive explicit generation; `GCSFile.url` also needs forwarding and is already noted in PR review.
4. `download_range` and `download_ranges` only log short reads. #920 validates only the disk-download boundary. Raising a shared integrity exception in the helpers would protect future BytesIO callers and centralize the invariant.
5. Broad fallback catches in HNS move paths can mask unexpected failures by switching to copy/delete. Narrow fallback to explicitly supported failure classes, while preserving the intentional UNKNOWN-layout fallback.

## Ranking

- Highest correctness value: complete generation propagation in #921 before merge.
- Highest performance value: bounded `_fetch_range_split` worker scheduling.
- Next performance value: bounded inventory listing concurrency.
- Defensive follow-ups: central short-read validation and narrower move fallback exceptions.

## Second-scan PR inventory

- Cache precision: #870 centralizes dircache update strategies; #871 adds version-aware HNS move invalidation.
- MRD lifecycle: #854 and #856 fix concurrent MRD reuse/refcounting; #875 changes exhausted-pool waiting; #884 removes the obsolete single-request capability branch.
- Metadata and type routing: #878 avoids repeated `_info`; #879 resolves the filesystem class dynamically; #880 and #882 repair generation forwarding.
- Runtime and measurement: #892 parallelizes E2E provisioning/tests; #893 increases delete batch size; #881 fixes monitor shutdown latency; #883 surfaces benchmark child failures; #917 excludes data generation from write timing.

The second scan will look for adjacent code that still repeats metadata calls, duplicates lifecycle state, launches unbounded work, or includes setup/teardown in measured durations.

### Metadata and generation observations

- #878's local lazy-size memoization pattern is sound, but the inherited standard `_process_limits` still performs its own generation-blind `_info` for negative ranges; this reinforces the first scan's #921 extension.
- #880/#882 show that accepting `generation` in a signature is insufficient unless it is threaded into every constructor and metadata lookup. Remaining high-value gaps are standard concurrent size resolution and Zonal MRD-pool selection, already recorded above.
- The dynamic-class problem fixed by #879 appears localized: production `GCSMap` now resolves `gcsfs.GCSFileSystem`; remaining direct `GCSFile` construction is the standard filesystem's intended factory path.
- A new N+1 candidate exists in `ExtendedGcsFileSystem._expand_path_with_details`, which calls `_info` for each expanded path even when surrounding listing operations may already have detailed entries. It needs call-path inspection before promotion to a recommendation.

### Cache and MRD observations

- #871 was closed without merge. Current #870 code still removes cached HNS entries by name only. In version-aware mode, moving or batch-deleting generation 123 can remove cached generation 456 as well, producing a false-negative cached listing. Revive #871's generation matcher and apply it to both move and batch-delete updates.
- The HNS cache representation is a list. `_cache_drop_entries` and `_cache_upsert_entry` rebuild an entire parent listing for point mutations, while folder rename scans all dircache keys. High-cardinality HNS directories would benefit from a name/generation index or dict-backed internal representation.
- #875 closed without merge because #884 made MRDs shareable, eliminating the exhausted-queue branch. A related issue remains: `MRDPool.initialize()` and `get_mrd()` hold the pool lock while awaiting network-backed MRD creation. This serializes pool warm-up and makes `close()` wait for creation latency. Reserve capacity under lock, create outside it, then re-check closed state.

### Benchmark correctness observations

- #883 makes nonzero child-process exit codes fatal, but the listing multiprocess worker never calls `Future.result()`: it converts the future list to a list and lets the executor context wait. Thread exceptions therefore remain stored in futures, the child exits zero, and #883 cannot detect failure. Replace with explicit result collection and add a regression test.
- Open and info benchmark operations catch `FileNotFoundError` and report success. Missing test data can therefore produce deceptively fast benchmark results. Fail the case or record/validate an error count.
- #917 removes random-data generation from the configured write window, but fixed-duration write throughput divides bytes by configured runtime while final close/flush occurs after the loop. Report steady-state bytes/s and drain/finalization latency separately, or use measured end-to-end duration for committed throughput.
- Resource monitoring samples the process tree at a one-second default interval. Short-lived child processes can start and exit between samples, so peak CPU/RSS can be understated even though #881 fixed shutdown latency. Consider a shorter benchmark-specific interval or a final process-tree sample before stop.

### Delete and E2E observations

- #893 raises `GCSFileSystem._rm`'s default batch size to 100, but `ExtendedGcsFileSystem._rm` and `_rm_bucket_paths` still default to 20. Because the extended class is the default exported filesystem when experimental support is enabled, flat-bucket deletes routed through it lose most of #893's benefit.
- Do not simply change the extended default to 100 everywhere: `_perform_rm` also uses the same value as concurrent HNS directory-delete fan-out, while 100 is specifically the JSON batch-operation limit for file deletes. Split `file_batch_size=100` from a smaller `directory_concurrency`.
- #892 starts roughly one `gcloud` process per worker-specific bucket variant with no provisioning cap, and cleanup does the same. Current defaults create/delete dozens concurrently. Add a configurable bounded job pool and generate a single bucket manifest consumed by both create and cleanup to prevent worker-count drift and quota spikes.

### Upload and lifecycle observations

- #894 aligns non-final chunks in buffered `GCSFile._upload_chunk`, but the standard `_pipe_file` and `_put_file` resumable loops accept arbitrary public `chunksize` values and send slices directly to `upload_chunk`. Their comments say the size should be a 256 KiB multiple, but no validation or carry buffer exists. Centralize chunk-size validation/alignment across all resumable upload entry points.
- #855 correctly fixes inclusive Range offset recursion in `upload_chunk`; no second manual shortfall recursion was found.
- #857 establishes dependency-ordered teardown (MRD cache before gRPC transport). The remaining question is whether `_close_resources` is wired into production filesystem shutdown or only tests/finalizers; inspect before recommending lifecycle integration.

Follow-up inspection confirmed that `ExtendedGcsFileSystem` exposes no `close`, `aclose`, or context-manager method. `_close_resources` is called by test fixtures only. Garbage-collection finalization closes the MRD cache and memmove executor, but not the gRPC or Storage Control transports. Add an explicit idempotent `aclose()` plus synchronous wrapper/context-manager support, preserving #857's dependency order; keep finalizers as fallback only.

### Lower-value checks

- #858's per-bucket deletion routing is structurally sound; its unbounded bucket-group fan-out matters only for unusually large cross-bucket requests.
- #853's swapped `_process_object` arguments could be prevented with stronger type annotations/static checking, but no second production call-site mismatch was found.
- The tentative `_expand_path_with_details` metadata concern is not a broad N+1 over all descendants: it performs `_info` primarily for explicit roots missing from detailed find/glob output. It is not promoted as a recommendation.

## Second-scan ranking

1. Correctness: revive version-aware HNS cache matching from closed PR #871 and apply it to move plus batch delete.
2. Performance: split Extended filesystem file-delete batch size (100) from HNS directory concurrency so #893's gain reaches the default exported class.
3. Correctness: collect multiprocess-listing thread results so #883 can actually observe failures; stop swallowing missing-target errors in open/info benchmarks.
4. Correctness: centralize 256 KiB alignment validation for `_pipe_file` and `_put_file`, extending #894 beyond buffered writes.
5. Lifecycle: add explicit idempotent filesystem shutdown for MRD, gRPC, Storage Control, executor, and HTTP resources in dependency order.
6. Concurrency: move network-backed MRD creation outside the pool lock.
7. Larger cache mechanism: add a name/generation index for targeted HNS listing mutations and subtree invalidation.

## Creative mechanism exploration

Constraints confirmed by user:

- Cover throughput/latency, memory/cache cost, and correctness/reliability one by one.
- Each mechanism must be backward-compatible and opt-in initially.

### Runtime throughput/latency approaches

1. **Opportunistic range-request fusion (recommended).** Add a per-object/generation micro-batcher that collects concurrent range requests for a very short window, merges overlapping or nearby ranges, submits true MRD multi-range requests for Zonal buckets, and fans results back to callers. The existing `zb_hns_utils.download_ranges` helper is unused, while `_fetch_range_split` currently turns adjacent chunks into separate tasks and single-range MRD calls. This directly reduces RPC/task overhead for random small reads without speculative prefetch.
2. **Feedback-controlled concurrency.** Replace static concurrency with an additive-increase/multiplicative-decrease controller driven by moving latency, throughput, and retry/error signals. It adapts to network and object size but is harder to make deterministic and can oscillate across mixed workloads.
3. **Hot-object session pinning.** Keep initialized MRDs attached to a file/session and prewarm pool capacity for repeatedly accessed objects. This reduces setup latency but consumes idle resources and helps fewer workloads than request fusion.

Recommended first design: request fusion, because it reuses an existing but unused multi-range primitive, does not duplicate prefetch behavior, and can be bounded by request count, bytes, and delay.

### Memory/cache-cost approaches

1. **Unified byte-budget broker (recommended).** Introduce a filesystem-scoped `MemoryBudget` that grants explicit leases before read buffers, speculative prefetch, or retained cache chunks are allocated. Foreground demand has priority over cache fills, which have priority over speculative prefetch. Under pressure it first cancels/shrinks prefetch, then evicts retained read segments, then bypasses caching; it never blocks a foreground read indefinitely.
2. **Independent byte caps per subsystem.** Add byte limits separately to `ReadAheadChunked`, `BackgroundPrefetcher`, dircache, and MRD cache. This is simpler but cannot prevent aggregate oversubscription or duplicate buffering between cache and prefetch layers.
3. **RSS-feedback throttling.** Observe process RSS and shrink behavior when a high-water mark is crossed. This captures third-party allocations but is noisy, reactive, and difficult to test deterministically.

Recommended design: the unified broker, initially covering read/prefetch payload bytes only. MRDs and dircache can join later through weighted resource units or estimated sizes. Evidence: `ReadAheadChunked` retains a deque without a byte budget; `BackgroundPrefetcher` uses its own sizing and queue; `DirectMemmoveBuffer` allocates the entire expected result; `MRDPoolCache` is bounded by counts rather than resource cost; cache and prefetch overlap is explicitly acknowledged in `ZonalFile`.

### Correctness/reliability approaches

1. **Immutable object snapshot envelope (recommended).** Resolve an object once into a `ResolvedObject` carrying bucket, key, generation, size, hashes, finalized state, and billing context. All transport calls, range calculations, retries, cache keys, and integrity checks consume this envelope instead of reparsing a path and separately rediscovering metadata. Mutations return a receipt describing exact successful effects for cache updates.
2. **Central validation decorators.** Wrap existing read/write functions with generation and size assertions. Lower change cost, but still allows metadata and transport arguments to drift internally.
3. **Operation journal/state machine.** Model downloads, uploads, moves, and deletes as durable step machines with compensation. Strongest failure recovery, but much larger than the current correctness problems justify.

Recommended design: snapshot envelope. It addresses the recurring class of bugs behind generation loss, TOCTOU size mismatches, short reads, retrying against a newer generation, and imprecise cache mutation. Roll out as `consistency_mode="snapshot"` and keep current behavior as default.

## Creative mechanism expansion to ten

The additional mechanisms are intentionally orthogonal to the initial three.

4. **Metadata singleflight and capability cache.** Concurrent callers currently repeat `_info`, storage-layout discovery, and MRD initialization. `_lookup_bucket_type` only caches a completed non-UNKNOWN result, so a cold bucket can stampede and transient UNKNOWN results are immediately retried. Coalesce identical in-flight metadata/capability work by key, give negative/transient results short bounded TTLs, and isolate waiter cancellation from the leader request. Opt in with `metadata_singleflight=True`.

5. **In-place HTTP range assembler.** The standard concurrent `_cat_file` path creates one `bytes` result per range and then `b"".join`s them, causing a second full-size copy. The disk-download comments already identify `DirectMemmoveBuffer` promotion as the missing mechanism. Add a standard-bucket assembler that assigns each request an exact writable slice of one preallocated destination and validates each slice length before publishing it. Opt in with `range_assembler="inplace"`.

6. **Parallel composite upload engine.** Standard `_put_file` and `_pipe_file` serialize every resumable chunk, while core already exposes server-side compose through `_merge`. For sufficiently large non-HNS objects, upload independently retryable temporary components concurrently, compose them into the final object in bounded fan-in stages, apply final metadata/preconditions only to the destination, and garbage-collect components through a session manifest. Opt in with `upload_strategy="composite"`; retain resumable upload as the fallback.

7. **Generation-bound resumable downloads.** `_get_file` deletes the destination after any failure, discarding completed ranges. Download into a sidecar partial file plus a compact manifest containing the resolved generation, expected size/checksum, and completed aligned extents. A retry resumes only missing extents; a changed generation or invalid extent discards the state; successful validation ends in atomic rename. Opt in with `resume_download=True`.

8. **Budgeted hedged reads.** A small number of slow range requests can dominate user-visible latency even when concurrency is otherwise healthy. After a dynamically measured or configured delay, issue at most one duplicate against a different MRD/connection for eligible foreground reads, accept the first response that passes snapshot/length validation, and cancel or drain the loser. A token bucket caps extra bytes and request rate. Opt in with `hedged_reads=True`.

9. **Structured operation scope.** Task ownership is spread across manual `create_task`, `gather`, `cancel`, pending-write sets, weakref finalizers, prefetch producers, and MRD leases. Introduce an operation-scoped owner for child tasks, buffer leases, MRD pools, file descriptors, temp paths, and callbacks, with deterministic dependency-ordered teardown and one cancellation/error policy. Initially use it only when `structured_operations=True` or another experimental mechanism requests it.

10. **Admission-controlled generational segment cache.** Existing retained read chunks use FIFO-like deques and unconditional admission; one-hit scans can evict genuinely hot data, and the cache/prefetch overlap can duplicate bytes. Key aligned segments by object snapshot, admit with a TinyLFU-style frequency sketch, retain with segmented LRU, and optionally spill validated cold segments to a bounded local store. The byte-budget broker controls capacity; this mechanism controls which bytes deserve it. Opt in with `segment_cache="tinylfu"`, with disk spill separately disabled by default.

### Dependency order

- Foundations: snapshot envelope (3), byte-budget broker (2), structured operation scope (9).
- Read path: metadata singleflight (4), in-place assembler (5), request fusion (1), hedged reads (8).
- Retention and recovery: segment cache (10), resumable downloads (7).
- Write path: composite uploads (6), isolated from the read stack except for shared operation/snapshot primitives.

### Portfolio validation and rollout boundaries

| # | Mechanism | Primary gain | Main risk | Initial compatibility boundary |
|---|---|---|---|---|
| 1 | Range fusion | Fewer small-read RPCs/tasks | Micro-batch delay | Disabled; max delay/range/bytes caps |
| 2 | Byte-budget broker | Predictable aggregate memory | Foreground starvation if priorities are wrong | Disabled; foreground bypass and timeout |
| 3 | Snapshot envelope | Generation-consistent operations | Extra metadata call on some paths | `consistency_mode="snapshot"` only |
| 4 | Metadata singleflight | Removes cold-start RPC stampedes | Leader failure shared by waiters | Disabled; bounded negative TTL, no persistent error cache |
| 5 | In-place assembler | Removes full-result copy | Unsafe publication before all slices validate | Disabled; publish only after exact-length completion |
| 6 | Composite upload | Parallel large-file writes | Orphan parts and compose complexity | Disabled; standard buckets/large files only; resumable fallback |
| 7 | Resumable download | Retains completed work after failure | Stale/corrupt sidecar state | Disabled; generation/checksum-bound manifest and atomic rename |
| 8 | Hedged reads | Lower p95/p99 range latency | Increased request/egress cost | Disabled; one hedge, token bucket, eligible foreground reads only |
| 9 | Structured operation scope | Deterministic cancellation/resource cleanup | Broad internal refactor | Disabled; adopted per experimental path first |
| 10 | TinyLFU segment cache | Better hit rate per retained byte | Policy complexity/local disk management | New cache name; memory-only first, spill separately opt-in |

The mechanisms do not require changing existing defaults, path syntax, public return types, or existing cache names. Each can be introduced behind a distinct constructor/operation option and benchmarked against the current path. Mechanisms 2, 3, and 9 should expose internal primitives without forcing other features to use them until those features are enabled.

Alternatives rejected from the top ten:

- Adaptive transport routing/circuit breaking overlaps with existing bucket-type routing and is less immediately grounded than singleflight plus hedging.
- Double-buffered resumable upload improves overlap but cannot parallelize server acceptance; composite upload has a larger ceiling and remains independently retryable.
- Durable mutation journaling is stronger than resumable downloads but imposes a database/state-machine surface disproportionate to current failure modes.

## Hot-path micro-optimizations (open "Bolt" PRs)

Source: 9 open PRs on `Yonghui-Lee/gcsfs` (#1–#9, branches `bolt-optimize-*` / `perf-*`), plus the team playbook in `.jules/bolt.md`. This is a distinct improvement class from the bug-fixes above: **per-path / per-object CPU micro-optimizations**.

Playbook principle (`.jules/bolt.md`): in frequently-called path/list code, prefer native string slicing, direct indexing, and simple loops over general-purpose parsers (`urlsplit`, `parse_qs`, `posixpath.join`, `self._parent()`, `self.split_path()`) and temporary allocations (`set`, `list`, dict-comps).

Already covered by open PRs #1–#9 (do not duplicate): `_split_path`, `_strip_protocol`, `_coalesce_generation` (×2), and `_get_dirs_and_update_cache` (cache short-circuit + inlined parent extraction).

### New candidates (not covered by #1–#9)

1. **`_process_object` — `posixpath.join` in the hottest loop (high value).** `gcsfs/core.py:548` runs `result["name"] = posixpath.join(bucket, object_metadata["name"])` once per object in every listing (called at `gcsfs/core.py:865`, `[self._process_object(bucket, i) for i in items]`). `posixpath.join` does absolute-path detection + separator logic per object — the overhead PRs #7/#9 stripped from `_split_path`/`_strip_protocol`, but on a higher-traffic path (every object of every `ls`/`find`/`walk`).
   - Proposed, exactly behavior-preserving:
     ```python
     name = object_metadata["name"]
     result["name"] = name if name[:1] == "/" else bucket + "/" + name
     ```
   - The `name[:1] == "/"` guard preserves `posixpath.join`'s absolute-second-arg semantics (`join("b", "/x") == "/x"`). GCS names effectively never start with `/`, so the guard is one cheap slice-compare for zero semantic risk. `bucket` (from `split_path`) never has a trailing slash, so empty-name and no-double-slash cases already match.
   - Validation: `timeit` microbenchmark over deep keys / 10k objects; run existing `ls`/`find`/`info` tests; add a regression test asserting a leading-`/` name maps identically to the old output; add a `.jules/bolt.md` entry. Optionally drop the now-unused `import posixpath` (line 548 is its only use).

2. **`_dircache` inlining — evaluated, recommend NOT shipping standalone.** `gcsfs/_dircache.py:51` (`set(self._parent(p) for p in paths) | set(paths)`) and `:223` (`[(path, *self.split_path(path)) for path in paths]`) call general-purpose helpers per element, but on **mutation paths** (`rm`/`mv` cache updates), O(N) in files touched by one user action — orders of magnitude colder than the per-listed-object read loop. The `set()` at line 51 is **legitimate dedup** (collapsing many files under one parent into a single `invalidate_cache`), unlike `_coalesce_generation`'s wasteful set. Only inlining `self._parent` loosely matches the playbook, and on a cold path the gain is noise. Fold in only if `_dircache.py` is edited for another reason.

### Already-clean (checked, no action)

- `_parse_timestamp` uses C-backed `datetime.fromisoformat`, not `dateutil`.
- `url()` quoting is per-file, not a per-object loop.
- The `re.search` error parsing in `_call` (`gcsfs/core.py:1549-1561`) is on the error path only.
