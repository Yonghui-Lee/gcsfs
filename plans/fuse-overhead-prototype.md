# Design: FUSE Overhead Measurement Prototype

A small prototype that produces a **measured, attributable number for FUSE overhead** on gcsfs-shaped workloads. The output of this work is a decision input — not a production FUSE library, not a benchmark of gcsfs. It tells us whether driving gcsfs through a custom FUSE filesystem is a viable track in the eventual benchmark report, and which FUSE tunings actually matter.

This document is a complete spec. Sections marked **[Decision]** record a deliberate choice and its trade-off. Sections marked **[Contract]** are invariants the implementation must uphold.

---

## 1. Why this prototype exists

The broader benchmark project has three candidate bridges for `fio → gcsfs`:

- **(A)** Embedded CPython external ioengine (current `plans/fio-benchmark.md` spec)
- **(B)** External Python daemon over Unix socket + shared memory
- **(C)** Custom FUSE filesystem on top of `gcsfs`, fio runs against the mount

(C) is attractive because fio is unmodified and the harness is identical to how `gcsfuse` is benchmarked, but FUSE adds a layer whose overhead we have not measured. The benchmark report's stated goal is **measuring gcsfs performance and locating its bottlenecks**; (C) is only viable for that goal if FUSE overhead is small enough that it does not mask gcsfs-internal effects.

This prototype answers: *with the best feasible FUSE tuning, what overhead does FUSE add over a direct in-process call to the same backend, across the workloads we care about?*

## 2. Scope

**In scope (Phase 1, this spec):**

- A FUSE handler in Python (`pyfuse3`, trio-native).
- Two synthetic backends: `MemFs` (zero latency) and `LatencyFs` (synthetic per-op RTT).
- A direct in-process caller that exercises the same backends from the same Python process, used as the reference path.
- Three fio jobfiles covering small random reads, large sequential reads, and a stat storm.
- A tuning matrix (six rows) varying FUSE mount options.
- A runner that mounts FUSE with a given tuning, runs fio, captures fio JSON + `perf stat` counters.
- A reporter that produces a markdown comparison table.

**Deferred (Phase 2, gated on Phase 1 results):**

- `GcsfsBackend`, which requires a trio↔asyncio bridge to drive `gcsfs.GCSFileSystem(asynchronous=True)`.
- Bucket seeding script.
- Any real-network measurement.

**Out of scope (will not happen as part of this work):**

- A production-quality FUSE library for gcsfs end-users.
- Picking (A) vs (B) vs (C) for the final benchmark — this prototype only de-risks (C).
- Benchmarking gcsfs itself.

## 3. Architecture

```
fio  ──read()──▶  kernel FUSE  ──/dev/fuse──▶  pyfuse3 handler (trio)
                                                       │
                                                       │ Backend protocol (async, trio-native)
                                                       ▼
                                              MemFs / LatencyFs
```

A direct (no-FUSE) reference path runs the same backends from a second Python program:

```
direct_caller.py  ──▶  MemFs / LatencyFs
```

`fio_throughput − direct_caller_throughput`, on the same backend and matched concurrency / IO size, **is the FUSE overhead.**

## 4. Backend protocol **[Contract]**

All backends implement this surface. The handler depends on this surface only; no backend-specific code lives in the handler.

```python
class Backend(Protocol):
    async def stat(self, path: str) -> StatInfo: ...
    async def listdir(self, path: str) -> list[str]: ...
    async def open(self, path: str) -> FileHandle: ...
    async def read(self, fh: FileHandle, offset: int, size: int) -> bytes: ...
    async def close(self, fh: FileHandle) -> None: ...

class StatInfo:
    size: int
    mtime: float
    is_dir: bool
```

`FileHandle` is opaque to the handler. For Phase 1 backends it is simply the path string.

## 5. Backends

### 5.1 MemFs **[Decision]**

Pre-allocates `bytes` buffers from `os.urandom` for a fixed set of synthetic files. `read` is a slice — measurable in tens of nanoseconds. This is the **zero-latency floor**: any time observed in fio that does not appear in `direct_caller` is FUSE overhead.

Default fileset: **64 files × 16 MB each**, named `/f0000.bin`..`/f0063.bin`. Total ~1 GB resident; fits comfortably in RAM on any benchmark host.

Trade-off accepted: pre-allocating 1 GB at startup makes mounts slow (~1 s). Acceptable; the prototype mounts once per fio run, not per IO.

### 5.2 LatencyFs

Subclass of `MemFs`. Inserts `await trio.sleep(rtt_seconds)` on every `read` and `stat` call. `rtt_ms` is a constructor argument. Models the *shape* of a network-bound backend without making any network calls — the results are reproducible across machines.

This is what lets us answer "does FUSE overhead amortize as backend latency grows?" — we sweep `rtt_ms ∈ {1, 10, 50}` and watch the FUSE overhead percentage shrink (or fail to). 1 ms approximates a same-region cache hit; 50 ms approximates a cross-region GCS round-trip; 10 ms is the representative middle.

### 5.3 GcsfsBackend *(Phase 2, deferred)*

Will run an asyncio loop on a dedicated thread, own one `GCSFileSystem(asynchronous=True)`, and bridge to trio via `trio_asyncio.run_aio_future`. The same loop-thread pattern is reusable by bridge (B) if we go that direction. Not built in Phase 1.

## 6. FUSE handler **[Contract]**

- Implemented as a `pyfuse3.Operations` subclass.
- All callbacks are `async def`; pyfuse3's trio main loop drives them.
- Maintains an inode↔path table (in-memory dict, monotonically growing for the process lifetime). Inodes are never reused; this is correct for a read-only prototype.
- Maintains an open-file table mapping integer `fh` → backend `FileHandle`.
- Read-only. `open` with `O_WRONLY | O_RDWR` returns `EROFS`. No `create`, `unlink`, `mkdir`, `write`, `fsync`, `setattr`.
- `keep_cache` and `direct_io` on `FileInfo` are controlled by runner flags so the tuning matrix can flip page-cache behavior without code changes.
- `entry_timeout` and `attr_timeout` on returned `EntryAttributes` are controlled by a runner flag (the `+cache` tuning row).

The handler is the only file in the prototype that is "real" engineering work. Backends and runners are deliberately thin.

## 7. Direct caller **[Decision]**

`direct_caller.py` is a small async program that exercises the same backend from the same process — no FUSE, no kernel involvement. It accepts the same parameters as the matched fio jobfile (IO size, op count, concurrency, access pattern) and reports throughput / IOPS / p50 / p99 / p99.9.

**Trade-off accepted:** the submission pattern will not be byte-identical to fio's. fio uses `psync` (one syscall per IO, no batching); `direct_caller` uses asyncio with a `Semaphore`-bounded concurrency. The numbers are not directly comparable as absolute values — what *is* comparable is the **ratio** of fio-with-FUSE to direct-without-FUSE at matched concurrency, because each path's submission cost is constant across backends. We document this caveat in the report.

If the ratio looks suspicious, the fallback is to write a "psync-shaped" direct caller that does sequential blocking calls in N threads. Not in Phase 1.

## 8. Tuning matrix

Each row is one runner invocation. We do not sweep cartesian; we sweep progressively, with each row layering on top of the previous.

| Row | Mount options (delta from previous) |
|---|---|
| `baseline`  | libfuse3 defaults, splice off, `max_read=128KB` |
| `+splice`   | `splice_read,splice_write,splice_move` on |
| `+large_io` | `max_read=max_write=1MB`, kernel readahead 32 MB |
| `+clone_fd` | `clone_fd=1`, `max_background=256` |
| `+cache`    | `entry_timeout=600 attr_timeout=600` |
| `+io_uring` | fuse-over-io_uring backing (Linux ≥6.14 only; skipped if unavailable) |

The `+async` handler tuning is *not* a row, because pyfuse3 + trio is always async. We note this in the report rather than have a row that does nothing.

If a row regresses against the previous one on every workload, we drop it from the recommended config but keep the data — that is itself a finding.

## 9. Workloads

Three fio jobfiles. Each has `runtime=30 ramp_time=5 time_based=1 group_reporting=1 randrepeat=0`.

| Job              | rw       | bs   | numjobs | iodepth | ioengine | Stresses |
|------------------|----------|------|---------|---------|----------|----------|
| `randread_4k`    | randread | 4k   | 8       | 1       | psync    | per-request overhead |
| `seqread_1m`     | read     | 1M   | 4       | 4       | psync    | bulk throughput |
| `stat_storm`     | randread | 4k   | 4       | 1       | psync    | metadata caching: `nrfiles=10000 file_size=4k openfiles=16` per job, `create_on_open=0`, exists at mount time |

`stat_storm` exercises the `lookup`/`getattr` path more than `read`. With `+cache` off, every IO does a fresh `lookup` round-trip; with `+cache` on, only the first access of each file does. The delta between rows on this workload directly measures attr-caching value.

Three workloads × six tuning rows × four backend configs (MemFs + LatencyFs at 1/10/50 ms) = **72 runs** at ~35 s each (5 s ramp + 30 s runtime) = **~42 minutes wall-clock, single-threaded scheduling**. Manageable.

## 10. Measurements

For each run, capture:

- fio JSON: throughput (MB/s, IOPS), latency p50 / p99 / p99.9, clat histogram.
- `perf stat -e context-switches,cycles,instructions` wrapping the fio invocation.
- Optional, low-frequency: `perf record -F 99 -p <handler_pid>` for ~10 s mid-run, written to per-row perf data files. Off by default; flag to enable.

Results land in `prototype/results/{backend}-{row}-{job}.{json,perf}`.

## 11. Reporter

`prototype/report.py` reads the result files and emits `prototype/results/report.md`:

- One table per workload, columns = tuning rows, rows = backend (incl. RTT), cells = throughput + p99.
- One "FUSE overhead %" table per workload, computed as `(direct − fuse) / direct × 100`.
- A "tuning win/loss" summary: for each row, on how many (workload, backend) combinations did it help/hurt/no-change?
- A "decision" section that evaluates the criteria in §13 and emits PASS / FAIL / INCONCLUSIVE for each.

## 12. File layout

```
prototype/
├── backends/
│   ├── base.py            # Backend protocol, StatInfo
│   ├── memfs.py           # MemFs
│   └── latencyfs.py       # LatencyFs (subclass of MemFs)
├── fuse_handler.py        # pyfuse3 Operations subclass
├── run_mount.py           # mounts FUSE with a given tuning; trio.run main
├── direct_caller.py       # the reference path
├── jobs/
│   ├── randread_4k.fio
│   ├── seqread_1m.fio
│   └── stat_storm.fio
├── run_one.sh             # one (backend, row, job) cell of the matrix
├── run_all.sh             # the full matrix
├── report.py              # results → report.md
└── results/               # gitignored output
```

## 13. Decision criteria **[Decision]**

Written down **before** running. The report mechanically evaluates these from the data.

**Definition: "best tuning"** for a given (backend, workload) is the tuning row producing the lowest p99 latency on `randread_4k` and `stat_storm`, and the highest throughput on `seqread_1m`. If these point to different rows for the same backend, both are reported; the criterion uses whichever row wins on the workload being evaluated.

1. **MemFs + best tuning on `randread_4k`** — if FUSE adds **>2×** p99 latency over `direct_caller`, (C) is misleading for small-IO bottleneck analysis. Recommendation: skip (C), use (B) only.
2. **LatencyFs(10ms) + best tuning on `seqread_1m`** — if FUSE adds **<5%** throughput overhead over `direct_caller`, (C) is acceptable for the throughput sections of the report.
3. **Which tunings mattered** — the rows that contributed >5% to the best-tuning result on at least one (workload, backend) pair become the "required configuration" appendix in the eventual benchmark report.
4. **Is `+io_uring` necessary?** — if it is the only row that brings (1) under threshold, the benchmark report's environment requirements include "Linux ≥6.14 with fuse-over-io_uring".

The prototype succeeds when each criterion has a clear PASS/FAIL answer from data. The prototype does **not** need to make (C) win — a clear FAIL is a valid, useful result.

## 14. Dependencies

- `pyfuse3` (drags in `trio`; requires `libfuse3` ≥ 3.10 and `libfuse3-dev` at build time)
- `fio` ≥ 3.30 (for JSON output stability)
- `perf` (linux-tools)
- Linux ≥ 5.15 baseline; ≥ 6.14 to exercise the `+io_uring` row

All dev-only. None enter `gcsfs`'s runtime dependency set.

## 15. Open questions

- **Spec / file home.** Currently lives on `main` at `plans/fuse-overhead-prototype.md`. The related `plans/fio-benchmark.md` lives on `fio-design`. We may want to move this onto `fio-design` so the two specs are colocated.
- **Result publication format.** The report is markdown today. If the eventual benchmark report wants charts, we add a `matplotlib` pass; not in Phase 1.
- **CI integration.** Not wired up. The prototype is run manually on a benchmark host; it is not part of the test suite.

## 16. Non-goals (recap)

- No write path. `EROFS` is the correct answer.
- No POSIX compliance beyond what these three fio jobs need.
- No multi-mount, no per-user mounts, no FUSE-server daemonization.
- No real GCS traffic in Phase 1.
