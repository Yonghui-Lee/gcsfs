# Detailed Design: Direct Async Fio Engine for gcsfs

A custom fio external ioengine (`gcsfs_engine.so`) that embeds CPython inside each fio worker process to drive `gcsfs` async operations. Used to benchmark gcsfs head-to-head against other GCS client libraries (e.g. gcsfuse, Go/C++ clients) under an identical fio workload definition.

This document is a complete spec. Sections marked **[Decision]** record a deliberate choice and its trade-off; sections marked **[Contract]** are invariants the implementation must uphold.

---

## 1. Architecture Overview

Two components:

1. **`gcsfs_engine.c`** — fio external ioengine. Embeds CPython post-fork; pure C on the hot path; talks to Python only at submission and via a C completion callback invoked from the loop thread.
2. **`gcsfs_async_adapter.py`** — runs an asyncio loop on a dedicated thread; owns one `GCSFileSystem(asynchronous=True)`; submits gcsfs coroutines and invokes the C callback on completion.

Per-IO flow (steady state):

1. fio calls `queue(td, io_u)`.
2. Engine acquires GIL, calls `adapter.submit_io(io_u_ptr, op, path, offset, length, buf_ptr)`, releases GIL, returns `FIO_Q_QUEUED`.
3. Adapter schedules a coroutine on the loop thread via `loop.call_soon_threadsafe`.
4. Coroutine awaits the gcsfs operation. On completion (success, exception, or cancellation) it invokes the C completion callback with `(io_u_ptr, errno)`.
5. The C callback enqueues `io_u` into a per-worker SPSC ring and writes 1 byte to a per-worker eventfd.
6. `getevents` blocks on `epoll_wait` over the eventfd (no GIL held), drains the ring, returns the events.

---

## 2. Python Embedding & GIL Management

### 2.1 Process model **[Decision]**

Use fio's default process forking model (`thread=0`). Each fio worker is a separate OS process with its own CPython interpreter, GIL, and gcsfs instance.

**Trade-off accepted:** every worker maintains its own aiohttp connection pool, auth-token cache, and `DirCache`. This is *not* how a real long-running gcsfs application behaves (one shared pool). Cross-library comparison results must be interpreted with section 9 in mind.

### 2.2 Python version **[Decision]**

CPython 3.11+ required. Rationale:
- `python3-config --embed` is available (3.8+).
- Per-interpreter GIL work in 3.12 is irrelevant to us (we use one interpreter per process), and 3.11 is the floor where `PyConfig`-based embedding is stable.
- gcsfs is tested on 3.9+; we tighten to 3.11 so we don't have to special-case older `asyncio` semantics.

Pin in `setup.cfg`/CI: exact patch version, captured in the benchmark report.

### 2.3 Fork-after-import hazards **[Contract]**

`Py_Initialize()` **must not** run in the parent fio process. It runs only in `init()`, post-fork. fio invokes the engine's `setup()` in the parent and `init()` in each child after `fork()`; this is the only ordering that avoids:
- aiohttp / SSLContext file descriptors duplicated across workers
- libcurl / OpenSSL global state corruption
- Python atexit handlers firing on the wrong process
- `_PyImport_AcquireLock` deadlocks observed when Python is initialized then forked

`setup()` is restricted to: option parsing, workload validation (section 8), and reading credential paths. **No Python C API, no gcsfs imports, no socket creation.**

### 2.4 GIL discipline **[Contract]**

| Path | GIL held? |
|---|---|
| `setup()` | N/A (no Python yet) |
| `init()` | yes during `Py_Initialize`, released via `PyEval_SaveThread()` before returning |
| `queue()` | acquired via `PyGILState_Ensure`, released before return |
| `getevents()` | **never** holds the GIL while blocking on the eventfd |
| C completion callback (called from loop thread, GIL already held by Python) | held throughout; see 4.4 |
| `cleanup()` | acquired to drain & shut down adapter |

`PyEval_SaveThread()` in `init()` is critical — without it, the loop thread cannot acquire the GIL and gcsfs hangs on the first await.

---

## 3. Asyncio & Networking

### 3.1 Loop thread **[Contract]**

A single `threading.Thread` per worker runs `asyncio.new_event_loop().run_forever()`. The thread is **not** a daemon; teardown joins it explicitly (section 7).

### 3.2 GCSFileSystem construction **[Contract]**

`GCSFileSystem(asynchronous=True)` binds to the *current* running loop. It must be instantiated **inside** the loop thread, after `set_event_loop(loop)`, not in `__init__`. Construction is awaited on from the main thread via `run_coroutine_threadsafe(...).result()` so `init()` blocks until the fs is ready.

```python
def _start(self):
    fs_future = concurrent.futures.Future()
    def _bootstrap():
        asyncio.set_event_loop(self.loop)
        self.fs = gcsfs.GCSFileSystem(asynchronous=True, loop=self.loop)
        fs_future.set_result(None)
        self.loop.run_forever()
    self.thread = threading.Thread(target=_bootstrap, name="gcsfs-loop")
    self.thread.start()
    fs_future.result(timeout=30)  # block init() until fs is ready
```

### 3.3 CPU pinning

Pin fio workers to specific cores via `cpus_allowed` in the job file, leaving NIC IRQ cores free. Document this as part of the run protocol; not enforced by the engine.

### 3.4 Auth & credentials **[Decision]**

To avoid an N-way metadata-server token-fetch storm at `numjobs > 1` start-up:

- **Required**: set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account key file. Each worker reads the file independently (no network call). Document this as a benchmark prerequisite.
- **Forbidden in benchmark runs**: ADC via metadata server (introduces unbounded warmup latency that contaminates p99).
- Auth-refresh latency that occurs *during* the run is intrinsic to the library and stays in the measurement. Use fio `ramp_time ≥ 30s` so initial token fetch is excluded from steady-state stats.

---

## 4. C Engine Structure

### 4.1 `struct ioengine_ops` **[Contract]**

```c
static struct ioengine_ops ioengine = {
    .name        = "gcsfs",
    .version     = FIO_IOOPS_VERSION,
    .setup       = fio_gcsfs_setup,
    .init        = fio_gcsfs_init,
    .queue       = fio_gcsfs_queue,
    .getevents   = fio_gcsfs_getevents,
    .event       = fio_gcsfs_event,
    .open_file   = fio_gcsfs_open_file,
    .close_file  = fio_gcsfs_close_file,
    .get_file_size = fio_gcsfs_get_file_size,
    .cleanup     = fio_gcsfs_cleanup,
    .flags       = FIO_DISKLESSIO | FIO_NOEXTEND | FIO_NODISKUTIL,
    .options     = options,
    .option_struct_size = sizeof(struct gcsfs_options),
};
```

### 4.2 Function-by-function

**`setup`** — parent process. Validates workload (section 8). Rejects `td->o.td_ddir == TD_DDIR_RANDWRITE`, `TD_DDIR_RANDRW`, `TD_DDIR_TRIM`, and any `bs < 256KB` with `td_verror` and a clear message. **No Python.**

**`init`** — child process, post-fork.
1. Allocate per-thread state (`struct gcsfs_thread`): completion ring, eventfd, options.
2. Create eventfd: `eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)`. Add to an epoll fd.
3. `Py_Initialize()`. Adjust `sys.path`. Import adapter module.
4. Construct `GCSFSAsyncAdapter(c_complete_ptr=&c_complete_trampoline, thread_state=<opaque ptr>)`. Adapter's `_start` blocks until the loop thread has created the gcsfs instance (3.2).
5. `PyEval_SaveThread()` — releases the GIL so the loop thread can run.

**`queue`** —
1. `PyGILState_Ensure()`.
2. Build a `memoryview` over `io_u->xfer_buf` (writes: read-only; reads: writable). Use `PyMemoryView_FromMemory` with `PyBUF_READ` or `PyBUF_WRITE`. Refcount: the memoryview is consumed by `submit_io` and released when the coroutine completes; the C side holds no reference past the call.
3. Call `adapter.submit_io(...)`.
4. `Py_DECREF` all temporary objects (path string, memoryview wrapper if not consumed).
5. `PyGILState_Release()`. Return `FIO_Q_QUEUED`.

`queue` never blocks on network I/O. Submission overhead is bounded by GIL acquisition + one Python call — measured budget: < 5 µs.

**`getevents(min, max, timeout)`** — see section 4.3.

**`event(idx)`** — pop pre-staged event at `idx` from the per-thread events array (populated by `getevents`). Plain array indexing, no locking.

**`open_file`** — for reads: call `adapter.stat(path)` synchronously via `run_coroutine_threadsafe(...).result()`; cache size for `get_file_size`. For writes: no-op (object created on `_pipe_file`).

**`close_file`** — for writes that used streaming upload (rare; see 8.3): call `adapter.close_file(path)` to finalize the multipart upload. For single-shot writes: no-op.

**`get_file_size`** — return cached size from `open_file`.

**`cleanup`** — see section 7.

### 4.3 `getevents` design **[Contract]**

```c
int fio_gcsfs_getevents(struct thread_data *td, unsigned int min,
                       unsigned int max, const struct timespec *t) {
    struct gcsfs_thread *gt = td->io_ops_data;
    unsigned int reaped = 0;

    while (reaped < min) {
        // Drain ring without blocking.
        reaped += ring_drain(&gt->ring, &gt->events[reaped], max - reaped);
        if (reaped >= min) break;

        // Block on eventfd. GIL is NOT held here.
        struct epoll_event ev;
        int ms = timespec_to_ms_or_minus1(t);
        int n = epoll_wait(gt->epfd, &ev, 1, ms);
        if (n == 0) break;          // timeout
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return -errno;

        uint64_t cnt;
        read(gt->efd, &cnt, sizeof(cnt));  // drain counter
    }
    return reaped;
}
```

Key points:
- **No GIL** in `getevents`. fio's `getevents` cannot afford to serialize with the loop thread.
- The events array `gt->events` is per-thread; `event(idx)` reads it without locking.
- The `epoll_wait` timeout honors fio's `timespec`; pass through unchanged so `--io_submit_mode=offload` and timeouts work.

### 4.4 Completion callback (C, called from Python loop thread) **[Contract]**

```c
static void c_complete_trampoline(void *thread_state, void *io_u_ptr,
                                  int err) {
    // GIL is held by caller (we are in a Python C call).
    struct gcsfs_thread *gt = thread_state;
    struct io_u *io_u = io_u_ptr;
    io_u->error = err;

    // SPSC ring; sized to td->o.iodepth so it cannot fill.
    ring_push(&gt->ring, io_u);

    // Wake getevents. eventfd write is non-blocking up to UINT64_MAX-1
    // pending; with iodepth ≤ 2^32 this never blocks.
    uint64_t one = 1;
    write(gt->efd, &one, sizeof(one));
}
```

GIL handling: **the callback does not release the GIL.** Justification:
- Ring push is a single atomic store; eventfd write is a single non-blocking syscall.
- Ring is sized to `iodepth + 1`; fio's contract bounds in-flight IOs to `iodepth`, so the ring never fills and the producer never has to spin.
- Total callback time: < 1 µs. Releasing/reacquiring the GIL would cost more than the callback itself.

The ring is **single-producer, single-consumer**: the asyncio loop thread is the only producer (asyncio is single-threaded by construction); the fio worker thread calling `getevents` is the only consumer. Use a fixed-size power-of-two ring with `_Atomic` head/tail and `memory_order_acquire`/`memory_order_release` ordering. No mutex.

### 4.5 ctypes signatures

`c_complete_trampoline` is exposed to Python as a `ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_int)`. The first `c_void_p` is the per-thread state pointer captured by the adapter at construction; the second is the `io_u` pointer; the int is the errno. Both pointers are opaque to Python.

---

## 5. Data Transfer

### 5.1 Definition of zero-copy **[Decision]**

"Zero-copy" in this design means **no `memcpy`-equivalent across the C↔Python boundary**: the Python side reads or writes `xfer_buf` memory in place via a memoryview; no separate Python-heap `bytes` object is materialized and then copied across. Internal copies *inside* gcsfs/aiohttp/asyncio are out of scope — those are intrinsic to the library under test and we want them in the measurement.

### 5.2 Honest accounting **[Decision]**

| Path | Boundary copies | How |
|---|---|---|
| **Write** | **0** | `PyMemoryView_FromMemory(xfer_buf, len, PyBUF_READ)` is passed straight to `_pipe_file`. aiohttp's `BytesPayload` accepts the memoryview and writes it through the transport via `sendmsg` without re-buffering on the Python heap. **Conditional on retries being disabled — see 5.4.** |
| **Read** | **1, unavoidable** | `gcsfs._cat_file` returns a Python `bytes` object built by aiohttp's `StreamReader.read()`. Getting those bytes into `xfer_buf` requires one memcpy from the Python heap to the C buffer, regardless of how it's spelled (`ctypes.memmove`, `mv[:] = data`, or per-chunk `mv[off:off+n] = chunk` via `iter_chunked`). |

### 5.3 Why read-side zero-copy is not achievable here

aiohttp's HTTP response handler inherits from `asyncio.Protocol`, whose `data_received(data: bytes)` hands the protocol an already-allocated `bytes` object. asyncio does support `BufferedProtocol` (with `get_buffer(sizehint)` → writable buffer + `buffer_updated(nbytes)`, where the transport calls `sock.recv_into(buffer)` directly), but aiohttp does not use it. Eliminating the read-side boundary copy would require either:

1. **Patching aiohttp** to make its response handler a `BufferedProtocol` and to thread our memoryview down through `StreamReader` so each chunk recv lands in `xfer_buf`. This is a real fork of aiohttp's parser, not just an API tweak.
2. **Bypassing aiohttp** with a hand-rolled HTTP/1.1 client over `loop.sock_recv_into(mv)`. At that point we are no longer benchmarking gcsfs.

Neither is in scope. The read path is therefore documented as **"single boundary copy, intrinsic to gcsfs's aiohttp dependency"**, and any future change here is a gcsfs/aiohttp upstream conversation, not a benchmark-harness conversation.

### 5.4 Disabling retries on the write path **[Contract]**

Zero-copy writes require passing `xfer_buf` directly to aiohttp via memoryview. If aiohttp (or gcsfs's retry layer in `gcsfs/retry.py`) retries the request, it needs the buffer again — but by then fio may have reused `xfer_buf` for a different IO. The design therefore disables retries on the write path:

```python
self.fs = gcsfs.GCSFileSystem(asynchronous=True, loop=self.loop)
self.fs.retries = 1   # gcsfs class attribute; default is 6
```

This is the right default for a benchmark — retry behavior is a property of the production stack and amplifies throughput numbers in misleading ways. If retry behavior under load is itself a question of interest, it is a **separate workload** (see section 14): run with `retries=N` *and* accept the boundary copy by passing `bytes(memoryview(mv))` instead. The two configurations measure different things and should be reported separately.

### 5.5 Buffer lifetime **[Contract]**

For both directions, `xfer_buf` is owned by fio for the duration `queue()` → matching event returned via `getevents()`/`event()`. The adapter must not retain the memoryview past the coroutine that uses it. Specifically: do not stash it in a list, do not pass it to a `create_task` whose lifetime exceeds the await. With retries disabled (5.4), aiohttp will not request the buffer after the coroutine's first `await` returns.

---

## 6. Python Adapter (`gcsfs_async_adapter.py`)

```python
import asyncio
import concurrent.futures
import ctypes
import threading
import gcsfs

# Matches c_complete_trampoline signature.
_C_COMPLETE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)

class GCSFSAsyncAdapter:
    def __init__(self, c_complete_ptr: int, thread_state_ptr: int):
        self._c_complete = _C_COMPLETE(c_complete_ptr)
        self._thread_state = ctypes.c_void_p(thread_state_ptr)

        self.loop = asyncio.new_event_loop()
        self.fs: gcsfs.GCSFileSystem | None = None
        self._inflight: set[asyncio.Task] = set()
        self._shutting_down = False

        ready = concurrent.futures.Future()
        self.thread = threading.Thread(
            target=self._bootstrap, args=(ready,), name="gcsfs-loop"
        )
        self.thread.start()
        ready.result(timeout=30)

    def _bootstrap(self, ready):
        asyncio.set_event_loop(self.loop)
        self.fs = gcsfs.GCSFileSystem(asynchronous=True, loop=self.loop)
        self.fs.retries = 1   # zero-copy writes require single-shot requests; see 5.4
        ready.set_result(None)
        self.loop.run_forever()

    # Called from C with GIL held; must not block.
    def submit_io(self, io_u_ptr, op, path, offset, length, buf_ptr):
        if self._shutting_down:
            self._complete(io_u_ptr, 5)  # EIO
            return
        self.loop.call_soon_threadsafe(
            self._spawn, io_u_ptr, op, path, offset, length, buf_ptr
        )

    def _spawn(self, io_u_ptr, op, path, offset, length, buf_ptr):
        task = self.loop.create_task(
            self._do_io(io_u_ptr, op, path, offset, length, buf_ptr)
        )
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _do_io(self, io_u_ptr, op, path, offset, length, buf_ptr):
        # Contract: _complete must be called exactly once on every code path,
        # including exceptions and cancellation.
        try:
            if op == OP_READ:
                # Read path: one boundary copy is unavoidable (see 5.2/5.3).
                data = await self.fs._cat_file(path, start=offset, end=offset + length)
                if len(data) != length:
                    self._complete(io_u_ptr, 5)  # EIO: short read
                    return
                ctypes.memmove(buf_ptr, data, length)
                self._complete(io_u_ptr, 0)
            elif op == OP_WRITE:
                # Write path: zero boundary copies. Pass the memoryview directly;
                # aiohttp's BytesPayload streams it via sendmsg. Safe because
                # retries are disabled (see 5.4) so the buffer is only read once
                # and only during this await.
                mv = memoryview((ctypes.c_ubyte * length).from_address(buf_ptr))
                await self.fs._pipe_file(path, mv)
                self._complete(io_u_ptr, 0)
            elif op == OP_DELETE:
                await self.fs._rm_file(path)
                self._complete(io_u_ptr, 0)
            else:
                self._complete(io_u_ptr, 22)  # EINVAL
        except asyncio.CancelledError:
            self._complete(io_u_ptr, 125)  # ECANCELED
            raise
        except FileNotFoundError:
            self._complete(io_u_ptr, 2)  # ENOENT
        except Exception:
            self._complete(io_u_ptr, 5)  # EIO

    def _complete(self, io_u_ptr, err):
        # Single chokepoint so the "exactly once" contract is auditable.
        self._c_complete(self._thread_state, ctypes.c_void_p(io_u_ptr), err)

    def shutdown(self, timeout=30.0):
        self._shutting_down = True
        # Drain inflight, then stop the loop.
        async def _drain():
            if self._inflight:
                await asyncio.gather(*self._inflight, return_exceptions=True)
        fut = asyncio.run_coroutine_threadsafe(_drain(), self.loop)
        fut.result(timeout=timeout)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=timeout)
```

### 6.1 Notes on the implementation

- **Write path memoryview**: zero boundary copies. The memoryview is alive for the full lifetime of `_pipe_file`'s `await`; aiohttp consumes it once via `sendmsg`. Safety depends on `self.fs.retries = 1` (see 5.4) — without that, a retry would read freed/reused memory.
- **Read path memmove**: one boundary copy. Cannot be eliminated while gcsfs uses aiohttp's `StreamReader` (see 5.3). Documented and accepted.
- **Inflight set**: prevents `create_task` futures from being garbage-collected mid-flight (a real asyncio footgun).
- **`_rm_file`** not `_rm`: `_rm` is the recursive variant; `_rm_file` is the single-object delete and matches gcsfs/core.py:1594.
- **`_cat_file` end semantics**: in gcsfs (core.py:1238) `end` is **exclusive**, matching HTTP `Range: bytes=start-end-1`. We pass `offset + length` as `end`; this is correct for the gcsfs version pinned in `pyproject.toml`. Add a smoke test that asserts `len(data) == length`; fail loudly on regression.

---

## 7. Teardown **[Contract]**

`cleanup()` is the only place writes are guaranteed to finalize. The previous design used `daemon=True` which kills the loop mid-flight and corrupts multipart uploads.

Sequence in `fio_gcsfs_cleanup`:
1. `PyGILState_Ensure()`.
2. Call `adapter.shutdown(timeout=30)`. This drains inflight tasks, stops the loop, and joins the thread. Pending writes complete. If `shutdown` raises, log and continue — better to leak than to deadlock fio.
3. `Py_DECREF` adapter and module references.
4. **Skip `Py_Finalize()`**: well-known for crashing on threads in extension modules (especially aiohttp's SSL state). Process exit reclaims everything.
5. Close eventfd and epoll fd.
6. Free per-thread state.

Run integrity test: write objects in run A, read-and-verify in run B. If shutdown is broken, B will see truncated objects.

---

## 8. Workload Equivalence

This section makes "identical workload across libraries" auditable. The fio job file passed to gcsfs, gcsfuse, and any other library under test must be **literally the same file**, modulo `ioengine=`. To make that work, this engine restricts the supported subset and rejects everything else loudly.

### 8.1 Supported

| fio knob | Supported values | Notes |
|---|---|---|
| `rw` | `read`, `write`, `randread` | `randread` requires multiple distinct objects via `nrfiles>1` or a filename list |
| `bs` | ≥ 256 KiB; ranges allowed | Below 256 KiB is rejected; sub-MB GCS reads are unrealistic and dominated by RTT |
| `iodepth` | ≥ 1 | Implemented as the SPSC ring depth and as the `asyncio.Semaphore` cap on inflight gcsfs ops |
| `numjobs` | ≥ 1 | One Python interpreter + one gcsfs instance per worker. **See section 9.** |
| `size` | any | Per-object size; for `write` becomes the object size |
| `time_based`, `runtime`, `ramp_time` | as documented | `ramp_time ≥ 30s` recommended for auth warmup |
| `direct` | ignored | Meaningless against object storage; engine sets `td->o.odirect = 0` |

### 8.2 Mapped (with explicit warning)

| fio knob | Mapping |
|---|---|
| `rw=write` | Each "write IO" creates or fully overwrites an object via `_pipe_file(path, bytes_of_size)`. fio's per-IO `bs` becomes the object size. There is no append. |
| `verify=` | Only `verify=md5` and `verify=crc32c` against pre-written objects. Read-after-write within the same run uses GCS strong-consistency guarantees. |

### 8.3 Rejected (with `td_verror`)

| Pattern | Reason |
|---|---|
| `rw=randwrite`, `randrw`, `trim`, `randtrim` | GCS objects are immutable; no in-place partial write. Silent mapping would produce meaningless throughput numbers (e.g. 1 GB upload per "4 KB random write"). |
| `bs < 256 KiB` | Outside GCS realistic operating range; results would measure HTTP framing overhead, not the library. |
| `fsync`, `fdatasync` | No POSIX semantics. |
| `--client/--server` distributed mode | Not supported in v1. Run N copies and aggregate via HdrHistogram merge instead. |

### 8.4 Implementation in `setup()`

```c
if (td->o.td_ddir & (TD_DDIR_RANDWRITE | TD_DDIR_TRIM)) {
    td_verror(td, EINVAL, "gcsfs: random writes / trim unsupported");
    return 1;
}
if (td->o.min_bs[DDIR_READ] < 256*1024 || td->o.min_bs[DDIR_WRITE] < 256*1024) {
    td_verror(td, EINVAL, "gcsfs: bs must be ≥ 256KiB");
    return 1;
}
// ...
```

Loud failure beats silent distortion. A benchmark that ran but produced wrong numbers is worse than one that refused to start.

---

## 9. Cross-Library Comparison Notes **[Decision]**

The embedded-CPython, fork-per-worker model has a **modeling asymmetry** that must be reported alongside results:

- **gcsfs (this engine)**: `numjobs=N` produces N independent gcsfs instances, N aiohttp connection pools, N auth-token caches. Total connection count to GCS scales linearly with `numjobs`.
- **gcsfuse (fio libaio engine)**: one gcsfuse process serves all fio workers; one connection pool.
- **Go/C++ clients (their own fio engines)**: typically one client object per process; depends on the engine.

**Recommendation for fair comparisons**: run two configurations and report both.

1. **Concurrency-via-iodepth**: `numjobs=1, iodepth=64`. Gives gcsfs a single shared connection pool, matching the steady-state of a real application. This is the apples-to-apples number.
2. **Concurrency-via-jobs**: `numjobs=64, iodepth=1`. Stresses the fork/init path and shows worst-case per-process overhead.

Report both. Differences between them are themselves a finding.

---

## 10. Telemetry **[Decision]**

Commit to a minimal, always-on telemetry layer. No "optional OpenTelemetry" hand-wave.

Per-IO, recorded in the adapter:
- Wall time at `submit_io` entry (C → Python boundary).
- Wall time at `_cat_file` / `_pipe_file` entry (Python → gcsfs boundary).
- Wall time at coroutine completion.
- Number of HTTP retries (read from `gcsfs.retry` counter, if exposed; otherwise omit).

Aggregated per worker into an HdrHistogram, dumped as JSON next to fio's output on `cleanup`. This lets a reader correlate fio's reported clat with time spent inside gcsfs vs. crossing the language boundary. If the deltas are large, it informs whether the comparison is library-bound or harness-bound.

OpenTelemetry export is not in v1; revisit only if a span-based correlation across processes is needed.

---

## 11. Memory & Resource Discipline **[Contract]**

- **Refcount audit**: every Python C-API call in `queue()` and `cleanup()` has a paired DECREF, audited by reading the code with one developer pass and verified with `tracemalloc` snapshots before/after a 10M-IO run. Net allocation in worker should be flat.
- **`io_u` exactly-once completion**: the `_complete` chokepoint in the adapter is the only path that calls into C. The contract is that every coroutine path (success, every exception class, `CancelledError`, adapter shutdown short-circuit) reaches `_complete` exactly once. Code-reviewed against this checklist:
  - normal return → `_complete(0)`
  - `FileNotFoundError` → `_complete(ENOENT)`
  - other `Exception` → `_complete(EIO)`
  - `CancelledError` → `_complete(ECANCELED)` then re-raise
  - shutdown rejection in `submit_io` → `_complete(EIO)`
- **Buffer aliasing**: the read-path memoryview is created on demand and dropped at coroutine exit; no `_inflight` task may capture it past the await.
- **Soft assertion in debug builds**: `getevents` checks that the popped `io_u` is one fio submitted and not seen before (debug bitmap), to catch double-completion early.

---

## 12. Compilation

```bash
# Requires CPython 3.11+ with development headers
PY_CFLAGS=$(python3.11-config --cflags --embed)
PY_LDFLAGS=$(python3.11-config --ldflags --embed)
FIO_SRC=/path/to/fio/source       # tested against fio 3.36+

gcc -O3 -g -shared -fPIC -Wall -Wextra \
    -o gcsfs_engine.so gcsfs_engine.c \
    -I${FIO_SRC} \
    ${PY_CFLAGS} ${PY_LDFLAGS} \
    -pthread
```

CI matrix: {Linux x86_64, Linux arm64} × {Python 3.11, 3.12} × {fio 3.36, 3.38}. macOS not supported (fio external engine support is Linux-only in practice; benchmarks should run on Linux anyway).

---

## 13. Test Plan

1. **Unit (Python)**: adapter `_do_io` paths with a `gcsfs` stub — verify exactly-once completion, errno mapping.
2. **Integration (engine)**: against a fake GCS endpoint (e.g. `fake-gcs-server`) — `read`, `write`, mixed, `randread` over 100 objects, `numjobs=1..16`, `iodepth=1..64`. Assert: fio reports zero IO errors, written object size matches, read content matches (`verify=crc32c`).
3. **Stress**: 10M IOs at `iodepth=64` × `numjobs=4`. Assert: RSS flat (within 5%) over the run; no zombie processes; eventfd/epoll fd counts return to baseline after `cleanup`.
4. **Teardown**: kill -INT during steady state. Assert: no orphan multipart uploads (poll GCS list-multipart-uploads before next run).
5. **Workload-validation**: every rejected pattern from 8.3 fails `setup()` with the documented errno and message.
6. **Cross-library smoke**: same job file (8.1 subset) drives gcsfs and gcsfuse; both produce fio JSON; sanity-check that throughput numbers are within 5x (catches gross harness bugs).

---

## 14. Open Items

- **`numjobs` start-up auth**: even with key-file ADC, simultaneous TLS handshakes to `storage.googleapis.com` can exceed connection limits. May need `init()` to stagger by `td->thread_number * 50ms` if observed in practice.
- **HTTP/2 multiplexing**: aiohttp uses HTTP/1.1 with a connection pool. gRPC-based GCS clients use HTTP/2 with multiplexing. This is a library property, not a harness issue — but the comparison report should call out the protocol difference.
- **Long-tail latency under cancellation**: fio may cancel in-flight IOs at end-of-runtime. We map to `ECANCELED`; verify those don't get counted into clat percentiles incorrectly.
- **Retries-on workload, separate run**: the default benchmark disables retries (5.4) to keep the write path zero-copy and to avoid contaminating throughput numbers with retry amplification. If retry behavior under load is itself of interest, add a second run profile that sets `fs.retries = 6` (gcsfs default) and uses `bytes(memoryview(mv))` on the write path to keep the buffer alive across retries. Report it as a separate measurement, not as the headline number.
- **Read-side zero-copy via aiohttp `BufferedProtocol`**: tracked as a possible upstream contribution to aiohttp + gcsfs. Out of scope for this engine; only listed here so the trade-off is not forgotten.

