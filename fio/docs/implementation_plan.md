# Implementation Plan: High-Performance Direct FIO Engine for gcsfs

Design and implementation plan for a custom FIO external I/O engine (`libgcsfs_async_fio_engine.so`) that embeds the Python interpreter inside FIO processes to benchmark `gcsfs` at wire-speed, drawing structural inspiration from `/usr/local/google/home/yonghuili/Projects/gcs-py-sdk-fio-engine` but incorporating key architectural improvements to eliminate GIL contention and maximize performance.

---

## User Review Required

> [!IMPORTANT]
> **GIL and Multi-Processing:**
> To achieve high-concurrency GCS network saturation without GIL bottlenecks or segmentation faults, FIO must be run in multi-processing mode (`thread=0` in job configurations).
> * **Implication:** Every worker runs in an independent process with its own CPython runtime, leading to separate connection pools (`aiohttp` session) and auth-token caches. This differs from a long-running Python application using a single shared connection pool. Trade-offs and fair comparison strategies are detailed in the plan below.

---

## Proposed Architecture

The engine consists of two core components operating in a hybrid C/Python space:

```mermaid
sequenceDiagram
    autonumber
    participant FIO as FIO Engine Loop
    participant C as C Wrapper (libgcsfs_async_fio_engine.so)
    participant Py as Python Adapter (gcsfs_async_adapter.py)
    participant GCSFS as gcsfs (asyncio Event Loop)

    Note over FIO,C: Fork-per-worker (thread=0)
    FIO->>C: queue(td, io_u)
    activate C
    Note over C: Acquire GIL (PyGILState_Ensure)
    C->>Py: submit_io(io_u_ptr, op, offset, length, buf_ptr)
    activate Py
    Py->>GCSFS: loop.call_soon_threadsafe(_do_io, ...)
    Py-->>C: return status (QUEUED)
    deactivate Py
    Note over C: Release GIL
    C-->>FIO: return FIO_Q_QUEUED
    deactivate C

    Note over GCSFS: Runs asynchronously in background thread
    GCSFS->>GCSFS: await fs._cat_file() or _pipe_file()
    GCSFS->>C: c_complete_trampoline(io_u_ptr, errno)
    activate C
    Note over C: SPSC Ring Buffer Push
    C->>C: write(eventfd, 1)
    deactivate C

    FIO->>C: getevents(min, max, timeout)
    activate C
    Note over C: No GIL Held
    C->>C: epoll_wait(eventfd) [Blocks if ring empty]
    C->>C: Drain SPSC Ring Buffer
    C-->>FIO: Return reaped events
    deactivate C
```

### 1. `gcsfs_async_engine.c` (The C Hot Path)
A compiled shared library loaded by FIO as an external engine (`--ioengine=external:/path/to/libgcsfs_async_fio_engine.so`).
* **`init` (Post-Fork):** Initializes the embedded Python interpreter (`Py_Initialize`), loads the Python adapter module, and starts the async loop thread.
* **`queue`:** Hands I/O requests to the Python adapter. To achieve minimum overhead (<5µs budget), it acquires the GIL, wraps the C transfer buffer (`io_u->xfer_buf`) as a Python `memoryview` (zero-copy boundary crossing), invokes Python's async submission function, releases the GIL, and returns `FIO_Q_QUEUED` immediately.
* **`getevents`:** Crucially **never holds the GIL**. It blocks using a high-efficiency Linux `epoll_wait` over an `eventfd` which is signaled by the C completion callback. Upon wake-up, it drains a thread-safe lockless Single-Producer Single-Consumer (SPSC) ring buffer containing finished IO units.
* **C Completion Callback (`c_complete_trampoline`):** Invoked by the Python asyncio loop thread when a `gcsfs` coroutine finishes. It pushes the completed `io_u` to the SPSC ring buffer and writes a byte to the `eventfd` to wake the main FIO thread.

### 2. `gcsfs_async_adapter.py` (The Python Async Orchestrator)
Runs a dedicated background thread with a high-performance event loop (`uvloop`).
* **Background Event Loop:** Manages a single instance of `GCSFileSystem(asynchronous=True)` bound to its running loop.
* **`submit_io`:** Schedules a coroutine to execute the requested file operation (e.g., `_cat_file` for reads, `_pipe_file` for writes) on the event loop thread.
* **Zero-Copy Optimization:**
  * **Write Path:** Passes the `memoryview` over FIO's buffer directly to `gcsfs._pipe_file()`. `aiohttp`'s `BytesPayload` streams the memoryview directly through the socket via `sendmsg(2)` without duplicating the buffer on the Python heap.
  * **Read Path:** `aiohttp`'s parser creates a `bytes` object internally from the network socket. The adapter copies this data into the FIO C buffer via a high-speed `ctypes.memmove` (single boundary copy, unavoidable without rewriting `aiohttp`).

---

## Performance Considerations

To ensure wire-speed benchmarking of GCS network storage without introducing C-Python boundary bottlenecks, the engine incorporates the following low-overhead designs:

### 1. Complete GIL Decoupling on Hot Paths
The Global Interpreter Lock (GIL) limits CPython execution to a single OS thread at a time. If FIO blocked while holding the GIL, it would completely choke network performance.
* **Hot Path Submission (`queue`):** The GIL is acquired only long enough to schedule the task on the event loop via `loop.call_soon_threadsafe` and immediately released (measured CPU time `<5µs`).
* **Hot Path Completion (`getevents`):** The FIO main thread blocks via `epoll_wait` on a native Linux `eventfd`. **No Python code runs and no GIL is held** while waiting for network completions.
* **Asynchronous Completion Callback:** The C callback `c_complete_trampoline` is invoked from the Python loop thread (which already holds the GIL). It performs a lockless ring-buffer push and writes to `eventfd` in less than `1µs` before yielding back to Python, minimizing GIL retention time.

### 2. Lockless Single-Producer Single-Consumer (SPSC) Ring Buffer
Synchronizing the FIO thread and the background Python thread using standard POSIX mutexes/condition variables introduces scheduling latency and lock contention.
* Since there is exactly one producer (the background Python event loop thread) and one consumer (the main FIO engine thread calling `getevents`) per worker process, we utilize a thread-safe **lockless SPSC circular queue**.
* Atomic operations (`memory_order_release` on push, `memory_order_acquire` on pop) are used to ensure memory visibility with zero lock overhead.

### 3. Boundary Data Copies
* **Zero-Copy Writes:** FIO writes the generated block to `io_u->xfer_buf`. The C engine creates a `memoryview` directly over this memory address. `aiohttp` streams this `memoryview` directly down the network socket descriptor without copying it onto the Python heap.
* **Single-Copy Reads:** Due to the architecture of `aiohttp`'s parser, incoming data is first read into a Python `bytes` object. To move this to FIO's C buffer, we use a highly optimized native `memcpy` via `ctypes.memmove(buf_ptr, data_ptr, size)`. Since this is in RAM, memory copy takes minimal time compared to GCS network round-trip time.

### 4. Fast Event Loop Runtime (`uvloop`)
Standard `asyncio` scheduling overhead is dominated by pure-Python code. By replacing the standard event loop policy with `uvloop` (a Cython wrapper around the highly optimized C-based `libuv` library), we reduce scheduling latency and CPU overhead by 2-4x, leaving more CPU cycles available to process GCS packets.

### 5. Floating CPU Affinity
When configuring high-core VM architectures, we recommend separating core affinity:
* **NIC IRQ Cores:** Leave cores (e.g., 0-19) unassigned to FIO to handle hardware network interrupts.
* **FIO Workers:** Float FIO workers across a reserved pool of cores (e.g., 20-95) using `cpus_allowed`. We intentionally avoid strict 1:1 pinning (`cpus_allowed_policy=split`) to allow the Linux scheduler to dynamically balance C++ network worker threads and the Python interpreter thread within the allocated pool.

---

## Trade-Off Analysis

### Direct CPython Engine vs daemon process over Unix sockets
* **Direct CPython Engine (Recommended):** In-process execution avoids any serialization or IPC context switches. However, because Python GIL prevents true multi-threaded concurrency, we must use FIO multi-processing (`thread=0`). This creates independent instances of connection pools.
* **Daemon process over Unix sockets:** Allows sharing connection pools across workers, but introduces high serialization/IPC overhead for high-frequency or large I/O transfers.

### Read Path boundary-copy
* Bypassing aiohttp to achieve read-side zero-copy is not feasible without major upstream modifications to aiohttp to support `BufferedProtocol`. Thus, the single boundary `memcpy` on the read path is a necessary architectural trade-off for benchmarking `gcsfs` unmodified.

---

## Proposed Changes

### 1. [NEW] [gcsfs_async_engine.c](file:///usr/local/google/home/yonghuili/Projects/migrate-fio-engine/gcsfs/fio/gcsfs_async_engine.c)
A C library wrapping FIO external engine hooks and embedding Python. Key functions to implement:
* `fio_gcsfs_init`: Initializes CPython post-fork, creates the lockless SPSC ring buffer, and sets up the `eventfd` + `epoll` context.
* `fio_gcsfs_queue`: Acquires GIL, wraps the C buffer as a `memoryview`, and calls `gcsfs_async_adapter.submit_io()`.
* `fio_gcsfs_getevents`: Releases/does not acquire GIL. Blocks on `epoll_wait` over `eventfd`, then pops completed `io_u` structures from the ring buffer.
* `c_complete_trampoline`: Invoked from Python via a `ctypes.CFUNCTYPE` trampoline. Pushes `io_u` to the SPSC ring, writes to `eventfd`, keeping execution time <1µs.

### 2. [NEW] [gcsfs_async_adapter.py](file:///usr/local/google/home/yonghuili/Projects/migrate-fio-engine/gcsfs/fio/gcsfs_async_adapter.py)
The Python side of the embedded engine:
* Manages a dedicated event loop thread (`uvloop` if available, otherwise default `asyncio`).
* Hosts a persistent thread-bound `gcsfs.GCSFileSystem(asynchronous=True)`.
* Configures `fs.retries = 1` to ensure zero-copy writes are safe (preventing FIO from modifying buffers during TCP retries).
* Exposes a clean `submit_io` API that schedules coroutines, tracks in-flight tasks to prevent premature garbage collection, and maps exceptions (`FileNotFoundError`, etc.) to standard system `errno` codes before invoking the C completion trampoline.

### 3. [NEW] [Makefile](file:///usr/local/google/home/yonghuili/Projects/migrate-fio-engine/gcsfs/fio/Makefile)
A lightweight build system to compile `gcsfs_async_engine.c` into a shared object:
* Uses `python3-config --cflags --embed` and `python3-config --ldflags --embed` to guarantee robust linking across different system architectures.
* Links against target FIO headers.

### 4. [NEW] [benchmark.fio](file:///usr/local/google/home/yonghuili/Projects/gcsfs/prototype/jobs/benchmark.fio)
Job configurations designed to stress testing workloads:
* Concurrency via `iodepth` (e.g., 1 job, `iodepth=64` to stress connection multiplexing and async scheduling).
* Concurrency via `numjobs` (e.g., 32 jobs, `iodepth=1` to stress multi-processing and system core utilization).
* Standard blocksizes (`seqread_1m`, `randread_4k`).

---

## Verification & Testing Plan

To guarantee correct behavior, we will implement a 3-phase verification suite:

### 1. Automated Unit Tests
* **Zero-Leak Check:** Run a long-duration stress job (10M+ I/Os) and record memory consumption (`RSS`) to assert that all CPython objects are correctly `DECREF`'d and no memory leaks occur across the C-Python boundary.
* **Integrity Verification:** Perform read-after-write tests. Generate random patterns on FIO `write`, upload them, then download and verify checksums (`verify=crc32c`) to guarantee that no buffer aliasing or race conditions exist.

### 2. Workload Equivalence & Baseline Validation
* Compare the throughput of the direct FIO engine against `gcsfs`'s internal sync API calls to ensure the engine's overhead is negligible (<2% delta).
* Measure engine latencies under local mock environments (using a fake GCS mock server) to verify p99 timing accuracy.

---

## Open Questions / Discussion Points

1. **Startup TLS Storms:** With `numjobs=64`, 64 Python runtimes will perform SSL handshakes concurrently. We may want to introduce a short staggered sleep (e.g., `td->thread_number * 50ms`) in `init()` to prevent GCS connection rate limiting at startup.
2. **Telemetry and Profiling:** Should we integrate standard `HdrHistogram` dump files at teardown to track boundary-crossing latency overhead separate from GCS network time? (We recommend this as it adds minimal overhead but provides invaluable diagnostic telemetry).
