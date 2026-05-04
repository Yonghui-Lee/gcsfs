# Detailed Design: Direct Async Fio Engine for gcsfs

This document provides a detailed design for a simplified, direct asynchronous `fio` engine for `gcsfs` using a single C file and a Python adapter. It bypasses the complex multi-client framework previously discussed but incorporates key architectural decisions required to achieve high-performance Python networking and benchmarking accuracy.

---

## 1. Architecture Overview

The solution consists of two main parts:

1.  **`gcsfs_engine.c` (compiles to `gcsfs_engine.so`)**: A direct `fio` engine that embeds the Python interpreter.
2.  **`gcsfs_async_adapter.py`**: A Python script running a high-performance background event loop to handle `gcsfs` calls.

### Execution Flow:

1.  FIO calls `queue` in `gcsfs_engine.c` with an `io_u` struct.
2.  The C code acquires the GIL, calls the Python adapter to submit the task, and releases the GIL. It returns `FIO_Q_QUEUED` immediately.
3.  The Python adapter schedules the `gcsfs` operation on the background loop.
4.  Upon completion, the Python task invokes a C callback via `ctypes`.
5.  The C callback enqueues the completed `io_u` and signals `getevents`.
6.  `getevents` harvests completed events without holding the GIL while waiting.

---

## 2. Python Embedding & GIL Management (Crucial for Performance)

### The FIO Fork Model vs. Threading
Python's Global Interpreter Lock (GIL) is a major bottleneck for high-throughput network I/O when using C-threads. If FIO uses threads (`thread=1`), all threads will share a single Python interpreter and violently contend for the GIL.

- **Mandated Approach**: Use FIO's default process forking model (`thread=0`). This ensures that each FIO worker process gets its own separate Python interpreter, its own memory space, and its own GIL, allowing true parallel execution. Since each FIO worker embeds Python within its own address space, passing pointers directly to Python via `ctypes` remains safe and avoids cross-process IPC overhead.
- **Initialization Timing**: Because Python does not handle `fork()` well when embedded, `Py_Initialize()` **MUST NOT** be called in the main FIO process before forking. FIO executes `setup` in the main process before `fork()`, and `init` in the child process after `fork()`. Therefore, `Py_Initialize()` must strictly occur in the `init` function (post-fork), and never in `setup`.

### GIL Handling within a Worker
Even with separate processes, C calls to Python within the worker must acquire the GIL:

```c
PyGILState_STATE gstate = PyGILState_Ensure();
// ... Call Python C API ...
PyGILState_Release(gstate);
```
The C worker holds the GIL only briefly to dispatch tasks. Asynchronous network I/O is handled by the background Python event loop.

---

## 3. High-Performance Asyncio & Networking

### Event Loop
Standard `asyncio` will be used as the event loop for the Python adapter.

### CPU Pinning
To avoid clashing with Network Interface Card (NIC) hardware interrupts, it is highly recommended to pin FIO worker processes to specific CPU cores. Use FIO's `cpus_allowed` parameter in the job file to isolate FIO workers to dedicated cores, leaving other cores free for OS network stack processing.

---

## 4. C Engine Structure (`struct ioengine_ops`)

The engine exports a `struct ioengine_ops` utilizing asynchronous execution (`FIO_SYNCIO` is not used).

### Key Functions to Implement:

- **`setup`**:
  - Executes in the main process pre-fork. **Must NOT invoke any Python C API.**
  - Can be used to parse options and reject random writes as GCS does not support them.
- **`init`**:
  - Executes post-fork in the worker process.
  - Initialize the Python interpreter (`Py_Initialize()`).
  - Release the GIL (`PyEval_SaveThread()`) so the background loop can run.
  - Adjust `sys.path` and import `gcsfs_async_adapter`.
- **`queue`**:
  - Acquire GIL (`PyGILState_Ensure()`).
  - Call Python `adapter.submit_io()`, passing the `io_u` pointer, operation type, path, offset, length, and buffer pointer.
  - Release GIL and return `FIO_Q_QUEUED`.
- **`get_file_size`**:
  - Call Python `adapter.get_file_size(path)` to return object size to `fio`.
- **`close_file`**:
  - Call Python `adapter.close_file(path)`. Critical for writes to finalize multipart uploads.
- **`getevents`**:
  - Wait for completions on a condition variable or file descriptor without holding the GIL until at least `min` events are available.

---

## 5. Data Transfer (Zero-Copy)

To achieve high throughput, memory copies between C and Python must be minimized or eliminated.

- **Writes (C to Python)**: Use `PyMemoryView_FromMemory` to create a read-only Python memoryview pointing directly to FIO's `xfer_buf`. Pass this memoryview to the `gcsfs` write operations to avoid copying payload data.
- **Reads (Python to C)**: The C code passes the raw buffer pointer to Python. Once `gcsfs` fetches the data into a bytes object, use `ctypes.memmove(buffer_ptr, data, len(data))` to copy it directly into FIO's buffer, bypassing intermediate serialization.

---

## 6. Python Adapter Implementation (`gcsfs_async_adapter.py`)

```python
import asyncio
import threading
import ctypes
import time
import gcsfs

class GCSFSAsyncAdapter:
    def __init__(self, c_complete_ptr):
        c_complete_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)
        self.c_complete = c_complete_type(c_complete_ptr)

        self.fs = gcsfs.GCSFileSystem(asynchronous=True)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit_io(self, io_u_ptr, gcs_op, path, offset, length, buffer_ptr):
        asyncio.run_coroutine_threadsafe(
            self._do_io(io_u_ptr, gcs_op, path, offset, length, buffer_ptr),
            self.loop
        )

    async def _do_io(self, io_u_ptr, gcs_op, path, offset, length, buffer_ptr):
        error = 0
        # Optional: Start OpenTelemetry span here
        start_time = time.perf_counter()
        try:
            if gcs_op == "create_new":
                # Handle sequential write/pipe
                pass
            elif gcs_op == "read":
                data = await self.fs._cat_file(path, start=offset, end=offset+length)
                ctypes.memmove(buffer_ptr, data, len(data))
            elif gcs_op == "delete_object":
                await self.fs._rm(path)
        except Exception as e:
            print(f"I/O Error: {e}")
            error = 5  # EIO
        finally:
            # Optional: End span and record duration
            # duration = time.perf_counter() - start_time
            self.c_complete(io_u_ptr, error)
```

---

## 7. Telemetry & Observability

To align with broader benchmarking visibility goals, the `gcsfs` wrapper should capture internal operational metrics:
- **Metrics/Spans**: Wrap `_cat_file`, `_pipe_file`, and other `gcsfs` calls in OpenTelemetry spans or capture basic `time.perf_counter()` latency metrics.
- **Correlation**: This allows for correlating FIO-reported latency (which includes GIL acquisition and C-to-Python boundary crossing) with the actual time spent inside the `gcsfs` SDK and underlying HTTP requests.

---

## 8. Key Risks & Recommendations

- **GCS Immutability**: GCS objects are immutable and cannot be updated in place (e.g., random partial writes). FIO's random write patterns will fail unless mapped to completely overwriting the object. This engine is constrained to workload patterns that match object storage semantics: full object writes (`create_new`), arbitrary reads (`read`), and object deletions (`delete_object`).
- **Memory Leaks**: FIO may loop over millions of IOs. Extreme care must be taken with Python reference counting (`Py_DECREF`) in C. Leaking references during `queue` calls will rapidly consume memory, even within isolated worker processes.
- **Synchronous Operations**: Synchronous FIO operations like `open_file` and `close_file` must block on the Python event loop (e.g., using `asyncio.run_coroutine_threadsafe(...).result()`) to wait for completion, as `gcsfs` is async.

---

## 9. Compilation

You need FIO source headers and Python development libraries.

```bash
PY_CFLAGS=$(python3-config --cflags --embed)
PY_LDFLAGS=$(python3-config --ldflags --embed)
FIO_SRC=/path/to/fio/source

gcc -O3 -g -shared -fPIC \
    -o gcsfs_engine.so gcsfs_engine.c \
    -I${FIO_SRC} \
    ${PY_CFLAGS} ${PY_LDFLAGS}
```
