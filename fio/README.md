# High-Performance Embedded FIO Engines for `gcsfs`

This directory contains two high-performance, CPython-embedded external I/O engines for FIO designed to benchmark `gcsfs` at wire-speed directly in-process. By embedding the CPython interpreter directly inside FIO worker processes, these engines bypass the FUSE context-switching bottleneck, allowing direct performance evaluation of Python storage protocols.

---

## 1. Engine Options

Depending on your benchmarking goals, you can choose between two distinct runtime architectures:

### A. Lockless Asynchronous Engine (`gcsfs`)
* **Shared Library:** `libgcsfs_fio_engine.so`
* **Focus:** Maximum raw physical network/gRPC capacity.
* **Mechanism:** Completely decouples I/O submission and completion using a background thread running `uvloop` (asyncio), communicating finished operations back to the main FIO thread via an `eventfd` and a **lockless Single-Producer Single-Consumer (SPSC) circular ring buffer** using atomic memory operations. The Python Global Interpreter Lock (GIL) is only held during submission (<5µs) and is completely bypassed during completion blocking, allowing massive parallel pipeline concurrency.
* **API Path:** Uses GCSFS's internal private async block transport layer `_async_fetch_range`.

### B. Pure Synchronous Engine (`gcsfs_sync`)
* **Shared Library:** `libgcsfs_sync_fio_engine.so`
* **Focus:** Real-world application fidelity and fsspec cache strategy profiling.
* **Mechanism:** Runs inline in the FIO execution thread under standard Python GIL lifecycles, completing the block transfers synchronously on submission and returning `FIO_Q_COMPLETED` immediately.
* **API Path:** Directly invokes standard, public Python file-like methods: `f.seek(offset)` and `f.read(size)` / `f.write(data)`. This allows measuring the exact network, serialization, and cache-lookup latency of standard production Python GCSFS client code.

---

## 2. Architecture Comparison

### Lockless Async Engine (`gcsfs`)
```
  fio worker (process)
┌────────────────────────────────────────┐
│  fio core loop                         │
│     │                                  │
│     ├── queue() (Acquires GIL, <5µs)   │
│     │     ▼                            │
│     │  Python gcsfs_adapter            │
│     │     │                            │
│     │     └── asyncio loop thread      │
│     │           │                      │
│     │           └── gcsfs operations   │
│     │                                  │
│     └── getevents() (No GIL held)      │
│           ▲                            │
│           ├── epoll_wait(eventfd)      │
│           └── pops SPSC Ring Buffer    │
└────────────────────────────────────────┘
```

### Pure Synchronous Engine (`gcsfs_sync`)
```
  fio worker (process/thread)
┌────────────────────────────────────────┐
│  fio core loop                         │
│     │                                  │
│     └── queue()                        │
│           │                            │
│           ├── Acquire GIL              │
│           ├── f.seek(offset)           │
│           ├── f.read(size)             │
│           ├── Zero-copy memory copy    │
│           ├── Release GIL              │
│           ▼                            │
│     Return FIO_Q_COMPLETED             │
└────────────────────────────────────────┘
```

---

## 3. Prerequisites & Setup

Ensure FIO and Python development headers are installed on your Linux host:

```bash
sudo apt-get update
sudo apt-get install -y fio libfuse3-dev python3-dev
```

### Compilation
Compile both shared library engines by linking them against FIO and Python runtimes:

```bash
# Downloads FIO headers matching the system FIO version and compiles both shared engines
make FIO_SRC=fio-src
```
This produces:
1. `libgcsfs_fio_engine.so` (Asynchronous direct-fetch engine)
2. `libgcsfs_sync_fio_engine.so` (Synchronous file-API engine)

---

## 4. Running Benchmarks

Always export the required Python environment variables to target libraries and PYTHONPATH.

### A. Local Emulator Smoke Tests
A local write/read validation pipeline runs instantly against a local GCS emulator.

#### 1. Setup the GCS Emulator
Start the dockerized fake-gcs-server and seed the target benchmark bucket:
```bash
# Start the emulator
docker run -d -p 4443:4443 --name gcsfs_test fsouza/fake-gcs-server:latest -scheme http -public-host 0.0.0.0:4443 -external-url http://localhost:4443 -backend memory

# Create the test bucket
curl -X POST -H "Content-Type: application/json" -d '{"name": "gcsfs-fio-benchmark"}' "http://localhost:4443/storage/v1/b?project=dummy-project"
```

#### 2. Run the Asynchronous Engine (`gcsfs`)
Test max pipeline capability over standard 16MB blocks:
```bash
# Sequential Write
LD_PRELOAD="/usr/local/google/home/yonghuili/.pyenv/versions/3.14.3/lib/libpython3.14.so" \
STORAGE_EMULATOR_HOST="http://localhost:4443" \
PYTHONPATH="/usr/local/google/home/yonghuili/Projects/gcsfs" \
fio jobs/smoke_test_write.fio

# Sequential Read
LD_PRELOAD="/usr/local/google/home/yonghuili/.pyenv/versions/3.14.3/lib/libpython3.14.so" \
STORAGE_EMULATOR_HOST="http://localhost:4443" \
PYTHONPATH="/usr/local/google/home/yonghuili/Projects/gcsfs" \
fio jobs/smoke_test_read.fio
```

#### 3. Run the Synchronous Engine (`gcsfs_sync`)
Test production file-like `f.read()` and `f.write()` calls under standard caching regimes.

To configure synchronous cache structures, customize options inside the FIO job target:
* **`cache_type`:** The standard fsspec cache strategy (e.g. `"none"`, `"readahead"`).
* **`block_size`:** Read-ahead cache buffer buffer size in bytes (e.g. `8388608` for 8MB block bounds).

```bash
# Sequential Write (Uses standard sync write buffers)
LD_PRELOAD="/usr/local/google/home/yonghuili/.pyenv/versions/3.14.3/lib/libpython3.14.so" \
STORAGE_EMULATOR_HOST="http://localhost:4443" \
PYTHONPATH="/usr/local/google/home/yonghuili/Projects/gcsfs" \
fio jobs/smoke_test_sync_write.fio

# Sequential Read - No Caching (Serial HTTP block requests)
LD_PRELOAD="/usr/local/google/home/yonghuili/.pyenv/versions/3.14.3/lib/libpython3.14.so" \
STORAGE_EMULATOR_HOST="http://localhost:4443" \
PYTHONPATH="/usr/local/google/home/yonghuili/Projects/gcsfs" \
fio jobs/smoke_test_sync_read.fio

# Sequential Read - standard Read-Ahead buffering
LD_PRELOAD="/usr/local/google/home/yonghuili/.pyenv/versions/3.14.3/lib/libpython3.14.so" \
STORAGE_EMULATOR_HOST="http://localhost:4443" \
PYTHONPATH="/usr/local/google/home/yonghuili/Projects/gcsfs" \
fio jobs/smoke_test_sync_read_readahead.fio
```

---

## 5. Synchronous Engine Custom Options

When configuring job specs targeting the external synchronous engine (`libgcsfs_sync_fio_engine.so`), you can supply custom tuning metrics on distinct lines under FIO job blocks:

```fio
[sync_readahead_job]
rw=read
use_prefetch=1
concurrency=4
block_size=8388608
```

| Option Key | Data Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `use_prefetch` | Boolean | `1` (True) | Enable the adaptive background prefetcher (requires read mode). When enabled, it sets `cache_type` to `"none"` to avoid double-buffering and redundant memory copies. |
| `concurrency` | Integer | `4` | Number of concurrent requests GCSFS uses to fetch data. |
| `block_size` | Integer | `16777216` (16MB) | Read-ahead cache bounds buffer bounds passed directly to the GCSFS `fs.open()` API. |

---

## 6. Important Notes & Architectural Constraints

### Zonal Cache Restrictions under Emulator
* **Layout Fallbacks:** GCS Zonal buckets leverage dynamic location plane metadata calls (via the Storage Control API) to resolve layout profiles. Because the local fake-gcs-server emulator does not implement the Storage Control API plane, layout lookups fail and ExtendedGcsFileSystem gracefully falls back to standard HTTP/REST `GCSFile` structures.
* **`readahead_chunked` Constraint:** The custom cache strategy `readahead_chunked` relies on zonal parallel chunk retrieval pipelines (bulk fetch ranges accepting a `chunk_lengths` keyword argument). Standard non-zonal HTTP `GCSFile` objects do not support `chunk_lengths` in their fetch channels. Consequently, attempting to run `cache_type=readahead_chunked` against a standard HTTP fallback target (including local emulator runs) correctly and safely raises `TypeError`. To test `readahead_chunked`, configure testing profiles on actual Google Cloud Zonal buckets.
