# High-Performance Embedded FIO Engines for `gcsfs`

This directory provides two high-performance, **CPython-embedded FIO I/O engines** designed to benchmark the `gcsfs` library at close to physical wire-speed. By embedding the CPython interpreter directly inside FIO worker processes, these engines bypass the massive FUSE context-switching bottleneck, allowing standard FIO tools to perform direct in-process evaluation of Python storage protocols, parallel gRPC connections, and high-throughput caching architectures.

---

## 1. Engine Architectures

Depending on your benchmarking goals, you can choose between two distinct runtime engines:

### A. Lockless Asynchronous Engine (`gcsfs_async`)
* **Shared Library:** `libgcsfs_async_fio_engine.so`
* **Focus:** Maximum raw physical network, gRPC, and hardware pipeline capacity.
* **Mechanism:** Decouples I/O submission and completion using a background thread running a high-speed `uvloop` (asyncio). Finished operations communicate back to the main FIO thread via an `eventfd` and a **lockless Single-Producer Single-Consumer (SPSC) circular ring buffer** using atomic memory barriers. The Python Global Interpreter Lock (GIL) is only held during task queueing (<5µs) and is completely bypassed during block transfers and completion blocking, enabling massive pipeline concurrency.
* **API Path:** Leverages GCSFS's high-speed private asynchronous block transport layer `_async_fetch_range`.

### B. Pure Synchronous Engine (`gcsfs_sync`)
* **Shared Library:** `libgcsfs_sync_fio_engine.so`
* **Focus:** Real-world application fidelity and standard fsspec cache profiling.
* **Mechanism:** Runs inline in the main FIO execution thread under standard Python GIL lifecycles. It completes block transfers synchronously on submission and returns `FIO_Q_COMPLETED` immediately.
* **API Path:** Invokes standard, public Python file-like methods (`f.seek()`, `f.read()`, `f.write()`), allowing measurement of the exact network, serialization, and cache latency patterns of production user client code.
* **Storage Backends & Protocols:** Supports any standard generic `fsspec` filesystem. The engine determines the filesystem based on path format:
  * **Local Storage:** Any path starting with `/`, `./`, `../`, or prefixed with `file://` (e.g., `file:///tmp/fio_test`). This is fully self-contained and does not require Google Application Default Credentials (ADC).
  * **Google Cloud Storage (GCS):** Standard paths without protocol (e.g., `my-bucket/file`) or prefixed with `gs://` (e.g., `gs://my-bucket/file`) use the low-latency `ExtendedGcsFileSystem`.
  * **Other Protocols:** Any registered `fsspec` protocol (e.g., `s3://`, `abfs://`) is dynamically dispatched to its corresponding generic `fsspec` filesystem on demand.

---

## 2. Architecture Visual Comparison

### Lockless Async Engine (`gcsfs_async`)
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
│     │           └── GCS async requests │
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
│           ├── f.read(size) / f.write() │
│           ├── Zero-copy memory copy    │
│           ├── Release GIL              │
│           ▼                            │
│     Return FIO_Q_COMPLETED             │
└────────────────────────────────────────┘
```

---

## 3. Quick Start (Google Cloud or Local Storage)

Follow these steps to prepare the system and run either cloud-based or local file benchmarks.

### Prerequisites (Steps 1 & 2)

#### Step 1: Install System Prerequisites
Ensure you have FIO and Python development headers installed on your Linux host:

```bash
# On Debian/Ubuntu based systems:
sudo apt-get update && sudo apt-get install -y fio libfuse3-dev python3-dev
```

#### Step 2: Compile the Engines
Build both dynamic shared engines by running:
```bash
make
```
This automatically downloads matching FIO source headers and compiles:
* `libgcsfs_async_fio_engine.so` (Asynchronous engine)
* `libgcsfs_sync_fio_engine.so` (Synchronous engine)

---

### Option A: Google Cloud Benchmarks (Zonal HNS Buckets)

#### Step A3: Configure Authentication
Configure standard Google Application Default Credentials (ADC) to authorize access to your Google Cloud projects:
```bash
gcloud auth application-default login
```
Ensure your running user or service account has the **Storage Admin** role (`roles/storage.admin`) or **Storage Object Admin** role (`roles/storage.objectAdmin`) on the target project. This is a requirement for GCSFS's Storage Control API interaction (used under the hood to perform HNS bucket layout detection and atomic folder operations).

#### Step A4: Create a High-Performance Zonal HNS Bucket
Zonal buckets are optimized for low latency and high throughput, making them ideal for high-performance benchmarks. You must enable uniform bucket-level access and the Hierarchical Namespace (HNS) feature at the time of creation:

```bash
# Replace with your unique bucket name, region, and target zone (e.g. us-central1-a)
export BUCKET_NAME="my-zonal-hns-bucket"
export BUCKET_ZONE="us-central1-a"

gcloud storage buckets create gs://${BUCKET_NAME} \
    --location=${BUCKET_ZONE} \
    --enable-hierarchical-namespace \
    --uniform-bucket-level-access
```

#### Step A5: Run the Cloud Benchmark Smoke Tests
Trigger the pre-configured smoke test suite against your new cloud Zonal bucket:
```bash
# Ensure the variable is exported in your environment
export BUCKET_NAME="my-zonal-hns-bucket"

make run-smoke-all
```

---

### Option B: Local fsspec Benchmarks (No Cloud/Credentials Required)

If you don't have active GCP access or just want to benchmark standard local file operations with fsspec's caching configurations (such as standard read-ahead buffering), you can run entirely locally without cloud buckets or credentials!

#### Step B3: Run the Local Benchmark Smoke Tests
Trigger the pre-configured local storage benchmarks suite by running:
```bash
make run-smoke-local
```
This writes a 100MB temporary file to `./tmp/fio_local_test_file` and then reads it back using the synchronous engine with the standard `readahead` cache strategy enabled.

You can also run specific local jobs directly via `./run.sh` without setting the `BUCKET_NAME` environment variable:
```bash
# Run local synchronous write benchmark
./run.sh jobs/smoke_test_sync_write_local.fio

# Run local synchronous read benchmark
./run.sh jobs/smoke_test_sync_read_local.fio
```

#### Specifying Local Storage Paths in Custom Workloads
To configure your own FIO workloads/job profiles to target local files, define the `filename` option in your `.fio` file using either of the following formats:

* **Implicit Local Paths (Relative or Absolute):** Any path starting with `/`, `./`, or `../` is automatically routed to the local filesystem.
  ```fio
  filename=./tmp/fio_local_test_file
  # OR: filename=/tmp/fio_local_test_file
  ```

* **Explicit URI Scheme (`file://`):** If your target file sits directly in the root of your current directory or you prefer explicit schemas:
  ```fio
  filename=file://my_benchmark_file.bin
  # OR: filename=file:///tmp/fio_local_test_file
  ```

Under the hood, the engine dynamically routes these paths to standard `fsspec.filesystem("file")` and strips any GCS-exclusive capabilities (such as adaptive prefetching, connection pools, or multi-range vector cache properties). For writes, the engine also pre-creates parent folders automatically (like `./tmp/` or custom subdirectories) to ensure standard filesystems don't encounter folder missing errors.

---

## 4. The Smart Launcher Wrapper (`./run.sh`)

Python extension modules (especially binary dependencies like `cryptography` or `grpc`) require their parent Python library symbols to be globally visible. To solve this dynamically without setting complex manual paths, use the provided launcher:

```bash
# Example: Running a sequential zonal benchmark profile
export BUCKET_NAME="my-zonal-hns-bucket"
./run.sh jobs/seqread_prefetch_zonal.fio
```

### What `./run.sh` does under the hood:
1. **Validates requirements:** Verifies that `BUCKET_NAME` is defined when a `.fio` job spec is targeted.
2. **Auto-detects the active Python environment's shared library** (`libpython3.X.so`), resolving virtual environments (`venv`), `pyenv`, `conda`, and standard system structures.
3. **Preloads the library** via `LD_PRELOAD` to safely publish all global Python interpreter symbols.
4. **Sets the correct `PYTHONPATH`** pointing back to the repository root.
5. **Executes FIO** with all arguments passed through transparently.

---

## 5. Custom Engine Options

You can tune the embedded Python parameters directly from your FIO job files by declaring the custom keys on distinct lines:

```fio
[sync_chunked_job]
ioengine=external:./libgcsfs_sync_fio_engine.so
use_prefetch=0
cache_type=readahead_chunked
block_size=8388608
```

### A. Lockless Async Engine (`libgcsfs_async_fio_engine.so`) Options

| Option Key | Data Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `flush_every_write` | Boolean | `0` (False) | If true, forces the writer stream to flush back to storage after every append payload. |

### B. Pure Synchronous Engine (`libgcsfs_sync_fio_engine.so`) Options

| Option Key | Data Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `use_prefetch` | Boolean | `1` (True) | Enables the experimental adaptive background prefetcher (requires read mode). When active, sets `cache_type` to `"none"` by default to avoid double-buffering. |
| `cache_type` | String | `none` (if prefetch=1)<br>`readahead` (if prefetch=0) | GCSFS sync file-level cache strategy. Options include: `none`, `readahead`, `readahead_chunked`, `bytes`, `mmap`. |
| `concurrency` | Integer | `4` | Number of concurrent network connection requests GCSFS uses to fetch data. |
| `block_size` | Integer | `16777216` (16MB) | Read-ahead cache buffer size (bytes) passed directly to the GCSFS file constructor (`fs.open()`). |

---

## 6. Directory Structure & Job Profiles

* **`gcsfs_async_engine.c` / `gcsfs_async_adapter.py`**: Lockless Async dynamic engine C source and Python mapping logic.
* **`gcsfs_sync_engine.c` / `gcsfs_sync_adapter.py`**: Synchronous dynamic engine C source and Python fsspec-caching mapping.
* **`run.sh`**: Dynamic Python path environment helper tool.
* **`Makefile`**: Compilation actions, simple cloud smoke validation suite wrappers.
* **`jobs/`**: Pre-defined benchmark job templates:
  * `smoke_test_write.fio` / `smoke_test_read.fio`: Base async write/read test profiles (updated to sync by default).
  * `smoke_test_sync_write.fio`: Baseline synchronous write stream.
  * `smoke_test_sync_read.fio`: Baseline synchronous raw read (no cache, no prefetching).
  * `smoke_test_sync_read_prefetch.fio`: Synchronous read utilizing adaptive background prefetching.
  * `smoke_test_sync_read_readahead.fio`: Synchronous read utilizing standard fsspec read-ahead buffering.
  * `smoke_test_sync_read_readahead_chunked.fio`: Synchronous read utilizing chunked read-ahead caching (designed for high-speed Zonal HNS buckets).
  * `seqread_prefetch_zonal.fio`: Large 10G sequential read profile targeting dynamic background prefetching on a Zonal HNS bucket.
  * `seqread_16m_multi.fio` / `seqread_1m_multi.fio`: Zonal sequential read benchmarks with higher concurrency.
  * `smoke_test_sync_write_local.fio` / `smoke_test_sync_read_local.fio`: Local storage smoke tests verifying standard write/read and fsspec caching overhead with no cloud dependencies.

---

## 7. Performance Recommendations for Zonal Buckets

To maximize performance when running benchmarks on actual Google Cloud Zonal buckets, follow these general patterns:

* **Adaptive Prefetching (`use_prefetch=1`):** In read tests, enabling the background prefetcher yields the highest throughput because it spins up concurrent, pre-emptive read threads that bypass standard inline cache parsing. By setting `cache_type=none`, it completely avoids double-buffering.
* **Vectorized Read-Ahead (`cache_type=readahead_chunked`):** If you are not using background prefetching, the `"readahead_chunked"` cache strategy is the most optimized fsspec cache strategy for Zonal buckets. It taps into GCSFS's dynamic vector-range read logic (using high-throughput concurrent gRPC chunk downloads), maintaining standard read-ahead semantics while avoiding large data memory slice copying.
* **GIL Bypassing Async Engine (`libgcsfs_async_fio_engine.so`):** For maximum throughput tests designed to reach full line rate (especially over large multi-job benchmarks), always prefer the lockless asynchronous engine, as it releases the Python GIL during the I/O completion path to avoid processing bottlenecks.

---

## 8. Troubleshooting & Manual Execution

If you prefer not to use the automated `./run.sh` script, you can configure your shell environment variables manually:

1. **Locate your shared Python library path:**
   ```bash
   python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))"
   # Example output: /usr/lib/libpython3.10.so
   ```
2. **Preload the library manually:**
   ```bash
   export LD_PRELOAD="/usr/lib/libpython3.10.so"
   ```
3. **Configure PYTHONPATH pointing back to the repo root:**
   ```bash
   export PYTHONPATH="/path/to/gcsfs"
   ```
4. **Configure your target bucket name:**
   ```bash
   export BUCKET_NAME="my-zonal-hns-bucket"
   ```
5. **Invoke FIO directly:**
   ```bash
   fio jobs/smoke_test_read.fio
   ```

---

## 9. Advanced Environment Toggles

When profiling, benchmarking, or troubleshooting workloads with the dynamic FIO I/O engines, you can tap into the following environment variables to diagnose performance profiles or control networking paths:

### A. DirectPath Control (`GOOGLE_CLOUD_DISABLE_DIRECT_PATH`)
By default, the asynchronous gRPC client in GCSFS attempts to use Google Cloud **DirectPath** (direct network pipeline connectivity bypassing frontend proxies) to achieve maximum line-rate throughput when executing benchmarks from within supported Google Cloud environments (like GKE or GCE VMs).

If you encounter networking restrictions, DNS resolution bugs (e.g., DNS amplification overhead), or wish to explicitly route traffic through standard Google Frontend (GFE) proxies to measure proxy latency differences, you can disable DirectPath:

```bash
# Disable DirectPath connectivity (forces standard proxy endpoints)
export GOOGLE_CLOUD_DISABLE_DIRECT_PATH=true
```

*   **Syntax & Robustness:** The toggle is highly robust and accepts standard truthy values case-insensitively, including `"true"`, `"1"`, `"yes"`, and `"on"`. Strips any leading/trailing whitespace automatically.

### B. Library Diagnostics & Log redirection (`GCSFS_DEBUG` / `GCSFS_LOG_FILE`)
To capture internal diagnostics, raw connection state events, HNS layout cache hits/misses, or retry events without muddying standard FIO outputs (which can break machine-readable FIO json parsing engines):

1.  **Enable Debug Level Streams:**
    ```bash
    export GCSFS_DEBUG=DEBUG  # Supports: DEBUG, INFO, WARNING, ERROR
    ```
2.  **Redirect to a Logfile:** To prevent raw log entries from polluting FIO's standard terminal stdout output, redirect all GCSFS logging events directly to a file:
    ```bash
    export GCSFS_DEBUG=DEBUG
    export GCSFS_LOG_FILE=gcsfs_fio_run.log
    ```

*   **Format:** Logging to a file uses a standardized structured design: `YYYY-MM-DD HH:MM:SS [LEVEL] logger_name: message` (e.g., `2026-05-29 05:15:30 [DEBUG] gcsfs: GET: b/my-zonal-hns-bucket/o/...`).
