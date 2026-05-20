# High-Performance Direct FIO Engine for gcsfs

This directory contains a high-performance, CPython-embedded external I/O engine for FIO designed to benchmark `gcsfs` at wire-speed in-process. By embedding the CPython interpreter directly inside FIO worker processes and using lockless circular ring buffers, this engine completely avoids the Global Interpreter Lock (GIL) overhead and FUSE context switching.

---

## Reference Benchmark Results

The following table represents standard sequential read throughput benchmarks comparing GCSFS performance on **Rapid Buckets** (e.g., HNS/Zonal) vs. **Standard Buckets**:

### Sequential Reads Throughput (MB/s)

| IO Size | Processes | Rapid Bucket | Standard Bucket | Speedup Factor |
|---|---|---|---|---|
| **1 MB** | Single Process | 469.09 MB/s | 37.76 MB/s | **~12x** |
| **16 MB** | Single Process | 628.59 MB/s | 64.50 MB/s | **~9x** |
| **1 MB** | 48 Processes | 16,932.00 MB/s | 2,202.00 MB/s | **~7x** |
| **16 MB** | 48 Processes | 19,213.27 MB/s | 4,010.50 MB/s | **~4x** |

---

## Architecture

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

To bypass the GIL bottleneck, submission and completion paths are fully decoupled:
1. **GIL-free Completion:** The FIO thread blocks on `epoll_wait` over an `eventfd` without holding the GIL.
2. **Lockless SPSC Ring Buffer:** Communication of completed I/O requests between the asyncio thread and the main FIO thread uses atomic memory operations with zero locks.
3. **Zero-Copy Writes:** Passes FIO C buffers directly as `memoryview` objects to GCSFS resumable uploads, streaming bytes to GCS directly from FIO memory.

---

## Prerequisites & Setup

Ensure FIO and Python development headers are installed on your Linux system:

```bash
sudo apt-get update
sudo apt-get install -y fio libfuse3-dev python3-dev
```

### 1. Compilation

Compile the shared library `libgcsfs_fio_engine.so` by linking it against FIO header files. A local script compiles and links them:

```bash
# Downloads FIO headers matching the system FIO version and compiles the library
make FIO_SRC=fio-src
```

---

## Running Benchmarks

Always export the required Python environment variables to ensure correct loading of libraries and paths.

### 1. Local Smoke Test (Emulator)
You can run a local write/read test using the GCS emulator:

```bash
# Start the emulator in docker
docker run -d -p 4443:4443 --name gcsfs_test fsouza/fake-gcs-server:latest -scheme http -public-host 0.0.0.0:4443 -external-url http://localhost:4443 -backend memory

# Create the benchmark bucket
curl -X POST -H "Content-Type: application/json" -d '{"name": "gcsfs-fio-benchmark"}' "http://localhost:4443/storage/v1/b?project=dummy-project"

# Run Sequential Write Smoke Test
LD_PRELOAD="/usr/local/google/home/yonghuili/.pyenv/versions/3.14.3/lib/libpython3.14.so" \
STORAGE_EMULATOR_HOST="http://localhost:4443" \
PYTHONPATH="/usr/local/google/home/yonghuili/Projects/gcsfs" \
fio jobs/smoke_test_write.fio

# Run Sequential Read Smoke Test
LD_PRELOAD="/usr/local/google/home/yonghuili/.pyenv/versions/3.14.3/lib/libpython3.14.so" \
STORAGE_EMULATOR_HOST="http://localhost:4443" \
PYTHONPATH="/usr/local/google/home/yonghuili/Projects/gcsfs" \
fio jobs/smoke_test_read.fio
```

### 2. Real GCS Bucket Benchmark
To run a benchmark against a real bucket:

```bash
# Unset any emulator settings and let FIO run against real GCS
PYTHONPATH="/usr/local/google/home/yonghuili/Projects/gcsfs" \
LD_PRELOAD="/usr/local/google/home/yonghuili/.pyenv/versions/3.14.3/lib/libpython3.14.so" \
fio jobs/your_real_job_config.fio
```
