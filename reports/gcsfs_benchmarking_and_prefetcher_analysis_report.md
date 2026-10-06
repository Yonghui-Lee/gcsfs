# Ray Data Ingestion Performance Report: GCSFS vs. PyArrow C++ & Cache Optimization on Google Cloud Storage

---

## Key Findings at a Glance

- **GCSFS (No Cache) Dominates Throughput**: Achieves **4,910.55 MiB/s (4.80 GiB/s = 40.2 Gbps)** on Zonal storage at 32 workers, beating native PyArrow C++ (2,705.38 MiB/s) by **+81% (1.81×)**, and by **+149% (2.49×)** at 16 workers.
- **Pre-Buffer Coalescing Makes Caching Redundant**: PyArrow's Parquet reader (`pre_buffer=True`) coalesces columns into 28–200 MB ranges. GCSFS `ReadAheadCache` actively degrades performance by slicing these into 5 MiB blocks, while `BaseCache` streams them in continuous bursts.
- **Zero-Copy Arrow Mandate**: Default NumPy batch formatting instantiates 20.8 million Python `str` objects on The Pile, creating a 0.40s bottleneck that caps throughput at 950 MiB/s. Zero-copy Arrow (`batch_format="pyarrow"`) unlocks the full 4.91 GiB/s line rate.
- **Regional Storage Scaling**: On 64 files (200 MB each, 12.84 GB total) over HTTP/2 REST, GCSFS No Cache delivers **1,414.78 MiB/s** at 32 workers, leading all configurations.

---

## 1. Experimental Setup & System Topology

| Component | Specification & Environment Configuration |
|---|---|
| **Compute Instance** | Google Cloud Platform c4-standard-96 (Dedicated Host) |
| **CPU & Architecture** | 96 vCPUs (Intel Xeon Platinum 8581C 'Emerald Rapids' @ 2.60 GHz) |
| **System Memory** | 384 GiB DDR5 ECC RAM (178 GiB allocated to `/dev/shm` for Ray Plasma) |
| **Network Topology** | Google Virtual NIC (gvnic) with Tier-1 50 Gbps egress line rate |
| **Software Stack** | Linux 6.6.137+, Python 3.14.4, Ray 2.58.0, PyArrow 21.0.0, GCSFS 2026.1.0+ |
| **Zonal Target** | `gs://hf-pile-deduplicated-us-central1-b-gcsfs` (Zonal HNS via gRPC MRD, RTT < 0.2 ms) |
| **Regional Target** | `gs://yonghui-gcsfs-regional-us/ray_data_200mb` (Regional REST HTTP/2, RTT ~1.5 ms) |

---

## 2. Zonal Storage Workload: The Pile Deduplicated (256 Files / 65.2 GB)

The Zonal workload tests real-world pretraining ingestion on 256 Parquet files (~255 MB each) from Hugging Face The Pile Deduplicated. Each file contains 9 row groups (~53 MB uncompressed) with `text: string` columns, expanding to ~110 GB in-memory Arrow tables. Data is streamed using zero-copy Arrow (`batch_format="pyarrow"`).

| Backend / Configuration | 8 Workers | 16 Workers | 32 Workers | Line Rate (32W) | Speedup vs C++ |
|---|:---:|:---:|:---:|:---:|:---:|
| **GCSFS (No Cache)** | 1,685.36 MiB/s | 3,101.44 MiB/s | **4,910.55 MiB/s (4.80 GiB/s)** | **40.20 Gbps** | **+81% (1.81×)** |
| **GCSFS (Prefetch ON)** | 1,679.12 MiB/s | 3,087.47 MiB/s | 4,856.13 MiB/s (4.74 GiB/s) | 39.75 Gbps | +80% (1.80×) |
| **GCSFS (ReadAhead Cache)** | 1,574.91 MiB/s | 2,813.22 MiB/s | 4,512.87 MiB/s (4.41 GiB/s) | 36.96 Gbps | +67% (1.67×) |
| **PyArrow C++** | 973.89 MiB/s | 1,243.24 MiB/s | 2,705.38 MiB/s (2.64 GiB/s) | 22.15 Gbps | Baseline |

![Zonal Storage Streaming Throughput](charts/zonal_throughput_chart.png)
*Figure 1: Zonal Storage Streaming Throughput (The Pile Deduplicated, 256 Files / 65.2 GB)*

---

## 3. Regional Storage Workload: 64 Files (~200 MB Each / 12.84 GB Total)

The Regional benchmark evaluates 64 Parquet files (~200.7 MB each) stored in `gs://yonghui-gcsfs-regional-us/ray_data_200mb`. Transport occurs via standard HTTP/2 REST with connection pooling over regional network paths (RTT ~1.5 ms).

| Backend / Configuration | 8 Workers | 16 Workers | 32 Workers | Line Rate (32W) | Speedup vs ReadAhead |
|---|:---:|:---:|:---:|:---:|:---:|
| **GCSFS (No Cache)** | **438.81 MiB/s** | 749.02 MiB/s | **1,414.78 MiB/s (1.38 GiB/s)** | **11.58 Gbps** | **+18.5%** |
| **PyArrow C++** | 410.97 MiB/s | 775.04 MiB/s | 1,394.43 MiB/s (1.36 GiB/s) | 11.42 Gbps | +16.8% |
| **GCSFS (Prefetch ON)** | 366.46 MiB/s | **796.62 MiB/s** | 1,250.14 MiB/s (1.22 GiB/s) | 10.24 Gbps | +4.7% |
| **GCSFS (ReadAhead Cache)** | 394.68 MiB/s | 698.22 MiB/s | 1,193.55 MiB/s (1.17 GiB/s) | 9.77 Gbps | Baseline |

![Regional Storage Streaming Throughput](charts/regional_throughput_chart.png)
*Figure 2: Regional Storage Streaming Throughput (64 Files / 12.84 GB)*

---

## 4. Architectural Deep Dive: Why 'No Cache' Consistently Wins

In traditional POSIX filesystem benchmarks, disabling cache degrades performance. However, in Ray Data Parquet reading, GCSFS (No Cache / BaseCache) consistently delivers the highest throughput:

![Cache Coalescing Diagram](charts/cache_coalescing_diagram.png)
*Figure 3: PyArrow Range Coalescing vs. Filesystem ReadAhead Slicing Pathology*

- **Application Layer Range Coalescing**: PyArrow inspects the Parquet footer and coalesces all needed column chunks into large contiguous byte ranges (28 MB to 200 MB in a single read request). Redundant filesystem caching adds no value because the application layer has already planned the optimal byte range.
- **Eliminating Request Slicing**: `ReadAheadCache` defaults to a 5 MiB block size. When PyArrow requests 200 MB, `ReadAheadCache` slices it into 40 separate 5 MiB sequential HTTP range requests. `BaseCache` passes the entire 200 MB request straight through to high-speed async transport in one continuous burst.
- **Zero Async Context Switching**: The `BackgroundPrefetcher` runs coroutines, circular queues, and mutex locks. Under already-coalesced I/O, `BaseCache` avoids all intermediate memory copies and CPU scheduling overhead.
- **PyArrow C++ Lock Contention**: Native PyArrow C++ delegates I/O across thousands of worker threads via `google-cloud-cpp`, creating mutex lock thrashing in `libcurl` multi-handles. GCSFS's non-blocking `asyncio` event loop scales cleanly to 4.91 GiB/s.

---

## 5. Mathematical Analysis: The Zero-Copy Arrow Mandate

Ray Data pipelines are governed by:
$$T_{\text{pipeline}} = \max\left(T_{\text{I/O}}, T_{\text{decode}}, T_{\text{format}}, T_{\text{consume}}\right)$$

When reading unstructured text (The Pile), NumPy lacks a native UTF-8 string type. Ray Data's default `batch_format="numpy"` iterates over 81,405 rows per file and allocates individual Python `str` objects on the heap:

$$\text{Total String Allocations} = 81,405 \text{ rows/file} \times 256 \text{ files} = \mathbf{20,839,680 \text{ Python Heap Objects}}$$
$$T_{\text{format}} = 0.404 \text{ s per 250 MB file} \implies \text{Max Throughput} \approx \frac{250 \text{ MB}}{0.404 \text{ s}} \approx \mathbf{625 - 950 \text{ MiB/s}}$$

With `prefetch_batches=2`, Ray Data's executor halted 16 and 32 reader tasks with `ConcurrencyCap` backpressure. Switching to zero-copy Arrow (`batch_format="pyarrow"`) passes shared memory pointers (`/dev/shm`) directly, reducing $T_{\text{format}}$ to $\sim 0.0001\text{ s}$ and immediately unleashing line-rate throughput to **4.91 GiB/s**.

![Concurrency Scaling Linearity](charts/scaling_linearity_chart.png)
*Figure 4: Concurrency Scaling Linearity (Zonal gRPC vs. Regional HTTP/2 REST)*

---

## 6. Production Tuning & Recommendations

For maximum Ray Data ingestion throughput on Google Cloud Storage, configure your cluster `runtime_env` as follows:

```python
runtime_env = {
    "env_vars": {
        # 1. Enable Zonal Hierarchical Namespace gRPC acceleration
        "GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT": "true",
        # 2. Disable redundant caching to allow unfragmented range streaming
        "USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING": "false",
        "GCSFS_DEFAULT_CACHE_TYPE": "none",
        # 3. Ray Data Parquet reader thread pool tuning
        "RAY_DATA_PARQUET_READER_IO_THREAD_COUNT": "128",
        "RAY_DATA_PARQUET_READER_CPU_COUNT": "32",
        "RAY_DATA_PARQUET_FRAGMENT_BUFFER_SIZE": str(8 * 1024 * 1024),
    }
}

# Always consume as zero-copy Arrow Tables:
for batch in ds.iter_batches(batch_size=None, prefetch_batches=2, batch_format="pyarrow"):
    process_batch(batch)
```
