# Profiling & Optimization Plan: Monolithic Checkpointing over GCSFS

This document outlines the strategy, reproduction steps, and architectural hypotheses to profile monolithic checkpointing (`torch.save` over `gcsfs` on standard and zonal HNS buckets) to identify bottlenecks and optimize GCSFS throughput.

---

## 1. Architectural Bottleneck & Primary Hypothesis

Under monolithic saving (`torch.save` / `pickle.dump`), model and optimizer weights are gathered onto Rank 0 and written sequentially to a file-like stream using Python's standard `pickle` engine.

### The Hypothesis

* **Lack of Client-Side Buffering in `ZonalFile`**: While standard `GCSFile` inherits from `fsspec.spec.AbstractBufferedFile` (which buffers writes up to `block_size` in memory), `ZonalFile` overrides `write()` entirely:
  ```python
  def write(self, data):
      ...
      self._ensure_aaow()
      asyn.sync(self.gcsfs.loop, self.aaow.append, data)
      ...
  ```
  Every time `pickle` or `torch.save` issues a `.write(data)` call (often in tiny blocks of 1KB to 128KB), `ZonalFile` blocks the main thread to run `asyn.sync`.
* **Synchronization Overhead**: `asyn.sync` context-switches to the asyncio event loop thread and waits for the gRPC stream append to complete. For a 48 GB checkpoint, writing in small chunks triggers millions of thread-synchronization events, saturating the CPU and starving the high-performance gRPC pipeline.

---

## 2. Profiling Methodology

To isolate and confirm this bottleneck without the overhead of running a full 4-rank distributed training cluster, we will use a **lightweight, isolated micro-reproduction script**.

### A. Profiling Tools & Visualizations
We will profile the micro-reproduction script using:
1. **`cProfile`**: To capture deterministic function call counts and exact cumulative time spent inside `asyn.sync` and `ZonalFile.write`.
2. **`py-spy`**: A non-intrusive sampling profiler to generate an interactive **Flame Graph** showing CPU consumption in the main thread vs. the asyncio loop thread.
3. **Custom Instrumentation**: Injecting a temporary logging wrapper to record the exact chunk size distribution of write calls issued by `torch.save`.

---

## 3. Micro-Reproduction Script (`profile_monolithic.py`)

This standalone script creates simulated model/optimizer weights matching the Llama 3.1 8B payload size (~10 GB to 48 GB) and saves them directly to standard and zonal buckets while collecting performance metrics.

```python
import os
import time
import torch
import fsspec
import cProfile
import pstats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Target GCS paths (Passed via environment or arguments)
GCS_ZONAL_PATH = os.getenv("CKPT_WRITE_PATH", "gs://yonghui-us-central1-b") + "/profile_test_zonal.ckpt"

def generate_simulated_state_dict(num_params_billions=1):
    """Generates simulated weights + AdamW states matching real model sizes.
    1 Billion parameters is ~2 GB bf16 weights + ~4 GB AdamW optimizer moments = ~6 GB.
    """
    logging.info(f"Generating simulated state dict for {num_params_billions}B parameters...")
    # Simulate model weights in bf16
    weights = torch.randn(int(num_params_billions * 5e8), dtype=torch.bfloat16)
    # Simulate 2 AdamW moments
    moment1 = torch.zeros_like(weights)
    moment2 = torch.zeros_like(weights)
    
    return {
        "model": {"weights": weights},
        "optimizer": {
            "state": {
                0: {"exp_avg": moment1, "exp_avg_sq": moment2}
            }
        }
    }

def run_save(filepath, state_dict):
    logging.info(f"Opening remote file: {filepath}")
    fs = fsspec.open(filepath, "wb")
    
    start_time = time.perf_counter()
    with fs as f:
        # Wrap torch.save with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        torch.save(state_dict, f)
        
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(30)  # Print top 30 hot methods
        
    duration = time.perf_counter() - start_time
    size_gb = os.path.getsize(filepath) / (1024**3) if not filepath.startswith("gs://") else 0
    logging.info(f"Finished saving to {filepath} in {duration:.2f} seconds.")

if __name__ == "__main__":
    # Ensure gcsfs experimental support is active
    os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "true"
    
    # 1B parameters payload (~6 GB) for fast profiling iteration
    state_dict = generate_simulated_state_dict(num_params_billions=1)
    
    logging.info("--- Starting Monolithic Checkpoint Save (Zonal Bucket) ---")
    run_save(GCS_ZONAL_PATH, state_dict)
```

---

## 4. Proposed Optimizations in GCSFS

If profiling confirms high `asyn.sync` latency and call volume, we can implement the following enhancements in `gcsfs`:

### Phase 1: Client-Side Write Buffering in `ZonalFile`
Introduce a lightweight, thread-safe memory buffer in `ZonalFile.write` (e.g., using `bytearray`).
* Accumulate incoming bytes locally in Python space.
* Trigger `asyn.sync(self.gcsfs.loop, self.aaow.append, bytes_to_write)` **only** when the local buffer exceeds a configurable chunk threshold (e.g., `flush_interval_bytes` or `block_size`, defaults to 16 MiB).
* Flush any remaining buffer contents during `.flush()`, `.commit()`, and `.close()`.

### Phase 2: Async Write Offloading / Pipelining
Instead of synchronously blocking the serializing thread (`asyn.sync`), we can dispatch the append operation to the asyncio loop via a future queue with bounded backpressure. This lets `pickle` continue serializing the next tensor chunk immediately while the previous chunk is streamed in the background.

---

## 5. Step-by-Step Execution Roadmap

```text
  STEP 1: Profile Baseline
  Run profile_monolithic.py & py-spy flame graphs on standard vs. zonal buckets.
       │
       ▼
  STEP 2: Analyze Bottlenecks
  Validate if asyn.sync call frequency matches pickle write chunk frequency.
       │
       ▼
  STEP 3: Implement ZonalFile Write Buffer
  Modify gcsfs/zonal_file.py to implement client-side buffering.
       │
       ▼
  STEP 4: Verify Correctness & Throughput
  Re-run profile_monolithic.py to verify data integrity and measure speedup.
       │
       ▼
  STEP 5: Validate via Macrobenchmark
  Run run_macro.sh with USE_DCP=false to verify real-world Llama 3.1 8B benefits.
```
