import os
import time
import torch
import fsspec
import cProfile
import pstats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Target GCS path (Passed via env, default to local if not specified, but usually we run on GCS)
GCS_ZONAL_PATH = os.getenv("CKPT_WRITE_PATH", "gs://yonghui-us-central1-b") + "/profile_test_zonal.ckpt"

def generate_simulated_state_dict(num_params_billions=0.2):
    """Generates simulated weights + AdamW states matching real model sizes.
    0.2 Billion parameters is ~400 MB bf16 weights + ~800 MB AdamW optimizer moments = ~1.2 GB.
    This is large enough to show I/O profiling bottlenecks but small enough to run quickly.
    """
    logging.info(f"Generating simulated state dict for {num_params_billions}B parameters (~{num_params_billions * 6:.2f} GB)...")
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

def run_save(filepath, state_dict, block_size=None, prof_filename=None):
    if block_size is not None:
        logging.info(f"Opening remote file: {filepath} with block_size={block_size} (Unbuffered Async Offloaded)")
        fs = fsspec.open(filepath, "wb", block_size=block_size)
    else:
        logging.info(f"Opening remote file: {filepath} with default block_size (Buffered Async Offloaded)")
        fs = fsspec.open(filepath, "wb")
    
    start_time = time.perf_counter()
    with fs as f:
        # Wrap torch.save with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        torch.save(state_dict, f)
        
        profiler.disable()
        if prof_filename:
            logging.info(f"Dumping stats to {prof_filename}")
            profiler.dump_stats(prof_filename)
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        logging.info("--- Top 20 Profiler Stats (Sorted by Cumulative Time) ---")
        stats.print_stats(20)  # Print top 20 hot methods
        
    duration = time.perf_counter() - start_time
    logging.info(f"Finished saving to {filepath} in {duration:.2f} seconds.\n")
    return duration

if __name__ == "__main__":
    # Ensure gcsfs experimental support is active
    os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "true"
    
    # 0.2B parameters payload (~1.2 GB) for fast, robust profiling iteration
    state_dict = generate_simulated_state_dict(num_params_billions=0.2)
    
    logging.info("--- Experiment 1: Buffered Async Offloaded (Default Block Size) ---")
    os.environ["GCSFS_ZONAL_FORCE_SYNC_WRITE"] = "false"
    duration_buffered = run_save(GCS_ZONAL_PATH, state_dict, block_size=None, prof_filename="async_offload.prof")
    
    logging.info("--- Experiment 2: Original Unchanged Code (Forced Sync Write) ---")
    os.environ["GCSFS_ZONAL_FORCE_SYNC_WRITE"] = "true"
    duration_original = run_save(GCS_ZONAL_PATH, state_dict, block_size=None, prof_filename="origin.prof")
    
    # Reset to default/false
    os.environ["GCSFS_ZONAL_FORCE_SYNC_WRITE"] = "false"
    
    print("\n==================== COMPARISON SUMMARY ====================")
    print(f"Buffered Async Offloaded Save Time   : {duration_buffered:.2f} seconds")
    print(f"Original Unchanged Sync Save Time    : {duration_original:.2f} seconds")
    print("============================================================\n")
