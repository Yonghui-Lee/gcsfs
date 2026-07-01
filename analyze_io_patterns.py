import os
import io
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IOTelemetryWrapper(io.BytesIO):
    """A file-like object that wraps BytesIO to intercept and record every write size."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_sizes = []

    def write(self, b):
        size = len(b)
        self.write_sizes.append(size)
        return super().write(b)

def generate_simulated_state_dict(num_params_billions=0.2):
    """Generates simulated weights + AdamW states."""
    logging.info(f"Generating state dict for {num_params_billions}B parameters...")
    weights = torch.randn(int(num_params_billions * 5e8), dtype=torch.bfloat16)
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

def analyze_patterns():
    state_dict = generate_simulated_state_dict(0.2)
    
    # Wrap standard stream with telemetry
    telemetry_stream = IOTelemetryWrapper()
    
    logging.info("Running torch.save serialization and intercepting IO events...")
    torch.save(state_dict, telemetry_stream)
    
    sizes = np.array(telemetry_stream.write_sizes)
    total_bytes = np.sum(sizes)
    total_writes = len(sizes)
    
    logging.info("--- IO Write Size & Pattern Analysis ---")
    print(f"Total Writes Intercepted : {total_writes:,}")
    print(f"Total Bytes Serialized   : {total_bytes:,} bytes ({total_bytes / (1024**3):.2f} GB)")
    print(f"Minimum Write Size       : {np.min(sizes):,} bytes")
    print(f"Average Write Size       : {np.mean(sizes):,.2f} bytes")
    print(f"Median Write Size (p50)  : {np.percentile(sizes, 50):,} bytes")
    print(f"p90 Write Size           : {np.percentile(sizes, 90):,} bytes")
    print(f"p95 Write Size           : {np.percentile(sizes, 95):,} bytes")
    print(f"p99 Write Size           : {np.percentile(sizes, 99):,} bytes")
    print(f"Maximum Write Size       : {np.max(sizes):,} bytes")
    
    # Bucket categorization
    buckets = {
        "Tiny Metadata (< 1 KB)": sizes[sizes < 1024],
        "Small Configs (1 KB - 64 KB)": sizes[(sizes >= 1024) & (sizes < 64*1024)],
        "Medium Buffers (64 KB - 1 MB)": sizes[(sizes >= 64*1024) & (sizes < 1024*1024)],
        "Large Chunks (1 MB - 16 MB)": sizes[(sizes >= 1024*1024) & (sizes < 16*1024*1024)],
        "Monolithic Payloads (> 16 MB)": sizes[sizes >= 16*1024*1024],
    }
    
    print("\n--- IO Size Distribution Breakdown ---")
    print(f"{'Category':<32} | {'Count':<10} | {'Total Bytes':<15} | {'Percentage (%)':<10}")
    print("-" * 78)
    for name, bucket_sizes in buckets.items():
        count = len(bucket_sizes)
        b_sum = np.sum(bucket_sizes)
        pct = (b_sum / total_bytes) * 100 if total_bytes > 0 else 0
        print(f"{name:<32} | {count:<10,} | {b_sum:<15,} | {pct:<10.2f}%")

if __name__ == "__main__":
    analyze_patterns()
