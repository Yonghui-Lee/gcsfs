import subprocess
import sys
import json
import time

PYTHON = sys.executable

configurations = [
    # (bucket_type, backend, mode)
    # 1. ZONAL BUCKET
    ("zonal", "cpp", "count"),
    ("zonal", "gcsfs", "count"),
    ("zonal", "cpp", "materialize"),
    ("zonal", "gcsfs", "materialize"),
    
    # 2. REGIONAL BUCKET
    ("regional", "cpp", "count"),
    ("regional", "gcsfs", "count"),
    ("regional", "cpp", "materialize"),
    ("regional", "gcsfs", "materialize"),
]

all_results = []

for bucket_type, backend, mode in configurations:
    print(f"\n=======================================================")
    print(f"RUNNING: bucket={bucket_type}, backend={backend}, mode={mode}, ray_concurrency=8")
    print(f"=======================================================", flush=True)
    
    cmd = [
        PYTHON,
        "scratch/run_zonal_regional_bench.py",
        "--bucket-type", bucket_type,
        "--backend", backend,
        "--mode", mode,
        "--ray-concurrency", "8",
        "--warmup-runs", "1",
        "--measured-runs", "3",
    ]
    
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    if proc.returncode != 0:
        print(f"FAILED (returncode={proc.returncode}):")
        print("STDERR:\n", proc.stderr[-2000:])
        continue
    
    # Parse result
    result_data = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result_data = json.loads(line[len("RESULT_JSON:"):])
            break
            
    if result_data:
        all_results.append(result_data)
        print(f"DONE in {elapsed:.1f}s: Mean = {result_data['mean_throughput_mib_s']:.2f} MiB/s ({result_data['mean_throughput_mib_s']/1024:.2f} GiB/s), Per-worker = {result_data['per_worker_mib_s']:.2f} MiB/s")
    else:
        print("COULD NOT FIND RESULT_JSON IN OUTPUT:")
        print(proc.stdout[-2000:])

with open("scratch/master_benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n\nAll benchmarks completed successfully! Results written to scratch/master_benchmark_results.json")

