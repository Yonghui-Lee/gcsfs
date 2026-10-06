#!/usr/bin/env python3
import subprocess
import json
import os
import sys

BUCKETS = ["regional", "zonal"]
CONCURRENCIES = [8, 16, 32]
WARMUP = 1
MEASURED = 4

results = []

for bucket in BUCKETS:
    for c in CONCURRENCIES:
        cmd = [
            sys.executable,
            "scratch/run_zonal_regional_bench.py",
            "--bucket-type", bucket,
            "--backend", "gcsfs_no_cache",
            "--dataset", "40mb",
            "--num-files", "128",
            "--ray-concurrency", str(c),
            "--warmup-runs", str(WARMUP),
            "--measured-runs", str(MEASURED),
        ]
        print(f"\n=======================================================", flush=True)
        print(f"Running: Bucket={bucket}, Backend=gcsfs_no_cache, Concurrency={c}", flush=True)
        print(f"=======================================================", flush=True)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Parse RESULT_JSON
        found = False
        for line in res.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                data = json.loads(line[len("RESULT_JSON:"):])
                results.append(data)
                found = True
                print(f"SUCCESS: Median Throughput = {data['median_throughput_mib_s']:.2f} MiB/s", flush=True)
                break
        if not found:
            print("ERROR running benchmark:")
            print(res.stdout[-1500:])

output_file = "scratch/no_cache_benchmark_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved all results to {output_file}")
