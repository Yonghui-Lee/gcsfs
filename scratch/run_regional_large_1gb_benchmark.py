#!/usr/bin/env python3
"""
Benchmark Ray Data ingestion on large 1GB Parquet files on Regional bucket across:
1. GCSFS (No Cache / BaseCache)
2. GCSFS (Prefetch ON / BackgroundPrefetcher)
3. GCSFS (ReadAhead Cache)
4. PyArrow C++

Runs across 8 and 16 Ray workers using batch_format='pyarrow'.
"""

import subprocess
import sys
import json
import os
import time

BACKENDS = [
    "gcsfs_no_cache",
    "gcsfs_prefetch_on",
    "gcsfs_readahead",
    "pyarrow_cpp",
]

CONCURRENCIES = [8, 16]

PYTHON_EXEC = "/home/yonghuili_google_com/ray_venv/bin/python"
BENCH_SCRIPT = "/home/yonghuili_google_com/gcsfs/scratch/run_single_large_bench.py"
OUTPUT_JSON = "/home/yonghuili_google_com/gcsfs/scratch/regional_large_1gb_benchmark_results.json"

def main():
    all_results = []
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []
    
    for conc in CONCURRENCIES:
        for backend in BACKENDS:
            # Check if already done
            already_done = any(
                r.get("backend") == backend and r.get("concurrency") == conc
                for r in all_results
            )
            if already_done:
                print(f"Skipping already completed: backend={backend}, conc={conc}", flush=True)
                continue

            print(f"\n=======================================================", flush=True)
            print(f"Running Regional Large 1GB Benchmark: Backend={backend}, Concurrency={conc}", flush=True)
            print(f"=======================================================", flush=True)
            
            cmd = [
                PYTHON_EXEC,
                BENCH_SCRIPT,
                "--bucket-type", "regional",
                "--backend", backend,
                "--concurrency", str(conc),
                "--warmup", "1",
                "--runs", "2",
            ]
            
            t0 = time.time()
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            elapsed = time.time() - t0
            
            if res.returncode != 0:
                print(f"FAILED (elapsed {elapsed:.1f}s):", flush=True)
                print(res.stderr, flush=True)
                continue
                
            try:
                stdout = res.stdout
                for line in stdout.splitlines():
                    if line.startswith("RESULT_JSON:"):
                        payload = json.loads(line[len("RESULT_JSON:"):].strip())
                        all_results.append(payload)
                        print(f"SUCCESS: Median Throughput = {payload['median_throughput_mib_s']:.2f} MiB/s "
                              f"({payload['median_throughput_mib_s']/1024:.2f} GiB/s), "
                              f"Runs = {[round(r['throughput_mib_s'], 2) for r in payload['runs']]}", flush=True)
                        break
                else:
                    print(f"No RESULT_JSON found in output:\n{stdout}", flush=True)
            except Exception as e:
                print(f"Error parsing result: {e}\nStdout:\n{res.stdout}", flush=True)

            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nAll regional large file results saved to {OUTPUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
