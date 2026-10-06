#!/usr/bin/env python3
"""
Benchmark Ray Data ingestion on large 1GB Parquet files across:
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
OUTPUT_JSON = "/home/yonghuili_google_com/gcsfs/scratch/large_1gb_benchmark_results.json"

def main():
    all_results = []
    
    for conc in CONCURRENCIES:
        for backend in BACKENDS:
            print(f"\n=======================================================")
            print(f"Running Large 1GB Benchmark: Backend={backend}, Concurrency={conc}")
            print(f"=======================================================")
            
            cmd = [
                PYTHON_EXEC,
                BENCH_SCRIPT,
                "--backend", backend,
                "--concurrency", str(conc),
                "--warmup", "1",
                "--runs", "3",
            ]
            
            t0 = time.time()
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            elapsed = time.time() - t0
            
            if res.returncode != 0:
                print(f"FAILED (elapsed {elapsed:.1f}s):")
                print(res.stderr)
                continue
                
            try:
                # Find JSON output between markers
                stdout = res.stdout
                for line in stdout.splitlines():
                    if line.startswith("RESULT_JSON:"):
                        payload = json.loads(line[len("RESULT_JSON:"):].strip())
                        all_results.append(payload)
                        print(f"SUCCESS: Median Throughput = {payload['median_throughput_mib_s']:.2f} MiB/s "
                              f"({payload['median_throughput_mib_s']/1024:.2f} GiB/s), "
                              f"Runs = {[round(r['throughput_mib_s'], 2) for r in payload['runs']]}")
                        break
                else:
                    print(f"No RESULT_JSON found in output:\n{stdout}")
            except Exception as e:
                print(f"Error parsing result: {e}\nStdout:\n{res.stdout}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
