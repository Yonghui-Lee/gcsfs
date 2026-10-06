#!/usr/bin/env python3
import subprocess
import json
import os
import sys

concurrencies = [8, 16, 32]
backends = ["gcsfs", "gcsfs_no_cache", "cpp", "gcsfs_no_prefetch"]
dataset = "pile"
num_files = 256
output_file = "scratch/pile_256_benchmark_results.json"

all_results = []
if os.path.exists(output_file):
    try:
        with open(output_file, "r") as f:
            all_results = json.load(f)
    except Exception:
        all_results = []

for conc in concurrencies:
    for backend in backends:
        # Check if already completed
        already_done = False
        for entry in all_results:
            if (entry.get("backend") == backend and 
                entry.get("ray_concurrency") == conc and 
                entry.get("num_files") == num_files):
                already_done = True
                print(f"Skipping already completed: backend={backend}, conc={conc}", flush=True)
                break
        if already_done:
            continue

        cmd = [
            sys.executable,
            "scratch/run_zonal_regional_bench.py",
            "--bucket-type", "zonal",
            "--backend", backend,
            "--dataset", dataset,
            "--num-files", str(num_files),
            "--ray-concurrency", str(conc),
            "--warmup-runs", "1",
            "--measured-runs", "2",
        ]
        print(f"\n=======================================================", flush=True)
        print(f"Running PILE 256: Backend={backend}, Concurrency={conc} (2 runs, 64 GB each)", flush=True)
        print(f"=======================================================", flush=True)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = res.stdout
        
        found = False
        for line in output.splitlines():
            if line.startswith("RESULT_JSON:"):
                data = json.loads(line[len("RESULT_JSON:"):])
                data["dataset"] = "hf-pile-deduplicated"
                all_results.append(data)
                found = True
                runs_str = ", ".join(f"{r['throughput_mib_s']:.2f}" for r in data['runs'])
                print(f"SUCCESS: Backend={backend}, Conc={conc} -> Median={data['median_throughput_mib_s']:.2f} MiB/s (runs: [{runs_str}])", flush=True)
                break
        if not found:
            print(f"ERROR on backend={backend}, conc={conc}:")
            print(output[-1500:])

        # Save results immediately
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

print(f"\nAll 256-file Pile benchmarks complete. Saved to {output_file}")
