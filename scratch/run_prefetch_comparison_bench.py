import subprocess
import json
import os
import sys

concurrencies = [8, 16, 32]
bucket_types = ["zonal", "regional"]
backends = ["gcsfs", "gcsfs_no_prefetch", "cpp"]
num_files = 128
output_file = "scratch/prefetch_comparison_multi_round_results.json"

all_results = []
if os.path.exists(output_file):
    try:
        with open(output_file, "r") as f:
            all_results = json.load(f)
    except Exception:
        all_results = []

for bucket in bucket_types:
    for conc in concurrencies:
        for backend in backends:
            cmd = [
                sys.executable,
                "scratch/run_zonal_regional_bench.py",
                "--bucket-type", bucket,
                "--backend", backend,
                "--dataset", "40mb",
                "--num-files", str(num_files),
                "--ray-concurrency", str(conc),
                "--warmup-runs", "1",
                "--measured-runs", "4",
            ]
            print(f"\n=======================================================", flush=True)
            print(f"Running: bucket={bucket}, backend={backend}, concurrency={conc} (4 measured rounds)", flush=True)
            print(f"=======================================================", flush=True)
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = res.stdout
            
            # Extract RESULT_JSON
            found = False
            for line in output.splitlines():
                if line.startswith("RESULT_JSON:"):
                    data = json.loads(line[len("RESULT_JSON:"):])
                    # Remove any previous entry for this exact configuration
                    all_results = [
                        r for r in all_results
                        if not (r["bucket_type"] == bucket and r["backend"] == backend and r["ray_concurrency"] == conc)
                    ]
                    all_results.append(data)
                    with open(output_file, "w") as f:
                        json.dump(all_results, f, indent=2)
                    found = True
                    tputs = [f"{r['throughput_mib_s']:.1f}" for r in data['runs']]
                    print(f"--> ROUNDS: [{', '.join(tputs)}] MiB/s", flush=True)
                    print(f"--> MEAN: {data['mean_throughput_mib_s']:.2f} MiB/s (±{data['stdev_throughput_mib_s']:.2f}) | MEDIAN: {data['median_throughput_mib_s']:.2f} MiB/s", flush=True)
                    break
            if not found:
                print("Failed to find RESULT_JSON! Last 20 lines of output:", flush=True)
                print("\n".join(output.splitlines()[-20:]), flush=True)

print(f"\nAll benchmark rounds completed. Results saved to {output_file}", flush=True)
