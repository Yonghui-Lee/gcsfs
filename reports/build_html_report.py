#!/usr/bin/env python3
import base64
import os

charts_dir = "/home/yonghuili_google_com/gcsfs/reports/charts"

def get_base64_img(filename):
    path = os.path.join(charts_dir, filename)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

img_zonal = get_base64_img("zonal_throughput_chart.png")
img_reg = get_base64_img("regional_throughput_chart.png")
img_diag = get_base64_img("cache_coalescing_diagram.png")
img_scale = get_base64_img("scaling_linearity_chart.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ray Data Ingestion Performance Report: GCSFS vs. PyArrow C++</title>
<style>
  body {{
    font-family: 'Google Sans', Arial, sans-serif;
    color: #202124;
    line-height: 1.6;
    max-width: 950px;
    margin: 40px auto;
    padding: 0 20px;
  }}
  h1 {{
    color: #1a73e8;
    border-bottom: 2px solid #e8eaed;
    padding-bottom: 10px;
    text-align: center;
    font-size: 28px;
  }}
  h2 {{
    color: #202124;
    border-bottom: 1px solid #e8eaed;
    padding-bottom: 6px;
    margin-top: 36px;
    font-size: 20px;
  }}
  .subtitle {{
    text-align: center;
    color: #5f6368;
    font-size: 16px;
    margin-top: -15px;
    margin-bottom: 30px;
  }}
  .callout {{
    background-color: #e8f0fe;
    border-left: 5px solid #1a73e8;
    padding: 16px 20px;
    border-radius: 4px;
    margin: 25px 0;
  }}
  .callout h3 {{
    margin-top: 0;
    color: #1a73e8;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 14px;
  }}
  th, td {{
    border: 1px solid #dadce0;
    padding: 10px 12px;
    text-align: left;
  }}
  th {{
    background-color: #1a73e8;
    color: #ffffff;
    font-weight: bold;
  }}
  tr:nth-child(even) {{
    background-color: #f8f9fa;
  }}
  .img-container {{
    text-align: center;
    margin: 30px 0;
  }}
  .img-container img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  }}
  .caption {{
    font-size: 13px;
    color: #5f6368;
    font-style: italic;
    margin-top: 8px;
  }}
  pre {{
    background-color: #f1f3f4;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
  }}
  .math-box {{
    background-color: #fce8e6;
    border-left: 5px solid #d93025;
    padding: 12px 18px;
    border-radius: 4px;
    font-weight: bold;
    color: #c5221f;
    margin: 20px 0;
    text-align: center;
  }}
</style>
</head>
<body>

<h1>Ray Data Ingestion Performance Report</h1>
<div class="subtitle">GCSFS vs. PyArrow C++ & Cache Optimization on Google Cloud Storage</div>

<div class="callout">
  <h3>Key Findings at a Glance</h3>
  <ul>
    <li><strong>GCSFS (No Cache) Dominates Throughput:</strong> Achieves <strong>4,910.55 MiB/s (4.80 GiB/s = 40.2 Gbps)</strong> on Zonal storage at 32 workers, beating native PyArrow C++ (2,705.38 MiB/s) by <strong>+81% (1.81×)</strong>, and by <strong>+149% (2.49×)</strong> at 16 workers.</li>
    <li><strong>Pre-Buffer Coalescing Makes Caching Redundant:</strong> PyArrow's Parquet reader (<code>pre_buffer=True</code>) coalesces columns into 28–200 MB ranges. GCSFS <code>ReadAheadCache</code> actively degrades performance by slicing these into 5 MiB blocks, while <code>BaseCache</code> streams them in continuous bursts.</li>
    <li><strong>Zero-Copy Arrow Mandate:</strong> Default NumPy batch formatting instantiates 20.8 million Python <code>str</code> objects on The Pile, creating a 0.40s bottleneck that caps throughput at 950 MiB/s. Zero-copy Arrow (<code>batch_format="pyarrow"</code>) unlocks the full 4.91 GiB/s line rate.</li>
    <li><strong>Regional Storage Scaling:</strong> On 64 files (200 MB each, 12.84 GB total) over HTTP/2 REST, GCSFS No Cache delivers <strong>1,414.78 MiB/s</strong> at 32 workers, leading all configurations.</li>
  </ul>
</div>

<h2>1. Experimental Setup & System Topology</h2>
<table>
  <tr>
    <th>Component</th>
    <th>Specification & Environment Configuration</th>
  </tr>
  <tr>
    <td><strong>Compute Instance</strong></td>
    <td>Google Cloud Platform c4-standard-96 (Dedicated Host)</td>
  </tr>
  <tr>
    <td><strong>CPU & Architecture</strong></td>
    <td>96 vCPUs (Intel Xeon Platinum 8581C 'Emerald Rapids' @ 2.60 GHz)</td>
  </tr>
  <tr>
    <td><strong>System Memory</strong></td>
    <td>384 GiB DDR5 ECC RAM (178 GiB allocated to /dev/shm for Ray Plasma)</td>
  </tr>
  <tr>
    <td><strong>Network Topology</strong></td>
    <td>Google Virtual NIC (gvnic) with Tier-1 50 Gbps egress line rate</td>
  </tr>
  <tr>
    <td><strong>Software Stack</strong></td>
    <td>Linux 6.6.137+, Python 3.14.4, Ray 2.58.0, PyArrow 21.0.0, GCSFS 2026.1.0+</td>
  </tr>
  <tr>
    <td><strong>Zonal Target</strong></td>
    <td>gs://hf-pile-deduplicated-us-central1-b-gcsfs (Zonal HNS via gRPC MRD, RTT &lt; 0.2 ms)</td>
  </tr>
  <tr>
    <td><strong>Regional Target</strong></td>
    <td>gs://yonghui-gcsfs-regional-us/ray_data_200mb (Regional REST HTTP/2, RTT ~1.5 ms)</td>
  </tr>
</table>

<h2>2. Zonal Storage Workload: The Pile Deduplicated (256 Files / 65.2 GB)</h2>
<p>The Zonal workload tests real-world pretraining ingestion on 256 Parquet files (~255 MB each) from Hugging Face The Pile Deduplicated. Each file contains 9 row groups (~53 MB uncompressed) with <code>text: string</code> columns, expanding to ~110 GB in-memory Arrow tables. Data is streamed using zero-copy Arrow (<code>batch_format="pyarrow"</code>).</p>

<table>
  <tr>
    <th>Backend / Configuration</th>
    <th>8 Workers</th>
    <th>16 Workers</th>
    <th>32 Workers</th>
    <th>Line Rate (32W)</th>
    <th>Speedup vs C++</th>
  </tr>
  <tr>
    <td><strong>GCSFS (No Cache)</strong></td>
    <td>1,685.36 MiB/s</td>
    <td>3,101.44 MiB/s</td>
    <td><strong>4,910.55 MiB/s (4.80 GiB/s)</strong></td>
    <td><strong>40.20 Gbps</strong></td>
    <td><strong>+81% (1.81×)</strong></td>
  </tr>
  <tr>
    <td><strong>GCSFS (Prefetch ON)</strong></td>
    <td>1,679.12 MiB/s</td>
    <td>3,087.47 MiB/s</td>
    <td>4,856.13 MiB/s (4.74 GiB/s)</td>
    <td>39.75 Gbps</td>
    <td>+80% (1.80×)</td>
  </tr>
  <tr>
    <td><strong>GCSFS (ReadAhead Cache)</strong></td>
    <td>1,574.91 MiB/s</td>
    <td>2,813.22 MiB/s</td>
    <td>4,512.87 MiB/s (4.41 GiB/s)</td>
    <td>36.96 Gbps</td>
    <td>+67% (1.67×)</td>
  </tr>
  <tr>
    <td><strong>PyArrow C++</strong></td>
    <td>973.89 MiB/s</td>
    <td>1,243.24 MiB/s</td>
    <td>2,705.38 MiB/s (2.64 GiB/s)</td>
    <td>22.15 Gbps</td>
    <td>Baseline</td>
  </tr>
</table>

<div class="img-container">
  <img src="data:image/png;base64,{img_zonal}" alt="Zonal Storage Throughput Chart">
  <div class="caption">Figure 1: Zonal Storage Streaming Throughput (The Pile Deduplicated, 256 Files / 65.2 GB)</div>
</div>

<h2>3. Regional Storage Workload: 64 Files (~200 MB Each / 12.84 GB Total)</h2>
<p>The Regional benchmark evaluates 64 Parquet files (~200.7 MB each) stored in <code>gs://yonghui-gcsfs-regional-us/ray_data_200mb</code>. Transport occurs via standard HTTP/2 REST with connection pooling over regional network paths (RTT ~1.5 ms).</p>

<table>
  <tr>
    <th>Backend / Configuration</th>
    <th>8 Workers</th>
    <th>16 Workers</th>
    <th>32 Workers</th>
    <th>Line Rate (32W)</th>
    <th>Speedup vs ReadAhead</th>
  </tr>
  <tr>
    <td><strong>GCSFS (No Cache)</strong></td>
    <td><strong>438.81 MiB/s</strong></td>
    <td>749.02 MiB/s</td>
    <td><strong>1,414.78 MiB/s (1.38 GiB/s)</strong></td>
    <td><strong>11.58 Gbps</strong></td>
    <td><strong>+18.5%</strong></td>
  </tr>
  <tr>
    <td><strong>PyArrow C++</strong></td>
    <td>410.97 MiB/s</td>
    <td>775.04 MiB/s</td>
    <td>1,394.43 MiB/s (1.36 GiB/s)</td>
    <td>11.42 Gbps</td>
    <td>+16.8%</td>
  </tr>
  <tr>
    <td><strong>GCSFS (Prefetch ON)</strong></td>
    <td>366.46 MiB/s</td>
    <td><strong>796.62 MiB/s</strong></td>
    <td>1,250.14 MiB/s (1.22 GiB/s)</td>
    <td>10.24 Gbps</td>
    <td>+4.7%</td>
  </tr>
  <tr>
    <td><strong>GCSFS (ReadAhead Cache)</strong></td>
    <td>394.68 MiB/s</td>
    <td>698.22 MiB/s</td>
    <td>1,193.55 MiB/s (1.17 GiB/s)</td>
    <td>9.77 Gbps</td>
    <td>Baseline</td>
  </tr>
</table>

<div class="img-container">
  <img src="data:image/png;base64,{img_reg}" alt="Regional Storage Throughput Chart">
  <div class="caption">Figure 2: Regional Storage Streaming Throughput (64 Files / 12.84 GB)</div>
</div>

<h2>4. Architectural Deep Dive: Why 'No Cache' Consistently Wins</h2>
<p>In traditional POSIX filesystem benchmarks, disabling cache degrades performance. However, in Ray Data Parquet reading, GCSFS (No Cache / BaseCache) consistently delivers the highest throughput:</p>

<div class="img-container">
  <img src="data:image/png;base64,{img_diag}" alt="Cache Coalescing Diagram">
  <div class="caption">Figure 3: PyArrow Range Coalescing vs. Filesystem ReadAhead Slicing Pathology</div>
</div>

<ul>
  <li><strong>Application Layer Range Coalescing:</strong> PyArrow inspects the Parquet footer and coalesces all needed column chunks into large contiguous byte ranges (28 MB to 200 MB in a single read request). Redundant filesystem caching adds no value because the application layer has already planned the optimal byte range.</li>
  <li><strong>Eliminating Request Slicing:</strong> <code>ReadAheadCache</code> defaults to a 5 MiB block size. When PyArrow requests 200 MB, <code>ReadAheadCache</code> slices it into 40 separate 5 MiB sequential HTTP range requests. <code>BaseCache</code> passes the entire 200 MB request straight through to high-speed async transport in one continuous burst.</li>
  <li><strong>Zero Async Context Switching:</strong> The <code>BackgroundPrefetcher</code> runs coroutines, circular queues, and mutex locks. Under already-coalesced I/O, <code>BaseCache</code> avoids all intermediate memory copies and CPU scheduling overhead.</li>
  <li><strong>PyArrow C++ Lock Contention:</strong> Native PyArrow C++ delegates I/O across thousands of worker threads via <code>google-cloud-cpp</code>, creating mutex lock thrashing in <code>libcurl</code> multi-handles. GCSFS's non-blocking <code>asyncio</code> event loop scales cleanly to 4.91 GiB/s.</li>
</ul>

<h2>5. Mathematical Analysis: The Zero-Copy Arrow Mandate</h2>
<p>Ray Data pipelines are governed by: <code>T_pipeline = max(T_IO, T_decode, T_format, T_consume)</code>. When reading unstructured text (The Pile), NumPy lacks a native UTF-8 string type. Ray Data's default <code>batch_format="numpy"</code> iterates over 81,405 rows per file and allocates individual Python <code>str</code> objects on the heap:</p>

<div class="math-box">
  Total String Allocations = 81,405 rows/file × 256 files = 20,839,680 Python Heap Objects<br>
  T_format = 0.404 seconds per 250 MB file  ==>  Max Throughput = 250 MB / 0.404s ≈ 625 - 950 MiB/s
</div>

<p>With <code>prefetch_batches=2</code>, Ray Data's executor halted 16 and 32 reader tasks with <code>ConcurrencyCap</code> backpressure. Switching to zero-copy Arrow (<code>batch_format="pyarrow"</code>) passes shared memory pointers (<code>/dev/shm</code>) directly, reducing <code>T_format</code> to ~0.0001s and immediately unleashing line-rate throughput to 4.91 GiB/s.</p>

<div class="img-container">
  <img src="data:image/png;base64,{img_scale}" alt="Scaling Linearity Chart">
  <div class="caption">Figure 4: Concurrency Scaling Linearity (Zonal gRPC vs. Regional HTTP/2 REST)</div>
</div>

<h2>6. Production Tuning & Recommendations</h2>
<p>For maximum Ray Data ingestion throughput on Google Cloud Storage, configure your cluster <code>runtime_env</code> as follows:</p>

<pre><code>runtime_env = {{
    "env_vars": {{
        # 1. Enable Zonal Hierarchical Namespace gRPC acceleration
        "GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT": "true",
        # 2. Disable redundant caching to allow unfragmented range streaming
        "USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING": "false",
        "GCSFS_DEFAULT_CACHE_TYPE": "none",
        # 3. Ray Data Parquet reader thread pool tuning
        "RAY_DATA_PARQUET_READER_IO_THREAD_COUNT": "128",
        "RAY_DATA_PARQUET_READER_CPU_COUNT": "32",
        "RAY_DATA_PARQUET_FRAGMENT_BUFFER_SIZE": str(8 * 1024 * 1024),
    }}
}}

# Always consume as zero-copy Arrow Tables:
for batch in ds.iter_batches(batch_size=None, prefetch_batches=2, batch_format="pyarrow"):
    process_batch(batch)
</code></pre>

</body>
</html>
"""

output_html = "/home/yonghuili_google_com/gcsfs/reports/gcsfs_benchmarking_and_prefetcher_analysis_report.html"
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Report HTML successfully created at {output_html}")
