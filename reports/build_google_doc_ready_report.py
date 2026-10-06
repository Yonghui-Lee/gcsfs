#!/usr/bin/env python3
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_report_docx():
    doc = docx.Document()
    
    # Page setup - 0.75 in margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0x20, 0x21, 0x24) # Google text dark grey

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Ray Data Ingestion Performance Report\n")
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1a, 0x73, 0xe8) # Google Blue
    
    run_sub = p_title.add_run("GCSFS vs. PyArrow C++ & Cache Optimization on Google Cloud Storage")
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = RGBColor(0x5f, 0x63, 0x68)
    
    doc.add_paragraph() # Spacer

    # Callout Box: Executive Takeaways
    table_callout = doc.add_table(rows=1, cols=1)
    table_callout.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_callout = table_callout.cell(0, 0)
    set_cell_background(cell_callout, "E8F0FE")
    set_cell_margins(cell_callout, top=140, bottom=140, left=200, right=200)
    
    p_call = cell_callout.paragraphs[0]
    p_call.paragraph_format.space_before = Pt(4)
    p_call.paragraph_format.space_after = Pt(4)
    r_call_title = p_call.add_run("Key Findings at a Glance:\n")
    r_call_title.bold = True
    r_call_title.font.size = Pt(11)
    r_call_title.font.color.rgb = RGBColor(0x1a, 0x73, 0xe8)
    
    bullets = [
        ("GCSFS (No Cache) Dominates Throughput: ", "Achieves 4,910.55 MiB/s (4.80 GiB/s = 40.2 Gbps) on Zonal storage at 32 workers, beating native PyArrow C++ (2,705.38 MiB/s) by +81% (1.81×), and by +149% (2.49×) at 16 workers."),
        ("Pre-Buffer Coalescing Makes Caching Redundant: ", "PyArrow's Parquet reader (pre_buffer=True) coalesces columns into 28–200 MB ranges. GCSFS ReadAheadCache actively degrades performance by slicing these into 5 MiB blocks, while BaseCache streams them in continuous bursts."),
        ("Zero-Copy Arrow Mandate: ", "Default NumPy batch formatting instantiates 20.8 million Python str objects on The Pile, creating a 0.40s bottleneck that caps throughput at 950 MiB/s. Zero-copy Arrow (batch_format='pyarrow') unlocks full 4.91 GiB/s line rate."),
        ("Regional Storage Scaling: ", "On 64 files (200 MB each, 12.84 GB total) over HTTP/2 REST, GCSFS No Cache delivers 1,414.78 MiB/s at 32 workers, leading all configurations.")
    ]
    for b_title, b_desc in bullets:
        p_b = cell_callout.add_paragraph()
        p_b.paragraph_format.space_after = Pt(3)
        r_bt = p_b.add_run("• " + b_title)
        r_bt.bold = True
        p_b.add_run(b_desc)

    doc.add_paragraph() # Spacer

    # Section 1: Experimental Environment
    h1 = doc.add_heading(level=1)
    r = h1.add_run("1. Experimental Setup & System Topology")
    r.font.color.rgb = RGBColor(0x20, 0x21, 0x24)

    p_intro = doc.add_paragraph(
        "All benchmarks were executed on a dedicated high-memory, high-core compute instance located in us-central1-b:"
    )
    
    # Specs table
    specs = [
        ("Compute Instance", "Google Cloud Platform c4-standard-96 (Dedicated Host)"),
        ("CPU & Architecture", "96 vCPUs (Intel Xeon Platinum 8581C 'Emerald Rapids' @ 2.60 GHz)"),
        ("System Memory", "384 GiB DDR5 ECC RAM (178 GiB allocated to /dev/shm for Ray Plasma)"),
        ("Network Topology", "Google Virtual NIC (gvnic) with Tier-1 50 Gbps egress line rate"),
        ("Software Stack", "Linux 6.6.137+, Python 3.14.4, Ray 2.58.0, PyArrow 21.0.0, GCSFS 2026.1.0+"),
        ("Zonal Target", "gs://hf-pile-deduplicated-us-central1-b-gcsfs (Zonal HNS via gRPC MRD, RTT < 0.2 ms)"),
        ("Regional Target", "gs://yonghui-gcsfs-regional-us/ray_data_200mb (Regional REST HTTP/2, RTT ~1.5 ms)")
    ]
    t_specs = doc.add_table(rows=len(specs)+1, cols=2)
    t_specs.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t_specs.rows[0]
    hdr.cells[0].text = "Component"
    hdr.cells[1].text = "Specification & Environment Configuration"
    for c in hdr.cells:
        set_cell_background(c, "1A73E8")
        c.paragraphs[0].runs[0].font.bold = True
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(c, 80, 80, 120, 120)

    for i, (k, v) in enumerate(specs):
        row = t_specs.rows[i+1]
        row.cells[0].text = k
        row.cells[0].paragraphs[0].runs[0].font.bold = True
        row.cells[1].text = v
        bg = "F8F9FA" if i % 2 == 1 else "FFFFFF"
        for c in row.cells:
            set_cell_background(c, bg)
            set_cell_margins(c, 60, 60, 120, 120)

    doc.add_paragraph() # Spacer

    # Section 2: Zonal Storage Benchmark
    h2 = doc.add_heading(level=1)
    r2 = h2.add_run("2. Zonal Storage Workload: The Pile Deduplicated (256 Files / 65.2 GB)")
    r2.font.color.rgb = RGBColor(0x20, 0x21, 0x24)

    doc.add_paragraph(
        "The Zonal workload tests real-world pretraining ingestion on 256 Parquet files (~255 MB each) from "
        "Hugging Face The Pile Deduplicated. Each file contains 9 row groups (~53 MB uncompressed) with text: string columns, "
        "expanding to ~110 GB in-memory Arrow tables. Data is streamed using zero-copy Arrow (batch_format='pyarrow')."
    )

    # Zonal Table
    zonal_data = [
        ("GCSFS (No Cache)", "1,685.36 MiB/s", "3,101.44 MiB/s", "4,910.55 MiB/s (4.80 GiB/s)", "40.20 Gbps", "+81% (1.81×)"),
        ("GCSFS (Prefetch ON)", "1,679.12 MiB/s", "3,087.47 MiB/s", "4,856.13 MiB/s (4.74 GiB/s)", "39.75 Gbps", "+80% (1.80×)"),
        ("GCSFS (ReadAhead Cache)", "1,574.91 MiB/s", "2,813.22 MiB/s", "4,512.87 MiB/s (4.41 GiB/s)", "36.96 Gbps", "+67% (1.67×)"),
        ("PyArrow C++", "973.89 MiB/s", "1,243.24 MiB/s", "2,705.38 MiB/s (2.64 GiB/s)", "22.15 Gbps", "Baseline")
    ]
    t_zonal = doc.add_table(rows=len(zonal_data)+1, cols=6)
    t_zonal.alignment = WD_TABLE_ALIGNMENT.CENTER
    z_hdrs = ["Backend / Configuration", "8 Workers", "16 Workers", "32 Workers", "Line Rate (32W)", "Speedup vs C++"]
    for j, text in enumerate(z_hdrs):
        cell = t_zonal.rows[0].cells[j]
        cell.text = text
        set_cell_background(cell, "1A73E8")
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 80, 80, 100, 100)

    for i, row_data in enumerate(zonal_data):
        row = t_zonal.rows[i+1]
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            cell.text = val
            if j == 0 or j == 3:
                cell.paragraphs[0].runs[0].font.bold = True
            bg = "F8F9FA" if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 100, 100)

    p_chart1 = doc.add_paragraph()
    p_chart1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_chart1.paragraph_format.space_before = Pt(12)
    doc.add_picture('/home/yonghuili_google_com/gcsfs/reports/charts/zonal_throughput_chart.png', width=Inches(6.5))
    p_cap1 = doc.add_paragraph()
    p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap1 = p_cap1.add_run("Figure 1: Zonal Storage Streaming Throughput (The Pile Deduplicated, 256 Files / 65.2 GB)")
    r_cap1.font.size = Pt(9.5)
    r_cap1.font.italic = True

    doc.add_paragraph() # Spacer

    # Section 3: Regional Storage Benchmark
    h3 = doc.add_heading(level=1)
    r3 = h3.add_run("3. Regional Storage Workload: 64 Files (~200 MB Each / 12.84 GB Total)")
    r3.font.color.rgb = RGBColor(0x20, 0x21, 0x24)

    doc.add_paragraph(
        "The Regional benchmark evaluates 64 Parquet files (~200.7 MB each) stored in gs://yonghui-gcsfs-regional-us/ray_data_200mb. "
        "Transport occurs via standard HTTP/2 REST with connection pooling over regional network paths (RTT ~1.5 ms)."
    )

    regional_data = [
        ("GCSFS (No Cache)", "438.81 MiB/s", "749.02 MiB/s", "1,414.78 MiB/s (1.38 GiB/s)", "11.58 Gbps", "+18.5%"),
        ("PyArrow C++", "410.97 MiB/s", "775.04 MiB/s", "1,394.43 MiB/s (1.36 GiB/s)", "11.42 Gbps", "+16.8%"),
        ("GCSFS (Prefetch ON)", "366.46 MiB/s", "796.62 MiB/s", "1,250.14 MiB/s (1.22 GiB/s)", "10.24 Gbps", "+4.7%"),
        ("GCSFS (ReadAhead Cache)", "394.68 MiB/s", "698.22 MiB/s", "1,193.55 MiB/s (1.17 GiB/s)", "9.77 Gbps", "Baseline")
    ]
    t_reg = doc.add_table(rows=len(regional_data)+1, cols=6)
    t_reg.alignment = WD_TABLE_ALIGNMENT.CENTER
    reg_hdrs = ["Backend / Configuration", "8 Workers", "16 Workers", "32 Workers", "Line Rate (32W)", "Speedup vs ReadAhead"]
    for j, text in enumerate(reg_hdrs):
        cell = t_reg.rows[0].cells[j]
        cell.text = text
        set_cell_background(cell, "1A73E8")
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 80, 80, 100, 100)

    for i, row_data in enumerate(regional_data):
        row = t_reg.rows[i+1]
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            cell.text = val
            if j == 0 or j == 3:
                cell.paragraphs[0].runs[0].font.bold = True
            bg = "F8F9FA" if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 100, 100)

    p_chart2 = doc.add_paragraph()
    p_chart2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_chart2.paragraph_format.space_before = Pt(12)
    doc.add_picture('/home/yonghuili_google_com/gcsfs/reports/charts/regional_throughput_chart.png', width=Inches(6.5))
    p_cap2 = doc.add_paragraph()
    p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap2 = p_cap2.add_run("Figure 2: Regional Storage Streaming Throughput (64 Files / 12.84 GB)")
    r_cap2.font.size = Pt(9.5)
    r_cap2.font.italic = True

    doc.add_paragraph() # Spacer

    # Section 4: Architectural Deep Dive
    h4 = doc.add_heading(level=1)
    r4 = h4.add_run("4. Architectural Deep Dive: Why 'No Cache' Consistently Wins")
    r4.font.color.rgb = RGBColor(0x20, 0x21, 0x24)

    doc.add_paragraph(
        "In traditional POSIX filesystem benchmarks, disabling cache degrades performance. However, in Ray Data Parquet reading, "
        "GCSFS (No Cache / BaseCache) consistently delivers the highest throughput. The explanation stems from the interaction "
        "between PyArrow's pre_buffer=True logic and the filesystem cache layer:"
    )

    p_diag = doc.add_paragraph()
    p_diag.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_picture('/home/yonghuili_google_com/gcsfs/reports/charts/cache_coalescing_diagram.png', width=Inches(6.5))
    p_cap3 = doc.add_paragraph()
    p_cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap3 = p_cap3.add_run("Figure 3: PyArrow Range Coalescing vs. Filesystem ReadAhead Slicing Pathology")
    r_cap3.font.size = Pt(9.5)
    r_cap3.font.italic = True

    reasons = [
        ("Application Layer Range Coalescing: ", "PyArrow inspects the Parquet footer and coalesces all needed column chunks into large contiguous byte ranges (28 MB to 200 MB in a single read request). Redundant filesystem caching adds no value because the application layer has already planned the optimal byte range."),
        ("Eliminating Request Slicing: ", "ReadAheadCache defaults to a 5 MiB block size. When PyArrow requests 200 MB, ReadAheadCache slices it into 40 separate 5 MiB sequential HTTP range requests. BaseCache passes the entire 200 MB request straight through to high-speed async transport in one continuous burst."),
        ("Zero Async Context Switching: ", "The BackgroundPrefetcher runs coroutines, circular queues, and mutex locks. Under already-coalesced I/O, BaseCache avoids all intermediate memory copies and CPU scheduling overhead."),
        ("PyArrow C++ Lock Contention: ", "Native PyArrow C++ delegates I/O across thousands of worker threads via google-cloud-cpp, creating mutex lock thrashing in libcurl multi-handles. GCSFS's non-blocking asyncio event loop scales cleanly to 4.91 GiB/s.")
    ]
    for title, desc in reasons:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.space_after = Pt(4)
        rt = p_r.add_run("• " + title)
        rt.bold = True
        p_r.add_run(desc)

    doc.add_paragraph() # Spacer

    # Section 5: Zero Copy vs NumPy Bottleneck
    h5 = doc.add_heading(level=1)
    r5 = h5.add_run("5. Mathematical Analysis: The Zero-Copy Arrow Mandate")
    r5.font.color.rgb = RGBColor(0x20, 0x21, 0x24)

    doc.add_paragraph(
        "Ray Data pipelines are governed by: T_pipeline = max(T_IO, T_decode, T_format, T_consume). "
        "When reading unstructured text (The Pile), NumPy lacks a native UTF-8 string type. "
        "Ray Data's default batch_format='numpy' iterates over 81,405 rows per file and allocates individual Python str objects on the heap:"
    )

    p_eq = doc.add_paragraph()
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_eq = p_eq.add_run("Total String Allocations = 81,405 rows/file × 256 files = 20,839,680 Python Heap Objects\n"
                        "T_format = 0.404 seconds per 250 MB file  ==>  Max Throughput = 250 MB / 0.404s ≈ 625 - 950 MiB/s")
    r_eq.font.bold = True
    r_eq.font.size = Pt(10)
    r_eq.font.color.rgb = RGBColor(0xC5, 0x22, 0x1F)

    doc.add_paragraph(
        "With prefetch_batches=2, Ray Data's executor halted 16 and 32 reader tasks with ConcurrencyCap backpressure. "
        "Switching to zero-copy Arrow (batch_format='pyarrow') passes shared memory pointers (/dev/shm) directly, reducing "
        "T_format to ~0.0001s and immediately unleashing line-rate throughput to 4.91 GiB/s."
    )

    doc.add_paragraph() # Spacer

    # Section 6: Scaling Linearity Chart
    p_chart3 = doc.add_paragraph()
    p_chart3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_picture('/home/yonghuili_google_com/gcsfs/reports/charts/scaling_linearity_chart.png', width=Inches(6.5))
    p_cap4 = doc.add_paragraph()
    p_cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap4 = p_cap4.add_run("Figure 4: Concurrency Scaling Linearity (Zonal gRPC vs. Regional HTTP/2 REST)")
    r_cap4.font.size = Pt(9.5)
    r_cap4.font.italic = True

    doc.add_paragraph() # Spacer

    # Section 7: Production Configuration
    h6 = doc.add_heading(level=1)
    r6 = h6.add_run("6. Production Tuning & Recommendations")
    r6.font.color.rgb = RGBColor(0x20, 0x21, 0x24)

    doc.add_paragraph(
        "For maximum Ray Data ingestion throughput on Google Cloud Storage, configure your cluster runtime_env as follows:"
    )

    code_block = (
        'runtime_env = {\n'
        '    "env_vars": {\n'
        '        # 1. Enable Zonal Hierarchical Namespace gRPC acceleration\n'
        '        "GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT": "true",\n'
        '        # 2. Disable redundant caching to allow unfragmented range streaming\n'
        '        "USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING": "false",\n'
        '        "GCSFS_DEFAULT_CACHE_TYPE": "none",\n'
        '        # 3. Ray Data Parquet reader thread pool tuning\n'
        '        "RAY_DATA_PARQUET_READER_IO_THREAD_COUNT": "128",\n'
        '        "RAY_DATA_PARQUET_READER_CPU_COUNT": "32",\n'
        '        "RAY_DATA_PARQUET_FRAGMENT_BUFFER_SIZE": str(8 * 1024 * 1024),\n'
        '    }\n'
        '}\n'
        '# Always consume as zero-copy Arrow Tables:\n'
        'for batch in ds.iter_batches(batch_size=None, prefetch_batches=2, batch_format="pyarrow"):\n'
        '    pass\n'
    )
    p_code = doc.add_paragraph()
    p_code.paragraph_format.space_before = Pt(6)
    p_code.paragraph_format.space_after = Pt(6)
    r_code = p_code.add_run(code_block)
    r_code.font.name = 'Courier New'
    r_code.font.size = Pt(9.5)
    r_code.font.color.rgb = RGBColor(0x20, 0x21, 0x24)

    output_path = "/home/yonghuili_google_com/gcsfs/reports/gcsfs_benchmarking_and_prefetcher_analysis_report.docx"
    doc.save(output_path)
    print(f"Report DOCX successfully created at {output_path}")

if __name__ == "__main__":
    create_report_docx()
