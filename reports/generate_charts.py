#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'

colors = {
    'gcsfs_no_cache': '#1a73e8',      # Google Blue
    'gcsfs_prefetch': '#34a853',      # Google Green
    'cpp': '#ea4335'                  # Google Red
}

def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(round(height))}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# ==========================================
# Chart 1: Zonal Storage Throughput Scaling
# ==========================================
workers = ['8 Workers', '16 Workers', '32 Workers']
x = np.arange(len(workers))
width = 0.24

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

rects1 = ax.bar(x - width, [1685.36, 3101.44, 4910.55], width, label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'])
rects2 = ax.bar(x, [1679.12, 3087.47, 4856.13], width, label='GCSFS (Prefetch ON)', color=colors['gcsfs_prefetch'])
rects3 = ax.bar(x + width, [973.89, 1243.24, 2705.38], width, label='PyArrow C++', color=colors['cpp'])

ax.set_ylabel('Throughput (MiB/s)')
ax.set_title('Zonal Storage (The Pile Deduplicated - 256 Files / 65.2 GB)\nZero-Copy Arrow Streaming Throughput')
ax.set_xticks(x)
ax.set_xticklabels(workers)
ax.legend(frameon=True, loc='upper left', shadow=True)
ax.set_ylim(0, 5600)

autolabel(ax, rects1)
autolabel(ax, rects2)
autolabel(ax, rects3)

# Add line rate reference
ax.axhline(5120, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.text(0.02, 5170, 'Approx. 40 Gbps Line Rate (~5,120 MiB/s)', color='#555555', fontsize=10, fontstyle='italic')

plt.tight_layout()
plt.savefig('/home/yonghuili_google_com/gcsfs/reports/charts/zonal_throughput_chart.png')
plt.close()
print("Generated zonal_throughput_chart.png")

# ==========================================
# Chart 2: Regional Storage Throughput Scaling
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

rects1 = ax.bar(x - width, [565.14, 1481.14, 2731.73], width, label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'])
rects2 = ax.bar(x, [697.12, 1485.76, 2820.77], width, label='GCSFS (Prefetch ON)', color=colors['gcsfs_prefetch'])
rects3 = ax.bar(x + width, [900.17, 1686.63, 2435.81], width, label='PyArrow C++', color=colors['cpp'])

ax.set_ylabel('Throughput (MiB/s)')
ax.set_title('Regional Storage (The Pile Deduplicated - 256 Files / 65.2 GB)\nZero-Copy Arrow Streaming Throughput')
ax.set_xticks(x)
ax.set_xticklabels(workers)
ax.legend(frameon=True, loc='upper left', shadow=True)
ax.set_ylim(0, 3300)

autolabel(ax, rects1)
autolabel(ax, rects2)
autolabel(ax, rects3)

plt.tight_layout()
plt.savefig('/home/yonghuili_google_com/gcsfs/reports/charts/regional_throughput_chart.png')
plt.close()
print("Generated regional_throughput_chart.png")

# ==========================================
# Chart 3: Scaling Linearity (Zonal vs Regional)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

worker_nums = [8, 16, 32]
# Zonal
ax1.plot(worker_nums, [1685.36, 3101.44, 4910.55], 'o-', label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'], linewidth=2.5, markersize=8)
ax1.plot(worker_nums, [1679.12, 3087.47, 4856.13], 's--', label='GCSFS (Prefetch ON)', color=colors['gcsfs_prefetch'], linewidth=2)
ax1.plot(worker_nums, [973.89, 1243.24, 2705.38], '^-', label='PyArrow C++', color=colors['cpp'], linewidth=2.5, markersize=8)
ax1.plot([8, 32], [1685.36, 1685.36 * 4], 'k:', label='Ideal Linear Scaling', alpha=0.5)

ax1.set_title('Zonal Storage Scaling (The Pile 65.2 GB)')
ax1.set_xlabel('Ray Worker Concurrency')
ax1.set_ylabel('Throughput (MiB/s)')
ax1.set_xticks(worker_nums)
ax1.set_ylim(0, 5500)
ax1.legend(frameon=True)

# Regional
ax2.plot(worker_nums, [565.14, 1481.14, 2731.73], 'o-', label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'], linewidth=2.5, markersize=8)
ax2.plot(worker_nums, [697.12, 1485.76, 2820.77], 's--', label='GCSFS (Prefetch ON)', color=colors['gcsfs_prefetch'], linewidth=2.5, markersize=8)
ax2.plot(worker_nums, [900.17, 1686.63, 2435.81], '^-', label='PyArrow C++', color=colors['cpp'], linewidth=2.5, markersize=8)
ax2.plot([8, 32], [697.12, 697.12 * 4], 'k:', label='Ideal Linear Scaling', alpha=0.5)

ax2.set_title('Regional Storage Scaling (The Pile 65.2 GB)')
ax2.set_xlabel('Ray Worker Concurrency')
ax2.set_ylabel('Throughput (MiB/s)')
ax2.set_xticks(worker_nums)
ax2.set_ylim(0, 3200)
ax2.legend(frameon=True)

plt.tight_layout()
plt.savefig('/home/yonghuili_google_com/gcsfs/reports/charts/scaling_linearity_chart.png')
plt.close()
print("Generated scaling_linearity_chart.png")

# ==========================================
# Chart 4: Execution Timeline: Row-Group Pipelining vs. Serialized Latency Bubbles
# ==========================================
fig, ax = plt.subplots(figsize=(15, 6.8), dpi=300)

c_net = "#1a73e8"      # Google Blue: Network Fetch
c_prefetch = "#0d904f" # Green: Background Prefetch
c_cpu = "#f29900"      # Amber: CPU Decode
c_stall = "#ea4335"    # Red: Idle Latency Bubble / Stall
c_lock = "#9334e6"     # Purple: Mutex / Futex Contention
c_copy = "#e37400"     # Dark Amber: Redundant Memcpy

# 1. GCSFS (Prefetch ON) - CPU Decode (y=3)
ax.broken_barh([(0.0, 0.35)], (2.7, 0.6), facecolors=c_stall, hatch="//", edgecolor="#b3261e", alpha=0.5)
ax.broken_barh([(0.35, 0.35)], (2.7, 0.6), facecolors=c_cpu, edgecolor="black", linewidth=1)
ax.broken_barh([(0.70, 0.35)], (2.7, 0.6), facecolors=c_cpu, edgecolor="black", linewidth=1)
ax.broken_barh([(1.05, 0.35)], (2.7, 0.6), facecolors=c_cpu, edgecolor="black", linewidth=1)
ax.broken_barh([(1.40, 0.35)], (2.7, 0.6), facecolors=c_cpu, edgecolor="black", linewidth=1)

ax.text(0.175, 3.0, "Initial Wait", ha="center", va="center", color="#5f6368", fontsize=8.5, fontweight="bold")
ax.text(0.525, 3.0, "CPU Decode\nRow Group 0", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
ax.text(0.875, 3.0, "CPU Decode\nRow Group 1", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
ax.text(1.225, 3.0, "CPU Decode\nRow Group 2", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
ax.text(1.575, 3.0, "CPU Decode\nRow Group 3", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

# 2. GCSFS (Prefetch ON) - Background Network (y=2)
ax.broken_barh([(0.0, 0.35)], (1.8, 0.6), facecolors=c_net, edgecolor="black", linewidth=1)
ax.broken_barh([(0.35, 0.35)], (1.8, 0.6), facecolors=c_prefetch, edgecolor="black", linewidth=1)
ax.broken_barh([(0.70, 0.35)], (1.8, 0.6), facecolors=c_prefetch, edgecolor="black", linewidth=1)
ax.broken_barh([(1.05, 0.35)], (1.8, 0.6), facecolors=c_prefetch, edgecolor="black", linewidth=1)
ax.broken_barh([(1.40, 0.35)], (1.8, 0.6), facecolors=c_prefetch, edgecolor="black", linewidth=1)

ax.text(0.175, 2.1, "Network GET\nRow Group 0", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
ax.text(0.525, 2.1, "BG Prefetch\nRow Group 1", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
ax.text(0.875, 2.1, "BG Prefetch\nRow Group 2", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
ax.text(1.225, 2.1, "BG Prefetch\nRow Group 3", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
ax.text(1.575, 2.1, "BG Prefetch\nRow Group 4", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

ax.annotate("Pipelined Async Overlap: CPU busy while Network streams in background\nResult: Line-rate throughput (1,364.88 MiB/s, 4.17× speedup)", 
            xy=(0.875, 3.32), xytext=(0.875, 3.65),
            ha="center", fontsize=9.5, fontweight="bold", color="#0d904f",
            arrowprops=dict(arrowstyle="->", color="#0d904f", lw=2),
            bbox=dict(boxstyle="round,pad=0.4", fc="#e6f4ea", ec="#137333", lw=1.2))

# 3. GCSFS (No Cache) - Serialized (y=1)
ax.broken_barh([(0.0, 0.35)], (0.9, 0.6), facecolors=c_net, edgecolor="black", linewidth=1)
ax.text(0.175, 1.2, "Network GET\nRG 0", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

ax.broken_barh([(0.35, 0.35)], (0.9, 0.6), facecolors=c_cpu, edgecolor="black", linewidth=1)
ax.text(0.525, 1.2, "CPU Decode\nRG 0", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

ax.broken_barh([(0.70, 0.20)], (0.9, 0.6), facecolors=c_stall, hatch="//", edgecolor="#b3261e", alpha=0.6)
ax.text(0.80, 1.2, "Latency\nBubble", ha="center", va="center", color="#b3261e", fontsize=7.5, fontweight="bold")

ax.broken_barh([(0.90, 0.35)], (0.9, 0.6), facecolors=c_net, edgecolor="black", linewidth=1)
ax.text(1.075, 1.2, "Network GET\nRG 1", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

ax.broken_barh([(1.25, 0.35)], (0.9, 0.6), facecolors=c_cpu, edgecolor="black", linewidth=1)
ax.text(1.425, 1.2, "CPU Decode\nRG 1", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

ax.broken_barh([(1.60, 0.20)], (0.9, 0.6), facecolors=c_stall, hatch="//", edgecolor="#b3261e", alpha=0.6)
ax.text(1.70, 1.2, "Latency\nBubble", ha="center", va="center", color="#b3261e", fontsize=7.5, fontweight="bold")

# 4. PyArrow C++ - Contention & Memory Copies (y=0)
ax.broken_barh([(0.0, 0.22)], (0.0, 0.6), facecolors=c_lock, hatch="\\\\", edgecolor="#5e35b1", alpha=0.8)
ax.text(0.11, 0.3, "Mutex Wait\n(libcurl)", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")

ax.broken_barh([(0.22, 0.35)], (0.0, 0.6), facecolors=c_net, edgecolor="black", linewidth=1)
ax.text(0.395, 0.3, "libcurl GET\nRG 0", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

ax.broken_barh([(0.57, 0.12)], (0.0, 0.6), facecolors=c_copy, edgecolor="black", linewidth=1)
ax.text(0.63, 0.3, "Copy\n(Buf)", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

ax.broken_barh([(0.69, 0.35)], (0.0, 0.6), facecolors=c_cpu, edgecolor="black", linewidth=1)
ax.text(0.865, 0.3, "CPU Decode\nRG 0", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

ax.broken_barh([(1.04, 0.24)], (0.0, 0.6), facecolors=c_lock, hatch="\\\\", edgecolor="#5e35b1", alpha=0.8)
ax.text(1.16, 0.3, "Mutex Wait\n(libcurl)", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")

ax.broken_barh([(1.28, 0.35)], (0.0, 0.6), facecolors=c_net, edgecolor="black", linewidth=1)
ax.text(1.455, 0.3, "libcurl GET\nRG 1", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

ax.broken_barh([(1.63, 0.12)], (0.0, 0.6), facecolors=c_copy, edgecolor="black", linewidth=1)
ax.text(1.69, 0.3, "Copy", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

ax.set_yticks([0.3, 1.2, 2.1, 3.0])
ax.set_yticklabels([
    "PyArrow C++ (google-cloud-cpp)\n[OS Mutex Contention & Memcpy]",
    "GCSFS (No Cache)\n[Serialized Synchronous I/O]",
    "GCSFS (Prefetch ON)\n[Async Network Task]",
    "GCSFS (Prefetch ON)\n[Worker CPU Task]"
], fontsize=10, fontweight="bold")

ax.set_xlim(-0.05, 1.85)
ax.set_ylim(-0.25, 4.05)
ax.set_xticks([])
ax.set_xlabel("Time Progression (Simulated Execution Flow across Parquet Row Groups)", fontsize=11, fontweight="bold", labelpad=10)

patches = [
    mpatches.Patch(color=c_net, label="Network Direct GET"),
    mpatches.Patch(color=c_prefetch, label="Async Speculative Prefetch"),
    mpatches.Patch(color=c_cpu, label="CPU Arrow Decode (Compute)"),
    mpatches.Patch(facecolor=c_stall, hatch="//", label="Idle Latency Bubble (REST RTT)", edgecolor="#b3261e"),
    mpatches.Patch(facecolor=c_lock, hatch="\\\\", label="Thread Mutex Lock Wait", edgecolor="#5e35b1"),
    mpatches.Patch(color=c_copy, label="Redundant Stream Buffer Memcpy"),
]
ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fontsize=9, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig('/home/yonghuili_google_com/gcsfs/reports/charts/cache_coalescing_diagram.png', dpi=300)
plt.close()
print("Generated cache_coalescing_diagram.png")

# ==========================================
# Chart 5: Large 1GB Files Throughput Scaling (Zonal vs Regional)
# ==========================================
workers_large = ['8 Workers', '16 Workers']
x_l = np.arange(len(workers_large))
width_l = 0.24

fig, (ax1_l, ax2_l) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Panel 1: Zonal
rects1_z = ax1_l.bar(x_l - width_l, [2711.09, 2594.10], width_l, label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'])
rects2_z = ax1_l.bar(x_l, [2732.95, 2747.37], width_l, label='GCSFS (Prefetch ON)', color=colors['gcsfs_prefetch'])
rects3_z = ax1_l.bar(x_l + width_l, [1089.40, 1102.25], width_l, label='PyArrow C++', color=colors['cpp'])

ax1_l.set_ylabel('Throughput (MiB/s)')
ax1_l.set_title('Zonal Storage: Large 1.04 GB Files (8.33 GB Total)\nHigh-Bandwidth DirectPath gRPC')
ax1_l.set_xticks(x_l)
ax1_l.set_xticklabels(workers_large)
ax1_l.legend(frameon=True, loc='upper right')
ax1_l.set_ylim(0, 3400)

# Panel 2: Regional
rects1_r = ax2_l.bar(x_l - width_l, [327.17, 652.08], width_l, label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'])
rects2_r = ax2_l.bar(x_l, [1364.88, 1234.64], width_l, label='GCSFS (Prefetch ON)', color=colors['gcsfs_prefetch'])
rects3_r = ax2_l.bar(x_l + width_l, [906.28, 966.95], width_l, label='PyArrow C++', color=colors['cpp'])

ax2_l.set_ylabel('Throughput (MiB/s)')
ax2_l.set_title('Regional Storage: Large 1.04 GB Files (8.33 GB Total)\nStandard GFE HTTP/2 REST (Latency Dominated)')
ax2_l.set_xticks(x_l)
ax2_l.set_xticklabels(workers_large)
ax2_l.legend(frameon=True, loc='upper right')
ax2_l.set_ylim(0, 1800)

autolabel(ax1_l, rects1_z)
autolabel(ax1_l, rects2_z)
autolabel(ax1_l, rects3_z)

autolabel(ax2_l, rects1_r)
autolabel(ax2_l, rects2_r)
autolabel(ax2_l, rects3_r)

plt.tight_layout()
plt.savefig('/home/yonghuili_google_com/gcsfs/reports/charts/large_files_throughput_chart.png', dpi=300)
plt.close()
print("Saved /home/yonghuili_google_com/gcsfs/reports/charts/large_files_throughput_chart.png")
