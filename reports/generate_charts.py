#!/usr/bin/env python3
import matplotlib.pyplot as plt
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
    'gcsfs_readahead': '#fbbc04',     # Google Yellow
    'cpp': '#ea4335'                  # Google Red
}

# ==========================================
# Chart 1: Zonal Storage Throughput Scaling
# ==========================================
workers = ['8 Workers', '16 Workers', '32 Workers']
x = np.arange(len(workers))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

rects1 = ax.bar(x - 1.5*width, [1685.36, 3101.44, 4910.55], width, label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'])
rects2 = ax.bar(x - 0.5*width, [1679.12, 3087.47, 4856.13], width, label='GCSFS (Prefetch ON)', color=colors['gcsfs_prefetch'])
rects3 = ax.bar(x + 0.5*width, [1574.91, 2813.22, 4512.87], width, label='GCSFS (ReadAhead Cache)', color=colors['gcsfs_readahead'])
rects4 = ax.bar(x + 1.5*width, [973.89, 1243.24, 2705.38], width, label='PyArrow C++', color=colors['cpp'])

ax.set_ylabel('Throughput (MiB/s)')
ax.set_title('Zonal Storage (The Pile Deduplicated - 256 Files / 65.2 GB)\nZero-Copy Arrow Streaming Throughput')
ax.set_xticks(x)
ax.set_xticklabels(workers)
ax.legend(frameon=True, loc='upper left', shadow=True)
ax.set_ylim(0, 5600)

# Add data labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(round(height))}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

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

rects1 = ax.bar(x - 1.5*width, [438.81, 749.02, 1414.78], width, label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'])
rects2 = ax.bar(x - 0.5*width, [410.97, 775.04, 1394.43], width, label='PyArrow C++', color=colors['cpp'])
rects3 = ax.bar(x + 0.5*width, [366.46, 796.62, 1250.14], width, label='GCSFS (Prefetch ON)', color=colors['gcsfs_prefetch'])
rects4 = ax.bar(x + 1.5*width, [394.68, 698.22, 1193.55], width, label='GCSFS (ReadAhead Cache)', color=colors['gcsfs_readahead'])

ax.set_ylabel('Throughput (MiB/s)')
ax.set_title('Regional Storage (64 Files / 12.84 GB - ~200 MB Each)\nZero-Copy Arrow Streaming Throughput')
ax.set_xticks(x)
ax.set_xticklabels(workers)
ax.legend(frameon=True, loc='upper left', shadow=True)
ax.set_ylim(0, 1650)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

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

ax1.set_title('Zonal Storage Scaling (The Pile 65 GB)')
ax1.set_xlabel('Ray Worker Concurrency')
ax1.set_ylabel('Throughput (MiB/s)')
ax1.set_xticks(worker_nums)
ax1.set_ylim(0, 5500)
ax1.legend(frameon=True)

# Regional
ax2.plot(worker_nums, [438.81, 749.02, 1414.78], 'o-', label='GCSFS (No Cache)', color=colors['gcsfs_no_cache'], linewidth=2.5, markersize=8)
ax2.plot(worker_nums, [410.97, 775.04, 1394.43], '^-', label='PyArrow C++', color=colors['cpp'], linewidth=2.5, markersize=8)
ax2.plot(worker_nums, [394.68, 698.22, 1193.55], 'd-.', label='GCSFS (ReadAhead Cache)', color=colors['gcsfs_readahead'], linewidth=2)
ax2.plot([8, 32], [438.81, 438.81 * 4], 'k:', label='Ideal Linear Scaling', alpha=0.5)

ax2.set_title('Regional Storage Scaling (64 Files 12.8 GB)')
ax2.set_xlabel('Ray Worker Concurrency')
ax2.set_ylabel('Throughput (MiB/s)')
ax2.set_xticks(worker_nums)
ax2.set_ylim(0, 1800)
ax2.legend(frameon=True)

plt.tight_layout()
plt.savefig('/home/yonghuili_google_com/gcsfs/reports/charts/scaling_linearity_chart.png')
plt.close()
print("Generated scaling_linearity_chart.png")

# ==========================================
# Chart 4: Diagram of Pre-Buffer Coalescing vs Fragmentation
# ==========================================
fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
ax.axis('off')

# Box styles
bbox_pyarrow = dict(boxstyle="round,pad=0.5", fc="#e8f0fe", ec="#1a73e8", lw=2)
bbox_nocache = dict(boxstyle="round,pad=0.5", fc="#e6f4ea", ec="#137333", lw=2)
bbox_readahead = dict(boxstyle="round,pad=0.5", fc="#fef7e0", ec="#b06000", lw=2)

ax.text(0.05, 0.8, "Application Layer:\nPyArrow ParquetReader (pre_buffer=True)", 
        bbox=bbox_pyarrow, fontsize=11, fontweight='bold', va='center')
ax.text(0.55, 0.8, "Coalesces Column Chunks into a Single Range:\n[Offset: 12,048, Length: 200,691,782 bytes]", 
        bbox=dict(boxstyle="square,pad=0.4", fc="#ffffff", ec="#888888", lw=1), fontsize=10, va='center')

# Arrows
ax.annotate("", xy=(0.25, 0.52), xytext=(0.25, 0.68),
            arrowprops=dict(arrowstyle="->", lw=2, color="#137333"))
ax.annotate("", xy=(0.75, 0.52), xytext=(0.75, 0.68),
            arrowprops=dict(arrowstyle="->", lw=2, color="#b06000"))

# Branches
ax.text(0.25, 0.35, "GCSFS No Cache (BaseCache):\n\n✓ Raw Passthrough to Transport\n✓ Single Continuous Stream\n✓ Line-Rate Saturation (4.91 GiB/s)\n✓ Zero Intermediate Buffering",
        bbox=bbox_nocache, fontsize=10, va='center', ha='center')

ax.text(0.75, 0.35, "GCSFS ReadAhead Cache:\n\n✗ Slices into 5 MiB Blocks\n✗ 40 Sequential HTTP Range Requests\n✗ Buffer Reallocations & Window Tracking\n✗ Event Loop & Mutex Congestion",
        bbox=bbox_readahead, fontsize=10, va='center', ha='center')

plt.tight_layout()
plt.savefig('/home/yonghuili_google_com/gcsfs/reports/charts/cache_coalescing_diagram.png')
plt.close()
print("Generated cache_coalescing_diagram.png")
