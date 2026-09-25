#!/usr/bin/env python3
"""
GPU Hardware Diagram — NVIDIA GeForce GTX 1650 (sm_75, Turing)

Renders the chip -> SM -> register file -> L2 -> DRAM -> host memory hierarchy
with measured specs. Outputs docs/gpu_hardware_diagram.png

Run:  /tmp/opencode/pl-env/bin/python scripts/hw_diagram.py
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')


def draw_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle='',
    color='#ecf0f1',
    edgecolor='#2c3e50',
    title_size=8,
    sub_size=6.5,
):
  rect = patches.FancyBboxPatch(
      (x, y),
      w,
      h,
      boxstyle='round,pad=0.04',
      facecolor=color,
      edgecolor=edgecolor,
      linewidth=1.1,
  )
  ax.add_patch(rect)
  if subtitle:
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha='center',
        va='center',
        fontsize=title_size,
        weight='bold',
        color='#111111',
    )
    ax.text(
        x + w / 2,
        y + h * 0.3,
        subtitle,
        ha='center',
        va='center',
        fontsize=sub_size,
        color='#333333',
    )
  else:
    ax.text(
        x + w / 2,
        y + h / 2,
        title,
        ha='center',
        va='center',
        fontsize=title_size,
        weight='bold',
        color='#111111',
    )


# 1. Host CPU RAM
draw_box(
    ax,
    3.0,
    9.0,
    8.0,
    0.65,
    'HOST CPU RAM (h_* buffers, malloc)',
    'PCIe Bus (~12-16 GB/s transfer via cudaMemcpy)',
    color='#d5dbdb',
    edgecolor='#34495e',
    title_size=8.5,
)
ax.annotate(
    '',
    xy=(7.0, 8.1),
    xytext=(7.0, 9.0),
    arrowprops=dict(arrowstyle='<->', lw=1.5, color='#2c3e50'),
)

# 2. GPU Die Boundary
gpu_outer = patches.FancyBboxPatch(
    (0.5, 0.3),
    13.0,
    7.8,
    boxstyle='round,pad=0.06',
    facecolor='#f8f9fa',
    edgecolor='#7f8c8d',
    linestyle='--',
    linewidth=1.4,
)
ax.add_patch(gpu_outer)
ax.text(
    0.7,
    7.9,
    'GPU Die — GTX 1650 (sm_75, Turing) | 16 SMs Total',
    fontsize=9.5,
    weight='bold',
    color='#2c3e50',
)

# Top of Die: VRAM / 4 GB GDDR DRAM
draw_box(
    ax,
    1.5,
    6.8,
    11.0,
    0.8,
    '4 GB GDDR DRAM (VRAM / Global Memory - d_* buffers)',
    'Internal Off-Chip Bandwidth = 128 GB/s (4001 MHz, 128-bit bus) | Latency ~400-600 cycles',
    color='#e8daef',
    edgecolor='#8e44ad',
    title_size=8.5,
)
ax.annotate(
    '',
    xy=(7.0, 6.0),
    xytext=(7.0, 6.8),
    arrowprops=dict(arrowstyle='<->', lw=1.5, color='#8e44ad'),
)

# Middle: 1 MB L2 Cache
draw_box(
    ax,
    1.5,
    5.0,
    11.0,
    0.95,
    '1 MB L2 Cache (Shared across ALL 16 SMs)',
    'Coalescing, atomic resolution, last stop before DRAM (~200 cycles latency)',
    color='#fdebd0',
    edgecolor='#d35400',
    title_size=8.5,
)
ax.annotate(
    '',
    xy=(7.0, 4.3),
    xytext=(7.0, 5.0),
    arrowprops=dict(arrowstyle='<->', lw=1.5, color='#d35400'),
)

# Bottom Section: Zoom-in or SM Detail Breakdown Container
sm_group = patches.FancyBboxPatch(
    (1.5, 0.5),
    11.0,
    3.7,
    boxstyle='round,pad=0.05',
    facecolor='#ebf5fb',
    edgecolor='#2980b9',
    linewidth=1.1,
)
ax.add_patch(sm_group)
ax.text(
    1.7,
    3.95,
    'Streaming Multiprocessor (SM) Detail View (16 physical SMs on die, showing breakdown of 1 SM or macro block representation):',
    fontsize=8,
    weight='bold',
    color='#1b4f72',
)

# Detailed internal view of SMs side-by-side inside bottom container
sm_configs = [
    ('SM 0 (Active)', 1.8),
    ('SM 1 (Active)', 5.5),
    ('... SM 2..15 (14 more)', 9.2),
]
for sm_label, sx in sm_configs:
  sub_sm = patches.FancyBboxPatch(
      (sx, 0.7),
      3.0,
      3.1,
      boxstyle='round,pad=0.03',
      facecolor='#d4e6f1',
      edgecolor='#2980b9',
      linewidth=0.9,
  )
  ax.add_patch(sub_sm)
  ax.text(
      sx + 1.5,
      3.6,
      sm_label,
      ha='center',
      va='center',
      fontsize=8,
      weight='bold',
      color='#154360',
  )

  # Inside SM: Register File (256 KB)
  draw_box(
      ax,
      sx + 0.15,
      2.6,
      2.7,
      0.7,
      '256 KB Register File',
      '65,536 regs/SM (0-1 cycle)\nPrivate to 1 thread (~64 regs/th maxocc)',
      color='#a9cce3',
      edgecolor='#1f618d',
      title_size=7,
      sub_size=5.5,
  )

  # Inside SM: Shared / L1 (64 KB)
  draw_box(
      ax,
      sx + 0.15,
      1.75,
      2.7,
      0.7,
      '64 KB Shared / L1',
      '49 KB usable/block (~20-30 cyc)\nShared by threads of ONE block',
      color='#a3e4d7',
      edgecolor='#117864',
      title_size=7,
      sub_size=5.5,
  )

  # Inside SM: Execution Lanes (FP32)
  draw_box(
      ax,
      sx + 0.15,
      0.9,
      2.7,
      0.75,
      'FP32 / INT / Warp Scheduler',
      '128 FP32 lanes/SM @ 1560 MHz\nExecutes warps of 32 threads',
      color='#fadbd8',
      edgecolor='#c0392b',
      title_size=7,
      sub_size=5.5,
  )

plt.tight_layout()
plt.savefig('scripts/gtx1650_detailed_sm_vram.svg', format='svg')
plt.show()