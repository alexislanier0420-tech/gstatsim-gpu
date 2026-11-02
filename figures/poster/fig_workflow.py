"""Poster column 2: how the workflow runs, and how the GPU runs SGS."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import COL_W, GPU, INK, INK2, MUTED, save, style

style()
fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.98))
ax.set_xlim(0, 100)
ax.set_ylim(0, 98)
ax.axis("off")

GPU_FILL, NEUTRAL = "#e3eefb", "#f3f2ef"


def box(x, y, w, h, text, fc=NEUTRAL, ec=MUTED, fs=15, weight="normal", color=INK, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight,
            color=color, linespacing=1.15)


def arrow(p0, p1, color=INK2, lw=2.0, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


# 1. Preparation (CPU, seconds)
steps = ["Radar +\nrock", "Grid\n1 km", "Remove\ntrend", "Normal\nscores", "Vario-\ngram"]
w, gap = 16.4, 4.0
for i, s in enumerate(steps):
    x = 1 + i * (w + gap)
    box(x, 84, w, 11, s, fs=14)
    if i:
        arrow((x - gap + 0.5, 89.5), (x - 0.5, 89.5))
ax.text(50, 80.3, "Preparation: CPU, seconds", ha="center", fontsize=13, color=MUTED)

# 2. The slow steps, on the GPU
arrow((30, 78), (24, 72.5))
arrow((70, 78), (76, 72.5))
box(2, 59, 44, 12.5, "Kriging\ncells are independent:\nsolve them all at once", fc=GPU_FILL, ec=GPU, fs=14)
box(54, 59, 44, 12.5, "SGS\ncells depend on earlier ones:\none GPU block per realization",
    fc=GPU_FILL, ec=GPU, fs=14)
arrow((76, 57.6), (40, 50.5), color=GPU, lw=1.8)

# 3. Inside the SGS kernel: one realization in detail...
ax.add_patch(FancyBboxPatch((3, 3), 57.5, 46, boxstyle="round,pad=0.3,rounding_size=1.6",
                            fc="white", ec=GPU, lw=2.4))
ax.text(31, 45.0, "Realization 1", ha="center", fontsize=15, color=GPU, fontweight="bold")
ax.text(31, 41.8, "one thread block, 256 threads", ha="center", fontsize=12.5, color=INK2)
sub = ["Find the 96 nearest neighbors\n(8 octants × 12)", "Build covariance matrix\nin fast shared memory",
       "Cholesky solve:\nkriging mean + variance", "Draw the value,\nadd it to the data"]
for k, s in enumerate(sub):
    y = 31.5 - k * 9.2
    box(8, y, 40, 7.4, s, fs=12.5, fc=NEUTRAL, ec=AXIS_C if (AXIS_C := "#c3c2b7") else MUTED, lw=1.2)
    if k < 3:
        arrow((28, y - 0.4), (28, y - 1.8), lw=1.6)
# loop back: next cell
arrow((48.6, 6.2), (48.6, 35.2), color=GPU, lw=2.0, connectionstyle="arc3,rad=0.25")
ax.text(56.4, 20.7, "next\ncell", ha="center", va="center", fontsize=12.5, color=GPU, fontweight="bold")

# ...and the rest running side by side
for k in range(3):
    ax.add_patch(FancyBboxPatch((66 + k * 2.2, 10 + k * 2.2), 26, 30,
                                boxstyle="round,pad=0.3,rounding_size=1.6", fc="white", ec=GPU, lw=1.8,
                                zorder=3 + k))
ax.text(83.4, 29.0, "Realizations\n2 … N", ha="center", va="center", fontsize=15, color=GPU,
        fontweight="bold", zorder=10)
ax.text(83.4, 20.0, "same steps,\nrunning side by\nside on the GPU", ha="center", va="center", fontsize=12.5,
        color=INK2, zorder=10)
ax.text(50, -2.2, "One launch simulates every cell of every realization: the GPU never waits for Python.",
        ha="center", fontsize=12.5, color=INK2)
save(fig, "workflow")
