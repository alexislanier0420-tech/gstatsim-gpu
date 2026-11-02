"""Poster column 3: four SGS realizations, and the ensemble's uncertainty."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from common import (
    BED_CMAP,
    BED_NORM,
    COL_W,
    DIFF_CMAP,
    GL,
    INK2,
    RESULTS,
    SD_CMAP,
    SHELF,
    StudyArea,
    node_mesh,
    save,
    style,
    to_grid,
)

style()
r = np.load(RESULTS / "rubble.npz")
area = StudyArea()
ex, ey = node_mesh(r, area)


def map_axes(ax, left=True, bottom=True):
    area.shelf(ax)
    area.grounding_line(ax)
    area.frame(ax, left=left, bottom=bottom)


# ---------------------------------------------------------------- 4 realizations
fig, axs = plt.subplots(2, 2, figsize=(COL_W, COL_W * 1.08),
                        gridspec_kw=dict(wspace=0.06, hspace=0.08))
for i, ax in enumerate(axs.flat):
    im = ax.pcolormesh(ex, ey, to_grid(r, r["beds"][i]), cmap=BED_CMAP, norm=BED_NORM,
                       shading="flat", rasterized=True, zorder=1)
    map_axes(ax, left=i % 2 == 0, bottom=i >= 2)
    ax.text(0.03, 0.97, f"Realization {i + 1}", transform=ax.transAxes, va="top", fontsize=16,
            color="#0b0b0b", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            zorder=7)
area.scale_bar(axs[0, 0], fs=14)
area.north_arrow(axs[0, 1], fs=15)
cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.045, pad=0.06, aspect=40,
                  extend="both", ticks=[-1000, -500, 0, 500, 1000, 1500, 2000])
cb.set_label("Bed elevation (m above sea level)")
cb.outline.set_visible(False)
fig.legend(handles=[Patch(fc=SHELF, label="Ice shelf (no bed data)"),
                    plt.Line2D([], [], color=GL, lw=2, label="Grounding line / coast")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=15)
save(fig, "realizations")

# ---------------------------------------------------------------- uncertainty
beds = r["beds"]
sd = to_grid(r, beds.std(axis=0))
diff = to_grid(r, beds.mean(axis=0) - r["bedmachine"][r["nodes"]])
fig, axs = plt.subplots(1, 2, figsize=(COL_W, COL_W * 0.62), gridspec_kw=dict(wspace=0.06))
im0 = axs[0].pcolormesh(ex, ey, sd, cmap=SD_CMAP, vmin=0, vmax=300, shading="flat",
                        rasterized=True, zorder=1)
im1 = axs[1].pcolormesh(ex, ey, diff, cmap=DIFF_CMAP, vmin=-400, vmax=400, shading="flat",
                        rasterized=True, zorder=1)
map_axes(axs[0])
map_axes(axs[1], left=False)
panels = ((axs[0], im0, "Uncertainty: std. dev.\nof 20 realizations (m)", [0, 100, 200, 300]),
          (axs[1], im1, "Ensemble mean minus\nBedMachine v3 (m)", [-400, -200, 0, 200, 400]))
for ax, im, label, ticks in panels:
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.1, aspect=22,
                      extend="max" if ticks[0] == 0 else "both", ticks=ticks)
    cb.set_label(label, fontsize=16)
    cb.outline.set_visible(False)
med = np.nanmedian(np.abs(diff))
axs[1].text(0.97, 0.03, f"median |diff| {med:.0f} m", transform=axs[1].transAxes, ha="right",
            va="bottom", fontsize=14, color=INK2,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=7)
save(fig, "uncertainty")
