"""Poster column 1: where the study area is, and what data constrain the bed."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from common import (
    _LOCAL,  # north-up projection string
    BED_CMAP,
    BED_NORM,
    COL_W,
    GL,
    INK,
    INK2,
    ROOT,
    SHELF,
    StudyArea,
    save,
    style,
    to_local_km,
)

ROCK = "#8f8d87"

style()
area = StudyArea()
d = np.load(ROOT / "figures" / "data" / "study_area.npz")

fig = plt.figure(figsize=(COL_W, COL_W * 1.12))
ax = fig.add_axes([0.10, 0.17, 0.86, 0.80])

# Context fills: ice shelf, exposed rock; grounded ice stays white
area.shelf(ax)
rock = np.where(d["mask"] == 1, 1.0, np.nan)
ax.pcolormesh(area.ex, area.ey, rock, cmap=ListedColormap([ROCK]), shading="flat",
              rasterized=True, zorder=1)
# Radar bed picks (500 m cells), on the same elevation scale as the results
rx, ry = to_local_km(d["radar_x"], d["radar_y"])
sc = ax.scatter(rx, ry, c=d["radar_bed"], cmap=BED_CMAP, norm=BED_NORM, s=2.2, marker="s",
                linewidths=0, rasterized=True, zorder=2)
area.grounding_line(ax)
area.frame(ax)
area.scale_bar(ax)
area.north_arrow(ax, side="left")

# Rubble Glacier, at its published position (71°20'S, 68°18'W)
gx, gy = pyproj.Transformer.from_crs(4326, _LOCAL, always_xy=True).transform(-68.3, -71 - 20 / 60)
gx, gy = gx / 1e3, gy / 1e3
ax.plot(gx, gy, "o", ms=16, mfc="none", mec=INK, mew=2.5, zorder=7)
ax.annotate("Rubble\nGlacier", xy=(gx - 3, gy), xytext=(gx - 40, gy - 4), fontsize=17,
            fontweight="bold", color=INK, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.8), zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
for text, lat, lon in (("ALEXANDER\nISLAND", -71.55, -69.85), ("George VI\nIce Shelf", -71.05, -67.95),
                       ("PALMER\nLAND", -71.75, -66.8)):
    x, y = pyproj.Transformer.from_crs(4326, _LOCAL, always_xy=True).transform(lon, lat)
    ax.text(x / 1e3, y / 1e3, text, ha="center", va="center", fontsize=14, color=INK2,
            style="italic" if "Shelf" in text else "normal", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# Colorbar and legend under the map
cax = fig.add_axes([0.10, 0.075, 0.86, 0.022])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both",
                  ticks=[-1000, -500, 0, 500, 1000, 1500, 2000])
cb.set_label("Radar bed elevation (m above sea level)", fontsize=16)
cb.outline.set_visible(False)
fig.legend(handles=[Patch(fc=ROCK, label="Exposed rock (bed = surface)"),
                    Patch(fc=SHELF, label="Ice shelf (no bed data)"),
                    Line2D([], [], color=GL, lw=2, label="Grounding line / coast")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.53, -0.035), fontsize=13.5, handlelength=1.4,
           columnspacing=1.0)

# Locator: Antarctica with the study box
m = np.load(ROOT / "figures" / "data" / "antarctica_mask_5km.npz")
ins = fig.add_axes([0.715, 0.725, 0.24, 0.24])
cls = m["mask"].astype(float)
ins.pcolormesh(m["x"] / 1e3, m["y"] / 1e3, np.where(cls == 0, np.nan, cls),
               cmap=ListedColormap(["#8f8d87", "#f4f4f2", SHELF, "#f4f4f2"]), vmin=0.5, vmax=4.5,
               shading="nearest", rasterized=True)
ins.add_patch(Rectangle((d["x"].min() / 1e3, d["y"].min() / 1e3), 200, 200, fc="none", ec="#c22f2f",
                        lw=2.5, zorder=5))
ins.set_aspect("equal")
ins.set_xlim(-2800, 2800)
ins.set_ylim(-2400, 2400)
ins.set_xticks([])
ins.set_yticks([])
ins.set_facecolor("#dfe9f2")
for s in ins.spines.values():
    s.set_visible(True)
    s.set_color(INK2)
save(fig, "study_area")
