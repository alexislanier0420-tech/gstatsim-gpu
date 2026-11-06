"""Poster column 4: the large region (Amundsen Sea Embayment), kriged on the GPU.

Reads results/poster/ase.npz (compute_region.py) and the HiPerGator timings.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib.colors import ListedColormap, Normalize

from common import BED_CMAP, COL_W, GL, INK, INK2, MUTED, REGION, RESULTS, SHELF, save, style

style()
r = np.load(RESULTS / "ase.npz")
steps = {}
for f in REGION[1000]:
    steps |= json.loads(f.read_text(encoding="utf-8"))["steps"]
t_gpu, t_cpu = steps["gpu_kriging"]["seconds"], steps["gstatsim_kriging"]["seconds"]

# North-up polar stereographic on the region's central meridian, as for Rubble
LON_C = -103.0
to_loc = pyproj.Transformer.from_crs(3031, f"+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0={LON_C} "
                                     "+datum=WGS84 +units=m +no_defs", always_xy=True)
ll_to_loc = pyproj.Transformer.from_crs(4326, to_loc.target_crs, always_xy=True)


def km(x, y):
    lx, ly = to_loc.transform(np.asarray(x, float), np.asarray(y, float))
    return lx / 1e3, ly / 1e3


def corners(xc, yc, res):
    """Cell-corner mesh (local km) for a regular EPSG:3031 grid, x ascending, y descending."""
    xe = np.concatenate([xc - res / 2, [xc[-1] + res / 2]])
    ye = np.concatenate([yc + res / 2, [yc[-1] - res / 2]])
    return km(*np.meshgrid(xe, ye))


rows, cols, res = int(r["rows"]), int(r["cols"]), float(r["res"])
pred = r["pred_xy"]
X, Y = pred[:, 0].reshape(rows, cols), pred[:, 1].reshape(rows, cols)
bed = np.full(rows * cols, np.nan)
bed[r["nodes"]] = r["sk_bed"]
bed = bed.reshape(rows, cols)

fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.47))
fig.subplots_adjust(left=0, right=0.56, bottom=0, top=1)

# Floating ice and ocean (no bed data) as a pale fill, then the kriged bed
sx, sy, mask = r["x"].astype(float), r["y"].astype(float), r["mask"]
ex, ey = corners(sx, sy, 500)
ax.pcolormesh(ex, ey, np.where(np.isin(mask, (1, 2)), np.nan, 1.0), cmap=ListedColormap([SHELF]),
              shading="flat", rasterized=True, zorder=0)
mx, my = corners(X[0], Y[:, 0], res)
norm = Normalize(-2000, 1500)
im = ax.pcolormesh(mx, my, bed, cmap=BED_CMAP, norm=norm, shading="flat", rasterized=True, zorder=1)
cx, cy = km(*np.meshgrid(sx, sy))
ax.contour(cx, cy, np.isin(mask, (1, 2)).astype(float), levels=[0.5], colors=GL, linewidths=1.0, zorder=3)

for name, lon, lat in (("Thwaites", -106.5, -75.4), ("Pine Island", -99.5, -75.2)):
    x, y = ll_to_loc.transform(lon, lat)
    ax.text(x / 1e3, y / 1e3, name, fontsize=14, color=INK, fontweight="bold", ha="center", va="center",
            zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

# View: the region's corners in local km
bx, by = km(np.array([sx.min(), sx.max(), sx.min(), sx.max()]), np.array([sy.min(), sy.min(), sy.max(), sy.max()]))
# The box is rotated in the north-up view: crop to the rectangle inside it
xs, ys = np.sort(bx), np.sort(by)
ax.set_xlim(xs[1], xs[2])
ax.set_ylim(ys[1], ys[2])
bx, by = xs[1:3], ys[1:3]
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_color("#c3c2b7")
x0, y0 = bx.min() + 0.06 * np.ptp(bx), by.min() + 0.05 * np.ptp(by)
ax.plot([x0, x0 + 100], [y0, y0], color=INK, lw=4, solid_capstyle="butt", zorder=6)
ax.text(x0 + 50, y0 + 0.02 * np.ptp(by), "100 km", ha="center", va="bottom", fontsize=14, zorder=6,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
ax.annotate("N", xy=(bx.max() - 0.08 * np.ptp(bx), by.max() - 0.04 * np.ptp(by)),
            xytext=(bx.max() - 0.08 * np.ptp(bx), by.max() - 0.15 * np.ptp(by)),
            ha="center", va="center", fontsize=15, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2), zorder=6)

# Right side: what it took, and the colour key
fig.text(0.6, 0.97, f"{len(r['nodes']):,} cells\nat 1 km, kriged in", fontsize=16, color=INK2, va="top")
fig.text(0.6, 0.80, f"{t_gpu:.0f} s", fontsize=34, color="#2a78d6", fontweight="bold", va="top")
fig.text(0.6, 0.66, "on the GPU", fontsize=16, color=INK2, va="top")
fig.text(0.6, 0.55, f"GStatSim: {t_cpu / 60:.0f} min", fontsize=16, color="#eb6834", fontweight="bold",
         va="top")
cax = fig.add_axes([0.62, 0.16, 0.33, 0.045])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
cb.set_ticks([-2000, -1000, 0, 1000])
cb.ax.tick_params(labelsize=13)
cb.outline.set_visible(False)
cax.set_title("Bed (m above sea level)", fontsize=14, color=INK2, pad=6)
fig.text(0.6, 0.02, "Pale: floating ice / ocean", fontsize=12, color=MUTED, va="bottom")
save(fig, "amundsen")
