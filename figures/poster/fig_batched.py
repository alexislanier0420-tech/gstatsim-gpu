"""Poster column 2: exact SGS vs simulating many cells at once."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import BATCHED, COL_W, GPU, INK, INK2, MUTED, RESULTS, save, style

style()
b = np.load(RESULTS / "batched.npz")
r = np.load(RESULTS / "rubble.npz")
lags = b["lags"]
box = int(b["sub"][2])
MAXLAG = 15
keep = lags <= MAXLAG


def grid_semivariogram(field):
    g = []
    for h in lags[keep]:
        dx = (field[:, h:] - field[:, :-h]) ** 2
        dy = (field[h:, :] - field[:-h, :]) ** 2
        v = np.concatenate([dx[np.isfinite(dx)], dy[np.isfinite(dy)]])
        g.append(0.5 * v.mean())
    return np.array(g)


# The data's own variogram in the same sub-area, from the same grid offsets
pts = b["pts"]
where = {(x, y): i for i, (x, y) in enumerate(map(tuple, pts))}
data_field = np.full(box * box, np.nan)
for x, y, z in zip(r["data_x"], r["data_y"], r["data_nres"], strict=True):
    i = where.get((x, y))
    if i is not None:
        data_field[i] = z
data_gamma = grid_semivariogram(data_field.reshape(box, box))
coverage = np.isfinite(data_field).mean()

fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.62))

# --- realization variograms vs the data's, same sub-area
ax.grid(True, axis="y")
ax.plot(lags[keep], data_gamma, "o", color=INK, ms=8, label="Radar data", zorder=5)
for key, color, label in (("1", GPU, "Exact SGS (this project)"), ("4096", BATCHED, "Whole grid at once")):
    g = b[f"gamma_{key}"][:, keep]
    lo, hi = np.percentile(g, [10, 90], axis=0)
    ax.fill_between(lags[keep], lo, hi, color=color, alpha=0.2, linewidth=0, zorder=1)
    ax.plot(lags[keep], g.mean(axis=0), color=color, lw=3, label=label, zorder=4)
ex, al = b["gamma_1"].mean(axis=0)[0], b["gamma_4096"].mean(axis=0)[0]
ax.annotate(f"+{(al / ex - 1) * 100:.0f}% rougher at 1 km", xy=(1, al), xytext=(3.2, al - 0.02),
            fontsize=15, color=INK2, va="center",
            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
ax.set_xlim(0, MAXLAG)
ax.set_ylim(0, None)
ax.set_xlabel("Distance between cells (km)")
ax.set_ylabel("Semivariance\n(normal scores)")
ax.legend(loc="lower right", fontsize=15)
ax.text(0.0, -0.24, f"60 × 60 km of Alexander Island ({coverage:.0%} of cells have radar). "
        f"Bands: 10–90% of {b['gamma_1'].shape[0]} realizations.\n"
        "Batches of 64 and 512 cells stay within 4% of exact.",
        transform=ax.transAxes, fontsize=13, color=MUTED, va="top")
save(fig, "batched")
