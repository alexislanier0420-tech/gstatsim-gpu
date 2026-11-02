"""Shared style and map helpers for the poster figures.

Figures are drawn at their printed size on the 36 x 24 in poster, with type in
poster-scale points, so they go into PowerPoint at 100% with no rescaling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pyproj
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "poster"
RESULTS = ROOT / "results" / "poster"

# Poster grid: 4 columns between 0.75 in margins and 0.5 in gutters
COL_W = 8.25          # in
FULL_W = 2 * COL_W + 0.5

# Roles, fixed across every chart (validated: all pairs pass CVD + normal-vision floors)
GPU = "#2a78d6"       # this project
CPU = "#eb6834"       # GStatSim on the CPU
BATCHED = "#1baf7a"   # batched shortcut (low contrast on white: always direct-labeled)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SHELF = "#e4e3de"     # floating ice shelf, no bed data (neutral, so it never reads as a data value)
GL = "#16324f"        # grounding line / coast

# Bed elevation: brightness rises steadily with elevation (a colormap that jumps
# in brightness at sea level turns near-0 m noise into speckle). gist_earth is
# also what the GStatSim book uses.
BED_CMAP = mpl.colormaps["gist_earth"]
BED_NORM = Normalize(vmin=-1000, vmax=2000)
# Uncertainty: one hue, light -> dark (reference sequential blue ramp)
SD_CMAP = LinearSegmentedColormap.from_list(
    "sd", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# Differences: blue <-> red with a neutral gray at zero
DIFF_CMAP = LinearSegmentedColormap.from_list("diff", ["#1c5cab", "#86b6ef", "#f0efec", "#ef8f8e", "#c22f2f"])


def style():
    mpl.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 18, "axes.labelsize": 19, "axes.titlesize": 21,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 1.0,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100,
        "lines.linewidth": 2.0,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print("wrote", OUT / f"{name}.png")


# ------------------------------------------------------------------ maps
# North-up polar stereographic centered on Rubble Glacier's meridian (as in the
# method figure), so north is straight up on every map.
LON_C = -68.3
_LOCAL = (f"+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0={LON_C} "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
_PS_TO_LOC = pyproj.Transformer.from_crs(3031, _LOCAL, always_xy=True)
_LOC_TO_LL = pyproj.Transformer.from_crs(_LOCAL, 4326, always_xy=True)


def to_local_km(x, y):
    """EPSG:3031 metres -> north-up local km."""
    lx, ly = _PS_TO_LOC.transform(np.asarray(x, float), np.asarray(y, float))
    return lx / 1e3, ly / 1e3


def grid_corners(xc, yc, res):
    """Cell-corner coordinates (local km) for a regular EPSG:3031 grid given
    its 1-D center coordinates; for pcolormesh on the rotated grid."""
    xe = np.concatenate([xc - res / 2, [xc[-1] + res / 2]]) if xc[1] > xc[0] else \
        np.concatenate([xc + res / 2, [xc[-1] - res / 2]])
    ye = np.concatenate([yc + res / 2, [yc[-1] - res / 2]]) if yc[1] < yc[0] else \
        np.concatenate([yc - res / 2, [yc[-1] + res / 2]])
    X, Y = np.meshgrid(xe, ye)
    return to_local_km(X, Y)


class StudyArea:
    """BedMachine mask of the study box, for grounding lines and shelf fills."""

    def __init__(self):
        d = np.load(ROOT / "figures" / "data" / "study_area.npz")
        self.x, self.y, self.mask = d["x"].astype(float), d["y"].astype(float), d["mask"]
        X, Y = np.meshgrid(self.x, self.y)
        self.cx, self.cy = to_local_km(X, Y)              # cell centers, local km
        self.ex, self.ey = grid_corners(self.x, self.y, 500)
        # View: the largest north-up square inside the study box. EPSG:3031's axes
        # sit ~68 degrees off north-up at this longitude, so the box is rotated
        # here (same construction as the method figure).
        x0, x1 = self.x.min() - 250, self.x.max() + 250
        y0, y1 = self.y.min() - 250, self.y.max() + 250
        cx, cy = to_local_km((x0 + x1) / 2, (y0 + y1) / 2)
        rx, ry = to_local_km(x1, (y0 + y1) / 2)
        theta = np.arctan2(ry - cy, rx - cx)
        half = 0.97 * (x1 - x0) / 2e3 / (abs(np.cos(theta)) + abs(np.sin(theta)))
        self.xlim = (float(cx - half), float(cx + half))
        self.ylim = (float(cy - half), float(cy + half))

    def shelf(self, ax):
        """Floating ice shelf (no bed data) as a pale fill."""
        m = np.where(self.mask == 3, 1.0, np.nan)
        ax.pcolormesh(self.ex, self.ey, m, cmap=mpl.colors.ListedColormap([SHELF]),
                      shading="flat", rasterized=True, zorder=0)

    def grounding_line(self, ax, lw=1.6):
        grounded = np.isin(self.mask, (1, 2)).astype(float)
        ax.contour(self.cx, self.cy, grounded, levels=[0.5], colors=GL, linewidths=lw, zorder=4)

    def frame(self, ax, lat_ticks=(-71.0, -71.5), lon_ticks=(-70.0, -69.0, -68.0, -67.0),
              left=True, bottom=True):
        """North-up limits, lat/lon edge ticks, no km axes."""
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(AXIS)
        ax.set_xticks(*self._edge("x", lon_ticks) if bottom else ([], []))
        ax.set_yticks(*self._edge("y", lat_ticks) if left else ([], []))
        ax.tick_params(length=4, color=AXIS)

    def _edge(self, axis, values):
        n = 400
        if axis == "x":   # bottom edge: longitude
            along = np.linspace(*self.xlim, n)
            lon, _ = _LOC_TO_LL.transform(along * 1e3, np.full(n, self.ylim[0]) * 1e3)
            v = lon
        else:             # left edge: latitude
            along = np.linspace(*self.ylim, n)
            _, lat = _LOC_TO_LL.transform(np.full(n, self.xlim[0]) * 1e3, along * 1e3)
            v = lat
        order = np.argsort(v)
        pos, lab = [], []
        for t in values:
            if v.min() <= t <= v.max():
                pos.append(np.interp(t, v[order], along[order]))
                deg = abs(t)
                txt = f"{int(deg)}°" + (f"{round((deg % 1) * 60):d}′" if deg % 1 else "")
                lab.append(txt + ("W" if axis == "x" else "S"))
        return pos, lab

    def scale_bar(self, ax, km=25, color=INK, fs=15):
        x0 = self.xlim[0] + 0.05 * (self.xlim[1] - self.xlim[0])
        y0 = self.ylim[0] + 0.05 * (self.ylim[1] - self.ylim[0])
        ax.plot([x0, x0 + km], [y0, y0], color=color, lw=4, solid_capstyle="butt", zorder=6)
        ax.text(x0 + km / 2, y0 + 0.02 * (self.ylim[1] - self.ylim[0]), f"{km} km",
                ha="center", va="bottom", fontsize=fs, color=color, zorder=6)

    def north_arrow(self, ax, fs=16, side="right"):
        frac = 0.93 if side == "right" else 0.07
        x = self.xlim[0] + frac * (self.xlim[1] - self.xlim[0])
        y = self.ylim[1] - 0.16 * (self.ylim[1] - self.ylim[0])
        ax.annotate("N", xy=(x, y + 0.11 * (self.ylim[1] - self.ylim[0])), xytext=(x, y),
                    ha="center", va="center", fontsize=fs, fontweight="bold", color=INK,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2), zorder=6)


def node_mesh(r, study):
    """Corner mesh (local km) for the 1 km prediction grid stored in rubble.npz."""
    rows, cols = int(r["rows"]), int(r["cols"])
    X = r["pred_xy"][:, 0].reshape(rows, cols)
    Y = r["pred_xy"][:, 1].reshape(rows, cols)
    return grid_corners(X[0], Y[:, 0], 1000)


def to_grid(r, values_on_nodes):
    rows, cols = int(r["rows"]), int(r["cols"])
    g = np.full(rows * cols, np.nan)
    g[r["nodes"]] = values_on_nodes
    return g.reshape(rows, cols)
