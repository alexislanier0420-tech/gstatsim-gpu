"""Project figure for the GStatSim GPU-backend project (figure milestone).

Panel a: Antarctica locator map with the study area.
Panel b: study area (Lab 1 domain): BedMachine ice mask, grounding line,
          radar bed elevation, lat/lon graticule and place names.
Panel c: the GStatSim trend-interpolation tutorial workflow, with each step
          tagged by where it will run.
Panel d: why kriging parallelizes across grid nodes.
Panel e: why SGS does not (sequential dependency along the random path).
Panel f: why the experimental variogram is a GPU target (pair count ~ N^2).

Map data comes from data/*.npz, made by prep_map_data.py.
Run:  python make_method_figure.py   -> method_figure.{png,pdf,svg}
"""
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Polygon

HERE = Path(__file__).parent
OUT = HERE / "method_figure"
DATA = HERE / "data"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# Okabe-Ito based, colour-blind safe
STATUS = {
    "cpu":  dict(edge="#7a7a7a", fill="#eeeeee", ls="-",  label="Stays on CPU"),
    "gpu":  dict(edge="#0072B2", fill="#d5e7f4", ls="-",
                 label="Existing GPU code (gstatsim_gpu, CuPy): integrate + validate"),
    "cuda": dict(edge="#D55E00", fill="#fbe0cc", ls="-",
                 label="Proposed new CUDA C++ kernel (this project)"),
    "cand": dict(edge="#7a7a7a", fill="#ffffff", ls="--", label="GPU candidate: profile first"),
}
DATA_C = "#1a1a1a"

# BedMachine mask classes: 0 ocean, 1 ice-free land, 2 grounded ice,
# 3 floating ice, 4 Lake Vostok (drawn as grounded ice)
SURFACE = [("Open ocean", "#b9d4ea"), ("Exposed rock", "#9a9a9a"),
           ("Grounded ice", "#f7f7f7"), ("Floating ice shelf", "#dde3ea"),
           (None, "#f7f7f7")]
MASK_CMAP = ListedColormap([c for _, c in SURFACE])
MASK_NORM = BoundaryNorm(np.arange(-0.5, 5), MASK_CMAP.N)
GL_C = "#16324f"          # grounding line
BOX_C = "#000000"         # study-area outlines

TO_XY = pyproj.Transformer.from_crs(4326, 3031, always_xy=True)
TO_LL = pyproj.Transformer.from_crs(3031, 4326, always_xy=True)

# Panel b uses the same polar stereographic projection rotated so that the
# Rubble Glacier meridian is vertical, i.e. north is up on the map.
LON_C = -68.3
LOCAL = (f"+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0={LON_C} "
         "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
PS_TO_LOC = pyproj.Transformer.from_crs(3031, LOCAL, always_xy=True)
LOC_TO_PS = pyproj.Transformer.from_crs(LOCAL, 3031, always_xy=True)
LL_TO_LOC = pyproj.Transformer.from_crs(4326, LOCAL, always_xy=True)
LOC_TO_LL = pyproj.Transformer.from_crs(LOCAL, 4326, always_xy=True)


def loc_km(x_km, y_km):
    """EPSG:3031 km -> panel-b (north-up) km."""
    x, y = PS_TO_LOC.transform(np.asarray(x_km) * 1e3, np.asarray(y_km) * 1e3)
    return x / 1e3, y / 1e3


def ll_loc_km(lat, lon):
    x, y = LL_TO_LOC.transform(lon, lat)
    return np.asarray(x) / 1e3, np.asarray(y) / 1e3


def xy_km(lat, lon):
    x, y = TO_XY.transform(lon, lat)
    return np.asarray(x) / 1e3, np.asarray(y) / 1e3


def box(ax, xc, yc, w, h, title, func, status, num=None):
    s = STATUS[status]
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc=s["fill"], ec=s["edge"], lw=1.0, ls=s["ls"]))
    head = f"{num}  {title}" if num else title
    ax.text(xc, yc + 1.2, head, ha="center", va="center", fontsize=6.6,
            fontweight="bold", linespacing=1.05)
    if func:
        ax.text(xc, yc - h / 2 + 1.5, func, ha="center", va="center",
                fontsize=5.4, family="DejaVu Sans Mono", color="#333333")


def arrow(ax, p0, p1, **kw):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333",
                                shrinkA=0, shrinkB=0, mutation_scale=7, **kw))


def panel_label(ax, s, x=-0.12, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom")


def draw_mask(ax, x, y, mask):
    """Mask image plus ice-front and grounding-line contours; x, y in km."""
    if y[0] < y[-1]:
        y, mask = y[::-1], mask[::-1]
    dx = abs(x[1] - x[0]) / 2
    ax.imshow(mask, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest",
              extent=[x[0] - dx, x[-1] + dx, y[-1] - dx, y[0] + dx],
              origin="upper", zorder=0)
    ax.contour(x, y, (mask > 0).astype(float), [0.5], colors="#6b7b8c",
               linewidths=0.4, zorder=1)
    ax.contour(x, y, np.isin(mask, [1, 2, 4]).astype(float), [0.5],
               colors=GL_C, linewidths=0.7, zorder=2)


def halo(lw=2.0):
    return [pe.withStroke(linewidth=lw, foreground="white")]


def edge_ticks(edge_xy, values, coord, to_ll=TO_LL):
    """Positions along a straight map edge where lat or lon == each value."""
    lon, lat = to_ll.transform(edge_xy[0] * 1e3, edge_xy[1] * 1e3)
    along = edge_xy[0] if np.ptp(edge_xy[0]) > 0 else edge_xy[1]
    v = lat if coord == "lat" else lon
    order = np.argsort(v)
    pos, labels = [], []
    for target in values:
        if v.min() <= target <= v.max():
            pos.append(np.interp(target, v[order], along[order]))
            deg = abs(target)
            txt = f"{int(deg)}°" + (f"{round((deg % 1) * 60):d}′" if deg % 1 else "")
            labels.append(f"{txt}{'S' if coord == 'lat' else 'W'}")
    return pos, labels


fig = plt.figure(figsize=(7.2, 9.2))
outer = GridSpec(3, 1, figure=fig, height_ratios=[1.75, 1.2, 1], hspace=0.24,
                 left=0.03, right=0.985, top=0.925, bottom=0.07)

fig.suptitle("Where the GPU fits in GStatSim: accelerating Antarctic "
             "subglacial-topography reconstruction",
             fontsize=9.5, fontweight="bold", x=0.03, ha="left", y=0.988)
fig.text(0.03, 0.962,
         "Study area: Rubble Glacier and George VI Sound, Antarctic Peninsula; "
         "workflow from the GStatSim tutorial 'Interpolation with a trend'",
         fontsize=7, color="#444444", ha="left")

# ======================================================= row 1: maps (a, b)
maps = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1, 1.12],
                               wspace=0.2)
study = np.load(DATA / "study_area.npz")
sx, sy = study["x"] / 1e3, study["y"] / 1e3
dom = (sx.min() - 0.25, sx.max() + 0.25, sy.min() - 0.25, sy.max() + 0.25)

# Panel b view: largest north-up square that fits inside the (rotated) Lab 1 box
cx, cy = loc_km((dom[0] + dom[1]) / 2, (dom[2] + dom[3]) / 2)
ex, ey = loc_km(dom[1], (dom[2] + dom[3]) / 2)
theta = np.arctan2(ey - cy, ex - cx)
half = 0.97 * (dom[1] - dom[0]) / 2 / (abs(np.cos(theta)) + abs(np.sin(theta)))
view = (cx - half, cx + half, cy - half, cy + half)
corners = np.array([[view[0], view[2]], [view[1], view[2]],
                    [view[1], view[3]], [view[0], view[3]]])
view_ps = np.column_stack(LOC_TO_PS.transform(corners[:, 0] * 1e3,
                                              corners[:, 1] * 1e3)) / 1e3

# ---------------------------------------------------------------- panel a
ax = fig.add_subplot(maps[0])
ax.set_anchor("N")
ant = np.load(DATA / "antarctica_mask_5km.npz")
draw_mask(ax, ant["x"] / 1e3, ant["y"] / 1e3, ant["mask"])
for lat in (-60, -70, -80):
    lon = np.linspace(-180, 180, 361)
    ax.plot(*xy_km(np.full_like(lon, lat), lon), lw=0.35, color="#8aa0b4",
            ls=(0, (3, 2)), zorder=1)
for lon in range(-180, 180, 30):
    lat = np.linspace(-60, -88, 50)
    ax.plot(*xy_km(lat, np.full_like(lat, lon)), lw=0.35, color="#8aa0b4",
            ls=(0, (3, 2)), zorder=1)
ax.add_patch(Polygon(view_ps, closed=True, fill=False, ec=BOX_C, lw=1.4, zorder=5))
ax.annotate("Study area\n(panel b)", tuple(view_ps[0]),
            xytext=(-2500, -1500), fontsize=6.4, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=BOX_C), zorder=6)
for text, lat, lon, kw in [
        ("West\nAntarctica", -80, -100, {}),
        ("Antarctic\nPeninsula", -66, -45, dict(fontsize=6)),
        ("East\nAntarctica", -78, 80, {}),
        ("Ross\nIce Shelf", -81.5, -175, dict(style="italic", color=GL_C)),
        ("70°S", -70, 135, dict(color="#5d7387", fontsize=5.8)),
        ("80°S", -80, 135, dict(color="#5d7387", fontsize=5.8))]:
    ax.text(*xy_km(lat, lon), text, ha="center", va="center",
            **{"fontsize": 6, **kw}, path_effects=halo(), zorder=6)
ax.set_xlim(-3100, 3100)
ax.set_ylim(-2700, 3000)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor(SURFACE[0][1])
panel_label(ax, "a", x=0.0)
ax.set_title("Antarctica (polar stereographic)", fontsize=7.2, pad=3)

ax.legend(handles=[Patch(fc=c, ec="#888888", lw=0.4, label=n)
                   for n, c in SURFACE[:4]]
          + [Line2D([], [], color=GL_C, lw=0.9, label="Grounding line"),
             Line2D([], [], color="#6b7b8c", lw=0.6, label="Ice front / coast"),
             Line2D([], [], marker="o", ls="", ms=2.5, mfc="#5aae9f", mec="none",
                    label="Radar bed cell (500 m), colored\nby elevation (color bar, b)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
          frameon=False, fontsize=6.2, handlelength=1.6, columnspacing=0.8,
          handletextpad=0.4, labelspacing=0.35, title="BedMachine v3 mask · Bedmap3 radar",
          title_fontsize=6.4)

# ---------------------------------------------------------------- panel b
ax = fig.add_subplot(maps[1])
ax.set_anchor("N")
# BedMachine mask, drawn cell by cell in the north-up projection
mask = study["mask"]
dx = abs(sx[1] - sx[0]) / 2
xe = np.append(sx - np.sign(sx[1] - sx[0]) * dx, sx[-1] + np.sign(sx[1] - sx[0]) * dx)
ye = np.append(sy - np.sign(sy[1] - sy[0]) * dx, sy[-1] + np.sign(sy[1] - sy[0]) * dx)
XE, YE = loc_km(*np.meshgrid(xe, ye))
ax.pcolormesh(XE, YE, mask, cmap=MASK_CMAP, norm=MASK_NORM, rasterized=True, zorder=0)
XC, YC = loc_km(*np.meshgrid(sx, sy))
ax.contour(XC, YC, (mask > 0).astype(float), [0.5], colors="#6b7b8c",
           linewidths=0.5, zorder=1)
ax.contour(XC, YC, np.isin(mask, [1, 2, 4]).astype(float), [0.5],
           colors=GL_C, linewidths=0.9, zorder=2)
sc = ax.scatter(*loc_km(study["radar_x"] / 1e3, study["radar_y"] / 1e3), s=1.6,
                c=study["radar_bed"], cmap="BrBG_r", lw=0,
                norm=TwoSlopeNorm(vmin=-1500, vcenter=0, vmax=1500),
                rasterized=True, zorder=3)
lats = list(np.arange(-70.0, -73.01, -0.25))
lons = list(np.arange(-72.0, -64.9, 1.0))
for lat in lats:
    lon = np.linspace(-76, -61, 300)
    ax.plot(*ll_loc_km(np.full_like(lon, lat), lon), lw=0.4, color="#7f8c99",
            ls=(0, (3, 2)), zorder=4)
for lon in lons:
    lat = np.linspace(-69.5, -73.5, 200)
    ax.plot(*ll_loc_km(lat, np.full_like(lat, lon)), lw=0.4, color="#7f8c99",
            ls=(0, (3, 2)), zorder=4)

ax.set_xlim(view[0], view[1])
ax.set_ylim(view[2], view[3])
ax.set_aspect("equal")
edge = np.linspace(0, 1, 400)
bottom = (view[0] + edge * (view[1] - view[0]), np.full_like(edge, view[2]))
left = (np.full_like(edge, view[0]), view[2] + edge * (view[3] - view[2]))
ax.set_xticks(*edge_ticks(bottom, lons, "lon", LOC_TO_LL))
ax.set_yticks(*edge_ticks(left, lats, "lat", LOC_TO_LL))
ax.tick_params(labelsize=6.4, length=2.5)
ax.set_xlabel("Longitude", fontsize=6.6, labelpad=2)
ax.set_ylabel("Latitude", fontsize=6.6, labelpad=2)

# Rubble Glacier at its published coordinates (71°20'S, 68°18'W); region
# labels sit on the matching BedMachine mask class (checked against the mask).
rx, ry = ll_loc_km(-71.333, -68.30)
ax.scatter(rx, ry, marker="o", s=34, facecolor="none", ec="black", lw=1.2, zorder=7)
ax.annotate("Rubble Glacier", (rx, ry), xytext=(rx - 34, ry + 22), fontsize=7,
            fontweight="bold", ha="center", va="center", path_effects=halo(), zorder=7,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="black", shrinkB=4))


def label_at(text, lat, lon, **kw):
    ax.text(*ll_loc_km(lat, lon), text, ha="center", va="center",
            path_effects=halo(), zorder=7, **{"fontsize": 6.6, **kw})


label_at("George VI\nIce Shelf", -71.05, -67.85, style="italic", color=GL_C)
label_at("ALEXANDER\nISLAND", -71.55, -69.4, color="#444444", fontsize=6.8)
label_at("PALMER\nLAND", -71.55, -66.95, color="#444444", fontsize=6.8)

span = view[1] - view[0]
# 25 km scale bar, bottom left
sb_x, sb_y = view[0] + 0.05 * span, view[2] + 0.05 * span
ax.plot([sb_x, sb_x + 25], [sb_y, sb_y], color="black", lw=1.8,
        solid_capstyle="butt", zorder=7, path_effects=halo(3))
ax.text(sb_x + 12.5, sb_y + 0.02 * span, "25 km", ha="center", va="bottom",
        fontsize=6.4, path_effects=halo(), zorder=7)
# north arrow along the local meridian (points straight up here)
na = np.array([view[1] - 0.07 * span, view[3] - 0.17 * span])
lon0, lat0 = LOC_TO_LL.transform(*(na * 1e3))
tip = np.array(ll_loc_km(lat0 + 0.05, lon0))
d = (tip - na) / np.hypot(*(tip - na)) * 0.08 * span
ax.annotate("", xy=na + d, xytext=na, zorder=7,
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black", mutation_scale=9))
ax.text(*(na + d * 1.3), "N", ha="center", va="bottom", fontsize=7,
        fontweight="bold", path_effects=halo(), zorder=7)
panel_label(ax, "b", x=-0.2, y=1.02)
ax.set_title("Rubble Glacier: radar-derived bed elevation", fontsize=7.2, pad=3)
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.11,
                  aspect=30, extend="both")
cb.set_label("Radar bed elevation (m above geoid ≈ sea level)", fontsize=6.6)
cb.ax.tick_params(labelsize=6.2)

# =============================================================== panel c
ax = fig.add_subplot(outer[1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")
panel_label(ax, "c", x=0.0, y=0.98)

W, H = 16, 9.5
R1, RK, RS = 33, 19.5, 7.5          # row-1, kriging, SGS centres
box(ax, 10, R1, W, H, "Radar bed data\n(sparse x, y, z)", "input data", "cpu")
box(ax, 29, R1, W, H, "Grid radar\ndata", "gs.Gridding.grid_data", "cpu", 1)
box(ax, 48, R1, W, H, "Fit RBF\ntrend", "gs.rbf_trend", "cand", 2)
box(ax, 67, R1, W, H, "Normal-score\nresiduals", "QuantileTransformer", "cand", 3)
box(ax, 86, R1, W, H, "Experimental\nvariogram", "O(N²) point pairs", "cuda", 4)

box(ax, 29, RK, W, H, "Simple kriging\n(best estimate)", "gs.Interpolation.skrige", "gpu", 5)
box(ax, 29, RS, W, H, "SGS (ensemble\nof realizations)", "...skrige_sgs", "gpu", 6)
box(ax, 52, 13.5, W, H, "Back-transform\n+ add trend", "elementwise array ops", "cand", 7)
box(ax, 71, 13.5, W, H, "Validate:\nvariogram check", "realizations vs data", "cpu", 8)
box(ax, 90, 13.5, W, H, "Bed topography\nmaps (m)", "output", "cpu")

for x0 in (18, 37, 56, 75):
    arrow(ax, (x0, R1), (x0 + 3, R1))
# variogram model feeds both estimators: drop below row 1, run left, split
yb = R1 - H / 2 - 1.6
ax.plot([86, 86, 16, 16], [R1 - H / 2, yb, yb, RS], lw=0.8, color="#333333",
        solid_joinstyle="miter")
ax.text(62, yb - 0.7, "variogram model + trend-removed data", fontsize=5.6,
        ha="center", va="top", color="#333333", style="italic")
arrow(ax, (16, RK), (21, RK))
arrow(ax, (16, RS), (21, RS))
ax.plot([37, 40.5, 40.5, 37], [RK, RK, RS, RS], lw=0.8, color="#333333")
arrow(ax, (40.5, 13.5), (44, 13.5))
arrow(ax, (60, 13.5), (63, 13.5))
arrow(ax, (79, 13.5), (82, 13.5))

ax.legend(handles=[Patch(fc=s["fill"], ec=s["edge"], ls=s["ls"], lw=1.0,
                         label=s["label"]) for s in STATUS.values()],
          loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2,
          frameon=False, fontsize=6.2, handlelength=2.2, columnspacing=1.6)

# ======================================================= row 3: d, e, f
mini = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.38)
gx, gy = np.meshgrid(np.arange(7), np.arange(7))
data_xy = np.array([[0.4, 5.6], [1.6, 3.3], [2.8, 0.7], [3.2, 4.8],
                    [4.6, 2.4], [5.7, 5.5], [6.3, 1.2], [0.9, 1.0]])


def base(ax):
    ax.scatter(gx, gy, s=7, c="#c9c9c9", lw=0, zorder=1)
    ax.scatter(*data_xy.T, marker="^", s=22, c=DATA_C, zorder=4)
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(-0.6, 6.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999999")


def nearest(pt, pts, k):
    return pts[np.argsort(np.hypot(*(pts - pt).T))[:k]]


# ---------------------------------------------------------------- panel d
ax = fig.add_subplot(mini[0])
base(ax)
panel_label(ax, "d")
ax.set_title("Kriging: nodes independent", fontsize=7.2, pad=3)
ax.scatter(gx, gy, s=9, c=STATUS["gpu"]["edge"], lw=0, zorder=3)
for t in ([1, 4], [4, 2], [5, 5]):
    t = np.array(t)
    for d in nearest(t, data_xy, 3):
        ax.plot([t[0], d[0]], [t[1], d[1]], lw=0.7, color=STATUS["gpu"]["edge"],
                zorder=2)
    ax.scatter(*t, s=34, c=STATUS["gpu"]["edge"], ec="white", lw=0.6, zorder=5)
ax.set_xlabel("All nodes solved at once: one batch of k×k\n"
              "systems (k = nearest radar points, here 3)",
              fontsize=6.2, labelpad=3)

# ---------------------------------------------------------------- panel e
ax = fig.add_subplot(mini[1])
base(ax)
panel_label(ax, "e")
ax.set_title("SGS: nodes depend on earlier ones", fontsize=7.2, pad=3)
path = np.array([[1, 5], [3, 2], [5, 4], [2, 3], [4, 5], [3, 4]])
cols = plt.cm.Oranges(np.linspace(0.35, 0.95, len(path)))
cur = path[-1]
for p in list(path[:-1]) + list(nearest(cur, data_xy, 2)):
    ax.plot([cur[0], p[0]], [cur[1], p[1]], lw=0.7, color=STATUS["cuda"]["edge"],
            zorder=2)
for i, (p, c) in enumerate(zip(path, cols, strict=False), 1):
    ax.scatter(*p, s=70, color=c, ec="white", lw=0.6, zorder=5)
    ax.text(*p, str(i), ha="center", va="center", fontsize=5.6,
            fontweight="bold", color="white" if i > 2 else "black", zorder=6)
ax.set_xlabel("Numbers = random-path order. Node 6 uses nodes\n"
              "1–5 → parallelize within a step, not along the path",
              fontsize=6.2, labelpad=3)

fig.legend(handles=[
    Line2D([], [], marker="^", ls="", color=DATA_C, ms=4.5, label="radar measurement"),
    Line2D([], [], marker="o", ls="", color="#c9c9c9", ms=3.5, label="grid node (unsolved)"),
    Line2D([], [], marker="o", ls="", color=STATUS["gpu"]["edge"], ms=3.5,
           label="node solved in the same GPU batch (d)"),
    Line2D([], [], marker="o", ls="", color=cols[3], ms=5,
           label="simulated node, path order (e)"),
    Line2D([], [], ls="-", lw=0.8, color="#555555",
           label="neighbors used in the estimate"),
], loc="lower left", bbox_to_anchor=(0.04, 0.0), ncol=3, frameon=False,
    fontsize=6, handletextpad=0.3, columnspacing=1.0)

# ---------------------------------------------------------------- panel f
ax = fig.add_subplot(mini[2])
panel_label(ax, "f", x=-0.34)
ax.set_title("Variogram cost grows as N²", fontsize=7.2, pad=3)
n = np.logspace(3, 6, 200)
ax.loglog(n, n * (n - 1) / 2, color=STATUS["cuda"]["edge"], lw=1.4)
for nn in (1e3, 1e4, 1e5, 1e6):
    pairs = nn * (nn - 1) / 2
    ax.scatter(nn, pairs, s=14, color=STATUS["cuda"]["edge"], zorder=3)
    mant, exp = f"{pairs:.0e}".split("e")
    ax.annotate(f"${mant}\\times10^{{{int(exp)}}}$",
                (nn, pairs), xytext=(6, -1) if nn == 1e3 else (-5, 4),
                textcoords="offset points", fontsize=5.6,
                ha="left" if nn == 1e3 else "right",
                va="top" if nn == 1e3 else "bottom")
# radar cells visible in panel b
rbx, rby = loc_km(study["radar_x"] / 1e3, study["radar_y"] / 1e3)
n_study = int(((rbx >= view[0]) & (rbx <= view[1]) &
               (rby >= view[2]) & (rby <= view[3])).sum())
p_study = n_study * (n_study - 1) / 2
ax.scatter(n_study, p_study, marker="*", s=70, c="black", ec="white", lw=0.5, zorder=4)
ax.annotate(f"panel b: N = {n_study:,}\nradar cells → {p_study / 1e6:,.0f}\nmillion pairs",
            (n_study, p_study), xytext=(6, -8), textcoords="offset points",
            fontsize=5.8, ha="left", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
ax.set_xlabel("Radar data points, N", fontsize=6.6)
ax.set_ylabel("Point pairs to bin, N(N−1)/2", fontsize=6.6)
ax.tick_params(labelsize=6, which="both")
ax.grid(True, which="major", lw=0.4, color="#dddddd")
ax.set_xlim(2e2, 3e6)
ax.set_ylim(1e5, 3e12)
ax.text(0.04, 0.96, "every pair is independent\n→ one GPU thread per pair",
        transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
        color="#333333")

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=600)
print("wrote", OUT.with_suffix(".png"))
