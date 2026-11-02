"""Poster column 4: is the GPU right? Kriging cell by cell, SGS as ensembles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import COL_W, CPU, GPU, INK, ROOT, save, style

style()
kr = ROOT / "results" / "kriging"
cpu = np.load(next(kr.glob("*_tut3_simple_cpu_n22500.npz")))
gpu = np.load(next(kr.glob("*_tut3_simple_gpu_n22500.npz")))
ens = np.load(ROOT / "results" / "validation" / "sgs_ensembles.npz")

fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, COL_W * 0.52), gridspec_kw=dict(wspace=0.45))

# --- (a) kriging: how different is each cell's answer?
diff = np.abs(cpu["est"] - gpu["est"])
same = diff <= 1e-6
logd = np.log10(np.maximum(diff, 1e-17))
bins = np.arange(-17, 0.01, 0.5)
a.hist(logd[same], bins=bins, color=GPU, label="Identical", rwidth=0.85)
a.hist(logd[~same], bins=bins, color=CPU, label="Tied neighbors", rwidth=0.85)
# Exactly-equal cells can't sit on a log axis: they're binned at the far left and labeled 0
a.set_xticks([-16.75, -12, -8, -4, 0], ["0", "1e-12", "1e-8", "1e-4", "1"])
a.set_ylim(0, a.get_ylim()[1] * 1.4)
a.set_xlabel("|GPU − GStatSim|")
a.set_ylabel("Cells")
a.set_title("Kriging, 22,500 cells", fontsize=17, color=INK, loc="left")
a.text(0.03, 0.96, f"{same.mean():.0%} identical:\nrounding only", transform=a.transAxes, va="top",
       fontsize=14, color=GPU, fontweight="bold")
a.text(0.98, 0.62, f"{(~same).mean():.0%}\ntied\npoints", transform=a.transAxes, ha="right",
       va="top", fontsize=14, color=CPU, fontweight="bold")


# --- (b) SGS: ensembles agree (variograms of realizations)
def semivariogram(field, lags):
    out = []
    for h in lags:
        dx = (field[:, h:] - field[:, :-h]) ** 2
        dy = (field[h:, :] - field[:-h, :]) ** 2
        out.append(0.5 * np.concatenate([dx.ravel(), dy.ravel()]).mean())
    return np.array(out)


n = int(np.sqrt(ens["points"].shape[0]))
lags = np.arange(1, 13)
b.grid(True, axis="y")
for key, color, label in ((("A_gstatsim", "B_gstatsim"), CPU, "GStatSim, 40"), (("C_gpu",), GPU, "GPU, 20")):
    real = np.concatenate([ens[k] for k in key])
    g = np.stack([semivariogram(r.reshape(n, n), lags) for r in real])
    lo, hi = np.percentile(g, [10, 90], axis=0)
    b.fill_between(lags, lo, hi, color=color, alpha=0.18, linewidth=0)
    b.plot(lags, g.mean(axis=0), color=color, lw=3, label=label)
b.set_xlabel("Distance (km)")
b.set_ylabel("Semivariance")
b.set_title("SGS realizations", fontsize=17, color=INK, loc="left")
b.set_ylim(0, None)
b.legend(loc="lower right", fontsize=14, handlelength=1.2)
save(fig, "validation")
