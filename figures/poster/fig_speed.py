"""Poster column 4: how much faster, measured on one laptop in one run."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from common import COL_W, CPU, GPU, INK, INK2, MUTED, RESULTS, save, style

style()
sp = json.loads((RESULTS / "speed.json").read_text())
rows = sp["workloads"][::-1]   # first workload on top

fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, COL_W * 1.02),
                           gridspec_kw=dict(height_ratios=[1.1, 1], hspace=0.55))


def fmt(s):
    return f"{s:.1f} s" if s < 60 else f"{s / 60:.1f} min"


# --- (a) GStatSim vs GPU per workload (log time)
for i, r in enumerate(rows):
    cpu, gpu = r["gstatsim_s"], r["gpu_s"]
    many = r.get("gpu_per_realization_32_s")
    fast = many if many else gpu
    a.plot([fast, cpu], [i, i], color="#c3c2b7", lw=3, zorder=1, solid_capstyle="round")
    a.plot(cpu, i, "o", ms=13, color=CPU, zorder=3, markeredgecolor="white", markeredgewidth=2)
    a.plot(gpu, i, "o", ms=13, color=GPU, zorder=3, markeredgecolor="white", markeredgewidth=2)
    if many:
        a.plot(many, i, "o", ms=13, mfc="white", mec=GPU, mew=2.5, zorder=4)
        a.text(many / 1.35, i, f"{cpu / many:.0f}×", ha="right", va="center", fontsize=15,
               color=GPU, fontweight="bold")
        a.text(gpu, i + 0.33, f"{cpu / gpu:.0f}× alone", ha="center", va="bottom", fontsize=14, color=INK2)
    else:
        a.text(gpu / 1.35, i, f"{cpu / gpu:.0f}×", ha="right", va="center", fontsize=15, color=GPU,
               fontweight="bold")
    a.text(cpu * 1.35, i, fmt(cpu), ha="left", va="center", fontsize=15, color=INK2)
a.set_xscale("log")
a.set_yticks(range(len(rows)), [f"{r['label']}\n{r['cells']:,} cells" for r in rows], fontsize=16)
a.set_ylim(-0.6, len(rows) - 0.4)
a.set_xlim(0.08, 2000)
a.set_xticks([0.1, 1, 10, 100, 1000], ["0.1 s", "1 s", "10 s", "100 s", "1000 s"])
a.grid(True, axis="x")
a.spines["left"].set_visible(False)
a.tick_params(axis="y", length=0)
a.set_title("Time per run (per realization for SGS)", fontsize=18, color=INK, loc="left")
handles = [plt.Line2D([], [], marker="o", ls="", ms=12, color=CPU, label="GStatSim (CPU)"),
           plt.Line2D([], [], marker="o", ls="", ms=12, color=GPU, label="GPU"),
           plt.Line2D([], [], marker="o", ls="", ms=12, mfc="white", mec=GPU, mew=2.5,
                      label="GPU, 32 realizations together")]
a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.12), ncol=3, fontsize=14,
         handletextpad=0.3, columnspacing=1.0)

# --- (b) SGS: time per realization vs. how many run side by side
sc = {int(k): v for k, v in sp["scaling"].items()}
S = np.array(sorted(sc))
t = np.array([sc[s] for s in S])
gst = next(r["gstatsim_s"] for r in sp["workloads"] if r["label"].startswith("SGS · Rubble"))
b.axhline(gst, color=CPU, lw=2.5)
b.text(S[-1], gst / 1.25, f"GStatSim: {fmt(gst)} each", color=INK, ha="right", va="top", fontsize=16)
b.plot(S, t, "-o", color=GPU, lw=3, ms=9, markeredgecolor="white", markeredgewidth=1.5)
b.text(S[-1] * 1.08, t[-1], f"{fmt(t[-1])} each\n({gst / t[-1]:.0f}× faster)", color=INK, va="center",
       fontsize=16)
b.set_xscale("log", base=2)
b.set_yscale("log")
b.set_xticks(S, [str(s) for s in S])
b.set_yticks([1, 10, 100], ["1 s", "10 s", "100 s"])
b.set_ylim(0.45, gst * 1.6)
b.set_xlim(S[0] / 1.3, S[-1] * 4.5)
b.grid(True, axis="y", which="major")
b.set_xlabel("Realizations simulated side by side")
b.set_ylabel("Seconds per\nrealization")
b.set_title("Rubble Glacier SGS: more realizations, less time each", fontsize=18, color=INK, loc="left")
fig.text(0.0, -0.035, "All times on one laptop (Intel CPU, NVIDIA RTX 5060 8 GB). GPU times exclude "
         "one-time start-up (~3 s).", fontsize=12, color=MUTED)
save(fig, "speed")
