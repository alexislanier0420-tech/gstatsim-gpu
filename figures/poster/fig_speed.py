"""Poster column 4: how much faster. Rubble Glacier on one HiPerGator L4 node in one
run; the large region (Thwaites + Pine Island) from benchmarks/bench_region.py."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from common import COL_W, CPU, GPU, INK, INK2, MUTED, REGION, SPEED, save, style

style()
sp = json.loads(SPEED.read_text(encoding="utf-8"))


def region_rows(res):
    steps, n = {}, None
    for f in REGION[res]:
        d = json.loads(f.read_text(encoding="utf-8"))
        steps |= {k: v["seconds"] / v.get("realizations", 1) for k, v in d["steps"].items()}
        n = d["n_nodes"]
    name = f"Amundsen {res / 1000:g} km" if res >= 1000 else f"Amundsen {res:.0f} m"
    return [dict(label=f"Kriging · {name}", cells=n, gstatsim_s=steps["gstatsim_kriging"],
                 gpu_s=steps["gpu_kriging"]),
            dict(label=f"SGS · {name}", cells=n, gstatsim_s=steps["gstatsim_sgs_x1"],
                 gpu_s=steps.get("gpu_sgs_x1"), gpu_per_realization_32_s=steps["gpu_sgs_x32"])]


rubble = [r for r in sp["workloads"] if "Rubble" in r["label"]]
rows = (rubble + region_rows(1000) + region_rows(500))[::-1]   # first workload on top

fig, a = plt.subplots(figsize=(COL_W, COL_W * 0.64))


def fmt(s):
    return f"{s:.1f} s" if s < 60 else f"{s / 60:.1f} min" if s < 3600 else f"{s / 3600:.1f} h"


# --- (a) GStatSim vs GPU per workload (log time)
for i, r in enumerate(rows):
    cpu, gpu = r["gstatsim_s"], r["gpu_s"]
    many = r.get("gpu_per_realization_32_s")
    fast = many if many else gpu
    a.plot([fast, cpu], [i, i], color="#c3c2b7", lw=3, zorder=1, solid_capstyle="round")
    a.plot(cpu, i, "o", ms=13, color=CPU, zorder=3, markeredgecolor="white", markeredgewidth=2)
    if gpu:
        a.plot(gpu, i, "o", ms=13, color=GPU, zorder=3, markeredgecolor="white", markeredgewidth=2)
    if many:
        a.plot(many, i, "o", ms=13, mfc="white", mec=GPU, mew=2.5, zorder=4)
        a.text(many / 1.35, i, f"{cpu / many:.0f}×", ha="right", va="center", fontsize=15,
               color=GPU, fontweight="bold")
        if gpu:
            a.text(gpu, i + 0.3, f"{cpu / gpu:.0f}× alone", ha="center", va="bottom", fontsize=14, color=INK2)
    else:
        a.text(gpu / 1.35, i, f"{cpu / gpu:.0f}×", ha="right", va="center", fontsize=15, color=GPU,
               fontweight="bold")
    a.text(cpu * 1.35, i, fmt(cpu), ha="left", va="center", fontsize=15, color=INK2)
a.set_xscale("log")
a.set_yticks(range(len(rows)), [f"{r['label']}\n{r['cells']:,} cells" for r in rows], fontsize=16)
a.set_ylim(-0.6, len(rows) - 0.4)
a.set_xlim(0.12, 2.5e5)
a.set_xticks([1, 60, 3600, 36000], ["1 s", "1 min", "1 h", "10 h"])
a.grid(True, axis="x")
a.spines["left"].set_visible(False)
a.tick_params(axis="y", length=0)
a.set_title("Time per map (per realization for SGS)", fontsize=18, color=INK, loc="left")
handles = [plt.Line2D([], [], marker="o", ls="", ms=12, color=CPU, label="GStatSim (CPU)"),
           plt.Line2D([], [], marker="o", ls="", ms=12, color=GPU, label="GPU"),
           plt.Line2D([], [], marker="o", ls="", ms=12, mfc="white", mec=GPU, mew=2.5,
                      label="GPU, 32 realizations together")]
a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.07), ncol=3, fontsize=14,
         handletextpad=0.3, columnspacing=1.0)

fig.text(0.0, -0.06, "Amundsen: Amundsen Sea Embayment (Thwaites + Pine Island), a 650 × 700 km box.\n"
         "GPU: NVIDIA L4 (24 GB) on an AMD EPYC 9655P node. GStatSim: 4 cores of the same node\n"
         "for Rubble, older EPYC 7702 / 75F3 nodes for Amundsen. GPU times exclude start-up.",
         fontsize=12, color=MUTED, va="top")
save(fig, "speed")
