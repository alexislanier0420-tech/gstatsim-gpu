"""Make GPU versions of GStatSim's demo notebooks (demos/*.ipynb).

Each demo with a kriging or simulation step gets a copy, <name>_GPU.ipynb, in
the same folder. Only the slow calls change: gs.Interpolation.<func>(...)
becomes the matching gstatsim_gpu call with backend=BACKEND, timed with
%%time. Text, data, seeds and plots stay as the demo has them, and a shared
random Generator (seed=rng) is passed through, so on the GPU the demo
produces GStatSim's realizations (up to tied neighbors).

    python scripts/make_gstatsim_gpu_demos.py path/to/GStatSim/demos
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import nbformat as nbf

DEMOS = [
    "3_Simple_kriging_and_ordinary_kriging.ipynb",
    "4_Sequential_Gaussian_Simulation.ipynb",
    "5_Variogram_interpolation_comparison.ipynb",
    "6_interpolation_with_anisotropy.ipynb",
    "7_non-stationary_SGS_example1.ipynb",
    "8_non-stationary_SGS_example2.ipynb",
    "9_interpolation_with_a_trend.ipynb",
    "10_cokriging_and_cosimulation_MM1.ipynb",
]

SEED = r"(?:, seed=(\w+))?"
CALL = re.compile(r"gs\.Interpolation\.(skrige|okrige|skrige_sgs|okrige_sgs)\("
                  r"(\w+), (\w+), '(\w+)', '(\w+)', '(\w+)', (\w+), (\w+), (\w+)" + SEED + r"\)")
CLUSTER_CALL = re.compile(r"gs\.Interpolation\.cluster_sgs\((\w+), (\w+), '(\w+)', '(\w+)', '(\w+)', "
                          r"'(\w+)', (\w+), (\w+), (\w+)" + SEED + r"\)")
MM1_CALL = re.compile(r"gs\.Interpolation\.(cokrige_mm1|cosim_mm1)\((\w+), (\w+), '(\w+)', '(\w+)', '(\w+)',"
                      r"\s*(\w+), '(\w+)', '(\w+)', '(\w+)', (\w+), (\w+), (\w+), (\w+)" + SEED + r"\)")

HEADER = """\
> ⚡ **GPU version** of [{src}]({src}). Only the kriging and simulation calls change: they run
> through `gstatsim_gpu` with `backend=BACKEND` and are timed with `%%time`. Text, data, seeds and
> plots are the original demo's, so the results match it (to rounding).
>
> Needs an NVIDIA GPU and `pip install gstatsim[gpu]`. Without a GPU, `BACKEND` is set to `"cpu"`,
> which runs GStatSim itself."""

SETUP = """\
# GPU version: use the GPU if there is one; otherwise "cpu" runs GStatSim itself
import gstatsim_gpu as gsg

BACKEND = "gpu" if gsg.gpu_status().available else "cpu"
print("Backend:", BACKEND, "|", gsg.gpu_status().device_name or gsg.gpu_status().reason)"""

# Calls that stay on the CPU: (demo, variogram, "predict" or "sgs") -> comment
# added above the cell.
DEMO5 = "5_Variogram_interpolation_comparison.ipynb"
CPU_ONLY = {
    (DEMO5, "vario4", "predict"): "vario4 is a Matern model, which has no GPU version yet, so it runs on the CPU",
    (DEMO5, "vario4", "sgs"): "vario4 is a Matern model, which has no GPU version yet, so it runs on the CPU",
    (DEMO5, "vario1", "sgs"): ("vario1 is Gaussian with no nugget, so on this 1-unit grid the kriging systems are "
                               "numerically singular. GStatSim solves them by least squares, which GPU SGS can't "
                               "do mid-simulation, so this one runs on the CPU"),
}


def backend_arg(demo: str, vario: str, kind: str) -> str:
    return 'backend="cpu"' if (demo, vario, kind) in CPU_ONLY else "backend=BACKEND"


def converter(demo: str):
    def call(m: re.Match) -> str:
        func, grid, df, xx, yy, zz, k, vario, rad, seed = m.groups()
        method = "ordinary" if func.startswith("o") else "simple"
        name = "sgs" if func.endswith("_sgs") else "predict"
        seed_arg = f", seed={seed}" if seed else ""
        return (f"gsg.{name}({df}, {grid}, {vario}, num_points={k}, radius={rad}, method='{method}', "
                f"xx='{xx}', yy='{yy}', zz='{zz}'{seed_arg}, {backend_arg(demo, vario, name)})")

    def cluster(m: re.Match) -> str:
        grid, df, xx, yy, zz, kk, k, gamma, rad, seed = m.groups()
        seed_arg = f", seed={seed}" if seed else ""
        return (f"gsg.cluster_sgs({df}, {grid}, {gamma}, num_points={k}, radius={rad}, cluster_col='{kk}', "
                f"xx='{xx}', yy='{yy}', zz='{zz}'{seed_arg}, backend=BACKEND)")

    def mm1(m: re.Match) -> str:
        func, grid, df1, xx, yy, zz, df2, xx2, yy2, zz2, k, vario, rad, rho, seed = m.groups()
        seed_arg = f", seed={seed}" if seed else ""
        return (f"gsg.{func}({df1}, {df2}, {grid}, {vario}, num_points={k}, radius={rad}, corrcoef={rho},\n"
                f"    xx='{xx}', yy='{yy}', zz='{zz}', xx2='{xx2}', yy2='{yy2}', zz2='{zz2}'{seed_arg}, "
                "backend=BACKEND)")

    def convert_code(src: str) -> str:
        new = CALL.sub(call, src)
        new = CLUSTER_CALL.sub(cluster, new)
        new = MM1_CALL.sub(mm1, new)
        notes = [note for (d, v, kind), note in CPU_ONLY.items()
                 if d == demo and re.search(rf"gsg\.{kind}\(\w+, \w+, {v}, ", new)]
        if notes:
            new = "".join(f"# GPU version: {n}\n" for n in notes) + new
        # Demo 4 simulates two more realizations one after the other; on the GPU
        # they run side by side in one call (same Generator, same results).
        two = re.search(r"^sim2 = gsg\.sgs\((.*), seed=(\w+), backend=BACKEND\) *$", new, flags=re.M)
        if two and re.search(r"^sim3 = gsg\.sgs\(", new, flags=re.M):
            args, rng = two.groups()
            new = re.sub(r"^# simulation [23]\n", "", new, flags=re.M)
            new = re.sub(r"^sim[23] = gsg\.sgs\(.*\n", "", new, flags=re.M)
            new = new.replace(
                "# simulate multiple random realizations\n",
                "# simulate multiple random realizations\n"
                "# GPU version: both in one call, simulated side by side (same as two calls in a row)\n"
                f"sim2, sim3 = gsg.realizations({args}, seeds=[{rng}, {rng}], backend=BACKEND)\n", 1)
        return new

    return convert_code


def convert(src: Path, dst: Path) -> int:
    nb = nbf.read(src, as_version=4)
    convert_code = converter(src.name)
    out = copy.deepcopy(nb)
    out.cells = [nbf.v4.new_markdown_cell(HEADER.format(src=src.name)), nbf.v4.new_code_cell(SETUP)]
    changed = 0
    for cell in nb.cells:
        cell = copy.deepcopy(cell)
        if cell.cell_type == "code":
            new = convert_code(cell.source)
            if new != cell.source:
                changed += 1
                cell.source = "%%time\n" + new
            cell.outputs, cell.execution_count = [], None
            if "gs.Interpolation." in new and any(f in new for f in ("krige", "sgs", "cosim")):
                raise ValueError(f"{src.name}: a GStatSim call wasn't converted:\n{new}")
        out.cells.append(cell)
    nbf.write(out, dst)
    return changed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("demos", type=Path, help="GStatSim's demos folder")
    a = p.parse_args(argv)
    for name in DEMOS:
        dst = a.demos / name.replace(".ipynb", "_GPU.ipynb")
        n = convert(a.demos / name, dst)
        print(f"wrote {dst.name} ({n} cells changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
