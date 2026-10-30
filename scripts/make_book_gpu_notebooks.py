"""Make GPU versions of the GStatSim Jupyter Book chapters.

Copies each supported chapter from a checkout of
https://github.com/GatorGlaciology/gstatsimbook and changes only the slow steps:
GStatSim's kriging and SGS calls become gstatsim_accel calls with a CPU/GPU
switch, timed with %%time next to the CPU time recorded in the published book.
All text, data and other code stay as the book has them.

    git clone https://github.com/GatorGlaciology/gstatsimbook
    python scripts/make_book_gpu_notebooks.py path/to/gstatsimbook
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import nbformat as nbf

CHAPTERS = {
    "3_Simple_kriging_and_ordinary_kriging.ipynb": "Simple and ordinary kriging",
    "4_Sequential_Gaussian_Simulation.ipynb": "Sequential Gaussian simulation",
    "5_interpolation_with_anisotropy.ipynb": "Interpolation with anisotropy",
    "6_non-stationary_SGS_example1.ipynb": "Non-stationary SGS, example 1",
    "7_non-stationary_SGS_example2.ipynb": "Non-stationary SGS, example 2",
    "8_interpolation_with_a_trend.ipynb": "Interpolation with a trend",
}
BOOK_URL = "https://gatorglaciology.github.io/gstatsimbook/"
BOOK_COMMIT = "9365b2d4ef645d6f6bc1362dbe1221b52b714e4b"  # gstatsimbook commit these notebooks were made from
KERNEL = {"name": "gstatsim-gpu", "display_name": "Python (gstatsim-gpu)", "language": "python"}

CALL = re.compile(r"gs\.Interpolation\.(skrige|okrige|skrige_sgs|okrige_sgs)\("
                  r"(\w+), (\w+), '(\w+)', '(\w+)', '(\w+)', (\w+), (\w+), (\w+)\)")
CLUSTER_CALL = re.compile(r"gs\.Interpolation\.cluster_sgs\((\w+), (\w+), '(\w+)', '(\w+)', '(\w+)', "
                          r"'(\w+)', (\w+), (\w+), (\w+)\)")
PROGRESS = re.compile(r"(\d+)/(\d+) \[(\d+):(\d+)<")

SETUP = '''\
# GPU version: pick the backend, and fetch the book's data if it's missing.
from pathlib import Path
from urllib.request import urlretrieve

from gstatsim_accel import gpu_status, kriging, simulation

BACKEND = "gpu" if gpu_status().available else "cpu"
print("Backend:", BACKEND, "|", gpu_status().device_name or gpu_status().reason)

DATA_URL = "https://raw.githubusercontent.com/GatorGlaciology/GStatSim/main/demos/data/"
for name in ("greenland_test_data.csv", "greenland_surface_data.csv"):
    if not (Path("data") / name).exists():
        Path("data").mkdir(exist_ok=True)
        urlretrieve(DATA_URL + name, Path("data") / name)

# The book's plotting helpers, pinned to the reviewed commit.
if not Path("plot_utils.py").exists():
    urlretrieve("https://raw.githubusercontent.com/GatorGlaciology/gstatsimbook/"
                "BOOK_COMMIT/plot_utils.py", "plot_utils.py")'''


def replace_call(m: re.Match) -> str:
    func, grid, df, xx, yy, zz, k, vario, rad = m.groups()
    method = "ordinary" if func.startswith("o") else "simple"
    common = (f"{df}, {grid}, {vario}, num_points={k}, radius={rad}, method=\"{method}\", "
              f"xx='{xx}', yy='{yy}', zz='{zz}', backend=BACKEND")
    return f"kriging.predict({common})" if not func.endswith("_sgs") else f"simulation.sgs({common})"


def replace_cluster_call(m: re.Match) -> str:
    grid, df, xx, yy, zz, kk, k, gamma, rad = m.groups()
    return (f"simulation.cluster_sgs({df}, {grid}, {gamma}, num_points={k}, radius={rad}, "
            f"cluster_col='{kk}', xx='{xx}', yy='{yy}', zz='{zz}', backend=BACKEND)")


def book_times(cell) -> list[str]:
    """Wall times of GStatSim's progress bars in the book's executed cell."""
    text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", []))
    found = {}
    for done, total, mm, ss in PROGRESS.findall(text):
        if done == total:
            found[len(found)] = f"{int(mm)} min {int(ss):02d} s"
    return list(found.values())


def convert(src: Path, dst: Path, title: str) -> None:
    nb = nbf.read(src, as_version=4)
    out = copy.deepcopy(nb)
    out.cells = []
    out.cells.append(nbf.v4.new_markdown_cell(
        f"> ⚡ **GPU version** of the GStatSim book chapter *{title}*  \n"
        f"> Original: [{src.stem}]({BOOK_URL}{src.stem}.html), Gator Glaciology Lab. "
        "Only the slow steps changed: GStatSim's kriging and SGS calls run through "
        "`gstatsim_accel` (`backend=BACKEND`), timed with `%%time` next to the CPU time "
        "recorded in the published book. Everything else is the book's own text and code."))
    out.cells.append(nbf.v4.new_code_cell(SETUP.replace("BOOK_COMMIT", BOOK_COMMIT)))

    for cell in nb.cells:
        cell = copy.deepcopy(cell)
        if cell.cell_type == "code":
            src_code = cell.source
            new = CALL.sub(replace_call, src_code)
            new = CLUSTER_CALL.sub(replace_cluster_call, new)
            # Several chapters predate GStatSim's model-type entry (the first
            # GStatSim was exponential-only); today's needs it.
            new = re.sub(r"(vario\d* = \[azimuth, nugget, major_range, minor_range, sill)\]",
                         r"\1, 'Exponential']  # model type: required since GStatSim 1.1", new)
            new = re.sub(r"^(gam\d = \[[^'\]]*)\]", r"\1, 'Exponential']", new, flags=re.M)
            new = new.replace("var.parameters[1]] for var in variograms]",
                              "var.parameters[1],\n            'Exponential'] for var in variograms]"
                              "  # model type: required since GStatSim 1.1")
            sim2 = re.search(r"^sim2 = simulation\.sgs\((.*)\) *$", new, flags=re.M)
            if sim2 and re.search(r"^sim3 = simulation\.sgs\(", new, flags=re.M):
                # Two more realizations: one call, simulated side by side on the GPU.
                args = sim2.group(1).replace(", backend=BACKEND", "")
                new = re.sub(r"^sim[23] = simulation\.sgs\(.*\n", "", new, flags=re.M)
                new = new.replace(
                    "# simulate multiple random realizations\n",
                    "# simulate multiple random realizations\n"
                    "# GPU version: both realizations in one call, simulated side by side\n"
                    f"sim2, sim3 = simulation.realizations({args}, seeds=[2, 3], backend=BACKEND)\n", 1)
            if any(call in new for call in ("kriging.predict(", "simulation.", "rbf_trend(")):
                times = book_times(cell)
                note = (f"# Published book (GStatSim on CPU): {', '.join(times)}\n" if times else "")
                new = "%%time\n" + note + new
                cell.outputs, cell.execution_count = [], None
            cell.source = new
        out.cells.append(cell)

    out.metadata["kernelspec"] = KERNEL
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(out, dst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("book", type=Path, help="checkout of GatorGlaciology/gstatsimbook")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "notebooks" / "book")
    a = p.parse_args(argv)
    for name, title in CHAPTERS.items():
        convert(a.book / name, a.out / name.replace(".ipynb", "_GPU.ipynb"), title)
        print("wrote", a.out / name.replace(".ipynb", "_GPU.ipynb"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
