# GStatSim book, GPU versions ⚡

GPU versions of the chapters of the [GStatSim Jupyter Book](https://gatorglaciology.github.io/gstatsimbook/)
(Gator Glaciology Lab) whose slow steps `gstatsim_accel` supports. Each notebook is
the book's own chapter, same text, data and code, with only GStatSim's kriging and
SGS calls replaced by `kriging.predict` / `simulation.sgs` (`backend=BACKEND`). Every
replaced cell is timed with `%%time` and shows the CPU time recorded in the
published book.

| Chapter | Slow steps | Book (GStatSim, CPU) | GPU (RTX 5060 laptop) |
|---|---|---|---|
| [3 Simple & ordinary kriging](3_Simple_kriging_and_ordinary_kriging_GPU.ipynb) | 2 kriging runs | 4 min 11 s | 4.3 s |
| [4 Sequential Gaussian simulation](4_Sequential_Gaussian_Simulation_GPU.ipynb) | 3 SGS realizations | 7 min 58 s | 19.2 s |
| [5 Interpolation with anisotropy](5_interpolation_with_anisotropy_GPU.ipynb) | 2 kriging + 2 SGS | 10 min 25 s | 33.9 s |
| [8 Interpolation with a trend](8_interpolation_with_a_trend_GPU.ipynb) | kriging + SGS (+ RBF trend, CPU) | 5 min 23 s | 22.1 s |

GPU times include about 3 s of one-time GPU start-up in each notebook's first call.
The book's times come from a different computer, so compare the ratios, not the exact numbers.

**Not converted:**
- Chapters 1–2 (variograms) run in seconds already.
- Chapters 6–7 need `cluster_sgs` and chapter 9 needs `cokrige_mm1`/`cosim_mm1`,
  which have no GPU version yet.

**Changes beyond the swapped calls:**
- Chapters 4 and 5 define the variogram without a model type, which only the first
  GStatSim (exponential-only) accepted. The GPU versions add `'Exponential'`, the
  model the chapter fits.
- Chapter 4's two extra realizations run in one `simulation.realizations` call
  with seeds 2 and 3, simulated side by side on the GPU.

**Running them:** open with the **Python (gstatsim-gpu)** kernel (see the main
README). The first cell downloads the book's Greenland data and its `plot_utils.py`
(pinned to book commit `9365b2d`) if they're missing.

**Regenerating** after the book changes:
`python scripts/make_book_gpu_notebooks.py path/to/gstatsimbook`
