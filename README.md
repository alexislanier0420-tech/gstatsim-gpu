# gstatsim-accel

Opt-in CPU/GPU backends for [GStatSim](https://github.com/GatorGlaciology/GStatSim)
workflows. Final project for the UF geostatistics course; plan and milestones are in
[TRACKER.md](TRACKER.md).

```python
from gstatsim_accel import kriging

est, var = kriging.predict(df_grid, pred_xy, vario,
                           num_points=100, radius=50_000,
                           method="simple", backend="cpu", zz="Nbed")
```

- `backend="cpu"` calls GStatSim unchanged and is the reference for correctness.
- `backend="gpu"` (Phase 2, not implemented yet) raises `BackendUnavailableError`
  when CuPy or a CUDA device is missing. It never falls back to the CPU on its own.
- Inputs match GStatSim's: a DataFrame of conditioning data, an `(M, 2)` array of
  prediction points, and the variogram list
  `[azimuth, nugget, major_range, minor_range, sill, vtype(, s)]`.

## Setup

```bash
conda env create -f environment.yml     # isolated from the course env
conda activate gstatsim-gpu
pytest
```

The env pins OpenBLAS (MKL 2026.1 crashes SciPy on Windows) and installs
`cupy-cuda12x[ctk]`, which bundles the CUDA libraries. Without `[ctk]`, CuPy detects
the GPU but fails on the first linear-algebra call.

## Layout

| Path | What |
|---|---|
| `src/gstatsim_accel/kriging.py` | Notebook-facing `predict()` and backend dispatch |
| `src/gstatsim_accel/backend.py` | GPU detection, `BackendUnavailableError` |
| `src/gstatsim_accel/validation.py` | Input checks shared by both backends |
| `src/gstatsim_accel/bench.py`, `envinfo.py` | Timing harness and environment metadata |
| `benchmarks/bench_kriging.py` | Tutorial 3 workload; writes JSON record + `.npz` outputs |
| `benchmarks/compare_outputs.py` | CPU-vs-GPU output comparison against a set tolerance |
| `scripts/env_report.py` | Phase 0 hardware/software report and GPU smoke test |
| `scripts/hpg/env_report.sbatch` | Same report on a HiPerGator GPU node |
| `docs/existing_port_review.md` | Review of the prior CuPy port and early CPU profile |
| `figures/` | Method figure (Rubble Glacier study area) |

## Benchmarks

```bash
# Tutorial 3 data: GStatSim repo, demos/data/greenland_test_data.csv -> data/
python benchmarks/bench_kriging.py --csv data/greenland_test_data.csv --method simple --repeats 3
python scripts/env_report.py --label laptop
```

Records go to `results/` (git-ignored). Each holds the environment, the config,
raw timings and a correctness field. Records with `correctness: null` must not be
quoted as speedups.
