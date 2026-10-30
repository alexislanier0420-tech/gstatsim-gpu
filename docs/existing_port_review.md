# Review: existing GPU port (badger-lord/gstatsim_gpu)

Reviewed 2026-10-29 against GStatSim `8d7804b` (v1.2.0; the course env has 1.1.5).
**Correction (same day):** items 3 and 4 were first listed as port bugs. The port
actually matches GStatSim 1.1.5, the version current when it was written; 1.2.0
changed both. They are differences from today's GStatSim, not mistakes in the port.
Port commit `3982187` (Feb 2026, single commit, CuPy + RAPIDS cuML, written for HiPerGator B200/L4 nodes).

## What it is

`gstatsim_custom` ports kriging and SGS to the GPU by processing a batch of
thousands of grid cells per step: neighbors are found with a fixed grid stencil,
covariance matrices are built for the whole batch, and all systems are solved
with one batched `cp.linalg.solve` (OK) or Cholesky (SK) call. Its README reports
~4,500 pts/s on an L4 and ~55,000 pts/s on a B200 for SGS. There are no tests
and no CPU comparison, so those figures are throughput only, not validated speedups.

## Differences from GStatSim that change results

| # | Area | GStatSim (reference) | Port | Effect |
|---|---|---|---|---|
| 1 | **SGS batching** | Each simulated cell is added to the conditioning data before the next cell is visited | All cells in a batch (default up to 200k) are simulated at once, conditioned only on cells from *earlier* batches | Cells in the same batch ignore each other. Changes the simulation algorithm; spatial correlation at short lags will likely be too weak. This is the "changing stochastic semantics" risk in the timeline (section 2). |
| 2 | **Neighbor selection** | Octant search: up to `num_points // 8` nearest per 45° sector | Plain K-nearest inside the radius (`use_sector_balance`/`n_sectors` accepted but unused) | Different neighbors → different estimates, especially near data-dense flight lines. |
| 3 | Spherical covariance | 1.2.0: `(sill − nug)·(1 − 1.5h + 0.5h³)`, 0 beyond range | `sill − nug − 1.5h + 0.5h³`; `sill − 1` beyond range | **Same as GStatSim ≤ 1.1.5**, which had this formula; GStatSim fixed it in 1.2.0 (Jan 2026). Correct only when `sill − nug = 1` and `sill = 1`. Not a port bug, but outdated. |
| 4 | Kriging matrix diagonal | 1.2.0: full `sill` (nugget on the diagonal) | `sill − nugget` plus tiny jitter | **Same as GStatSim ≤ 1.1.5**; changed in 1.2.0 ("update nugget"). Differs from 1.2.0 whenever nugget > 0. |
| 5 | OK weights | Used as solved | Renormalized to sum to 1 after solving | Small differences; masks ill-conditioned solves instead of reporting them. |
| 6 | `krige_gpu` output | Estimate **and** variance | Estimate only; variance computed then discarded | Can't reproduce Tutorial 3's uncertainty maps. |
| 7 | Normal-score transform | Done by the user in the notebook | Done inside `krige_gpu`/`sgs_gpu` with cuML | Different workflow and API; cuML ties it to RAPIDS. |
| 8 | Input format | Scattered points (DataFrame) + prediction points | Regular 2-D grids only (`xx`, `yy`, `grid` with NaNs) | Can't take raw bed picks directly; needs pre-gridding. |
| 9 | float32 | n/a | README: "currently producing NaNs" | Only float64 is usable. |

Also: no license file (README says "[Insert your license here]"), so its code
can't be copied into this project without permission. Ideas (batching, GEMM
distances, stencil search) can be credited and reimplemented.

## What this means for the project

- The port shows batched GPU kriging is feasible on HiPerGator, so this project
  should credit it and not claim to be the first GPU version.
- What it lacks is what the RFC asks for: a validated CPU reference, documented
  tolerances, GStatSim-compatible inputs and outputs (octant search, variance), and
  SGS that keeps sequential semantics. That gap is this project's contribution.
- The port's SGS throughput should not be compared directly to GStatSim's SGS
  (difference 1). If cited, label it as a different algorithm.
- Items 2–4 are concrete test cases: the GPU backend here must match GStatSim
  1.2.0 on them within tolerance (done: see `tests/test_batched.py`).

## Early CPU profile (laptop, for Phase 1 planning)

GStatSim simple kriging, Tutorial 3 workload (3,650 conditioning cells,
k=100, 50 km radius), 300 prediction points under cProfile:

| Component | Share of time |
|---|---|
| Neighbor search (pandas copy, sort, `pd.cut` per point) | ~79% |
| Covariance matrix | ~5% |
| `lstsq` solve | ~6% |

Linear algebra is not the bottleneck on the CPU; the per-point pandas search
is. So (a) a GPU path must batch the neighbor search, not just the solves,
and (b) a CPU-only fix (e.g. KD-tree + vectorized octants) could give a large
speedup by itself. Report both, so a GPU speedup isn't really a measure of pandas overhead.
Full grid is 22,500 points at ~10 ms/point ≈ 3–4 min per run.
