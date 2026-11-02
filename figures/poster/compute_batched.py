"""Exact vs. batched SGS: does simulating many cells at once change the result?

Batched SGS (the shortcut an earlier GPU port takes) visits the random path in
chunks and kriges every cell in a chunk from the data and *earlier* chunks only,
so cells in the same chunk never condition on each other. Exact SGS (batch 1,
GStatSim's algorithm, here the CUDA kernel) adds every cell to the conditioning
set before the next. This compares the variogram of their realizations with the
variogram model fitted to the data, on a land-only sub-area of Rubble Glacier.
Writes results/poster/batched.npz.

    python figures/poster/compute_batched.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rubble_sgs import rubble_workflow  # noqa: E402

from gstatsim_accel import _batched, simulation  # noqa: E402

BATCHES = [1, 64, 512, 4096]  # 1 = exact; 4096 covers the whole sub-area at once
N_REAL = 10
BOX = 60                       # sub-area side, cells (1 km)
LAGS = np.arange(1, 26)        # km
K, RADIUS = 100, 50_000


def batched_sgs(points, X, Y, Z, vario, seed, batch, mean):
    """SGS where each chunk of `batch` path cells is kriged at once from the data and
    earlier chunks only. Same random path and draws as exact SGS for the seed."""
    rng = np.random.default_rng(seed)
    path = np.arange(len(points))
    rng.shuffle(path)
    draws = rng.standard_normal(len(points))
    # As in GStatSim (nugget 0): a cell on a datum copies it and is NOT added to the
    # conditioning set (a duplicate location would make the system singular), and the
    # k-th simulated cell takes the k-th random draw.
    data_xy = set(zip(X, Y, strict=True))
    on_datum = np.array([(x, y) in data_xy for x, y in points[path]])
    eps = np.zeros(len(points))
    eps[~on_datum] = draws[:(~on_datum).sum()]
    Xc, Yc, Zc = list(X), list(Y), list(Z)
    out = np.empty(len(points))
    for start in range(0, len(points), batch):
        idx = path[start:start + batch]
        est, var = _batched.krige(cp, points[idx], np.array(Xc), np.array(Yc), np.array(Zc),
                                  vario, K, RADIUS, ordinary=False, global_mean=mean)
        new = ~on_datum[start:start + len(idx)]
        val = np.where(new, est + np.sqrt(np.abs(var)) * eps[start:start + len(idx)], est)
        out[idx] = val
        Xc += list(points[idx[new], 0])
        Yc += list(points[idx[new], 1])
        Zc += list(val[new])
    return out


def grid_semivariogram(field):
    """Semivariance at each lag (cells), averaged over x and y shifts; NaNs ignored."""
    g = []
    for h in LAGS:
        dx = (field[:, h:] - field[:, :-h]) ** 2
        dy = (field[h:, :] - field[:-h, :]) ** 2
        g.append(0.5 * np.nanmean(np.concatenate([dx[np.isfinite(dx)], dy[np.isfinite(dy)]])))
    return np.array(g)


def main() -> int:
    w = rubble_workflow()
    df, pred, land, vario = w["df_grid"], w["pred_xy"], w["land"], w["vario"]
    rows, cols = w["rows"], w["cols"]
    # Sub-area with the most land
    L = land.reshape(rows, cols)
    best, r0c0 = -1, (0, 0)
    for r0 in range(0, rows - BOX, 5):
        for c0 in range(0, cols - BOX, 5):
            f = L[r0:r0 + BOX, c0:c0 + BOX].mean()
            if f > best:
                best, r0c0 = f, (r0, c0)
    r0, c0 = r0c0
    grid_idx = (np.arange(rows)[:, None] * cols + np.arange(cols)[None, :])[r0:r0 + BOX, c0:c0 + BOX].ravel()
    pts = pred[grid_idx]
    print(f"sub-area rows {r0}-{r0 + BOX}, cols {c0}-{c0 + BOX}, land {best:.0%}", flush=True)

    X, Y, Z = (df[c].to_numpy() for c in ("X", "Y", "NRes"))
    mean = float(df["NRes"].mean())
    gammas, examples, times = {}, {}, {}
    for b in BATCHES:
        t = time.perf_counter()
        if b == 1:
            reals = simulation.realizations(df, pts, vario, seeds=list(range(N_REAL)), backend="gpu",
                                            num_points=K, radius=RADIUS, zz="NRes")
        else:
            reals = np.stack([batched_sgs(pts, X, Y, Z, vario, s, b, mean) for s in range(N_REAL)])
        times[b] = time.perf_counter() - t
        fields = reals.reshape(N_REAL, BOX, BOX)
        gammas[b] = np.stack([grid_semivariogram(f) for f in fields])
        examples[b] = fields[0]
        print(f"batch {b:5d}: {times[b]:.1f} s, gamma(1 km) {gammas[b][:, 0].mean():.3f}", flush=True)

    nug, rng_, sill = vario[1], vario[2], vario[4]
    model = nug + (sill - nug) * (1 - np.exp(-3 * LAGS * 1000 / rng_))
    out = ROOT / "results" / "poster"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "batched.npz", batches=np.array(BATCHES), lags=LAGS, model=model,
                        **{f"gamma_{b}": gammas[b] for b in BATCHES},
                        **{f"example_{b}": examples[b] for b in BATCHES},
                        **{f"time_{b}": times[b] for b in BATCHES},
                        sub=np.array([r0, c0, BOX]), data_x=X, data_y=Y, pts=pts)
    print("wrote", out / "batched.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
