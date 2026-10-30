"""SGS as one CUDA kernel: a whole realization per thread block, one launch.

The CuPy version (_sgs.py) issues ~60 small GPU operations per simulated
node; for a single realization that launch overhead makes the GPU no faster
than the CPU. Here one block of threads walks the entire random path itself:
for each node it computes distances, picks octant neighbors, builds the
covariance matrix in shared memory, factors it (Cholesky), solves, and draws
the sample, then moves to the next node. Realizations run as separate blocks
at the same time.

Same results as GStatSim 1.2.0 (skrige_sgs / okrige_sgs) with the same seed,
up to neighbor ties, exactly like _sgs.py: same path, same random draws,
neighbors ordered octant-major and nearest first, ties by row order.
"""

from __future__ import annotations

import functools

import numpy as np

import gstatsim as gs

from ._batched import OCTANT_EDGES

_VTYPES = {"exponential": 0, "gaussian": 1, "spherical": 2}

_SOURCE = r"""
#define MAXK __MAXK__
#define INF (__longlong_as_double(0x7ff0000000000000ULL))

__constant__ double EDGES[9];

__device__ __forceinline__ double covar(double h, int vtype, double c_struct) {
    // Covariance.covar from GStatSim 1.2.0
    if (vtype == 0) return c_struct * exp(-3.0 * h);
    if (vtype == 1) return c_struct * exp(-3.0 * (h * h));
    return h <= 1.0 ? c_struct * (1.0 - 1.5 * h + 0.5 * (h * h * h)) : 0.0;
}

__device__ __forceinline__ bool before(double d1, int i1, double d2, int i2) {
    // (distance, row) order: nearest first, ties to the earlier row
    return d1 < d2 || (d1 == d2 && i1 < i2);
}

extern "C" __global__ void sgs_kernel(
    double* X, double* Y, double* Z,                 // (S, L) conditioning set, grows as nodes are simulated
    const double* P,                                 // (M, 2) node coordinates
    const long long* node, const long long* datum,   // (S, M) path order; datum to copy (-1 = simulate)
    const double* eps,                               // (S, M) standard normal for each step
    double* dist, int* octv, int* cand,              // (S, L) scratch: candidates within the radius
    double* out, int* flags,                         // (S, M) result; (S) error flags
    const int N, const int M, const int L,
    const double radius, const int oct_count, const int ordinary,
    const int vtype, const double nug, const double sill,
    const double R00, const double R01, const double R10, const double R11,
    const double global_mean)
{
    const int s = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, nwarps = (bs + 31) >> 5;
    X += (size_t)s * L; Y += (size_t)s * L; Z += (size_t)s * L;
    dist += (size_t)s * L; octv += (size_t)s * L; cand += (size_t)s * L;
    node += (size_t)s * M; datum += (size_t)s * M; eps += (size_t)s * M; out += (size_t)s * M;

    __shared__ double last_d[8];
    __shared__ int last_i[8];
    __shared__ double red_d[8][32];
    __shared__ int red_i[8][32];
    __shared__ int slot[MAXK], nb[MAXK];
    __shared__ int m_sh, bad_sh, ncand_sh;
    __shared__ double rx[MAXK], ry[MAXK], zz[MAXK], c0[MAXK], y0[MAXK], y1[MAXK];
    __shared__ double A[MAXK * (MAXK + 1) / 2];   // lower triangle, packed by rows

    const double c_struct = sill - nug;
    const int K = 8 * oct_count;

    for (int i = 0; i < M; ++i) {
        const int z = (int)node[i];
        const double px = P[2 * z], py = P[2 * z + 1];
        const int n = N + i;
        if (datum[i] >= 0) {               // node sits on a datum (nugget 0): copy it
            if (tid == 0) out[z] = Z[datum[i]];
            __syncthreads();
            continue;
        }

        // 1. Distance and octant of every point within the radius
        //    (NearestNeighbor.nearest_neighbor_search), collected into a
        //    candidate list so the selection rounds below skip the rest.
        if (tid == 0) ncand_sh = 0;
        if (tid < 8) { last_d[tid] = -1.0; last_i[tid] = -1; }
        __syncthreads();
        for (int c = tid; c < n; c += bs) {
            const double dx = X[c] - px, dy = Y[c] - py;
            const double d = sqrt(dx * dx + dy * dy);
            if (!(d < radius)) continue;                       // also drops NaN placeholders
            const double a = atan2(dx, dy);
            int o = -1;
            for (int e = 0; e < 9; ++e) o += (EDGES[e] < a);   // pd.cut, right-closed bins
            if (o < 0) continue;
            const int q = atomicAdd(&ncand_sh, 1);
            cand[q] = c; dist[q] = d; octv[q] = o;
        }
        __syncthreads();
        const int ncand = ncand_sh;

        // 2. The oct_count nearest per octant: each round takes the next
        //    candidate after the last one picked, in (distance, row) order.
        for (int r = 0; r < oct_count; ++r) {
            double bd[8]; int bi[8];
            for (int k = 0; k < 8; ++k) { bd[k] = INF; bi[k] = 0x7fffffff; }
            // Candidates are in arbitrary order (atomic append); the
            // (distance, row) comparison makes the result independent of it.
            for (int q = tid; q < ncand; q += bs) {
                const int o = octv[q], c = cand[q];
                const double d = dist[q];
                if (!before(last_d[o], last_i[o], d, c)) continue;
                if (before(d, c, bd[o], bi[o])) { bd[o] = d; bi[o] = c; }
            }
            for (int k = 0; k < 8; ++k) {
                for (int off = 16; off > 0; off >>= 1) {
                    const double od = __shfl_down_sync(0xffffffff, bd[k], off);
                    const int oi = __shfl_down_sync(0xffffffff, bi[k], off);
                    if (before(od, oi, bd[k], bi[k])) { bd[k] = od; bi[k] = oi; }
                }
                if (lane == 0) { red_d[k][warp] = bd[k]; red_i[k][warp] = bi[k]; }
            }
            __syncthreads();
            if (tid < 8) {
                double best_d = INF; int best_i = 0x7fffffff;
                for (int w = 0; w < nwarps; ++w)
                    if (before(red_d[tid][w], red_i[tid][w], best_d, best_i)) {
                        best_d = red_d[tid][w]; best_i = red_i[tid][w];
                    }
                slot[tid * oct_count + r] = best_d < INF ? best_i : -1;
                last_d[tid] = best_d; last_i[tid] = best_i;
            }
            __syncthreads();
        }

        // 3. Neighbors in GStatSim's order (octant-major, nearest first)
        if (tid == 0) {
            int m = 0;
            for (int q = 0; q < K; ++q) if (slot[q] >= 0) nb[m++] = slot[q];
            m_sh = m; bad_sh = 0;
            if (m == 0) atomicOr(&flags[s], 1);
        }
        __syncthreads();
        const int m = m_sh;
        if (m == 0) { if (tid == 0) out[z] = 0.0; __syncthreads(); continue; }

        // 4. Kriging system (Covariance.make_covariance_matrix / _array), rotated coordinates
        const double prx = px * R00 + py * R10, pry = px * R01 + py * R11;
        for (int q = tid; q < m; q += bs) {
            const int j = nb[q];
            const double x = X[j], y = Y[j];
            rx[q] = x * R00 + y * R10; ry[q] = x * R01 + y * R11; zz[q] = Z[j];
            const double e0 = rx[q] - prx, e1 = ry[q] - pry;
            c0[q] = covar(sqrt(e0 * e0 + e1 * e1), vtype, c_struct);
            y0[q] = c0[q]; y1[q] = 1.0;
        }
        __syncthreads();
        for (int e = tid; e < m * m; e += bs) {
            const int a = e / m, b = e % m;
            if (b > a) continue;
            const double d0 = rx[a] - rx[b], d1 = ry[a] - ry[b];
            A[a * (a + 1) / 2 + b] = (a == b) ? sill : covar(sqrt(d0 * d0 + d1 * d1), vtype, c_struct);
        }
        __syncthreads();

        // 5. Cholesky factorization C = L L^T, in place, one barrier per column:
        //    step k updates the trailing block with column k unscaled (divided
        //    by the pivot) while column k-1, which step k doesn't read, is scaled.
        for (int k = 0; k <= m; ++k) {
            if (k > 0) {                                   // finish column k-1
                const int kk1 = (k - 1) * k / 2 + (k - 1);
                double p1 = A[kk1];
                if (p1 <= 0.0) { if (tid == 0) bad_sh = 1; p1 = 1.0; }
                const double piv = sqrt(p1);
                for (int a = k + tid; a < m; a += bs) A[a * (a + 1) / 2 + k - 1] /= piv;
                if (tid == 0) A[kk1] = piv;
            }
            if (k < m) {
                const double p = A[k * (k + 1) / 2 + k];
                const double inv_p = p > 0.0 ? 1.0 / p : 1.0;
                for (int a = k + 1 + warp; a < m; a += nwarps) {   // one warp per row
                    const int ra = a * (a + 1) / 2;
                    const double lak = A[ra + k] * inv_p;
                    for (int b = k + 1 + lane; b <= a; b += 32) A[ra + b] -= lak * A[b * (b + 1) / 2 + k];
                }
            }
            __syncthreads();
        }

        // 6. Solve C w = c0 (and C v = 1 for ordinary kriging). Little arithmetic
        //    and a dependency at every step, so one warp does it with cheap
        //    warp barriers instead of block-wide ones.
        if (warp == 0) {
            for (int k = 0; k < m; ++k) {                 // forward: L y = rhs
                const double lkk = A[k * (k + 1) / 2 + k];
                const double v0 = y0[k] / lkk, v1 = y1[k] / lkk;
                __syncwarp();
                if (lane == 0) { y0[k] = v0; y1[k] = v1; }
                for (int a = k + 1 + lane; a < m; a += 32) {
                    const double l = A[a * (a + 1) / 2 + k];
                    y0[a] -= l * v0; y1[a] -= l * v1;
                }
                __syncwarp();
            }
            for (int k = m - 1; k >= 0; --k) {            // backward: L^T x = y
                const int row = k * (k + 1) / 2;
                const double v0 = y0[k] / A[row + k], v1 = y1[k] / A[row + k];
                __syncwarp();
                if (lane == 0) { y0[k] = v0; y1[k] = v1; }
                for (int a = lane; a < k; a += 32) { y0[a] -= A[row + a] * v0; y1[a] -= A[row + a] * v1; }
                __syncwarp();
            }
        }
        __syncthreads();

        // 7. Estimate, variance and the simulated value (skrige_sgs / okrige_sgs)
        if (tid == 0) {
            double est, var = sill;
            if (ordinary) {
                // Bordered system via the Schur complement: w = C^-1 c0 - mu C^-1 1, sum(w) = 1
                double s0 = 0.0, s1 = 0.0, mean = 0.0;
                for (int q = 0; q < m; ++q) { s0 += y0[q]; s1 += y1[q]; mean += zz[q]; }
                mean /= m;
                const double mu = (s0 - 1.0) / s1;
                est = 0.0;
                for (int q = 0; q < m; ++q) {
                    const double w = y0[q] - mu * y1[q];
                    est += w * (zz[q] - mean); var -= w * c0[q];
                }
                est += mean;
            } else {
                est = 0.0;
                for (int q = 0; q < m; ++q) { est += y0[q] * (zz[q] - global_mean); var -= y0[q] * c0[q]; }
                est += global_mean;
            }
            const double value = est + sqrt(fabs(var)) * eps[i];
            out[z] = value;
            X[n] = px; Y[n] = py; Z[n] = value;
            if (bad_sh) atomicOr(&flags[s], 2);
        }
        __syncthreads();
    }
}
"""


@functools.lru_cache(maxsize=None)
def _kernel(maxk: int):
    import cupy as cp

    module = cp.RawModule(code=_SOURCE.replace("__MAXK__", str(maxk)),
                          # No fused multiply-add, so distances round exactly like NumPy's.
                          options=("--fmad=false", "-std=c++17"))
    edges = cp.ndarray((9,), cp.float64, module.get_global("EDGES"))
    edges[...] = cp.asarray(OCTANT_EDGES, dtype=cp.float64)
    return module.get_function("sgs_kernel")


# The packed covariance matrix lives in shared memory: 96 neighbors is
# 96*97/2 doubles = 37 KB, inside the 48 KB a block gets by default.
MAX_NEIGHBORS = 96


def supports(vario, num_points) -> bool:
    return vario[5].lower() in _VTYPES and 8 <= 8 * (num_points // 8) <= MAX_NEIGHBORS


def sgs_ensemble(points, X, Y, Z, vario, num_points, radius, ordinary, seeds, global_mean,
                 threads=256):
    """One realization per seed (NumPy arrays in, NumPy (len(seeds), M) out)."""
    import cupy as cp

    if not supports(vario, num_points):
        raise NotImplementedError("CUDA SGS supports exponential/Gaussian/spherical variograms "
                                  f"and up to {MAX_NEIGHBORS} neighbors (num_points < {MAX_NEIGHBORS + 8})")
    S, M, N = len(seeds), len(points), len(X)
    oct_count = num_points // 8

    # Path, datum copies and random draws: identical to _sgs.sgs_ensemble.
    first_datum = {}
    if float(vario[1]) == 0:
        for i, xy in enumerate(zip(X, Y)):
            first_datum.setdefault(xy, i)
    datum_of = np.array([first_datum.get((x, y), -1) for x, y in points])
    node = np.empty((S, M), dtype=np.int64)
    eps = np.zeros((S, M))
    for s, seed in enumerate(seeds):
        rng = gs.get_random_generator(seed)
        path = np.arange(M)
        rng.shuffle(path)
        draws = rng.standard_normal(M)
        simulated = datum_of[path] < 0
        node[s] = path
        eps[s, simulated] = draws[:simulated.sum()]
    datum = datum_of[node]

    L = N + M
    Xa = cp.full((S, L), cp.nan)
    Ya = cp.full((S, L), cp.nan)
    Za = cp.zeros((S, L))
    Xa[:, :N], Ya[:, :N], Za[:, :N] = cp.asarray(X), cp.asarray(Y), cp.asarray(Z)
    out = cp.empty((S, M))
    flags = cp.zeros(S, dtype=cp.int32)
    R = gs.make_rotation_matrix(vario[0], vario[2], vario[3])

    maxk = 32 * -(-8 * oct_count // 32)  # round up so few kernel variants get compiled
    _kernel(maxk)(
        (S,), (threads,),
        (Xa, Ya, Za, cp.asarray(np.ascontiguousarray(points, dtype=np.float64)),
         cp.asarray(node), cp.asarray(datum), cp.asarray(eps),
         cp.empty((S, L)), cp.empty((S, L), dtype=cp.int32), cp.empty((S, L), dtype=cp.int32), out, flags,
         np.int32(N), np.int32(M), np.int32(L), np.float64(radius), np.int32(oct_count),
         np.int32(bool(ordinary)), np.int32(_VTYPES[vario[5].lower()]),
         np.float64(vario[1]), np.float64(vario[4]),
         np.float64(R[0, 0]), np.float64(R[0, 1]), np.float64(R[1, 0]), np.float64(R[1, 1]),
         np.float64(global_mean)),
    )
    f = int(flags.max().get()) if S else 0
    if f & 1:
        raise ValueError("Unable to find nearest neighbors for some nodes. Try increasing the search radius.")
    if f & 2:
        raise FloatingPointError("Kriging matrix not positive definite (duplicate points with zero nugget?)")
    return out.get()
