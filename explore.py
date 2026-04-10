#!/usr/bin/env python3
"""
explore.py  –  Systematic ANN algorithm exploration on RTX 5090

Algorithms:
  BF-FP32   sequential scan, FP32 heap  (baseline)
  BF-FP16   same but X stored as FP16   (2x less L2 traffic)
  BF-INT8   DP4A dot-product, INT8 X    (4x less L2 traffic)
  BF-GEMM   tiled cuBLAS QX^T + norms + running top-k merge
  LSH       FastLSHIndex (lsh_fast.py): E2LSH + CUDA bucket lookup + fused rerank
  IVF-Flat  sklearn K-means + GPU sequential BF within clusters

Theoretical model:
  For each algo: AI (FLOPs/byte), L2 traffic, predicted time vs actual.
  Ridge point = compute_peak / L2_bandwidth.
"""
import argparse, contextlib, io, json, math, re, time
from typing import Optional

import numpy as np
import cupy as cp

try:
    from sklearn.cluster import MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from lsh import LSHParams
    from lsh_fast import FastLSHIndex
    HAS_LSH = True
except ImportError:
    HAS_LSH = False

try:
    from cuvs.neighbors import cagra
    HAS_CAGRA = True
except ImportError:
    HAS_CAGRA = False

# ─────────────────────────────────────────────────────────────────────────────
# Shared device helper (heap_push) — inlined into every kernel via __device__
# ─────────────────────────────────────────────────────────────────────────────
_HEAP_DEVICE = r"""
__device__ __forceinline__ void heap_push(
    float* ld, int* li, int* ln,
    float* lmax, int* lmax_pos,
    float dist, int idx, int K)
{
    if (*ln < K) {
        ld[*ln] = dist; li[*ln] = idx; (*ln)++;
        if (*ln == K) {
            *lmax_pos = 0;
            for (int j = 1; j < K; j++) if (ld[j] > ld[*lmax_pos]) *lmax_pos = j;
            *lmax = ld[*lmax_pos];
        }
    } else if (dist < *lmax) {
        ld[*lmax_pos] = dist; li[*lmax_pos] = idx;
        *lmax_pos = 0;
        for (int j = 1; j < K; j++) if (ld[j] > ld[*lmax_pos]) *lmax_pos = j;
        *lmax = ld[*lmax_pos];
    }
}
__device__ __forceinline__ void heap_merge(
    float* sh_d, int* sh_i,
    float* gd, int* gi, int* gn,
    float* gmax, int* gmax_pos,
    int BLOCK, int K)
{
    for (int t = 0; t < BLOCK; t++)
        for (int j = 0; j < K; j++) {
            float d = sh_d[t*K+j]; int id = sh_i[t*K+j];
            if (id < 0) continue;
            heap_push(gd, gi, gn, gmax, gmax_pos, d, id, K);
        }
    for (int i = 0; i < K-1; i++)
        for (int j = i+1; j < K; j++)
            if (gd[j] < gd[i]) {
                float td = gd[i]; gd[i] = gd[j]; gd[j] = td;
                int   ti = gi[i]; gi[i] = gi[j]; gi[j] = ti;
            }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# BF-FP32 kernel
# ─────────────────────────────────────────────────────────────────────────────
_BF32_SRC = _HEAP_DEVICE + r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__
extern "C" __global__ void bf_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ X,
    int*         __restrict__ out_ids,
    float*       __restrict__ out_dists,
    int m)
{
    int qi = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float smem[];
    float* q_sh = smem;
    float* sh_d = smem + D_VAL;
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);
    for (int j = tid; j < D_VAL; j += BLOCK)
        q_sh[j] = Q[qi*D_VAL + j];
    __syncthreads();

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0; float lmax = 3.402823e+38f; int lmax_pos = 0;

    for (int ci = tid; ci < m; ci += BLOCK) {
        const float* xp = X + (long long)ci * D_VAL;
        float dist = 0.f;
#if D_VAL <= 96
        #pragma unroll
#endif
        for (int di = 0; di < D_VAL; di++) { float dif = q_sh[di]-xp[di]; dist += dif*dif; }
        heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, ci, K_VAL);
    }
    for (int i = 0; i < K_VAL; i++) { sh_d[tid*K_VAL+i]=ld[i]; sh_i[tid*K_VAL+i]=li[i]; }
    __syncthreads();
    if (tid == 0) {
        float gd[K_VAL]; int gi[K_VAL];
        for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
        int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
        heap_merge(sh_d, sh_i, gd, gi, &gn, &gmax, &gmax_pos, BLOCK, K_VAL);
        for (int i = 0; i < K_VAL; i++) { out_ids[qi*K_VAL+i]=gi[i]; out_dists[qi*K_VAL+i]=gd[i]; }
    }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# BF-FP16 kernel
# ─────────────────────────────────────────────────────────────────────────────
_BF16_SRC = r"#include <cuda_fp16.h>" + _HEAP_DEVICE + r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__
extern "C" __global__ void bf_fp16(
    const float*  __restrict__ Q,
    const __half* __restrict__ X,
    int*          __restrict__ out_ids,
    float*        __restrict__ out_dists,
    int m)
{
    int qi = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float smem[];
    float* q_sh = smem;
    float* sh_d = smem + D_VAL;
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);
    for (int j = tid; j < D_VAL; j += BLOCK)
        q_sh[j] = Q[qi*D_VAL + j];
    __syncthreads();

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0; float lmax = 3.402823e+38f; int lmax_pos = 0;

    for (int ci = tid; ci < m; ci += BLOCK) {
        const __half* xp = X + (long long)ci * D_VAL;
        float dist = 0.f;
#if D_VAL <= 96
        #pragma unroll
#endif
        for (int di = 0; di < D_VAL; di++) {
            float dif = q_sh[di] - __half2float(xp[di]); dist += dif*dif;
        }
        heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, ci, K_VAL);
    }
    for (int i = 0; i < K_VAL; i++) { sh_d[tid*K_VAL+i]=ld[i]; sh_i[tid*K_VAL+i]=li[i]; }
    __syncthreads();
    if (tid == 0) {
        float gd[K_VAL]; int gi[K_VAL];
        for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
        int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
        heap_merge(sh_d, sh_i, gd, gi, &gn, &gmax, &gmax_pos, BLOCK, K_VAL);
        for (int i = 0; i < K_VAL; i++) { out_ids[qi*K_VAL+i]=gi[i]; out_dists[qi*K_VAL+i]=gd[i]; }
    }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# BF-INT8 kernel  (DP4A — 4 INT8 MACs per __dp4a call)
# D_VAL must be a multiple of 4; for d=8 → 2 DP4A calls per (q,x) pair.
# Distance: ||q-x||^2 = q_norm2 + x_norm2 - 2*dot(q,x)
#           dot approximated as __dp4a(q_packed, x_packed) * scale^2
# ─────────────────────────────────────────────────────────────────────────────
_BF8_SRC = _HEAP_DEVICE + r"""
#define K_VAL   __K__
#define D_VAL   __D__
#define D_Q     (D_VAL / 4)    /* int32 quads per vector */
#define BLOCK   __BLOCK__

extern "C" __global__ void bf_int8(
    const int*   __restrict__ Q_i32,     /* [n, D_Q]  queries as packed int32  */
    const int*   __restrict__ X_i32,     /* [m, D_Q]  database as packed int32 */
    int*         __restrict__ out_ids,
    float*       __restrict__ out_dists,
    const float* __restrict__ q_norms2,  /* [n]  ||q_fp32||^2                  */
    const float* __restrict__ x_norms2,  /* [m]  ||x_fp32||^2                  */
    float        scale2,                 /* scale^2: dot_int * scale2 ~= dot_fp */
    int m)
{
    int qi = blockIdx.x, tid = threadIdx.x;
    extern __shared__ int smemi[];
    int*   q_sh = smemi;
    float* sh_d = (float*)(smemi + D_Q);
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);

    for (int j = tid; j < D_Q; j += BLOCK)
        q_sh[j] = Q_i32[qi*D_Q + j];
    __syncthreads();

    float qn2 = q_norms2[qi];
    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0; float lmax = 3.402823e+38f; int lmax_pos = 0;

    for (int ci = tid; ci < m; ci += BLOCK) {
        const int* xp = X_i32 + (long long)ci * D_Q;
        int dot = 0;
#if D_Q <= 96
        #pragma unroll
#endif
        for (int qi4 = 0; qi4 < D_Q; qi4++)
            dot = __dp4a(q_sh[qi4], xp[qi4], dot);
        float dist = qn2 + x_norms2[ci] - 2.f * dot * scale2;
        heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, ci, K_VAL);
    }
    for (int i = 0; i < K_VAL; i++) { sh_d[tid*K_VAL+i]=ld[i]; sh_i[tid*K_VAL+i]=li[i]; }
    __syncthreads();
    if (tid == 0) {
        float gd[K_VAL]; int gi[K_VAL];
        for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
        int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
        heap_merge(sh_d, sh_i, gd, gi, &gn, &gmax, &gmax_pos, BLOCK, K_VAL);
        for (int i = 0; i < K_VAL; i++) { out_ids[qi*K_VAL+i]=gi[i]; out_dists[qi*K_VAL+i]=gd[i]; }
    }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# IVF: assign each X vector to nearest centroid (one thread per vector)
# ─────────────────────────────────────────────────────────────────────────────
_IVF_ASSIGN_SRC = r"""
#define D_VAL __D__
extern "C" __global__ void ivf_assign(
    const float* __restrict__ X,
    const float* __restrict__ centroids,
    int*         __restrict__ assign_out,
    int m, int C)
{
    int xi = blockIdx.x * 256 + threadIdx.x;
    if (xi >= m) return;
    const float* xp = X + xi * D_VAL;
    float best = 3.402823e+38f; int best_c = 0;
    for (int c = 0; c < C; c++) {
        const float* cp = centroids + c * D_VAL;
        float dist = 0.f;
#if D_VAL <= 96
        #pragma unroll
#endif
        for (int d = 0; d < D_VAL; d++) { float dif = xp[d]-cp[d]; dist += dif*dif; }
        if (dist < best) { best = dist; best_c = c; }
    }
    assign_out[xi] = best_c;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# IVF search kernel
# X_sorted: X vectors reordered by cluster → sequential (coalesced) reads
# sorted_ids[ci]: original index of the ci-th reordered vector
# q_probes[qi, p]: which cluster to probe for query qi at probe p
# ─────────────────────────────────────────────────────────────────────────────
_IVF_SEARCH_SRC = _HEAP_DEVICE + r"""
#define K_VAL   __K__
#define D_VAL   __D__
#define NPROBE  __NPROBE__
#define BLOCK   __BLOCK__

extern "C" __global__ void ivf_search(
    const float* __restrict__ Q,
    const float* __restrict__ X_sorted,       /* [m, d] reordered by cluster  */
    const int*   __restrict__ sorted_ids,     /* [m] original indices          */
    const int*   __restrict__ cluster_offsets,/* [C+1]                         */
    const int*   __restrict__ q_probes,       /* [n, NPROBE]                   */
    int*         __restrict__ out_ids,
    float*       __restrict__ out_dists,
    int n)
{
    int qi = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float smem[];
    float* q_sh = smem;
    float* sh_d = smem + D_VAL;
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);

    for (int j = tid; j < D_VAL; j += BLOCK)
        q_sh[j] = Q[qi*D_VAL + j];
    __syncthreads();

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0; float lmax = 3.402823e+38f; int lmax_pos = 0;

    for (int p = 0; p < NPROBE; p++) {
        int c  = q_probes[qi*NPROBE + p];
        int lo = cluster_offsets[c];
        int hi = cluster_offsets[c+1];
        for (int ci = lo + tid; ci < hi; ci += BLOCK) {
            const float* xp = X_sorted + (long long)ci * D_VAL;  /* sequential! */
            float dist = 0.f;
#if D_VAL <= 96
            #pragma unroll
#endif
            for (int di = 0; di < D_VAL; di++) { float dif = q_sh[di]-xp[di]; dist += dif*dif; }
            heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, sorted_ids[ci], K_VAL);
        }
    }
    for (int i = 0; i < K_VAL; i++) { sh_d[tid*K_VAL+i]=ld[i]; sh_i[tid*K_VAL+i]=li[i]; }
    __syncthreads();
    if (tid == 0) {
        float gd[K_VAL]; int gi[K_VAL];
        for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
        int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
        heap_merge(sh_d, sh_i, gd, gi, &gn, &gmax, &gmax_pos, BLOCK, K_VAL);
        for (int i = 0; i < K_VAL; i++) { out_ids[qi*K_VAL+i]=gi[i]; out_dists[qi*K_VAL+i]=gd[i]; }
    }
}
"""

# IVF: find top-NPROBE centroids per query (one thread per query)
_IVF_PROBE_SRC = r"""
#define D_VAL   __D__
#define NPROBE  __NPROBE__
extern "C" __global__ void ivf_probe(
    const float* __restrict__ Q,
    const float* __restrict__ centroids,
    int*         __restrict__ q_probes,
    int n, int C)
{
    int qi = blockIdx.x * 128 + threadIdx.x;
    if (qi >= n) return;
    const float* qp = Q + qi * D_VAL;
    float best_d[NPROBE]; int best_c[NPROBE];
    #pragma unroll
    for (int i = 0; i < NPROBE; i++) { best_d[i]=3.402823e+38f; best_c[i]=-1; }
    for (int c = 0; c < C; c++) {
        const float* cp = centroids + c * D_VAL;
        float dist = 0.f;
#if D_VAL <= 96
        #pragma unroll
#endif
        for (int d = 0; d < D_VAL; d++) { float dif=qp[d]-cp[d]; dist+=dif*dif; }
        int worst = 0;
        #pragma unroll
        for (int j = 1; j < NPROBE; j++) if (best_d[j] > best_d[worst]) worst = j;
        if (dist < best_d[worst]) { best_d[worst]=dist; best_c[worst]=c; }
    }
    /* insertion sort ascending */
    for (int i = 0; i < NPROBE-1; i++)
        for (int j = i+1; j < NPROBE; j++)
            if (best_d[j] < best_d[i]) {
                float td=best_d[i]; best_d[i]=best_d[j]; best_d[j]=td;
                int   tc=best_c[i]; best_c[i]=best_c[j]; best_c[j]=tc;
            }
    int* op = q_probes + qi * NPROBE;
    #pragma unroll
    for (int i = 0; i < NPROBE; i++) op[i] = best_c[i];
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_KERN_CACHE: dict = {}
# Increment when any *_SRC CUDA string changes so cached RawKernel is not reused.
_KERNEL_SRC_VERSION = 4

def _compile(src: str, name: str, subs: dict) -> cp.RawKernel:
    key = (_KERNEL_SRC_VERSION, name, tuple(sorted(subs.items())))
    if key in _KERN_CACHE:
        return _KERN_CACHE[key]
    code = src
    for k, v in subs.items():
        code = code.replace(f'__{k}__', str(v))
    kern = cp.RawKernel(code, name)
    _KERN_CACHE[key] = kern
    return kern

def _choose_block(k: int, d: int, items_per_thread: int = 1) -> int:
    """Choose block size fitting (d + 2*BLOCK*k)*4 bytes in 47 KB shared mem.
    Empirically BLOCK=128 outperforms 256 on RTX 5090 (fewer serial merge ops,
    better SM occupancy). Register cap applied for large d."""
    limit = 47 * 1024
    reg_cap = max(32, 65536 // max(64, d + k + 10))
    # Prefer 128 first, then check others
    for b in [128, 64, 32, 256, 512]:
        if b > reg_cap:
            continue
        if (d + 2 * b * k) * 4 < limit:
            return b
    return 32

def _tms(fn, warmup: int = 10, reps: int = 20) -> float:
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    ev0 = cp.cuda.Event(); ev1 = cp.cuda.Event()
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record(); ev1.synchronize()
    return float(cp.cuda.get_elapsed_time(ev0, ev1)) / reps

def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    k = min(k, pred.shape[1], gt.shape[1])
    # Plain int() so int32 pred and int64 gt match in sets (numpy scalar hashing)
    hits = 0
    for i in range(len(pred)):
        ps = {int(pred[i, j]) for j in range(k)}
        gs = {int(gt[i, j]) for j in range(k)}
        hits += len(ps & gs)
    return hits / (len(pred) * k)

def ground_truth(Q_gpu: cp.ndarray, X_gpu: cp.ndarray, k: int,
                 batch: int = 200) -> np.ndarray:
    n, m = Q_gpu.shape[0], X_gpu.shape[0]
    x_sq = cp.sum(X_gpu ** 2, axis=1)
    ids_all = []
    for i in range(0, n, batch):
        Qb = Q_gpu[i:i+batch]
        q_sq = cp.sum(Qb ** 2, axis=1, keepdims=True)
        dists = q_sq - 2 * (Qb @ X_gpu.T) + x_sq[None, :]
        part = cp.argpartition(dists, k-1, axis=1)[:, :k]
        rd = dists[cp.arange(len(Qb))[:, None], part]
        srt = cp.argsort(rd, axis=1)
        ids_all.append(part[cp.arange(len(Qb))[:, None], srt].get())
    return np.concatenate(ids_all)

# ─────────────────────────────────────────────────────────────────────────────
# LSH (FastLSHIndex) — same stack as lsh_fast.py, tuned grids per d
# ─────────────────────────────────────────────────────────────────────────────
def lsh_config_grid(d: int, seed: int):
    """Return (short_name, LSHParams) pairs. High d → fewer tables (build argsort cost)."""
    if d <= 12:
        return [
            ("LSH L100 K3 w3 m500",  LSHParams(n_tables=100, n_projections=3,
                bucket_width=3.0, max_cands_per_table=500, seed=seed)),
            ("LSH L100 K3 w3 m2k",   LSHParams(n_tables=100, n_projections=3,
                bucket_width=3.0, max_cands_per_table=2000, seed=seed)),
            ("LSH L100 K3 w1.5 m1k", LSHParams(n_tables=100, n_projections=3,
                bucket_width=1.5, max_cands_per_table=1000, seed=seed)),
            ("LSH L128 K3 w2 m800",  LSHParams(n_tables=128, n_projections=3,
                bucket_width=2.0, max_cands_per_table=800, seed=seed)),
            ("LSH L256 K2 w2 m400",  LSHParams(n_tables=256, n_projections=2,
                bucket_width=2.0, max_cands_per_table=400, seed=seed)),
        ]
    if d <= 32:
        return [
            ("LSH L80 K3 w0 m800",   LSHParams(n_tables=80, n_projections=3,
                bucket_width=0.0, max_cands_per_table=800, seed=seed)),
            ("LSH L100 K3 w0 m1k",   LSHParams(n_tables=100, n_projections=3,
                bucket_width=0.0, max_cands_per_table=1000, seed=seed)),
            ("LSH L100 K3 w2 m1.5k", LSHParams(n_tables=100, n_projections=3,
                bucket_width=2.0, max_cands_per_table=1500, seed=seed)),
        ]
    # d >= 33: fewer tables; auto w; fused rerank is O(C·d)
    return [
        ("LSH L32 K3 w0 m500",   LSHParams(n_tables=32, n_projections=3,
            bucket_width=0.0, max_cands_per_table=500, seed=seed)),
        ("LSH L48 K3 w0 m600",   LSHParams(n_tables=48, n_projections=3,
            bucket_width=0.0, max_cands_per_table=600, seed=seed)),
        ("LSH L64 K3 w0 m800",   LSHParams(n_tables=64, n_projections=3,
            bucket_width=0.0, max_cands_per_table=800, seed=seed)),
    ]


def bench_lsh_fast(X_gpu: cp.ndarray, Q_gpu: cp.ndarray, k: int,
                   params, oversample: int = 4, reps: int = 20):
    """Build + timed search (search only in reported ms). Returns (ids_np, search_ms, build_ms)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        idx = FastLSHIndex(params)
        t0 = time.perf_counter()
        idx.build(X_gpu)
        build_ms = (time.perf_counter() - t0) * 1000.0
    cp.cuda.Stream.null.synchronize()
    for _ in range(3):
        idx.search_fast(Q_gpu, k, oversample=oversample, quiet=True)
    cp.cuda.Stream.null.synchronize()

    def fn():
        idx.search_fast(Q_gpu, k, oversample=oversample, quiet=True)

    ms = _tms(fn, warmup=10, reps=reps)
    cp.cuda.Stream.null.synchronize()
    out = idx.search_fast(Q_gpu, k, oversample=oversample, quiet=True)
    return cp.asnumpy(out), ms, build_ms


def cagra_search_grid(d: int):
    """
    Return (short_name, SearchParams) pairs.

    The default graph build params are already strong; search knobs are the
    main speed/recall trade-off surface we want to expose in this benchmark.
    """
    if not HAS_CAGRA:
        return []
    return [
        ("CAGRA(itopk=32,sw=1)", cagra.SearchParams(itopk_size=32, search_width=1)),
        ("CAGRA(itopk=64,sw=1)", cagra.SearchParams(itopk_size=64, search_width=1)),
        ("CAGRA(itopk=128,sw=2)", cagra.SearchParams(itopk_size=128, search_width=2)),
    ]


def build_cagra_index(X_gpu: cp.ndarray, metric: str = "sqeuclidean"):
    """
    Build a CAGRA index once and reuse it across multiple search parameter sets.
    Returns (index, build_ms).
    """
    if not HAS_CAGRA:
        raise RuntimeError("cuVS CAGRA is not available")

    build_params = cagra.IndexParams(metric=metric)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    index = cagra.build(build_params, X_gpu)
    cp.cuda.Stream.null.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0
    return index, build_ms


def run_cagra_search(index, Q_gpu: cp.ndarray, k: int, search_params, reps: int = 10):
    """Timed CAGRA search. Returns (ids_np, ms)."""
    if not HAS_CAGRA:
        raise RuntimeError("cuVS CAGRA is not available")

    def fn():
        _, neighbors = cagra.search(search_params, index, Q_gpu, k)
        return cp.asarray(neighbors)

    ms = _tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize()
    ids = fn()
    return cp.asnumpy(ids), ms


# ─────────────────────────────────────────────────────────────────────────────
# BF-FP32
# ─────────────────────────────────────────────────────────────────────────────
def run_bf_fp32(Q_gpu, X_gpu, k, reps=10):
    n, d = Q_gpu.shape; m = X_gpu.shape[0]
    B = _choose_block(k, d)
    kern = _compile(_BF32_SRC, 'bf_fp32', dict(K=k, D=d, BLOCK=B))
    oi = cp.zeros((n, k), cp.int32); od = cp.zeros((n, k), cp.float32)
    smem = (d + 2 * B * k) * 4
    def fn(): kern((n,), (B,), (Q_gpu, X_gpu, oi, od, np.int32(m)), shared_mem=smem)
    ms = _tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize(); fn()
    return oi.get(), ms

# ─────────────────────────────────────────────────────────────────────────────
# BF-FP16
# ─────────────────────────────────────────────────────────────────────────────
def run_bf_fp16(Q_gpu, X_fp16, k, reps=10):
    n, d = Q_gpu.shape; m = X_fp16.shape[0]
    B = _choose_block(k, d)
    kern = _compile(_BF16_SRC, 'bf_fp16', dict(K=k, D=d, BLOCK=B))
    oi = cp.zeros((n, k), cp.int32); od = cp.zeros((n, k), cp.float32)
    smem = (d + 2 * B * k) * 4
    def fn(): kern((n,), (B,), (Q_gpu, X_fp16, oi, od, np.int32(m)), shared_mem=smem)
    ms = _tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize(); fn()
    return oi.get(), ms

# ─────────────────────────────────────────────────────────────────────────────
# BF-INT8  (requires d % 4 == 0)
# ─────────────────────────────────────────────────────────────────────────────
def _quantize(arr: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return np.clip(np.round((arr - offset) / scale), -127, 127).astype(np.int8)

def run_bf_int8(Q_gpu, X_gpu, k, reps=10):
    n, d = Q_gpu.shape; m = X_gpu.shape[0]
    assert d % 4 == 0, f"INT8 requires d%4==0, got d={d}"
    X_np = X_gpu.get(); Q_np = Q_gpu.get()

    # Unified scale: center at midpoint so values map to [-127, +127]
    glob_min = min(X_np.min(), Q_np.min())
    glob_max = max(X_np.max(), Q_np.max())
    scale = (glob_max - glob_min) / 254.0 if glob_max > glob_min else 1e-6
    center = (glob_min + glob_max) / 2.0        # = glob_min + 127*scale

    X_q = _quantize(X_np, scale, center)            # (m, d) int8, values in [-127,127]
    Q_q = _quantize(Q_np, scale, center)            # (n, d) int8

    # Correct formula: ||q-x||^2 = scale^2 * ||q_q - x_q||^2
    #                            = scale^2 * (||q_q||^2 + ||x_q||^2 - 2*dot(q_q,x_q))
    # Precompute scale^2 * ||q_q||^2 and scale^2 * ||x_q||^2 (NOT original FP32 norms)
    scale2 = np.float32(scale ** 2)
    x_norms2 = (scale2 * np.sum(X_q.astype(np.float32) ** 2, axis=1)).astype(np.float32)
    q_norms2 = (scale2 * np.sum(Q_q.astype(np.float32) ** 2, axis=1)).astype(np.float32)

    # Reinterpret int8 arrays as int32 (packing 4 int8 per int32 for dp4a)
    X_i32 = X_q.view(np.int32).reshape(m, d // 4)
    Q_i32 = Q_q.view(np.int32).reshape(n, d // 4)

    X_i32_gpu = cp.asarray(X_i32)
    Q_i32_gpu = cp.asarray(Q_i32)
    xn2_gpu   = cp.asarray(x_norms2)
    qn2_gpu   = cp.asarray(q_norms2)

    B = _choose_block(k, d // 4)
    kern = _compile(_BF8_SRC, 'bf_int8', dict(K=k, D=d, BLOCK=B))
    oi = cp.zeros((n, k), cp.int32); od = cp.zeros((n, k), cp.float32)
    # smem: D_Q ints for q_sh + BLOCK*K floats + BLOCK*K ints
    smem = (d // 4 + 2 * B * k) * 4

    def fn():
        kern((n,), (B,),
             (Q_i32_gpu, X_i32_gpu, oi, od, qn2_gpu, xn2_gpu, scale2, np.int32(m)),
             shared_mem=smem)
    ms = _tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize(); fn()
    return oi.get(), ms

# ─────────────────────────────────────────────────────────────────────────────
# BF-GEMM  (tiled cuBLAS QX^T + norm trick + running top-k merge)
# NOTE: cp.matmul uses cuBLAS (fast on Blackwell), but cp.argpartition and
#       cp.concatenate use CuPy elementwise kernels (very slow on SM_120 due to
#       lack of precompiled PTX). Expect ~500-700 ms dominated by CuPy overhead.
# ─────────────────────────────────────────────────────────────────────────────
def run_bf_gemm(Q_gpu, X_gpu, k, tile=50_000, reps=3):
    """dist^2 = ||q||^2 - 2*Q@X^T + ||x||^2, computed in tiles to bound memory."""
    n, d = Q_gpu.shape; m = X_gpu.shape[0]

    x_sq = cp.sum(X_gpu ** 2, axis=1)          # (m,)  precomputed
    q_sq = cp.sum(Q_gpu ** 2, axis=1)[:, None]  # (n,1)

    def fn():
        best_d = cp.full((n, k), cp.inf, dtype=cp.float32)
        best_i = cp.full((n, k), -1, dtype=cp.int32)
        for t0 in range(0, m, tile):
            t1 = min(t0 + tile, m)
            Xt  = X_gpu[t0:t1]
            xst = x_sq[t0:t1]
            dt  = q_sq - 2.0 * (Q_gpu @ Xt.T) + xst[None, :]
            T   = t1 - t0; ki = min(k, T)
            idx_t = cp.argpartition(dt, ki-1, axis=1)[:, :ki]
            dv_t  = dt[cp.arange(n)[:, None], idx_t]
            idx_g = (idx_t + t0).astype(cp.int32)
            cat_d = cp.concatenate([best_d, dv_t], axis=1)
            cat_i = cp.concatenate([best_i, idx_g], axis=1)
            ki2   = min(k, cat_d.shape[1])
            sel   = cp.argpartition(cat_d, ki2-1, axis=1)[:, :ki2]
            best_d = cat_d[cp.arange(n)[:, None], sel]
            best_i = cat_i[cp.arange(n)[:, None], sel]
        return best_i

    ms = _tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize()
    ids = fn()
    return ids.get(), ms

# ─────────────────────────────────────────────────────────────────────────────
# IVF-Flat
# ─────────────────────────────────────────────────────────────────────────────
class IVFIndex:
    def __init__(self, n_clusters: int, nprobe: int):
        self.C = n_clusters
        self.nprobe = nprobe

    def build(self, X_np: np.ndarray, d: int, verbose: bool = False):
        self.d = d
        t0 = time.perf_counter()
        km = MiniBatchKMeans(n_clusters=self.C, n_init=3, random_state=42,
                             batch_size=min(10_000, len(X_np)))
        labels = km.fit_predict(X_np)
        t1 = time.perf_counter()
        self.build_ms = 1000 * (t1 - t0)

        # Sort vectors by cluster → sequential access in search kernel
        order = np.argsort(labels, kind='stable')
        cluster_sizes = np.bincount(labels, minlength=self.C)
        offsets = np.zeros(self.C + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(cluster_sizes)

        X_sorted = X_np[order].astype(np.float32)  # contiguous per cluster

        self.centroids_gpu     = cp.asarray(km.cluster_centers_.astype(np.float32))
        self.X_sorted_gpu      = cp.asarray(X_sorted)
        self.sorted_ids_gpu    = cp.asarray(order.astype(np.int32))
        self.cluster_offsets   = cp.asarray(offsets)
        if verbose:
            print(f"    build {self.build_ms:.0f} ms  "
                  f"avg_cluster={len(X_np)/self.C:.0f}")

    def search(self, Q_gpu: cp.ndarray, k: int, reps: int = 10):
        n = Q_gpu.shape[0]; d = self.d; C = self.C; nprobe = self.nprobe
        B = _choose_block(k, d)

        kern_probe  = _compile(_IVF_PROBE_SRC,  'ivf_probe',
                               dict(D=d, NPROBE=nprobe))
        kern_search = _compile(_IVF_SEARCH_SRC, 'ivf_search',
                               dict(K=k, D=d, NPROBE=nprobe, BLOCK=B))
        smem = (d + 2 * B * k) * 4

        q_probes = cp.zeros((n, nprobe), cp.int32)
        oi       = cp.zeros((n, k),      cp.int32)
        od       = cp.zeros((n, k),      cp.float32)

        def fn():
            kern_probe(
                (math.ceil(n / 128),), (128,),
                (Q_gpu, self.centroids_gpu, q_probes, np.int32(n), np.int32(C))
            )
            kern_search(
                (n,), (B,),
                (Q_gpu, self.X_sorted_gpu, self.sorted_ids_gpu,
                 self.cluster_offsets, q_probes, oi, od, np.int32(n)),
                shared_mem=smem
            )

        ms = _tms(fn, reps=reps)
        cp.cuda.Stream.null.synchronize(); fn()
        return oi.get(), ms

# ─────────────────────────────────────────────────────────────────────────────
# IVF2: 2-phase parallel — one block per (query, cluster), then merge
# Phase 1: n*nprobe blocks each scan 1 cluster → partial top-k
# Phase 2: n blocks each merge nprobe partial top-k arrays
# Goal: maximize GPU occupancy vs IVF1's 1000 long-running blocks
# ─────────────────────────────────────────────────────────────────────────────
_IVF2_PARTIAL_SRC = _HEAP_DEVICE + r"""
#define K_VAL   __K__
#define D_VAL   __D__
#define NPROBE  __NPROBE__
#define BLOCK   __BLOCK__

extern "C" __global__ void ivf2_partial(
    const float* __restrict__ Q,
    const float* __restrict__ X_sorted,
    const int*   __restrict__ sorted_ids,
    const int*   __restrict__ cluster_offsets,
    const int*   __restrict__ q_probes,       /* [n, NPROBE] */
    int*         __restrict__ part_ids,       /* [n*NPROBE, K] */
    float*       __restrict__ part_dists,     /* [n*NPROBE, K] */
    int n)
{
    /* one block per (query, probe) pair */
    int pair = blockIdx.x;
    int qi   = pair / NPROBE;
    int p    = pair % NPROBE;
    if (qi >= n) return;
    int tid  = threadIdx.x;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* sh_d = smem + D_VAL;
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);

    for (int j = tid; j < D_VAL; j += BLOCK)
        q_sh[j] = Q[qi * D_VAL + j];
    __syncthreads();

    int c  = q_probes[qi * NPROBE + p];
    int lo = cluster_offsets[c];
    int hi = cluster_offsets[c + 1];

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0; float lmax = 3.402823e+38f; int lmax_pos = 0;

    for (int ci = lo + tid; ci < hi; ci += BLOCK) {
        const float* xp = X_sorted + (long long)ci * D_VAL;
        float dist = 0.f;
#if D_VAL <= 96
        #pragma unroll
#endif
        for (int di = 0; di < D_VAL; di++) { float dif = q_sh[di]-xp[di]; dist += dif*dif; }
        heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, sorted_ids[ci], K_VAL);
    }
    for (int i = 0; i < K_VAL; i++) { sh_d[tid*K_VAL+i]=ld[i]; sh_i[tid*K_VAL+i]=li[i]; }
    __syncthreads();
    if (tid == 0) {
        float gd[K_VAL]; int gi[K_VAL];
        for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
        int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
        heap_merge(sh_d, sh_i, gd, gi, &gn, &gmax, &gmax_pos, BLOCK, K_VAL);
        int* oi = part_ids   + pair * K_VAL;
        float* od = part_dists + pair * K_VAL;
        for (int i = 0; i < K_VAL; i++) { oi[i]=gi[i]; od[i]=gd[i]; }
    }
}
"""

_IVF2_MERGE_SRC = _HEAP_DEVICE + r"""
#define K_VAL   __K__
#define NPROBE  __NPROBE__
#define BLOCK   256

extern "C" __global__ void ivf2_merge(
    const int*   __restrict__ part_ids,    /* [n*NPROBE, K] */
    const float* __restrict__ part_dists,  /* [n*NPROBE, K] */
    int*         __restrict__ out_ids,     /* [n, K] */
    float*       __restrict__ out_dists,   /* [n, K] */
    int n)
{
    /* one thread per query - merge NPROBE*K candidates into K */
    int qi = blockIdx.x * BLOCK + threadIdx.x;
    if (qi >= n) return;
    float gd[K_VAL]; int gi[K_VAL];
    for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
    int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
    for (int p = 0; p < NPROBE; p++) {
        const int*   src_i = part_ids   + (qi * NPROBE + p) * K_VAL;
        const float* src_d = part_dists + (qi * NPROBE + p) * K_VAL;
        for (int j = 0; j < K_VAL; j++) {
            if (src_i[j] < 0) continue;
            heap_push(gd, gi, &gn, &gmax, &gmax_pos, src_d[j], src_i[j], K_VAL);
        }
    }
    /* sort ascending */
    for (int i = 0; i < K_VAL-1; i++)
        for (int j = i+1; j < K_VAL; j++)
            if (gd[j] < gd[i]) {
                float td=gd[i]; gd[i]=gd[j]; gd[j]=td;
                int   ti=gi[i]; gi[i]=gi[j]; gi[j]=ti;
            }
    int* oi = out_ids   + qi * K_VAL;
    float* od = out_dists + qi * K_VAL;
    for (int i = 0; i < K_VAL; i++) { oi[i]=gi[i]; od[i]=gd[i]; }
}
"""

class IVFIndex2:
    """2-phase IVF: one GPU block per (query, cluster) for better occupancy."""
    def __init__(self, n_clusters: int, nprobe: int):
        self.C = n_clusters; self.nprobe = nprobe

    def build(self, X_np, d):
        """Shared build with IVFIndex."""
        self.d = d
        t0 = time.perf_counter()
        km = MiniBatchKMeans(n_clusters=self.C, n_init=3, random_state=42,
                             batch_size=min(10_000, len(X_np)))
        labels = km.fit_predict(X_np)
        t1 = time.perf_counter()
        self.build_ms = 1000 * (t1 - t0)
        order = np.argsort(labels, kind='stable')
        offsets = np.zeros(self.C + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(labels, minlength=self.C))
        self.centroids_gpu   = cp.asarray(km.cluster_centers_.astype(np.float32))
        self.X_sorted_gpu    = cp.asarray(X_np[order].astype(np.float32))
        self.sorted_ids_gpu  = cp.asarray(order.astype(np.int32))
        self.cluster_offsets = cp.asarray(offsets)

    def search(self, Q_gpu, k, reps=10):
        n = Q_gpu.shape[0]; d = self.d; C = self.C; nprobe = self.nprobe
        B = _choose_block(k, d)

        kern_probe   = _compile(_IVF_PROBE_SRC,    'ivf_probe',
                                dict(D=d, NPROBE=nprobe))
        kern_partial = _compile(_IVF2_PARTIAL_SRC, 'ivf2_partial',
                                dict(K=k, D=d, NPROBE=nprobe, BLOCK=B))
        kern_merge   = _compile(_IVF2_MERGE_SRC,   'ivf2_merge',
                                dict(K=k, NPROBE=nprobe))
        smem = (d + 2 * B * k) * 4

        q_probes   = cp.zeros((n, nprobe),        cp.int32)
        part_ids   = cp.zeros((n * nprobe, k),    cp.int32)
        part_dists = cp.full ((n * nprobe, k), 1e38, cp.float32)
        oi         = cp.zeros((n, k),             cp.int32)
        od         = cp.zeros((n, k),             cp.float32)

        def fn():
            kern_probe(
                (math.ceil(n / 128),), (128,),
                (Q_gpu, self.centroids_gpu, q_probes, np.int32(n), np.int32(C))
            )
            kern_partial(
                (n * nprobe,), (B,),
                (Q_gpu, self.X_sorted_gpu, self.sorted_ids_gpu,
                 self.cluster_offsets, q_probes, part_ids, part_dists, np.int32(n)),
                shared_mem=smem
            )
            kern_merge(
                (math.ceil(n / 256),), (256,),
                (part_ids, part_dists, oi, od, np.int32(n))
            )

        ms = _tms(fn, reps=reps)
        cp.cuda.Stream.null.synchronize(); fn()
        return oi.get(), ms

# ─────────────────────────────────────────────────────────────────────────────
# IVF-INT8: IVF search with INT8-quantized cluster vectors (DP4A distances)
# Quantize X_sorted to INT8; reduces cluster BF traffic by 4x
# ─────────────────────────────────────────────────────────────────────────────
_IVF_INT8_SEARCH_SRC = _HEAP_DEVICE + r"""
#define K_VAL   __K__
#define D_VAL   __D__
#define D_Q     (D_VAL / 4)
#define NPROBE  __NPROBE__
#define BLOCK   __BLOCK__

extern "C" __global__ void ivf_int8_search(
    const int*   __restrict__ Q_i32,
    const int*   __restrict__ X_i32_sorted,   /* [m, D_Q] cluster-ordered */
    const int*   __restrict__ sorted_ids,
    const int*   __restrict__ cluster_offsets,
    const int*   __restrict__ q_probes,
    int*         __restrict__ out_ids,
    float*       __restrict__ out_dists,
    const float* __restrict__ q_norms2,       /* [n]  scale^2*||q_q||^2 */
    const float* __restrict__ x_norms2,       /* [m]  scale^2*||x_q||^2 */
    float        scale2,
    int n)
{
    int qi = blockIdx.x, tid = threadIdx.x;
    extern __shared__ int smemi[];
    int*   q_sh = smemi;
    float* sh_d = (float*)(smemi + D_Q);
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);

    for (int j = tid; j < D_Q; j += BLOCK)
        q_sh[j] = Q_i32[qi * D_Q + j];
    __syncthreads();
    float qn2 = q_norms2[qi];

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0; float lmax = 3.402823e+38f; int lmax_pos = 0;

    for (int p = 0; p < NPROBE; p++) {
        int c  = q_probes[qi * NPROBE + p];
        int lo = cluster_offsets[c];
        int hi = cluster_offsets[c + 1];
        for (int ci = lo + tid; ci < hi; ci += BLOCK) {
            const int* xp = X_i32_sorted + (long long)ci * D_Q;
            int dot = 0;
#if D_Q <= 96
            #pragma unroll
#endif
            for (int qi4 = 0; qi4 < D_Q; qi4++)
                dot = __dp4a(q_sh[qi4], xp[qi4], dot);
            float dist = qn2 + x_norms2[ci] - 2.f * dot * scale2;
            heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, sorted_ids[ci], K_VAL);
        }
    }
    for (int i = 0; i < K_VAL; i++) { sh_d[tid*K_VAL+i]=ld[i]; sh_i[tid*K_VAL+i]=li[i]; }
    __syncthreads();
    if (tid == 0) {
        float gd[K_VAL]; int gi[K_VAL];
        for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
        int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
        heap_merge(sh_d, sh_i, gd, gi, &gn, &gmax, &gmax_pos, BLOCK, K_VAL);
        for (int i = 0; i < K_VAL; i++) { out_ids[qi*K_VAL+i]=gi[i]; out_dists[qi*K_VAL+i]=gd[i]; }
    }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Product Quantization kernels
# Splits d-dim vectors into M subspaces of KSUB=d/M dims.
# Encodes each subspace as 1 byte (index into 256-entry codebook).
# X_pq: m × M bytes  (for d=128, M=8: only 4 MB vs 256 MB for FP32!)
#
# dist_table[qi, s, c] = ||q[qi, s*KSUB:(s+1)*KSUB] - codebook[s, c]||^2
# dist(q, x) ≈ sum_s dist_table[qi, s, X_pq[xi, s]]
# ─────────────────────────────────────────────────────────────────────────────
_PQ_DIST_TABLE_SRC = r"""
#define D_VAL    __D__
#define M_VAL    __M__
#define KSUB_VAL (D_VAL / M_VAL)

extern "C" __global__ void pq_dist_table(
    const float* __restrict__ Q,          /* [n, D_VAL] */
    const float* __restrict__ codebooks,  /* [M_VAL, 256, KSUB_VAL] */
    float*       __restrict__ dist_tables,/* [n, M_VAL, 256] */
    int n)
{
    int qi = blockIdx.x;
    int s  = blockIdx.y;
    int c  = threadIdx.x;   /* 0 .. 255 */
    if (qi >= n) return;

    const float* qp = Q         + qi * D_VAL        + s * KSUB_VAL;
    const float* cp = codebooks + (s * 256 + c) * KSUB_VAL;
    float dist = 0.f;
#if KSUB_VAL <= 96
    #pragma unroll
#endif
    for (int i = 0; i < KSUB_VAL; i++) { float dif = qp[i]-cp[i]; dist += dif*dif; }
    dist_tables[qi * M_VAL * 256 + s * 256 + c] = dist;
}
"""

# BF-PQ: scan all m PQ codes, use shared-memory dist-table lookup per vector
_PQ_SEARCH_SRC = _HEAP_DEVICE + r"""
#define K_VAL   __K__
#define M_VAL   __M__
#define BLOCK   __BLOCK__

extern "C" __global__ void pq_search(
    const unsigned char* __restrict__ X_pq,      /* [m, M_VAL] uint8 */
    const float*         __restrict__ dist_tables,/* [n, M_VAL*256]   */
    int*   __restrict__ out_ids,
    float* __restrict__ out_dists,
    int m)
{
    int qi = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float smem[];
    float* sh_dt = smem;                       /* [M_VAL * 256] - dist table */
    float* sh_d  = sh_dt + M_VAL * 256;
    int*   sh_i  = (int*)(sh_d + BLOCK * K_VAL);

    /* load this query's dist table into shared memory */
    const float* dt = dist_tables + (long long)qi * M_VAL * 256;
    for (int i = tid; i < M_VAL * 256; i += BLOCK) sh_dt[i] = dt[i];
    __syncthreads();

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i]=3.402823e+38f; li[i]=-1; }
    int ln=0; float lmax=3.402823e+38f; int lmax_pos=0;

    for (int ci = tid; ci < m; ci += BLOCK) {
        const unsigned char* xp = X_pq + (long long)ci * M_VAL;
        float dist = 0.f;
        #pragma unroll
        for (int s = 0; s < M_VAL; s++)
            dist += sh_dt[s * 256 + (int)xp[s]];
        heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, ci, K_VAL);
    }
    for (int i = 0; i < K_VAL; i++) { sh_d[tid*K_VAL+i]=ld[i]; sh_i[tid*K_VAL+i]=li[i]; }
    __syncthreads();
    if (tid == 0) {
        float gd[K_VAL]; int gi[K_VAL];
        for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
        int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
        heap_merge(sh_d, sh_i, gd, gi, &gn, &gmax, &gmax_pos, BLOCK, K_VAL);
        for (int i = 0; i < K_VAL; i++) { out_ids[qi*K_VAL+i]=gi[i]; out_dists[qi*K_VAL+i]=gd[i]; }
    }
}
"""

# IVF-PQ: probe p clusters, scan their PQ codes with dist-table lookup
_IVFPQ_SEARCH_SRC = _HEAP_DEVICE + r"""
#define K_VAL   __K__
#define M_VAL   __M__
#define NPROBE  __NPROBE__
#define BLOCK   __BLOCK__

extern "C" __global__ void ivfpq_search(
    const unsigned char* __restrict__ X_pq_sorted,  /* [m, M_VAL] cluster-ordered */
    const int*           __restrict__ sorted_ids,
    const int*           __restrict__ cluster_offsets,
    const int*           __restrict__ q_probes,      /* [n, NPROBE] */
    const float*         __restrict__ dist_tables,   /* [n, M_VAL*256] */
    int*   __restrict__ out_ids,
    float* __restrict__ out_dists,
    int n)
{
    int qi = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float smem[];
    float* sh_dt = smem;
    float* sh_d  = sh_dt + M_VAL * 256;
    int*   sh_i  = (int*)(sh_d + BLOCK * K_VAL);

    const float* dt = dist_tables + (long long)qi * M_VAL * 256;
    for (int i = tid; i < M_VAL * 256; i += BLOCK) sh_dt[i] = dt[i];
    __syncthreads();

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i]=3.402823e+38f; li[i]=-1; }
    int ln=0; float lmax=3.402823e+38f; int lmax_pos=0;

    for (int p = 0; p < NPROBE; p++) {
        int c  = q_probes[qi * NPROBE + p];
        int lo = cluster_offsets[c];
        int hi = cluster_offsets[c+1];
        for (int ci = lo + tid; ci < hi; ci += BLOCK) {
            const unsigned char* xp = X_pq_sorted + (long long)ci * M_VAL;
            float dist = 0.f;
            #pragma unroll
            for (int s = 0; s < M_VAL; s++)
                dist += sh_dt[s * 256 + (int)xp[s]];
            heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, sorted_ids[ci], K_VAL);
        }
    }
    for (int i = 0; i < K_VAL; i++) { sh_d[tid*K_VAL+i]=ld[i]; sh_i[tid*K_VAL+i]=li[i]; }
    __syncthreads();
    if (tid == 0) {
        float gd[K_VAL]; int gi[K_VAL];
        for (int i = 0; i < K_VAL; i++) { gd[i]=3.402823e+38f; gi[i]=-1; }
        int gn=0; float gmax=3.402823e+38f; int gmax_pos=0;
        heap_merge(sh_d, sh_i, gd, gi, &gn, &gmax, &gmax_pos, BLOCK, K_VAL);
        for (int i = 0; i < K_VAL; i++) { out_ids[qi*K_VAL+i]=gi[i]; out_dists[qi*K_VAL+i]=gd[i]; }
    }
}
"""

class IVFIndexINT8:
    """IVF-Flat with INT8-quantized cluster data (4× less L2 traffic per probe)."""
    def __init__(self, n_clusters: int, nprobe: int):
        self.C = n_clusters; self.nprobe = nprobe

    def build(self, X_np, d):
        assert d % 4 == 0
        self.d = d
        t0 = time.perf_counter()
        km = MiniBatchKMeans(n_clusters=self.C, n_init=3, random_state=42,
                             batch_size=min(10_000, len(X_np)))
        labels = km.fit_predict(X_np)
        t1 = time.perf_counter()
        self.build_ms = 1000 * (t1 - t0)
        order = np.argsort(labels, kind='stable')
        offsets = np.zeros(self.C + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(labels, minlength=self.C))

        X_sorted = X_np[order].astype(np.float32)

        # Quantize X_sorted to INT8 with centered scale
        glob_min, glob_max = X_sorted.min(), X_sorted.max()
        scale = (glob_max - glob_min) / 254.0 if glob_max > glob_min else 1e-6
        center = (glob_min + glob_max) / 2.0
        self._scale = scale; self._center = center

        X_q = _quantize(X_sorted, scale, center)          # (m, d) int8
        self._scale2 = np.float32(scale ** 2)
        x_norms2 = (self._scale2 * np.sum(X_q.astype(np.float32)**2, axis=1)).astype(np.float32)

        self.centroids_gpu      = cp.asarray(km.cluster_centers_.astype(np.float32))
        self.X_i32_sorted_gpu   = cp.asarray(X_q.view(np.int32).reshape(-1, d // 4))
        self.sorted_ids_gpu     = cp.asarray(order.astype(np.int32))
        self.cluster_offsets    = cp.asarray(offsets)
        self.x_norms2_gpu       = cp.asarray(x_norms2)

    def search(self, Q_gpu, k, reps=10):
        n = Q_gpu.shape[0]; d = self.d; C = self.C; nprobe = self.nprobe
        B = _choose_block(k, d // 4)

        # Quantize queries
        Q_np = Q_gpu.get()
        Q_q = _quantize(Q_np, self._scale, self._center)
        q_norms2 = (self._scale2 * np.sum(Q_q.astype(np.float32)**2, axis=1)).astype(np.float32)
        Q_i32_gpu  = cp.asarray(Q_q.view(np.int32).reshape(n, d // 4))
        qn2_gpu    = cp.asarray(q_norms2)

        kern_probe  = _compile(_IVF_PROBE_SRC,       'ivf_probe',
                               dict(D=d, NPROBE=nprobe))
        kern_search = _compile(_IVF_INT8_SEARCH_SRC, 'ivf_int8_search',
                               dict(K=k, D=d, NPROBE=nprobe, BLOCK=B))
        smem = (d // 4 + 2 * B * k) * 4

        q_probes = cp.zeros((n, nprobe), cp.int32)
        oi       = cp.zeros((n, k),      cp.int32)
        od       = cp.zeros((n, k),      cp.float32)

        def fn():
            kern_probe(
                (math.ceil(n / 128),), (128,),
                (Q_gpu, self.centroids_gpu, q_probes, np.int32(n), np.int32(C))
            )
            kern_search(
                (n,), (B,),
                (Q_i32_gpu, self.X_i32_sorted_gpu, self.sorted_ids_gpu,
                 self.cluster_offsets, q_probes, oi, od,
                 qn2_gpu, self.x_norms2_gpu, self._scale2, np.int32(n)),
                shared_mem=smem
            )

        ms = _tms(fn, reps=reps)
        cp.cuda.Stream.null.synchronize(); fn()
        return oi.get(), ms

# ─────────────────────────────────────────────────────────────────────────────
# PQ Index: BF scan with PQ codes (d=128 → X_pq only 4 MB → L2-resident!)
# ─────────────────────────────────────────────────────────────────────────────
class PQIndex:
    """Product Quantization BF scan.
    Build: M CPU K-means (256 centers each). Search: GPU dist-table + scan.
    For d=128, M=8: X_pq = 4 MB — fits entirely in L2 → fast even for m=500k.
    """
    def __init__(self, M: int = 8):
        self.M = M

    def build(self, X_np: np.ndarray, d: int):
        assert d % self.M == 0, f"d={d} must be divisible by M={self.M}"
        self.d = d
        KSUB = d // self.M
        self._KSUB = KSUB

        t0 = time.perf_counter()
        codebooks = np.zeros((self.M, 256, KSUB), dtype=np.float32)
        codes     = np.zeros((len(X_np), self.M), dtype=np.uint8)
        for s in range(self.M):
            X_sub = X_np[:, s*KSUB:(s+1)*KSUB].copy()
            km = MiniBatchKMeans(n_clusters=256, n_init=3, random_state=42,
                                 batch_size=min(10_000, len(X_np)))
            labels = km.fit_predict(X_sub)
            codebooks[s] = km.cluster_centers_.astype(np.float32)
            codes[:, s]  = labels.astype(np.uint8)
        t1 = time.perf_counter()
        self.build_ms = 1000 * (t1 - t0)

        self.codebooks_gpu = cp.asarray(codebooks)  # (M, 256, KSUB) contiguous
        self.X_pq_gpu      = cp.asarray(codes)       # (m, M) uint8

    def search(self, Q_gpu: cp.ndarray, k: int, reps: int = 10):
        n = Q_gpu.shape[0]; M = self.M; d = self.d; KSUB = self._KSUB
        m = self.X_pq_gpu.shape[0]

        kern_dt  = _compile(_PQ_DIST_TABLE_SRC, 'pq_dist_table',
                            dict(D=d, M=M))
        kern_srch = _compile(_PQ_SEARCH_SRC, 'pq_search',
                             dict(K=k, M=M, BLOCK=128))

        dt_gpu = cp.zeros((n, M * 256), cp.float32)
        oi     = cp.zeros((n, k), cp.int32)
        od     = cp.zeros((n, k), cp.float32)
        smem   = (M * 256 + 2 * 128 * k) * 4   # dist_table + heap

        def fn():
            kern_dt(
                (n, M), (256,),
                (Q_gpu, self.codebooks_gpu, dt_gpu, np.int32(n))
            )
            kern_srch(
                (n,), (128,),
                (self.X_pq_gpu, dt_gpu, oi, od, np.int32(m)),
                shared_mem=smem
            )

        ms = _tms(fn, reps=reps)
        cp.cuda.Stream.null.synchronize(); fn()
        return oi.get(), ms


# ─────────────────────────────────────────────────────────────────────────────
# IVF-PQ Index: IVF partitioning + PQ scan within probed clusters
# Best of both worlds: reduces #vectors scanned (IVF) AND bytes per vector (PQ)
# ─────────────────────────────────────────────────────────────────────────────
class IVFPQIndex:
    """IVF + Product Quantization.
    X_pq_sorted: m × M bytes, cluster-ordered.
    For d=128, C=256, M=8: X_pq = 4 MB total, each cluster's PQ = 15 KB.
    All PQ codes fit in L2 → scan is L2-bound even in HBM regime.
    """
    def __init__(self, n_clusters: int, nprobe: int, M: int = 8):
        self.C = n_clusters; self.nprobe = nprobe; self.M = M

    def build(self, X_np: np.ndarray, d: int):
        assert d % self.M == 0
        self.d = d; KSUB = d // self.M; self._KSUB = KSUB

        t0 = time.perf_counter()
        # IVF assignment
        km_ivf = MiniBatchKMeans(n_clusters=self.C, n_init=3, random_state=42,
                                 batch_size=min(10_000, len(X_np)))
        labels = km_ivf.fit_predict(X_np)
        order  = np.argsort(labels, kind='stable')
        offsets = np.zeros(self.C + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(labels, minlength=self.C))
        X_sorted = X_np[order]

        # PQ encoding on sorted X
        codebooks = np.zeros((self.M, 256, KSUB), dtype=np.float32)
        codes     = np.zeros((len(X_np), self.M), dtype=np.uint8)
        for s in range(self.M):
            X_sub = X_sorted[:, s*KSUB:(s+1)*KSUB].copy()
            km_pq = MiniBatchKMeans(n_clusters=256, n_init=3, random_state=42,
                                    batch_size=min(10_000, len(X_np)))
            lbl = km_pq.fit_predict(X_sub)
            codebooks[s] = km_pq.cluster_centers_.astype(np.float32)
            codes[:, s]  = lbl.astype(np.uint8)
        t1 = time.perf_counter()
        self.build_ms = 1000 * (t1 - t0)

        self.centroids_gpu      = cp.asarray(km_ivf.cluster_centers_.astype(np.float32))
        self.X_pq_sorted_gpu    = cp.asarray(codes)
        self.codebooks_gpu      = cp.asarray(codebooks)
        self.sorted_ids_gpu     = cp.asarray(order.astype(np.int32))
        self.cluster_offsets    = cp.asarray(offsets)

    def search(self, Q_gpu: cp.ndarray, k: int, reps: int = 10):
        n = Q_gpu.shape[0]; M = self.M; d = self.d; KSUB = self._KSUB
        C = self.C; nprobe = self.nprobe

        kern_probe  = _compile(_IVF_PROBE_SRC,    'ivf_probe',
                               dict(D=d, NPROBE=nprobe))
        kern_dt     = _compile(_PQ_DIST_TABLE_SRC, 'pq_dist_table',
                               dict(D=d, M=M))
        kern_search = _compile(_IVFPQ_SEARCH_SRC,  'ivfpq_search',
                               dict(K=k, M=M, NPROBE=nprobe, BLOCK=128))

        smem = (M * 256 + 2 * 128 * k) * 4
        q_probes = cp.zeros((n, nprobe), cp.int32)
        dt_gpu   = cp.zeros((n, M * 256), cp.float32)
        oi       = cp.zeros((n, k), cp.int32)
        od       = cp.zeros((n, k), cp.float32)

        def fn():
            kern_probe(
                (math.ceil(n / 128),), (128,),
                (Q_gpu, self.centroids_gpu, q_probes, np.int32(n), np.int32(C))
            )
            kern_dt(
                (n, M), (256,),
                (Q_gpu, self.codebooks_gpu, dt_gpu, np.int32(n))
            )
            kern_search(
                (n,), (128,),
                (self.X_pq_sorted_gpu, self.sorted_ids_gpu,
                 self.cluster_offsets, q_probes, dt_gpu, oi, od, np.int32(n)),
                shared_mem=smem
            )

        ms = _tms(fn, reps=reps)
        cp.cuda.Stream.null.synchronize(); fn()
        return oi.get(), ms


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical AM model
# ─────────────────────────────────────────────────────────────────────────────
def am_model(m, d, n, k,
             bw_l2_theo=10_000,   # GB/s  — RTX 5090 theoretical L2 peak
             bw_l2_meas=3_900,    # GB/s  — measured effective BW (updated from run)
             bw_hbm=1_792,        # GB/s  — RTX 5090 HBM3e peak
             peak_fp32=104.8,     # TFLOP/s
             tile=50_000,
             l2_size=96,          # MB — L2 cache size
             M_pq=8):             # PQ subspaces (for PQ entries)
    """Return (table dict, ridge_l2_theo) for each algorithm.
    Distinguishes L2-resident vs HBM-bound regimes based on data size.
    """
    ridge_l2  = peak_fp32 * 1e3 / bw_l2_theo
    ridge_hbm = peak_fp32 * 1e3 / bw_hbm
    flops = n * m * d * 2
    l2_bytes = l2_size * 1e6

    def entry(name, total_traffic_bytes, dtype_bytes, note='', force_hbm=False):
        # If database footprint exceeds L2, scans are HBM-bound
        db_bytes = m * d * dtype_bytes
        if force_hbm or db_bytes > l2_bytes:
            bw_eff   = bw_hbm
            mem_tag  = 'HBM-MEM'
        else:
            bw_eff   = bw_l2_meas
            mem_tag  = 'L2-MEM'
        ai       = flops / total_traffic_bytes if total_traffic_bytes > 0 else float('inf')
        ridge    = ridge_hbm if (force_hbm or db_bytes > l2_bytes) else ridge_l2
        bound    = 'COMPUTE' if ai > ridge else mem_tag
        ms_theo  = total_traffic_bytes / (bw_l2_theo * 1e9) * 1e3
        ms_meas  = total_traffic_bytes / (bw_eff     * 1e9) * 1e3
        return dict(name=name, ai=ai, l2_GB=total_traffic_bytes/1e9,
                    pred_ms_theo=ms_theo, pred_ms_meas=ms_meas,
                    bound=bound, note=note,
                    data_MB=db_bytes/1e6)

    rows = {}
    rows['BF-FP32'] = entry('BF-FP32', n*m*d*4, 4)
    rows['BF-FP16'] = entry('BF-FP16', n*m*d*2, 2, 'X in FP16')
    rows['BF-INT8'] = entry('BF-INT8', n*m*d*1, 1, 'X in INT8, DP4A')

    # PQ: X_pq = m*M_pq bytes; if < L2, it is L2-resident across all n queries
    pq_db_bytes  = m * M_pq          # total PQ code storage
    pq_traffic   = n * pq_db_bytes   # n queries × m codes × M bytes each
    pq_force_hbm = pq_db_bytes > l2_bytes
    rows['BF-PQ'] = entry('BF-PQ', pq_traffic, M_pq,
                          f'M={M_pq} subspaces, X_pq={pq_db_bytes/1e6:.1f}MB',
                          force_hbm=pq_force_hbm)

    tiles    = math.ceil(m / tile)
    gemm_byt = n*m*d*4 + tiles*(n*tile*4)*2
    rows['BF-GEMM'] = entry('BF-GEMM', gemm_byt, 4, 'cuBLAS+CuPy topk (CuPy slow!)')

    for C in [32, 64, 128, 256, 512]:
        for p in [1, 2, 4, 8, 16, 32]:
            if p >= C: continue
            frac = p / C
            name = f'IVF(C={C},p={p})'
            scan_bytes = int(n * frac * m * d * 4)
            rows[name] = entry(name, scan_bytes, 4, f'recall≈f(data,C)')
            rows[name]['frac'] = frac

    # IVF-PQ entries: scan = n * p/C * m * M_pq bytes
    # Cluster PQ codes often L2-resident: cluster_pq = (m/C)*M_pq bytes
    for C in [64, 128, 256]:
        for p in [4, 8, 16]:
            if p >= C: continue
            frac = p / C
            name = f'IVF-PQ(C={C},p={p})'
            cluster_pq_bytes = (m // C) * M_pq
            all_pq_bytes     = m * M_pq
            # If all PQ codes fit in L2, scans are L2-bound
            force_hbm_pq = all_pq_bytes > l2_bytes
            scan_bytes = int(n * frac * m * M_pq)
            rows[name] = entry(name, scan_bytes, M_pq,
                               f'M={M_pq}, cluster_pq={cluster_pq_bytes/1024:.0f}KB',
                               force_hbm=force_hbm_pq)
            rows[name]['frac'] = frac

    return rows, ridge_l2


_ALG_PATTERN = re.compile(r'^(?P<family>[A-Z0-9\-]+)(?:\((?P<params>[^)]*)\))?$')


def _parse_algo_name(name: str) -> dict:
    m = _ALG_PATTERN.match(name)
    out = {"family": name, "params": {}}
    if not m:
        return out
    out["family"] = m.group("family")
    params = {}
    raw = m.group("params")
    if raw:
        for item in raw.split(","):
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                if "." in v:
                    params[k] = float(v)
                else:
                    params[k] = int(v)
            except ValueError:
                params[k] = v
    out["params"] = params
    return out


def _fit_affine_time_model(points: list[tuple[float, float]]) -> Optional[dict]:
    """Fit ms ~= intercept_ms + slope_ms_per_GB * traffic_GB via least squares."""
    if len(points) < 2:
        return None
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    A = np.stack([np.ones_like(xs), xs], axis=1)
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    intercept_ms = float(coef[0])
    slope_ms_per_GB = float(coef[1])
    yhat = A @ coef
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    bw_gbs = float("inf") if slope_ms_per_GB <= 0 else 1000.0 / slope_ms_per_GB
    return {
        "n_points": len(points),
        "intercept_ms": intercept_ms,
        "slope_ms_per_GB": slope_ms_per_GB,
        "bw_gbs": bw_gbs,
        "r2": r2,
    }


def _summarize_family_fits(results: list[tuple[str, float, float, float, Optional[float]]]) -> list[dict]:
    """Fit traffic->time models for families that have multiple scan fractions."""
    families = {"IVF1": [], "IVF2": [], "IVF-INT8": [], "IVF-PQ": []}
    for name, ms, _rc, _data_mb, l2_gb in results:
        if l2_gb is None:
            continue
        meta = _parse_algo_name(name)
        fam = meta["family"]
        if fam in families:
            families[fam].append((float(l2_gb), float(ms), meta["params"]))

    out = []
    for fam, pts in families.items():
        fit = _fit_affine_time_model([(gb, ms) for gb, ms, _ in pts])
        if fit is None:
            continue
        fit["family"] = fam
        fit["points"] = [
            {"traffic_GB": gb, "actual_ms": ms, **params}
            for gb, ms, params in pts
        ]
        out.append(fit)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Bandwidth measurement
# ─────────────────────────────────────────────────────────────────────────────
def measure_bw(size_MB=16) -> float:
    """Estimate effective memory bandwidth using a small (L2-resident) array."""
    n = size_MB * 1024 * 1024 // 4
    a = cp.ones(n, dtype=cp.float32)
    b = cp.zeros(n, dtype=cp.float32)
    for _ in range(5): b[:] = a[:]
    cp.cuda.Stream.null.synchronize()
    ev0 = cp.cuda.Event(); ev1 = cp.cuda.Event()
    ev0.record()
    for _ in range(20): b[:] = a[:]
    ev1.record(); ev1.synchronize()
    ms = float(cp.cuda.get_elapsed_time(ev0, ev1)) / 20
    return (n * 4 * 2) / (ms * 1e-3) / 1e9  # GB/s (read + write)

# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark driver (synthetic or ann-benchmarks HDF5 via explore_annb.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark_suite(
    args,
    X_np: np.ndarray,
    Q_np: np.ndarray,
    gt_np: Optional[np.ndarray] = None,
    skip_dsweep: bool = False,
    subtitle: str = "",
):
    """
    Run all explore benchmarks.  X_np: (m,d), Q_np: (n,d), gt_np: (n,k) int (optional).
    If gt_np is None, ground truth is computed on GPU (expensive for large m).
    """
    m, d = X_np.shape
    n, d2 = Q_np.shape
    assert d2 == d, f"Q dim {d2} != X dim {d}"
    if n < 1:
        raise ValueError("Need at least one query (Q_np is empty). Check ann-benchmarks filtering / --max-test.")
    k = args.k

    print("=" * 80)
    line = f"  ANN Exploration  |  m={m:,}  d={d}  n={n}  k={k}"
    if subtitle:
        line += f"  |  {subtitle}"
    print(line)
    print(f"  X (FP32):{m*d*4/1e6:.1f} MB  (FP16):{m*d*2/1e6:.1f} MB  "
          f"(INT8):{m*d/1e6:.1f} MB  |  L2=96 MB → X {'FITS' if m*d*4<96e6 else 'DOES NOT FIT'} in L2")
    print("=" * 80)

    X_gpu  = cp.asarray(X_np)
    Q_gpu  = cp.asarray(Q_np)
    X_fp16 = X_gpu.astype(cp.float16)

    bw = measure_bw()
    print(f"\n  Measured effective BW (16 MB L2 array): {bw:.0f} GB/s\n")

    if gt_np is not None:
        gt = np.ascontiguousarray(gt_np[:, :k], dtype=np.int64)
        assert gt.shape == (n, k), f"gt shape {gt.shape} != ({n},{k})"
        print("  Ground truth: from dataset (precomputed neighbors)\n")
    else:
        print("  Computing ground truth ...")
        gt = ground_truth(Q_gpu, X_gpu, k)

    # ── JIT warm-up ──────────────────────────────────────────────────────────
    print("  JIT warm-up ...")
    run_bf_fp32(Q_gpu, X_gpu, k, reps=1)
    run_bf_fp16(Q_gpu, X_fp16, k, reps=1)
    if d % 4 == 0:
        run_bf_int8(Q_gpu, X_gpu, k, reps=1)

    # ── Benchmarks ───────────────────────────────────────────────────────────
    results = []   # (name, ms, recall, data_MB_X, l2_GB)

    def bench(name, fn, data_MB, l2_GB):
        ids, ms = fn()
        rc = recall_at_k(ids, gt, k)
        results.append((name, ms, rc, data_MB, l2_GB))
        print(f"  {name:<24} {ms:7.2f} ms   recall={rc:.4f}")

    print("\n  ── Benchmarks ──────────────────────────────────────────────────")
    bench('BF-FP32',
          lambda: run_bf_fp32(Q_gpu, X_gpu, k),
          m*d*4/1e6, n*m*d*4/1e9)

    bench('BF-FP16',
          lambda: run_bf_fp16(Q_gpu, X_fp16, k),
          m*d*2/1e6, n*m*d*2/1e9)

    if d % 4 == 0:
        bench('BF-INT8',
              lambda: run_bf_int8(Q_gpu, X_gpu, k),
              m*d*1/1e6, n*m*d*1/1e9)
    else:
        print(f"  BF-INT8 skipped (d={d} not divisible by 4)")

    print("  BF-GEMM running (tiled cuBLAS) ...")
    bench('BF-GEMM',
          lambda: run_bf_gemm(Q_gpu, X_gpu, k),
          m*d*4/1e6, None)

    if HAS_CAGRA and not getattr(args, "no_cagra", False):
        print(f"\n  ── CAGRA (cuVS graph ANN) {'─'*39}")
        try:
            cagra_index, cagra_build_ms = build_cagra_index(X_gpu)
            print(f"  CAGRA build {cagra_build_ms:7.0f} ms")
            for cagra_name, cagra_params in cagra_search_grid(d):
                ids_c, ms_c = run_cagra_search(cagra_index, Q_gpu, k, cagra_params, reps=10)
                rc_c = recall_at_k(ids_c, gt, k)
                results.append((cagra_name, ms_c, rc_c, m * d * 4 / 1e6, None))
                print(f"  {cagra_name:<30} {ms_c:7.2f} ms  recall={rc_c:.4f}")
        except Exception as e:
            print(f"  CAGRA FAILED: {e}")
    elif not HAS_CAGRA:
        print("\n  CAGRA skipped (install cuvs)")
    else:
        print("\n  CAGRA skipped (--no-cagra)")

    # ── LSH (FastLSHIndex) ────────────────────────────────────────────────────
    if HAS_LSH and not args.no_lsh:
        print(f"\n  ── LSH Fast (E2LSH + CUDA stage7 + fused rerank, os={args.lsh_oversample}) {'─'*12}")
        for lsh_name, lsh_par in lsh_config_grid(d, args.seed):
            try:
                ids_l, ms_l, bms_l = bench_lsh_fast(
                    X_gpu, Q_gpu, k, lsh_par,
                    oversample=args.lsh_oversample, reps=20)
                rc_l = recall_at_k(ids_l, gt, k)
                Ltab = lsh_par.n_tables
                mc = lsh_par.max_cands_per_table
                rerank_reads_gb = n * Ltab * mc * 4 / 1e9
                results.append((lsh_name, ms_l, rc_l, m * d * 4 / 1e6, rerank_reads_gb))
                print(f"  {lsh_name:<30} {ms_l:7.2f} ms  "
                      f"build={bms_l:8.0f} ms  recall={rc_l:.4f}")
            except Exception as e:
                print(f"  {lsh_name:<30} FAILED: {e}")
    elif not HAS_LSH:
        print("\n  LSH skipped (import lsh.py / lsh_fast.py from same directory)")
    else:
        print("\n  LSH skipped (--no-lsh)")

    # ── IVF sweep ─────────────────────────────────────────────────────────────
    if HAS_SKLEARN:
        ivf_configs = [
            (64,  4), (64,  8), (64,  16),
            (128, 4), (128, 8), (128, 16),
            (256, 8), (256, 16),
        ]
        print(f"\n  ── IVF1 (1 block/query, sequential cluster scan) {'─'*20}")
        for C, p in ivf_configs:
            ivf = IVFIndex(C, p)
            ivf.build(X_np, d)
            frac = p / C
            bench(f'IVF1(C={C},p={p})',
                  lambda ivf=ivf: ivf.search(Q_gpu, k),
                  frac*m*d*4/1e6, n*frac*m*d*4/1e9)

        print(f"\n  ── IVF2 (1 block/(query,cluster), parallel) {'─'*24}")
        ivf2_configs = [(64, 8), (128, 8), (256, 8), (128, 16), (256, 16)]
        for C, p in ivf2_configs:
            ivf2 = IVFIndex2(C, p)
            ivf2.build(X_np, d)
            frac = p / C
            bench(f'IVF2(C={C},p={p})',
                  lambda ivf2=ivf2: ivf2.search(Q_gpu, k),
                  frac*m*d*4/1e6, n*frac*m*d*4/1e9)

        if d % 4 == 0:
            print(f"\n  ── IVF-INT8 (DP4A distances, 4× less traffic) {'─'*20}")
            ivf8_configs = [(64, 8), (64, 16), (128, 8), (128, 16), (256, 16)]
            for C, p in ivf8_configs:
                ivf8 = IVFIndexINT8(C, p)
                ivf8.build(X_np, d)
                frac = p / C
                bench(f'IVF-INT8(C={C},p={p})',
                      lambda ivf8=ivf8: ivf8.search(Q_gpu, k),
                      frac*m*d*1/1e6, n*frac*m*d*1/1e9)

        # ── PQ and IVF-PQ (critical for large d where X > L2) ─────────────────
        if d % 8 == 0 and not getattr(args, "skip_pq", False):
            M_pq = 8 if d >= 32 else max(1, d // 4)
            xpq_mb = m * M_pq / 1e6
            print(f"\n  ── BF-PQ (M={M_pq} subspaces, X_pq={xpq_mb:.1f} MB) {'─'*20}")
            pq = PQIndex(M=M_pq)
            print(f"    Building PQ ({M_pq} × K-means256)...")
            pq.build(X_np, d)
            print(f"    Build: {pq.build_ms:.0f} ms")
            bench(f'BF-PQ(M={M_pq})',
                  lambda pq=pq: pq.search(Q_gpu, k),
                  xpq_mb, n*m*M_pq/1e9)

            print(f"\n  ── IVF-PQ (IVF cluster selection + PQ scan) {'─'*20}")
            ivfpq_configs = [
                (64,  8), (64,  16),
                (128, 8), (128, 16),
                (256, 8), (256, 16),
            ]
            for C, p in ivfpq_configs:
                ivfpq = IVFPQIndex(C, p, M=M_pq)
                print(f"    Building IVF-PQ(C={C},p={p})...")
                ivfpq.build(X_np, d)
                frac = p / C
                bench(f'IVF-PQ(C={C},p={p})',
                      lambda ivfpq=ivfpq: ivfpq.search(Q_gpu, k),
                      frac*xpq_mb, n*frac*m*M_pq/1e9)
        elif d % 8 == 0:
            print("\n  PQ / IVF-PQ skipped (--skip-pq)")
    else:
        print("\n  IVF/PQ skipped (sklearn not available)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "─" * 85)
    print(f"  {'algorithm':<24} {'ms':>7} {'recall':>8} {'X-data':>8} {'L2-traffic':>12} {'AI':>6}  note")
    print("  " + "─" * 80)
    for name, ms, rc, data_MB, l2_GB in results:
        if l2_GB is not None:
            flops = n * m * d * 2
            ai_str = f"{flops / (l2_GB * 1e9):.2f}"
            l2_str = f"{l2_GB:.1f} GB"
        else:
            ai_str = "?"; l2_str = "tiled"
        print(f"  {name:<24} {ms:>7.2f} {rc:>8.4f} {data_MB:>6.1f}MB {l2_str:>12} {ai_str:>6} F/B")

    # ── Theoretical + corrected AM model ─────────────────────────────────────
    M_pq_am = 8 if d >= 32 else max(1, d // 4)
    print("\n" + "=" * 95)
    print("  THEORETICAL AM MODEL  (RTX 5090)")
    print(f"  X={m*d*4/1e6:.0f}MB (FP32) | L2={96}MB | X {'FITS' if m*d*4<96e6 else 'DOES NOT FIT'} in L2")
    print(f"  L2 BW (theoretical): 10,000 GB/s  |  L2 BW (measured): {bw:.0f} GB/s  |  HBM: 1,792 GB/s")
    print(f"  FP32 peak: 104.8 TFLOP/s   |  X_pq (M={M_pq_am}): {m*M_pq_am/1e6:.1f} MB"
          f"  {'← fits in L2!' if m*M_pq_am < 96e6 else '← HBM-bound'}")
    theo, ridge = am_model(m, d, n, k, bw_l2_meas=bw, M_pq=M_pq_am)
    print(f"  Ridge (L2→compute): {ridge:.1f} F/B  |  Ridge (HBM→compute): "
          f"{104800/1792:.1f} F/B\n")

    # collect actual results for efficiency
    actual_ms = {r[0]: r[1] for r in results}
    # map IVF1(C=X,p=Y) names to IVF(C=X,p=Y) for model lookup
    for a in list(actual_ms.keys()):
        if a.startswith('IVF1('):
            actual_ms[a.replace('IVF1(', 'IVF(')] = actual_ms[a]
        if a.startswith('BF-PQ(M='):
            actual_ms['BF-PQ'] = actual_ms[a]
        if a.startswith('IVF-PQ('):
            # IVF-PQ(C=X,p=Y) → IVF-PQ(C=X,p=Y) in theo model
            pass

    key_algos = ['BF-FP32', 'BF-FP16', 'BF-INT8', 'BF-PQ', 'BF-GEMM',
                 'IVF(C=64,p=4)',    'IVF(C=64,p=8)',    'IVF(C=64,p=16)',
                 'IVF(C=128,p=4)',   'IVF(C=128,p=8)',   'IVF(C=128,p=16)',
                 'IVF(C=256,p=8)',   'IVF(C=256,p=16)',
                 'IVF-PQ(C=64,p=8)', 'IVF-PQ(C=128,p=8)', 'IVF-PQ(C=256,p=8)',
                 'IVF-PQ(C=128,p=16)','IVF-PQ(C=256,p=16)']

    print(f"  {'algorithm':<26} {'AI':>6} {'traffic':>8} "
          f"{'pred(theo)':>11} {'pred(meas)':>11} {'actual':>9}  {'eff%':>5}  bound")
    print("  " + "─" * 90)
    for a in key_algos:
        if a not in theo: continue
        r = theo[a]
        act = actual_ms.get(a, float('nan'))
        eff = r['pred_ms_meas'] / act * 100 if not math.isnan(act) and act > 0 else float('nan')
        act_str = f"{act:7.2f}ms" if not math.isnan(act) else "  n/a   "
        eff_str = f"{eff:4.0f}%" if not math.isnan(eff) else "  n/a"
        print(f"  {a:<26} {r['ai']:>6.2f} {r['l2_GB']:>6.2f}GB "
              f"{r['pred_ms_theo']:>9.3f}ms {r['pred_ms_meas']:>9.3f}ms "
              f"{act_str:>10} {eff_str:>6}  {r['bound']}")

    family_fits = _summarize_family_fits(results)
    if family_fits:
        print("\n  FITTED TRAFFIC→TIME MODEL  (actual ms ~= overhead + traffic / BW_fit)\n")
        print(f"  {'family':<12} {'pts':>4} {'overhead':>11} {'BW_fit':>10} {'R^2':>7}")
        print("  " + "─" * 52)
        for fit in family_fits:
            bw_str = f"{fit['bw_gbs']:.0f} GB/s" if math.isfinite(fit["bw_gbs"]) else "inf"
            print(f"  {fit['family']:<12} {fit['n_points']:>4} "
                  f"{fit['intercept_ms']:>9.3f}ms {bw_str:>10} {fit['r2']:>7.3f}")

        print("\n  Interpretation:")
        print("  Large positive intercept => fixed launch/scheduling/top-k overhead not explained by pure traffic.")
        print("  Low fitted BW or low R^2    => the model needs another term (cache miss regime shift, quantization,"
              " centroid work, or merge serialization).")

    export_rows = []
    for name, ms, rc, data_mb, l2_gb in results:
        meta = _parse_algo_name(name)
        model_key = name
        if name.startswith('IVF1('):
            model_key = name.replace('IVF1(', 'IVF(')
        elif name.startswith('BF-PQ(M='):
            model_key = 'BF-PQ'
        theo_row = theo.get(model_key)
        export_rows.append({
            "name": name,
            "family": meta["family"],
            "params": meta["params"],
            "actual_ms": float(ms),
            "recall": float(rc),
            "data_MB": float(data_mb),
            "traffic_GB": None if l2_gb is None else float(l2_gb),
            "model": None if theo_row is None else {
                "ai": float(theo_row["ai"]),
                "pred_ms_theo": float(theo_row["pred_ms_theo"]),
                "pred_ms_meas": float(theo_row["pred_ms_meas"]),
                "bound": theo_row["bound"],
                "note": theo_row["note"],
                "data_MB": float(theo_row["data_MB"]),
            },
        })

    if getattr(args, "export_json", None):
        payload = {
            "meta": {
                "m": int(m),
                "d": int(d),
                "n": int(n),
                "k": int(k),
                "subtitle": subtitle,
                "measured_bw_GBs": float(bw),
                "ridge_l2_flops_per_byte": float(ridge),
            },
            "results": export_rows,
            "family_fits": family_fits,
        }
        with open(args.export_json, "w", encoding="ascii") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Exported model/benchmark data to {args.export_json}")

    # ── d-sweep (synthetic only; skip for real datasets) ───────────────────────
    if skip_dsweep:
        print("\n  Done (d-sweep skipped for non-synthetic run).")
        return

    print("\n" + "=" * 100)
    print("  d-SWEEP  (m=100k, n=1000, k=10)  –  BF-FP32 / INT8 / PQ; regime transitions")
    print("─" * 100)
    m2 = 100_000
    np.random.seed(getattr(args, 'seed', 42))
    X2np = np.random.randn(m2, 128).astype(np.float32)
    Q2np = np.random.randn(n,   128).astype(np.float32)
    X2g  = cp.asarray(X2np)
    Q2g  = cp.asarray(Q2np)
    print(f"  {'d':>4}  {'FP32':>8}  {'FP16':>8}  {'INT8':>8}  "
          f"{'BF-PQ':>8}  {'X_fp32':>8}  {'X_pq':>8}  note")
    print("  " + "─" * 80)
    for di in [4, 8, 16, 32, 64, 128]:
        Xd   = cp.ascontiguousarray(X2g[:m2, :di])
        Qd   = cp.ascontiguousarray(Q2g[:n,  :di])
        Xd16 = Xd.astype(cp.float16)
        try: _, ms32 = run_bf_fp32(Qd, Xd, k, reps=5)
        except Exception as e: ms32 = float('nan'); print(f"    fp32 d={di}: {e}")
        try: _, ms16 = run_bf_fp16(Qd, Xd16, k, reps=5)
        except Exception as e: ms16 = float('nan'); print(f"    fp16 d={di}: {e}")
        if di % 4 == 0:
            try: _, ms8 = run_bf_int8(Qd, Xd, k, reps=5)
            except Exception as e: ms8 = float('nan'); print(f"    int8 d={di}: {e}")
        else:
            ms8 = float('nan')
        # BF-PQ: M=min(d//4, 8) subspaces, needs d%M==0 and M*KSUB==d
        M_d = 0
        mspq = float('nan')
        if HAS_SKLEARN and di >= 8 and di % 8 == 0:
            M_d = min(8, di // 4)  # keep KSUB >= 4
            if di % M_d == 0:
                try:
                    pq_d = PQIndex(M=M_d)
                    pq_d.build(X2np[:m2, :di], di)
                    _, mspq = pq_d.search(Qd, k, reps=5)
                except Exception as e:
                    print(f"    pq d={di}: {e}")
        fits_fp32 = 'L2✓' if m2*di*4 < 96e6 else 'HBM'
        if M_d > 0:
            fits_pq = 'L2✓' if m2 * M_d < 96e6 else 'HBM'
            xpq_mb = m2 * M_d / 1e6
        else:
            fits_pq = '—'
            xpq_mb = float('nan')
        print(f"  {di:>4}  {ms32:>7.2f}ms  {ms16:>7.2f}ms  {ms8:>7.2f}ms  "
              f"{mspq:>7.2f}ms  {m2*di*4/1e6:>6.1f}MB  "
              f"{xpq_mb:>6.2f}MB  fp32:{fits_fp32}")

    print("\n  Done.")


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m',    type=int,   default=500_000)
    ap.add_argument('--d',    type=int,   default=8)
    ap.add_argument('--n',    type=int,   default=1_000)
    ap.add_argument('--k',    type=int,   default=10)
    ap.add_argument('--seed', type=int,   default=42)
    ap.add_argument('--no-cagra', action='store_true',
                    help='skip cuVS CAGRA benchmarks even if cuvs is installed')
    ap.add_argument('--no-lsh', action='store_true',
                    help='skip FastLSH benchmarks')
    ap.add_argument('--lsh-oversample', type=int, default=4,
                    help='k_mul for fused rerank before GPU dedup (lsh_fast)')
    ap.add_argument('--export-json', type=str, default=None,
                    help='write machine-readable benchmark/model data to JSON')
    ap.add_argument('--skip-pq', action='store_true',
                    help='skip PQ and IVF-PQ runs')
    ap.add_argument('--skip-dsweep', action='store_true',
                    help='skip the synthetic d-sweep at the end')
    args = ap.parse_args()
    m, d, n = args.m, args.d, args.n

    np.random.seed(args.seed)
    X_np = np.random.randn(m, d).astype(np.float32)
    Q_np = np.random.randn(n, d).astype(np.float32)

    run_benchmark_suite(args, X_np, Q_np, gt_np=None, skip_dsweep=args.skip_dsweep)


if __name__ == '__main__':
    run()
