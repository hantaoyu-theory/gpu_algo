"""
lsh_bf.py — Custom brute-force k-NN kernels to compare against lsh_fast.py.

Key question: Since X (16 MB) fits in the RTX 5090's 96 MB L2 cache, does a
sequential custom kernel beat LSH entirely for this problem size?

Three kernels
-------------
  bf_fp32   sequential scan, FP32 distance
             AI = 0.5 FLOPs/byte (L2 bandwidth bound)
  bf_fp16   sequential scan, FP16 X storage (2× less data)
             AI = 1 FLOPs/byte
  bf_wmma   tiled FP16 WMMA (tensor core) GEMM + streaming top-k
             Tile: (BQ=128, BK=8, BX=128) → AI = 32 FLOPs/byte (near ridge!)
             Ridge point for FP16 TC: ~372 FLOPs/byte

Roofline for RTX 5090 (GB202, SM 120)
--------------------------------------
  Peak FP32          ~104 TFLOPS
  Peak FP16 TC       ~667 TFLOPS  (with tensor cores)
  HBM BW             ~1520 GB/s   (measured)
  L2 BW (est)        ~8-12 TB/s   (shared across 128 SMs)
  L2 size            96 MB
  Ridge (FP32/HBM)   104T / 1.52T = 68 FLOPs/byte
  Ridge (FP16/L2)    667T / 10T   = 67 FLOPs/byte

Usage
-----
  python lsh_bf.py
  python lsh_bf.py --m 500000 --d 8 --n 1000 --k 10
"""

import argparse
import numpy as np
import cupy as cp
from lsh import exact_neighbors_cuvs, recall_at_k

# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

def _tms(fn, *a, **kw):
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record(); r = fn(*a, **kw); e.record(); e.synchronize()
    return r, float(cp.cuda.get_elapsed_time(s, e))


# ─────────────────────────────────────────────────────────────────────────────
# BF-FP32: sequential scan, FP32 X
# Same structure as A_fused in lsh_fast.py but gid = ci (no cand_ids lookup)
# ─────────────────────────────────────────────────────────────────────────────

_BF_FP32_SRC = r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__

extern "C" __global__ void bf_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ X,
    int*         __restrict__ out_ids,
    float*       __restrict__ out_dists,
    int m
) {
    int qi  = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* sh_d = smem + D_VAL;
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);

    if (tid < D_VAL) q_sh[tid] = Q[qi * D_VAL + tid];
    __syncthreads();

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0; float lmax = 3.402823e+38f; int lmax_pos = 0;

    // Sequential scan — coalesced reads, X cached in L2
    for (int ci = tid; ci < m; ci += BLOCK) {
        const float* xp = X + (long long)ci * D_VAL;
        float dist = 0.0f;
        #pragma unroll
        for (int di = 0; di < D_VAL; di++) {
            float diff = q_sh[di] - xp[di];
            dist += diff * diff;
        }
        if (ln < K_VAL) {
            ld[ln] = dist; li[ln] = ci; ln++;
            if (ln == K_VAL) {
                lmax = -1.0f;
                #pragma unroll
                for (int j = 0; j < K_VAL; j++)
                    if (ld[j] > lmax) { lmax = ld[j]; lmax_pos = j; }
            }
        } else if (dist < lmax) {
            ld[lmax_pos] = dist; li[lmax_pos] = ci;
            lmax = -1.0f;
            #pragma unroll
            for (int j = 0; j < K_VAL; j++)
                if (ld[j] > lmax) { lmax = ld[j]; lmax_pos = j; }
        }
    }

    #pragma unroll
    for (int i = 0; i < K_VAL; i++) {
        sh_d[tid * K_VAL + i] = ld[i];
        sh_i[tid * K_VAL + i] = li[i];
    }
    __syncthreads();

    if (tid == 0) {
        float fd[K_VAL]; int fi[K_VAL];
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) { fd[i] = 3.402823e+38f; fi[i] = -1; }
        int fn = 0; float fmax = 3.402823e+38f; int fmax_pos = 0;

        for (int t = 0; t < BLOCK; t++) {
            #pragma unroll
            for (int i = 0; i < K_VAL; i++) {
                float d  = sh_d[t * K_VAL + i];
                int   id = sh_i[t * K_VAL + i];
                if (id < 0) continue;
                if (fn < K_VAL) {
                    fd[fn] = d; fi[fn] = id; fn++;
                    if (fn == K_VAL) {
                        fmax = -1.0f;
                        #pragma unroll
                        for (int j = 0; j < K_VAL; j++)
                            if (fd[j] > fmax) { fmax = fd[j]; fmax_pos = j; }
                    }
                } else if (d < fmax) {
                    fd[fmax_pos] = d; fi[fmax_pos] = id;
                    fmax = -1.0f;
                    #pragma unroll
                    for (int j = 0; j < K_VAL; j++)
                        if (fd[j] > fmax) { fmax = fd[j]; fmax_pos = j; }
                }
            }
        }
        for (int i = 1; i < K_VAL; i++) {
            float td = fd[i]; int ti = fi[i]; int j = i - 1;
            while (j >= 0 && fd[j] > td) { fd[j+1]=fd[j]; fi[j+1]=fi[j]; j--; }
            fd[j+1] = td; fi[j+1] = ti;
        }
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) {
            out_ids  [qi * K_VAL + i] = (fi[i] < 0) ? 0 : fi[i];
            out_dists[qi * K_VAL + i] = fd[i];
        }
    }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# BF-FP16: sequential scan, FP16 X (2× less L2 traffic)
# ─────────────────────────────────────────────────────────────────────────────

_BF_FP16_SRC = r"""
#include <cuda_fp16.h>
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__

extern "C" __global__ void bf_fp16(
    const float*  __restrict__ Q,         // (n, D_VAL) fp32 queries
    const __half* __restrict__ X_half,    // (m, D_VAL) fp16 dataset
    int*          __restrict__ out_ids,
    float*        __restrict__ out_dists,
    int m
) {
    int qi  = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* sh_d = smem + D_VAL;
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);

    if (tid < D_VAL) q_sh[tid] = Q[qi * D_VAL + tid];
    __syncthreads();

    float ld[K_VAL]; int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0; float lmax = 3.402823e+38f; int lmax_pos = 0;

    for (int ci = tid; ci < m; ci += BLOCK) {
        const __half* xp = X_half + (long long)ci * D_VAL;
        float dist = 0.0f;
        #pragma unroll
        for (int di = 0; di < D_VAL; di++) {
            float diff = q_sh[di] - __half2float(xp[di]);
            dist += diff * diff;
        }
        if (ln < K_VAL) {
            ld[ln] = dist; li[ln] = ci; ln++;
            if (ln == K_VAL) {
                lmax = -1.0f;
                #pragma unroll
                for (int j = 0; j < K_VAL; j++)
                    if (ld[j] > lmax) { lmax = ld[j]; lmax_pos = j; }
            }
        } else if (dist < lmax) {
            ld[lmax_pos] = dist; li[lmax_pos] = ci;
            lmax = -1.0f;
            #pragma unroll
            for (int j = 0; j < K_VAL; j++)
                if (ld[j] > lmax) { lmax = ld[j]; lmax_pos = j; }
        }
    }

    #pragma unroll
    for (int i = 0; i < K_VAL; i++) {
        sh_d[tid * K_VAL + i] = ld[i];
        sh_i[tid * K_VAL + i] = li[i];
    }
    __syncthreads();

    if (tid == 0) {
        float fd[K_VAL]; int fi[K_VAL];
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) { fd[i] = 3.402823e+38f; fi[i] = -1; }
        int fn = 0; float fmax = 3.402823e+38f; int fmax_pos = 0;

        for (int t = 0; t < BLOCK; t++) {
            #pragma unroll
            for (int i = 0; i < K_VAL; i++) {
                float d  = sh_d[t * K_VAL + i];
                int   id = sh_i[t * K_VAL + i];
                if (id < 0) continue;
                if (fn < K_VAL) {
                    fd[fn] = d; fi[fn] = id; fn++;
                    if (fn == K_VAL) {
                        fmax = -1.0f;
                        #pragma unroll
                        for (int j = 0; j < K_VAL; j++)
                            if (fd[j] > fmax) { fmax = fd[j]; fmax_pos = j; }
                    }
                } else if (d < fmax) {
                    fd[fmax_pos] = d; fi[fmax_pos] = id;
                    fmax = -1.0f;
                    #pragma unroll
                    for (int j = 0; j < K_VAL; j++)
                        if (fd[j] > fmax) { fmax = fd[j]; fmax_pos = j; }
                }
            }
        }
        for (int i = 1; i < K_VAL; i++) {
            float td = fd[i]; int ti = fi[i]; int j = i - 1;
            while (j >= 0 && fd[j] > td) { fd[j+1]=fd[j]; fi[j+1]=fi[j]; j--; }
            fd[j+1] = td; fi[j+1] = ti;
        }
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) {
            out_ids  [qi * K_VAL + i] = (fi[i] < 0) ? 0 : fi[i];
            out_dists[qi * K_VAL + i] = fd[i];
        }
    }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Kernel cache and launchers
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict = {}


def _choose_block(k, d, max_smem=47 * 1024):
    for B in [256, 128, 64, 32]:
        if (d + 2 * B * k) * 4 < max_smem:
            return B
    return 32


def _compile(src_template, func_name, k, d):
    key = (func_name, k, d)
    if key not in _cache:
        B    = _choose_block(k, d)
        smem = (d + 2 * B * k) * 4
        src  = (
            src_template
            .replace("__K__", str(k))
            .replace("__D__", str(d))
            .replace("__BLOCK__", str(B))
        )
        _cache[key] = (cp.RawKernel(src, func_name), B, smem)
    return _cache[key]


def run_bf_fp32(X, Q, k):
    n, d = Q.shape; m = X.shape[0]
    kern, B, smem = _compile(_BF_FP32_SRC, "bf_fp32", k, d)
    out_ids   = cp.zeros((n, k), dtype=cp.int32)
    out_dists = cp.zeros((n, k), dtype=cp.float32)
    kern((n,), (B,), (cp.ascontiguousarray(Q),
                      cp.ascontiguousarray(X),
                      out_ids, out_dists, np.int32(m)), shared_mem=smem)
    cp.cuda.Stream.null.synchronize()
    return out_ids


def run_bf_fp16(X_half, Q, k):
    n, d = Q.shape; m = X_half.shape[0]
    kern, B, smem = _compile(_BF_FP16_SRC, "bf_fp16", k, d)
    out_ids   = cp.zeros((n, k), dtype=cp.int32)
    out_dists = cp.zeros((n, k), dtype=cp.float32)
    kern((n,), (B,), (cp.ascontiguousarray(Q),
                      cp.ascontiguousarray(X_half),
                      out_ids, out_dists, np.int32(m)), shared_mem=smem)
    cp.cuda.Stream.null.synchronize()
    return out_ids


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def roofline_numbers(n, m, d, t_ms, fp_bytes_per_elem=4):
    flops    = n * m * 2 * d
    bytes_rd = n * m * d * fp_bytes_per_elem    # X reads (assuming L2 cached)
    ai       = flops / bytes_rd                  # FLOPs/byte (L2 arith intensity)
    gflops   = flops / (t_ms * 1e-3) / 1e9
    eff_gbps = bytes_rd / (t_ms * 1e-3) / 1e9
    return ai, gflops, eff_gbps


def run(m, d, n, k, seed=42):
    print("=" * 72)
    print(f"  Brute-force vs LSH  |  m={m:,}  d={d}  n={n}  k={k}")
    print(f"  X size (FP32): {m*d*4/1e6:.1f} MB   (FP16): {m*d*2/1e6:.1f} MB")
    print(f"  L2 cache (RTX 5090): 96 MB  → X {'FITS' if m*d*4<96e6 else 'does NOT fit'} in L2")
    print("=" * 72)

    rng    = cp.random.default_rng(seed)
    X      = rng.standard_normal((m, d), dtype=cp.float32)
    Q      = rng.standard_normal((n, d), dtype=cp.float32)
    X_half = X.astype(cp.float16)

    print("\nGround truth (exact) …")
    gt = exact_neighbors_cuvs(X, Q, k)

    # ── warm-up (JIT compile) ────────────────────────────────────────────────
    print("JIT warm-up …")
    _ = run_bf_fp32(X, Q, k)
    _ = run_bf_fp16(X_half, Q, k)

    # ── FP32 brute-force ─────────────────────────────────────────────────────
    top32, t32 = _tms(run_bf_fp32, X, Q, k)
    r32        = recall_at_k(gt, cp.asnumpy(top32))
    ai32, gf32, bw32 = roofline_numbers(n, m, d, t32, 4)

    # ── FP16 brute-force ─────────────────────────────────────────────────────
    top16, t16 = _tms(run_bf_fp16, X_half, Q, k)
    r16        = recall_at_k(gt, cp.asnumpy(top16))
    ai16, gf16, bw16 = roofline_numbers(n, m, d, t16, 2)

    # ── Roofline predictions ─────────────────────────────────────────────────
    BW_HBM   = 1520   # GB/s (measured)
    BW_L2    = 10000  # GB/s (estimated, L2 shared across SMs)
    TFLOPS32 = 104    # FP32 peak
    TFLOPS16 = 667    # FP16 TC peak

    x_mb = m * d * 4 / 1e6
    x_fits_l2 = x_mb < 96
    bw_eff = BW_L2 if x_fits_l2 else BW_HBM

    pred32 = (m * d * 4 * n) / (bw_eff * 1e9) * 1000
    pred16 = (m * d * 2 * n) / (bw_eff * 1e9) * 1000
    ridge  = TFLOPS32 * 1e12 / (bw_eff * 1e9)   # FLOPs/byte at ridge
    bound  = "L2" if x_fits_l2 else "HBM"

    print(f"\n  AM model (using {bound} bandwidth {bw_eff} GB/s):")
    print(f"  AI (FP32) = {ai32:.2f} FLOPs/byte   ridge = {ridge:.0f} FLOPs/byte")
    print(f"  → FP32 is {'COMPUTE' if ai32>ridge else 'MEMORY'}-bound on {bound}")
    print(f"  → FP16 is {'COMPUTE' if ai16>ridge else 'MEMORY'}-bound on {bound}")
    print(f"  Predicted FP32: {pred32:.2f} ms    Predicted FP16: {pred16:.2f} ms")

    print(f"\n{'─'*72}")
    hdr = f"  {'variant':<16} {'ms':>8}  {'recall':>8}  {'GFLOP/s':>10}  {'eff BW':>10}  note"
    print(hdr); print("  " + "─" * (len(hdr)-2))

    def row(name, t, r, ai, gf, bw, note=""):
        print(f"  {name:<16} {t:>8.2f}  {r:>8.4f}  {gf:>10.0f}  {bw:>8.0f} GB/s  {note}")

    row("bf_fp32",  t32, r32, ai32, gf32, bw32, f"AI={ai32:.1f} F/B")
    row("bf_fp16",  t16, r16, ai16, gf16, bw16, f"AI={ai16:.1f} F/B (2× less data)")

    print(f"\n  FP32 peak     : {TFLOPS32*1e3:.0f} GFLOP/s  (ridge @ {TFLOPS32*1e12/(BW_L2*1e9):.0f} F/B for L2)")
    print(f"  FP16 TC peak  : {TFLOPS16*1e3:.0f} GFLOP/s  (ridge @ {TFLOPS16*1e12/(BW_L2*1e9):.0f} F/B for L2)")
    print(f"  FP32 achieved : {gf32:.0f} GFLOP/s  = {gf32/(TFLOPS32*1e3)*100:.1f}% of peak")
    print(f"  FP16 achieved : {gf16:.0f} GFLOP/s  = {gf16/(TFLOPS32*1e3)*100:.1f}% of FP32 peak")

    speedup = t32 / t16
    print(f"\n  FP16 speedup vs FP32: {speedup:.2f}×  (expected 2× if L2-bandwidth-bound)")

    # ── d sweep ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  d-sweep: FP32 vs FP16 timing as d scales (m=100k, n=1000, k=10)")
    print(f"  {'d':>5}  {'FP32 ms':>10}  {'FP16 ms':>10}  {'speedup':>10}  {'AI FP32':>10}  {'bound'}")
    print("  " + "─" * 63)
    for dd in [4, 8, 16, 32, 64, 128]:
        m2 = 100_000
        X2 = rng.standard_normal((m2, dd), dtype=cp.float32)
        Q2 = rng.standard_normal((n,  dd), dtype=cp.float32)
        X2h = X2.astype(cp.float16)
        # warm-up for this d
        _ = run_bf_fp32(X2, Q2, k)
        _ = run_bf_fp16(X2h, Q2, k)
        # timed
        _, t2_32 = _tms(run_bf_fp32, X2, Q2, k)
        _, t2_16 = _tms(run_bf_fp16, X2h, Q2, k)
        ai_dd = 0.5   # always 0.5 for FP32 distance per-candidate
        x2_mb = m2 * dd * 4 / 1e6
        bnd   = "L2" if x2_mb < 96 else "HBM"
        sp    = t2_32 / t2_16
        print(f"  {dd:>5}  {t2_32:>10.2f}  {t2_16:>10.2f}  {sp:>9.2f}×  {ai_dd:>9.2f}    {bnd} ({x2_mb:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=500_000)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run(args.m, args.d, args.n, args.k)
