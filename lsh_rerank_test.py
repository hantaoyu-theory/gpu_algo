"""
Benchmark four alternative implementations of stages 9-11 (gather + rerank + topk).
Also prints AM-model theoretical predictions for each stage to identify the gap.

AM Model (from paper):
  P streaming multiprocessors, each with M bytes of L1 cache.
  HBM bandwidth B (bytes/sec). Sending x bytes costs x/B + L.
  Computation is free. Minimize total time = Σ (bytes_i / B + L_i).

Stages covered
--------------
  5  hash_queries      : Q @ A  →  q_codes
  6  pack_keys         : q_codes → q_keys (int64 per table)
  7  candidate_lookup  : searchsorted × L tables → all_cands (n, C)
  8  dedup             : sort-based dedup → padded_gpu (n, n_unique)
  9  gather_vectors    : X[padded_gpu] → cand_vecs (n, n_unique, d)
 10  rerank_distances  : ||q - c||^2  → dists (n, n_unique)
 11  topk_selection    : argpartition + argsort → top_k

Variants tested for stages 9-11
---------------------------------
  baseline   : lsh.py as-is  (gather → einsum → argpartition)
  A_fused    : custom CUDA kernel, no intermediate matrices
  B_tiled    : tiled gather+compute, peak mem ∝ tile_size
  C_fp16     : fp16 einsum for dot-product (tensor cores)
  D_precomp  : precomputed ||x||^2  → skip c_sq computation

Usage
-----
  python lsh_rerank_test.py
  python lsh_rerank_test.py --m 500000 --d 8 --L 100 --K 3 --w 3.0 --max_cands 2000
"""

import argparse
import sys
import time
import numpy as np
import cupy as cp

from lsh import (
    LSHIndex, LSHParams,
    exact_neighbors_cuvs, recall_at_k, benchmark_exact_neighbors_trivial,
)

# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _timed(fn, *a, **kw):
    """Run fn(*a, **kw); return (result, elapsed_ms) via CUDA events."""
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record()
    result = fn(*a, **kw)
    e.record(); e.synchronize()
    return result, float(cp.cuda.get_elapsed_time(s, e))


def measure_hbm_bandwidth(size_bytes: int = 1 << 29) -> float:
    """Estimate peak HBM streaming bandwidth (GB/s) via a copy benchmark."""
    n = size_bytes // 4
    A = cp.ones(n, dtype=cp.float32)
    B = cp.zeros(n, dtype=cp.float32)
    B[:] = A[:]                                         # warm-up
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record()
    for _ in range(4):
        B[:] = A[:]
    e.record(); e.synchronize()
    ms = float(cp.cuda.get_elapsed_time(s, e)) / 4
    return (size_bytes * 2) / ms / 1e6                  # GB/s (read + write)


# ─────────────────────────────────────────────────────────────────────────────
# AM model analysis
# ─────────────────────────────────────────────────────────────────────────────

def am_analysis(m, d, n, k, L, K, max_cands, n_unique, bw_GBps):
    """
    Print theoretical lower-bound times for each stage under the AM model.
    We count every byte that must cross HBM ↔ SM (reads + writes).
    Random-access streams are flagged but counted the same as sequential
    (the experiment will reveal the penalty).
    """
    B = bw_GBps * 1e9          # bytes / s
    C = L * max_cands
    F = 4                       # float32 bytes
    I = 4                       # int32 bytes

    def ms(b): return b / B * 1e3

    rows = [
        ("──────────────────────────────────────────────────────────", "", "", ""),
        ("stage",               "bytes  (GB)",  "AM min (ms)", "access"),
        ("──────────────────────────────────────────────────────────", "", "", ""),
    ]

    # Stage 5: project + hash
    s5 = n*d*F + L*K*d*F + L*K*F + n*L*K*F
    rows.append(("5  hash_queries",   f"{s5/1e9:.3f}", f"{ms(s5):.2f}", "seq"))

    # Stage 6: pack keys
    s6 = n*L*K*F + n*L*8
    rows.append(("6  pack_keys",      f"{s6/1e9:.3f}", f"{ms(s6):.2f}", "seq"))

    # Stage 7: candidate lookup  (loop L tables, searchsorted + gather)
    # searchsorted reads O(n * log m * 8 bytes) per table (binary search)
    s7_ss   = L * n * int(np.ceil(np.log2(m))) * 8
    # gather: read si[offsets] n*max_c entries + write all_cands
    s7_idx  = L * (n*max_cands*I + n*max_cands*I)   # read si + write cands
    s7 = s7_ss + s7_idx
    rows.append(("7  candidate_lookup", f"{s7/1e9:.3f}", f"{ms(s7):.2f}", "rnd/seq"))

    # Stage 8: dedup (two sorts of (n, C) int32 matrix)
    s8 = 4 * n*C*I     # read+write × 2 sorts (lower bound: 2 passes)
    rows.append(("8  dedup",          f"{s8/1e9:.3f}", f"{ms(s8):.2f}", "seq"))

    # Stage 9: gather vectors
    s9 = n*n_unique*I + n*n_unique*d*F + n*n_unique*d*F  # read idx + read X + write cand_vecs
    rows.append(("9  gather_vectors", f"{s9/1e9:.3f}", f"{ms(s9):.2f}", "rnd"))

    # Stage 10: rerank distances
    s10 = n*n_unique*d*F + n*d*F + n*n_unique*F        # read cand_vecs + Q + write dists
    rows.append(("10 rerank_dists",   f"{s10/1e9:.3f}", f"{ms(s10):.2f}", "seq"))

    # Stage 11: top-k selection
    s11 = n*n_unique*F + n*k*F                          # read dists + write topk
    rows.append(("11 topk_select",    f"{s11/1e9:.3f}", f"{ms(s11):.2f}", "seq"))

    rows.append(("──────────────────────────────────────────────────────────", "", "", ""))
    total = s5 + s6 + s7 + s8 + s9 + s10 + s11
    rows.append(("TOTAL (stages 5-11)", f"{total/1e9:.3f}", f"{ms(total):.2f}", ""))

    rows.append(("──────────────────────────────────────────────────────────", "", "", ""))
    # Option A breakdown
    sA = n*n_unique*I + n*n_unique*d*F + n*d*F + n*k*F  # read idx + X(rnd) + Q + write out
    rows.append(("A_fused  (9-11)",   f"{sA/1e9:.3f}", f"{ms(sA):.2f}", "rnd"))
    rows.append(("baseline (9-11)",   f"{(s9+s10+s11)/1e9:.3f}",
                 f"{ms(s9+s10+s11):.2f}", "rnd+seq"))
    rows.append(("A / baseline ratio", f"—",
                 f"{(s9+s10+s11)/sA:.1f}×  (AM)", ""))
    rows.append(("──────────────────────────────────────────────────────────", "", "", ""))

    fmt = "{:<30}  {:>12}  {:>12}  {:>12}"
    print(f"\n{'─'*70}")
    print(f"  AM Model  (BW={bw_GBps:.0f} GB/s,  n={n}, m={m}, d={d}, "
          f"L={L}, K={K}, max_c={max_cands}, n_unique={n_unique})")
    print(f"{'─'*70}")
    for r in rows:
        if r[0].startswith("─"):
            print("─" * 70)
        elif r[0] == "stage":
            print(fmt.format("stage", "bytes (GB)", "AM min (ms)", "access"))
        else:
            print(fmt.format(*r))
    print()

    # Return key values for caller
    return {
        "s5": s5, "s6": s6, "s7": s7, "s8": s8,
        "s9": s9, "s10": s10, "s11": s11,
        "baseline_9_11": s9 + s10 + s11,
        "A_9_11": sA,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Re-run stages 5-8 with per-stage timings
# ─────────────────────────────────────────────────────────────────────────────

def run_stages_5_8_timed(idx, Q, k):
    """Re-run stages 5-8 with individual CUDA-event timing; return per-stage dict."""
    n = Q.shape[0]
    L, K_  = idx.p.n_tables, idx.p.n_projections
    max_c  = idx.p.max_cands_per_table
    m      = idx._m
    C      = L * max_c

    def _t(fn, *a, **kw):
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record(); r = fn(*a, **kw); e.record(); e.synchronize()
        return r, float(cp.cuda.get_elapsed_time(s, e))

    # Stage 5
    q_codes, t5 = _t(idx._project_and_hash, Q)

    # Stage 6
    q_keys, t6 = _t(idx._pack_keys, q_codes)

    # Stage 7
    s7 = cp.cuda.Event(); e7 = cp.cuda.Event()
    s7.record()
    all_cands    = cp.full((n, C), -1, dtype=cp.int32)
    offsets_base = cp.arange(max_c, dtype=cp.int32)[None, :]
    for l in range(L):
        sk = idx._sorted_keys[l]; si = idx._sorted_indices[l]
        left  = cp.searchsorted(sk, q_keys[:, l], side="left")
        right = cp.searchsorted(sk, q_keys[:, l], side="right")
        offsets  = left[:, None] + offsets_base
        valid    = offsets < right[:, None]
        offsets  = cp.clip(offsets, 0, m - 1)
        gathered = cp.where(valid, si[offsets], cp.int32(-1))
        col = l * max_c
        all_cands[:, col : col + max_c] = gathered
    e7.record(); e7.synchronize()
    t7 = float(cp.cuda.get_elapsed_time(s7, e7))

    # Stage 8
    INT_MAX = int(np.iinfo(np.int32).max)
    def _dedup():
        tmp  = cp.where(all_cands >= 0, all_cands, cp.int32(INT_MAX))
        tmp  = cp.sort(tmp, axis=1)
        is_v = tmp < INT_MAX
        is_n = cp.concatenate([cp.ones((n, 1), dtype=cp.bool_), tmp[:, 1:] != tmp[:, :-1]], axis=1)
        tmp  = cp.where(is_v & is_n, tmp, cp.int32(INT_MAX))
        tmp  = cp.sort(tmp, axis=1)
        n_u  = max(int((tmp < INT_MAX).sum(axis=1).max()), k)
        pg   = cp.where(tmp[:, :n_u] < INT_MAX, tmp[:, :n_u].astype(cp.int32), cp.int32(0))
        return pg, n_u
    (padded_gpu, n_unique), t8 = _t(_dedup)

    dup_frac = 1.0 - n_unique / C
    print(f"  C={C:,}  n_unique={n_unique:,}  dup_fraction={dup_frac:.1%}")
    return padded_gpu, n_unique, {"t5": t5, "t6": t6, "t7": t7, "t8": t8}


# ─────────────────────────────────────────────────────────────────────────────
# Option A: fused distance + top-k CUDA kernel
# ─────────────────────────────────────────────────────────────────────────────

_FUSED_SRC = r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  256

extern "C" __global__ void fused_l2_topk(
    const float* __restrict__ Q,
    const float* __restrict__ X,
    const int*   __restrict__ cand_ids,
    int*         __restrict__ out_ids,
    float*       __restrict__ out_dists,
    int C, int m
) {
    int qi  = blockIdx.x;
    int tid = threadIdx.x;

    // smem: query (D_VAL f32) | thread-local dists (BLOCK*K_VAL f32) | ids (BLOCK*K_VAL i32)
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

    for (int ci = tid; ci < C; ci += BLOCK) {
        int gid = cand_ids[qi * C + ci];
        const float* xp = X + (long long)gid * D_VAL;
        float dist = 0.0f;
        #pragma unroll
        for (int di = 0; di < D_VAL; di++) {
            float diff = q_sh[di] - xp[di];
            dist += diff * diff;
        }
        if (ln < K_VAL) {
            ld[ln] = dist; li[ln] = gid; ln++;
            if (ln == K_VAL) {
                lmax = -1.0f;
                #pragma unroll
                for (int j = 0; j < K_VAL; j++)
                    if (ld[j] > lmax) { lmax = ld[j]; lmax_pos = j; }
            }
        } else if (dist < lmax) {
            ld[lmax_pos] = dist; li[lmax_pos] = gid;
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
        // Insertion sort (k is tiny)
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

_kernel_cache: dict = {}

def _get_fused_kernel(k: int, d: int) -> cp.RawKernel:
    key = (k, d)
    if key not in _kernel_cache:
        src = _FUSED_SRC.replace("__K__", str(k)).replace("__D__", str(d))
        _kernel_cache[key] = cp.RawKernel(src, "fused_l2_topk")
    return _kernel_cache[key]


def search_A_fused(idx, Q, padded_gpu, n_unique, k) -> cp.ndarray:
    """Option A: fused L2+top-k — no intermediate distance matrix."""
    n = Q.shape[0]; d = idx._d; C = padded_gpu.shape[1]
    BLOCK = 256
    smem = (d + 2 * BLOCK * k) * 4

    kern      = _get_fused_kernel(k, d)
    out_ids   = cp.zeros((n, k), dtype=cp.int32)
    out_dists = cp.zeros((n, k), dtype=cp.float32)

    kern(
        (n,), (BLOCK,),
        (
            cp.ascontiguousarray(Q),
            cp.ascontiguousarray(idx._dataset),
            cp.ascontiguousarray(padded_gpu),
            out_ids, out_dists,
            np.int32(C), np.int32(idx._m),
        ),
        shared_mem=smem,
    )
    cp.cuda.Stream.null.synchronize()
    return out_ids


# ─────────────────────────────────────────────────────────────────────────────
# Option B: tiled rerank
# ─────────────────────────────────────────────────────────────────────────────

def search_B_tiled(idx, Q, padded_gpu, n_unique, k, tile_size: int = 4096) -> cp.ndarray:
    """Option B: tiled gather+compute — peak memory ∝ tile_size, not n_unique."""
    n = Q.shape[0]; d = idx._d
    q_sq       = (Q ** 2).sum(1, keepdims=True)
    top_dists  = cp.full((n, k), cp.float32(1e18))
    top_global = cp.zeros((n, k), dtype=cp.int32)

    for start in range(0, n_unique, tile_size):
        end      = min(start + tile_size, n_unique)
        tile_ids = padded_gpu[:, start:end]
        tvecs    = idx._dataset[tile_ids.reshape(-1)].reshape(n, end - start, d)
        c_sq     = (tvecs ** 2).sum(2)
        dot      = cp.einsum("nd,ncd->nc", Q, tvecs)
        tdists   = cp.maximum(q_sq + c_sq - 2.0 * dot, 0.0)

        comb_d   = cp.concatenate([top_dists,  tdists  ], axis=1)
        comb_g   = cp.concatenate([top_global, tile_ids], axis=1)
        sel      = cp.argpartition(comb_d, k - 1, axis=1)[:, :k]
        top_dists  = cp.take_along_axis(comb_d, sel, axis=1)
        top_global = cp.take_along_axis(comb_g, sel, axis=1)

    order = cp.argsort(top_dists, axis=1)
    return cp.take_along_axis(top_global, order, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Option C: fp16 einsum
# ─────────────────────────────────────────────────────────────────────────────

def search_C_fp16(idx, Q, padded_gpu, n_unique, k) -> cp.ndarray:
    """Option C: fp16 dot-product via tensor cores; c_sq/q_sq remain fp32."""
    n = Q.shape[0]; d = idx._d
    cand_vecs = idx._dataset[padded_gpu.reshape(-1)].reshape(n, n_unique, d)

    q_sq  = (Q ** 2).sum(1, keepdims=True)
    c_sq  = (cand_vecs ** 2).sum(2)
    dot   = cp.einsum(
        "nd,ncd->nc",
        Q.astype(cp.float16),
        cand_vecs.astype(cp.float16),
    ).astype(cp.float32)
    dists = cp.maximum(q_sq + c_sq - 2.0 * dot, 0.0)

    actual_k  = min(k, n_unique)
    top_local = cp.argpartition(dists, actual_k - 1, axis=1)[:, :actual_k]
    top_dists = cp.take_along_axis(dists, top_local, axis=1)
    order     = cp.argsort(top_dists, axis=1)
    return cp.take_along_axis(padded_gpu,
                              cp.take_along_axis(top_local, order, axis=1), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Option D: precomputed squared norms
# ─────────────────────────────────────────────────────────────────────────────

def search_D_precomp(idx, Q, padded_gpu, n_unique, k, x_sq) -> cp.ndarray:
    """Option D: c_sq via scalar gather from precomputed x_sq — skips one cand_vecs pass."""
    n = Q.shape[0]; d = idx._d
    cand_vecs = idx._dataset[padded_gpu.reshape(-1)].reshape(n, n_unique, d)

    q_sq  = (Q ** 2).sum(1, keepdims=True)
    c_sq  = x_sq[padded_gpu.reshape(-1)].reshape(n, n_unique)   # scalar gather
    dot   = cp.einsum("nd,ncd->nc", Q, cand_vecs)
    dists = cp.maximum(q_sq + c_sq - 2.0 * dot, 0.0)

    actual_k  = min(k, n_unique)
    top_local = cp.argpartition(dists, actual_k - 1, axis=1)[:, :actual_k]
    top_dists = cp.take_along_axis(dists, top_local, axis=1)
    order     = cp.argsort(top_dists, axis=1)
    return cp.take_along_axis(padded_gpu,
                              cp.take_along_axis(top_local, order, axis=1), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline (lsh.py stages 9-11 verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def search_baseline(idx, Q, padded_gpu, n_unique, k) -> cp.ndarray:
    """Baseline: explicit gather → einsum → argpartition + argsort."""
    n = Q.shape[0]; d = idx._d
    cand_vecs = idx._dataset[padded_gpu.reshape(-1)].reshape(n, n_unique, d)
    q_sq  = (Q ** 2).sum(1, keepdims=True)
    c_sq  = (cand_vecs ** 2).sum(2)
    dot   = cp.einsum("nd,ncd->nc", Q, cand_vecs)
    dists = cp.maximum(q_sq + c_sq - 2.0 * dot, 0.0)
    actual_k  = min(k, n_unique)
    top_local = cp.argpartition(dists, actual_k - 1, axis=1)[:, :actual_k]
    top_dists = cp.take_along_axis(dists, top_local, axis=1)
    order     = cp.argsort(top_dists, axis=1)
    return cp.take_along_axis(padded_gpu,
                              cp.take_along_axis(top_local, order, axis=1), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run(m, d, n, k, params, bf_query_batch=128, tile_size=4096, seed=42):
    L         = params.n_tables
    K         = params.n_projections
    max_cands = params.max_cands_per_table
    C         = L * max_cands

    print("=" * 70)
    print(f"  LSH rerank benchmark")
    print(f"  dataset m={m:,}  d={d}  queries n={n}  k={k}")
    print(f"  L={L}  K={K}  w={params.bucket_width}  max_cands={max_cands}  C={C:,}")
    print("=" * 70)

    # ── measure hardware bandwidth ────────────────────────────────────────────
    print("\n[1] Measuring HBM bandwidth …")
    bw = measure_hbm_bandwidth()
    print(f"    peak streaming BW ≈ {bw:.0f} GB/s")

    # ── generate data ─────────────────────────────────────────────────────────
    print("\n[2] Generating data …")
    rng = cp.random.default_rng(seed)
    X   = rng.standard_normal((m, d), dtype=cp.float32)
    Q   = rng.standard_normal((n, d), dtype=cp.float32)

    print("\n[3] Computing ground truth (brute-force) …")
    gt = exact_neighbors_cuvs(X, Q, k)
    _, bf_ms = benchmark_exact_neighbors_trivial(X, Q, k, query_batch=bf_query_batch)
    print(f"    brute-force: {bf_ms:.1f} ms  ({bf_ms/n:.3f} ms/query)")

    # ── build LSH index ───────────────────────────────────────────────────────
    print("\n[4] Building LSH index …")
    idx = LSHIndex(params)
    _, t_build = _timed(idx.build, X)
    x_sq = (X ** 2).sum(1)                              # precompute for option D
    print(f"    build time: {t_build:.1f} ms")

    # ── stages 5-8 ────────────────────────────────────────────────────────────
    print("\n[5] Running stages 5-8 (shared across all variants) …")
    padded_gpu, n_unique, t58 = run_stages_5_8_timed(idx, Q, k)

    # ── AM model analysis ─────────────────────────────────────────────────────
    print("\n[6] AM model predictions …")
    am = am_analysis(m, d, n, k, L, K, max_cands, n_unique, bw)

    # ── warm-up fused kernel (JIT compile) ────────────────────────────────────
    print("[7] Warming up fused kernel (JIT compile) …")
    _ = search_A_fused(idx, Q, padded_gpu, n_unique, k)

    # ── per-stage timings (5-8) ───────────────────────────────────────────────
    def eff(actual_ms, am_bytes):
        """Effective bandwidth in GB/s given actual_ms and AM traffic bytes."""
        return am_bytes / (actual_ms * 1e-3) / 1e9

    print("[8] Per-stage timing (5-8) and AM comparison:")
    fmt = "  {:25s}  {:>8.1f} ms  AM {:>7.2f} ms  ratio {:>5.1f}×  ({:.0f} GB/s eff)"
    for name, t_ms, bytes_key in [
        ("5_hash_queries",    t58["t5"], "s5"),
        ("6_pack_keys",       t58["t6"], "s6"),
        ("7_candidate_lookup",t58["t7"], "s7"),
        ("8_dedup",           t58["t8"], "s8"),
    ]:
        am_ms   = am[bytes_key] / (bw * 1e9) * 1e3
        ratio   = t_ms / am_ms if am_ms > 0 else float("inf")
        eff_bw  = eff(t_ms, am[bytes_key])
        print(fmt.format(name, t_ms, am_ms, ratio, eff_bw))

    t58_total = sum(t58.values())
    print(f"  {'stages 5-8 total':<25}  {t58_total:>8.1f} ms")

    # ── stages 9-11 variants ──────────────────────────────────────────────────
    variants = [
        ("baseline",  lambda: search_baseline  (idx, Q, padded_gpu, n_unique, k)),
        ("A_fused",   lambda: search_A_fused    (idx, Q, padded_gpu, n_unique, k)),
        ("B_tiled",   lambda: search_B_tiled    (idx, Q, padded_gpu, n_unique, k, tile_size)),
        ("C_fp16",    lambda: search_C_fp16     (idx, Q, padded_gpu, n_unique, k)),
        ("D_precomp", lambda: search_D_precomp  (idx, Q, padded_gpu, n_unique, k, x_sq)),
    ]

    print("\n[9] Stages 9-11 comparison  (all variants start from same padded_gpu):")
    base_am_bytes = am["baseline_9_11"]
    fused_am_bytes = am["A_9_11"]

    hdr = f"  {'variant':<14} {'ms (9-11)':>10}  {'recall':>8}  {'BF speedup':>11}  " \
          f"{'vs AM':>8}  {'eff BW':>9}"
    print(hdr); print("  " + "─" * (len(hdr) - 2))

    results = []
    for name, fn in variants:
        top, ms_911 = _timed(fn)
        r  = recall_at_k(gt, cp.asnumpy(top))
        sp = bf_ms / (t58_total + ms_911)
        am_b  = fused_am_bytes if name == "A_fused" else base_am_bytes
        am_ms = am_b / (bw * 1e9) * 1e3
        ratio = ms_911 / am_ms
        eff_bw = eff(ms_911, am_b)
        print(f"  {name:<14} {ms_911:>10.1f}  {r:>8.4f}  {sp:>10.2f}×  "
              f"{ratio:>7.1f}×  {eff_bw:>7.0f} GB/s")
        results.append((name, ms_911, r, sp, ratio, eff_bw))

    # Summary
    print(f"\n  brute-force total: {bf_ms:.1f} ms  (stages 5-8 would add {t58_total:.1f} ms)")
    print(f"  AM min stages 9-11 (baseline traffic): {base_am_bytes/(bw*1e9)*1e3:.2f} ms")
    print(f"  AM min stages 9-11 (A_fused  traffic): {fused_am_bytes/(bw*1e9)*1e3:.2f} ms")

    # Key takeaway
    base_ms  = next(r[1] for r in results if r[0] == "baseline")
    fused_ms = next(r[1] for r in results if r[0] == "A_fused")
    print(f"\n  speedup A_fused vs baseline: {base_ms/fused_ms:.2f}×")
    print(f"  AM-predicted speedup:        {am['baseline_9_11']/am['A_9_11']:.2f}×")
    print(f"  → gap (actual / AM ratio):   {(base_ms/fused_ms)/(am['baseline_9_11']/am['A_9_11']):.2f}×")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",         type=int,   default=500_000)
    parser.add_argument("--d",         type=int,   default=8)
    parser.add_argument("--n",         type=int,   default=1_000)
    parser.add_argument("--k",         type=int,   default=10)
    parser.add_argument("--L",         type=int,   default=100)
    parser.add_argument("--K",         type=int,   default=3)
    parser.add_argument("--w",         type=float, default=3.0)
    parser.add_argument("--max_cands", type=int,   default=2000)
    parser.add_argument("--tile_size", type=int,   default=4096)
    parser.add_argument("--bf_batch",  type=int,   default=128)
    args = parser.parse_args()

    run(
        m=args.m, d=args.d, n=args.n, k=args.k,
        params=LSHParams(
            n_tables=args.L, n_projections=args.K,
            bucket_width=args.w, max_cands_per_table=args.max_cands,
        ),
        bf_query_batch=args.bf_batch,
        tile_size=args.tile_size,
    )
