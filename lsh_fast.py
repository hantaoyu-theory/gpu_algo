"""
lsh_fast.py — LSH pipeline with custom CUDA kernels for all hot stages.

Root cause (from profiling on RTX 5090):
  CuPy's elementwise / reduction / einsum kernels are 50-150× slower than
  theoretical bandwidth on Blackwell (SM 120), likely because cupy-cuda12x
  was not compiled with SM 120 precompiled kernels and falls back to a slow
  compatibility path.  Simple gather/indexing is unaffected (~expected speed).
  Custom RawKernel code is JIT-compiled by the driver against SM 120 and runs
  at near-theoretical bandwidth.

Changes vs lsh.py
-----------------
  Stage 7  vectorised lookup  — single CUDA kernel, no Python loop over L tables.
           Was: 1036 ms (Python loop × 100 CuPy ops per iteration)
           Target: ~5 ms  (800 MB write + binary-search reads, fully parallelised)

  Stage 8  eliminated — duplicates are handled lazily:
           The fused kernel outputs k*oversample global indices; a cheap CPU
           pass deduplicates them to k unique neighbors.

  Stage 9-11  fused rerank+topk  — same custom kernel as A_fused in
           lsh_rerank_test.py, but now driven from all_cands (the raw
           C-wide candidate buffer with duplicates).

Pipeline timing target on RTX 5090:
  stage 5-6  ~22 ms  (CuPy hash — still Python, modest improvement possible)
  stage 7    ~5 ms   (custom kernel)
  stage 9-11 ~3 ms   (fused kernel on C=200k candidates, k*4 oversample)
  CPU dedup  ~0 ms
  ─────────────
  total      ~30 ms vs brute-force 673 ms → ~22× speedup at 95% recall

Usage
-----
  python lsh_fast.py
  python lsh_fast.py --m 500000 --d 8 --L 100 --K 3 --w 3.0 --max_cands 2000
"""

import argparse
import numpy as np
import cupy as cp

from lsh import (
    LSHIndex, LSHParams, CUDAProfiler,
    exact_neighbors_cuvs, recall_at_k, benchmark_exact_neighbors_trivial,
)

# ─────────────────────────────────────────────────────────────────────────────
# Stage-7 CUDA kernel: vectorised candidate lookup
#
# Grid: (n, L)  — one warp per (query, table) pair
# Block: WARP=32 threads (single warp for coalesced I/O)
#
# Thread 0: binary-search sorted_keys[l] for q_keys[qi, l]
# All 32: cooperatively copy up to MAX_C indices into all_cands[qi, l*MAX_C:]
# ─────────────────────────────────────────────────────────────────────────────

_STAGE7_SRC = r"""
#define MAX_C   __MAX_C__
#define WARP    32

extern "C" __global__ void stage7_lookup(
    const long long* __restrict__ sorted_keys,    // (L, m)
    const int*       __restrict__ sorted_indices, // (L, m)
    const long long* __restrict__ q_keys,         // (n, L)
    int*             __restrict__ all_cands,       // (n, L * MAX_C)
    int L, int m
) {
    int qi  = blockIdx.x;
    int l   = blockIdx.y;
    int tid = threadIdx.x;   // 0..WARP-1

    const long long* sk = sorted_keys  + (long long)l * m;
    const int*       si = sorted_indices + (long long)l * m;

    __shared__ int sh_left, sh_right;

    if (tid == 0) {
        long long target = q_keys[qi * L + l];

        // lower bound
        int lo = 0, hi = m;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (sk[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        sh_left = lo;

        // upper bound
        lo = 0; hi = m;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (sk[mid] <= target) lo = mid + 1;
            else hi = mid;
        }
        sh_right = lo;
    }
    __syncthreads();

    int left  = sh_left;
    int right = sh_right;
    int cnt   = right - left;
    if (cnt > MAX_C) cnt = MAX_C;

    int* dst = all_cands + (long long)qi * L * MAX_C + l * MAX_C;

    // coalesced copy: threads read/write consecutive positions each step
    for (int i = tid; i < cnt; i += WARP)
        dst[i] = si[left + i];
    for (int i = cnt + tid; i < MAX_C; i += WARP)
        dst[i] = -1;
}
"""

_stage7_cache: dict = {}

def _get_stage7_kernel(max_c: int) -> cp.RawKernel:
    if max_c not in _stage7_cache:
        src = _STAGE7_SRC.replace("__MAX_C__", str(max_c))
        _stage7_cache[max_c] = cp.RawKernel(src, "stage7_lookup")
    return _stage7_cache[max_c]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 9-11 fused kernel (same as A_fused in lsh_rerank_test.py)
# Used with K_VAL = k * oversample to allow lazy dedup afterward.
# ─────────────────────────────────────────────────────────────────────────────

_FUSED_SRC = r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__

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
        if (gid < 0) continue;                  // skip padding
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

_fused_cache: dict = {}

def _choose_block(k_val: int, d: int, max_smem: int = 47 * 1024) -> int:
    """Pick the largest BLOCK (power of 2) such that shared memory ≤ max_smem."""
    for B in [256, 128, 64, 32]:
        if (d + 2 * B * k_val) * 4 < max_smem:
            return B
    return 32


def _get_fused_kernel(k_val: int, d: int) -> tuple:
    """Return (kernel, block_size, smem_bytes)."""
    key = (k_val, d)
    if key not in _fused_cache:
        B    = _choose_block(k_val, d)
        smem = (d + 2 * B * k_val) * 4
        src  = (
            _FUSED_SRC
            .replace("__K__", str(k_val))
            .replace("__D__", str(d))
            .replace("__BLOCK__", str(B))
        )
        kern = cp.RawKernel(src, "fused_l2_topk")
        _fused_cache[key] = (kern, B, smem)
    return _fused_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# GPU dedup kernel: take first k unique values per row from k_os sorted results
# 1 thread per query, registers hold the seen-set (k_os ≤ 64)
# ─────────────────────────────────────────────────────────────────────────────

_DEDUP_SRC = r"""
#define K_OS  __K_OS__
#define K_OUT __K_OUT__

extern "C" __global__ void gpu_dedup(
    const int* __restrict__ in_ids,   // (n, K_OS) sorted by distance
    int*       __restrict__ out_ids,  // (n, K_OUT)
    int n
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n) return;

    const int* src = in_ids  + qi * K_OS;
    int*       dst = out_ids + qi * K_OUT;

    int seen[K_OS];
    int seen_n = 0, cnt = 0;

    for (int i = 0; i < K_OS && cnt < K_OUT; i++) {
        int val = src[i];
        bool dup = false;
        #pragma unroll
        for (int j = 0; j < K_OS; j++) {
            if (j >= seen_n) break;
            if (seen[j] == val) { dup = true; break; }
        }
        if (!dup) {
            dst[cnt++] = val;
            seen[seen_n++] = val;
        }
    }
    while (cnt < K_OUT) dst[cnt++] = 0;
}
"""

_dedup_cache: dict = {}

def _get_dedup_kernel(k_os: int, k_out: int) -> cp.RawKernel:
    key = (k_os, k_out)
    if key not in _dedup_cache:
        src = _DEDUP_SRC.replace("__K_OS__", str(k_os)).replace("__K_OUT__", str(k_out))
        _dedup_cache[key] = cp.RawKernel(src, "gpu_dedup")
    return _dedup_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# FastLSHIndex
# ─────────────────────────────────────────────────────────────────────────────

class FastLSHIndex(LSHIndex):
    """
    Extends LSHIndex with custom CUDA kernels for stages 7 and 9-11.
    Stage 8 (dedup) is eliminated; duplicates resolved lazily after top-k.
    """

    def build(self, X: cp.ndarray):
        super().build(X)
        L = self.p.n_tables
        # Stack per-table arrays into 2-D tensors for the stage-7 kernel
        self._sorted_keys_2d    = cp.stack(self._sorted_keys,    axis=0)  # (L, m)
        self._sorted_indices_2d = cp.stack(self._sorted_indices, axis=0)  # (L, m)

    def search_fast(self, Q: cp.ndarray, k: int,
                    oversample: int = 4, quiet: bool = False) -> cp.ndarray:
        """
        Fast search using custom CUDA kernels for stages 7 and 9-11.

        Returns (n, k) int32 array of approximate nearest-neighbor indices.
        quiet: if True, skip profiler report (for batch benchmarks).
        """
        prof   = CUDAProfiler()
        n      = Q.shape[0]
        L      = self.p.n_tables
        max_c  = self.p.max_cands_per_table
        C      = L * max_c
        BLOCK7 = 32         # one warp per (query, table) in stage 7
        k_os   = k * oversample   # output size before dedup (kernel block auto-selected)

        # ── stage 5 ──────────────────────────────────────────────────────────
        prof.start("5_query/hash")
        q_codes = self._project_and_hash(Q)
        prof.stop("5_query/hash")

        # ── stage 6 ──────────────────────────────────────────────────────────
        prof.start("6_query/pack_keys")
        q_keys = self._pack_keys(q_codes)       # (n, L) int64
        prof.stop("6_query/pack_keys")

        # ── stage 7 (custom kernel) ───────────────────────────────────────────
        prof.start("7_query/lookup_kernel")
        all_cands = cp.empty((n, C), dtype=cp.int32)
        kern7     = _get_stage7_kernel(max_c)
        kern7(
            (n, L), (BLOCK7,),
            (
                cp.ascontiguousarray(self._sorted_keys_2d),
                cp.ascontiguousarray(self._sorted_indices_2d),
                cp.ascontiguousarray(q_keys),
                all_cands,
                np.int32(L), np.int32(self._m),
            ),
        )
        cp.cuda.Stream.null.synchronize()
        prof.stop("7_query/lookup_kernel")

        # stage 8 eliminated — duplicates handled by oversampling in stage 9-11

        # ── stage 9-11 (fused kernel, k*oversample output) ───────────────────
        prof.start("9-11_query/fused_rerank_topk")
        kern9, BLOCK9, smem = _get_fused_kernel(k_os, self._d)
        out_ids   = cp.zeros((n, k_os), dtype=cp.int32)
        out_dists = cp.zeros((n, k_os), dtype=cp.float32)
        kern9(
            (n,), (BLOCK9,),
            (
                cp.ascontiguousarray(Q),
                cp.ascontiguousarray(self._dataset),
                cp.ascontiguousarray(all_cands),
                out_ids, out_dists,
                np.int32(C), np.int32(self._m),
            ),
            shared_mem=smem,
        )
        cp.cuda.Stream.null.synchronize()
        prof.stop("9-11_query/fused_rerank_topk")

        # ── GPU dedup of top (k*oversample) → k unique ───────────────────────
        prof.start("GPU_dedup")
        kern_d    = _get_dedup_kernel(k_os, k)
        final_ids = cp.zeros((n, k), dtype=cp.int32)
        BDEDUP    = 128
        kern_d(
            (int(np.ceil(n / BDEDUP)),), (BDEDUP,),
            (out_ids, final_ids, np.int32(n)),
        )
        cp.cuda.Stream.null.synchronize()
        prof.stop("GPU_dedup")

        if not quiet:
            prof.report("Fast Search Profile")
        return final_ids


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def _timed(fn, *a, **kw):
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record(); r = fn(*a, **kw); e.record(); e.synchronize()
    return r, float(cp.cuda.get_elapsed_time(s, e))


def run(m, d, n, k, params, bf_batch=128, oversample=4, seed=42):
    print("=" * 70)
    print(f"  lsh_fast.py  m={m:,}  d={d}  n={n}  k={k}")
    print(f"  L={params.n_tables}  K={params.n_projections}  "
          f"w={params.bucket_width}  max_cands={params.max_cands_per_table}")
    print(f"  oversample={oversample}  "
          f"(kernel outputs k*oversample={k*oversample}, then CPU dedup)")
    print("=" * 70)

    # Data
    rng = cp.random.default_rng(seed)
    X   = rng.standard_normal((m, d), dtype=cp.float32)
    Q   = rng.standard_normal((n, d), dtype=cp.float32)

    # Ground truth + brute-force baseline
    print("\nComputing ground truth …")
    gt = exact_neighbors_cuvs(X, Q, k)
    _, bf_ms = benchmark_exact_neighbors_trivial(X, Q, k, query_batch=bf_batch)
    print(f"  brute-force: {bf_ms:.1f} ms")

    # Build (includes stacking sorted arrays)
    print("\nBuilding FastLSHIndex …")
    idx = FastLSHIndex(params)
    _, t_build = _timed(idx.build, X)
    print(f"  build time: {t_build:.1f} ms")

    # Warm-up (JIT compile both kernels)
    print("\nWarming up kernels (JIT compile) …")
    _ = idx.search_fast(Q, k, oversample=oversample)

    # Timed run
    print("\nTimed run:")
    top, t_search = _timed(idx.search_fast, Q, k, oversample=oversample)
    r = recall_at_k(gt, cp.asnumpy(top))
    speedup = bf_ms / t_search

    print(f"\n  search time : {t_search:.1f} ms")
    print(f"  recall@{k}   : {r:.4f}")
    print(f"  speedup vs BF: {speedup:.2f}×")

    # Compare with original lsh.py search()
    print("\nOriginal lsh.py search() for comparison:")
    _, t_orig = _timed(idx.search, Q, k)
    r_orig = recall_at_k(gt, cp.asnumpy(idx.search(Q, k)[1]))
    print(f"  search time : {t_orig:.1f} ms")
    print(f"  recall@{k}   : {r_orig:.4f}")
    print(f"  speedup vs BF: {bf_ms/t_orig:.2f}×")

    print(f"\n  lsh_fast speedup vs lsh.py: {t_orig/t_search:.1f}×")


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
    parser.add_argument("--oversample",type=int,   default=4)
    parser.add_argument("--bf_batch",  type=int,   default=128)
    args = parser.parse_args()

    run(
        m=args.m, d=args.d, n=args.n, k=args.k,
        params=LSHParams(
            n_tables=args.L, n_projections=args.K,
            bucket_width=args.w, max_cands_per_table=args.max_cands,
        ),
        bf_batch=args.bf_batch,
        oversample=args.oversample,
    )
