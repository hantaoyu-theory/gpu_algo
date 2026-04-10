from __future__ import annotations

import cupy as cp
import numpy as np

from bf.fp32 import compile_kernel, choose_block, tms, kernel_attrs


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
"""


_TILED_PARTIAL_SRC = _HEAP_DEVICE + r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__
extern "C" __global__ void bf_fp32_tiled_partial(
    const float* __restrict__ Q,
    const float* __restrict__ X,
    int*         __restrict__ part_ids,
    float*       __restrict__ part_dists,
    int m,
    int tile_size)
{
    int qi = blockIdx.x;
    int tile = blockIdx.y;
    int tid = threadIdx.x;
    int tile_start = tile * tile_size;
    int tile_end = min(tile_start + tile_size, m);

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* sh_d = smem + D_VAL;
    int*   sh_i = (int*)(sh_d + BLOCK * K_VAL);
    for (int j = tid; j < D_VAL; j += BLOCK)
        q_sh[j] = Q[qi * D_VAL + j];
    __syncthreads();

    float ld[K_VAL];
    int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0;
    float lmax = 3.402823e+38f;
    int lmax_pos = 0;

    for (int ci = tile_start + tid; ci < tile_end; ci += BLOCK) {
        const float* xp = X + (long long)ci * D_VAL;
        float dist = 0.f;
#if D_VAL <= 96
        #pragma unroll
#endif
        for (int di = 0; di < D_VAL; di++) {
            float dif = q_sh[di] - xp[di];
            dist += dif * dif;
        }
        heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, ci, K_VAL);
    }

    #pragma unroll
    for (int i = 0; i < K_VAL; i++) {
        sh_d[tid * K_VAL + i] = ld[i];
        sh_i[tid * K_VAL + i] = li[i];
    }
    __syncthreads();

    if (tid == 0) {
        float fd[K_VAL];
        int fi[K_VAL];
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) { fd[i] = 3.402823e+38f; fi[i] = -1; }
        int fn = 0;
        float fmax = 3.402823e+38f;
        int fmax_pos = 0;

        for (int t = 0; t < BLOCK; t++) {
            #pragma unroll
            for (int i = 0; i < K_VAL; i++) {
                float d = sh_d[t * K_VAL + i];
                int id = sh_i[t * K_VAL + i];
                if (id < 0) continue;
                heap_push(fd, fi, &fn, &fmax, &fmax_pos, d, id, K_VAL);
            }
        }
        long long out_base = ((long long)qi * gridDim.y + tile) * K_VAL;
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) {
            part_ids[out_base + i] = fi[i];
            part_dists[out_base + i] = fd[i];
        }
    }
}
"""


_TILED_MERGE_SRC = _HEAP_DEVICE + r"""
#define K_VAL  __K__
#define BLOCK  __BLOCK__
extern "C" __global__ void bf_fp32_tiled_merge(
    const int*   __restrict__ part_ids,
    const float* __restrict__ part_dists,
    int*         __restrict__ out_ids,
    float*       __restrict__ out_dists,
    int n_tiles)
{
    int qi = blockIdx.x;
    int tid = threadIdx.x;

    float ld[K_VAL];
    int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = 3.402823e+38f; li[i] = -1; }
    int ln = 0;
    float lmax = 3.402823e+38f;
    int lmax_pos = 0;

    int total = n_tiles * K_VAL;
    int base = qi * total;
    for (int idx = tid; idx < total; idx += BLOCK) {
        float dist = part_dists[base + idx];
        int id = part_ids[base + idx];
        if (id < 0) continue;
        heap_push(ld, li, &ln, &lmax, &lmax_pos, dist, id, K_VAL);
    }

    extern __shared__ float smem[];
    float* sh_d = smem;
    int* sh_i = (int*)(sh_d + BLOCK * K_VAL);
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) {
        sh_d[tid * K_VAL + i] = ld[i];
        sh_i[tid * K_VAL + i] = li[i];
    }
    __syncthreads();

    if (tid == 0) {
        float fd[K_VAL];
        int fi[K_VAL];
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) { fd[i] = 3.402823e+38f; fi[i] = -1; }
        int fn = 0;
        float fmax = 3.402823e+38f;
        int fmax_pos = 0;
        for (int t = 0; t < BLOCK; t++) {
            #pragma unroll
            for (int i = 0; i < K_VAL; i++) {
                float d = sh_d[t * K_VAL + i];
                int id = sh_i[t * K_VAL + i];
                if (id < 0) continue;
                heap_push(fd, fi, &fn, &fmax, &fmax_pos, d, id, K_VAL);
            }
        }
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) {
            out_ids[qi * K_VAL + i] = fi[i];
            out_dists[qi * K_VAL + i] = fd[i];
        }
    }
}
"""


def run_bf_fp32_tiled(
    Q_gpu: cp.ndarray,
    X_gpu: cp.ndarray,
    k: int,
    tile_size: int = 125_000,
    reps: int = 10,
):
    n, d = Q_gpu.shape
    m = X_gpu.shape[0]
    n_tiles = (m + tile_size - 1) // tile_size
    block = choose_block(k, d)
    partial_kernel = compile_kernel(
        _TILED_PARTIAL_SRC,
        "bf_fp32_tiled_partial",
        dict(K=k, D=d, BLOCK=block),
    )
    merge_kernel = compile_kernel(
        _TILED_MERGE_SRC,
        "bf_fp32_tiled_merge",
        dict(K=k, BLOCK=block),
    )

    part_ids = cp.full((n, n_tiles, k), -1, dtype=cp.int32)
    part_dists = cp.full((n, n_tiles, k), cp.inf, dtype=cp.float32)
    out_ids = cp.full((n, k), -1, dtype=cp.int32)
    out_dists = cp.full((n, k), cp.inf, dtype=cp.float32)

    partial_smem = (d + 2 * block * k) * 4
    merge_smem = 2 * block * k * 4

    def fn():
        partial_kernel(
            (n, n_tiles),
            (block,),
            (Q_gpu, X_gpu, part_ids, part_dists, np.int32(m), np.int32(tile_size)),
            shared_mem=partial_smem,
        )
        merge_kernel(
            (n,),
            (block,),
            (part_ids, part_dists, out_ids, out_dists, np.int32(n_tiles)),
            shared_mem=merge_smem,
        )

    ms = tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize()
    fn()
    return {
        "ids": out_ids.get(),
        "ms": ms,
        "block": block,
        "tile_size": tile_size,
        "n_tiles": n_tiles,
        "partial_shared_mem_bytes": partial_smem,
        "merge_shared_mem_bytes": merge_smem,
        "partial_kernel_attrs": kernel_attrs(partial_kernel),
        "merge_kernel_attrs": kernel_attrs(merge_kernel),
    }
