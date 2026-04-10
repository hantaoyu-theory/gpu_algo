from __future__ import annotations

import cupy as cp
import numpy as np

from bf.fp32 import compile_kernel, tms, kernel_attrs


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


_BF32_Q2_SRC = _HEAP_DEVICE + r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__
extern "C" __global__ void bf_fp32_q2(
    const float* __restrict__ Q,
    const float* __restrict__ X,
    int*         __restrict__ out_ids,
    float*       __restrict__ out_dists,
    int m,
    int n)
{
    int pair = blockIdx.x;
    int q0 = pair * 2;
    int q1 = q0 + 1;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* q0_sh = smem;
    float* q1_sh = smem + D_VAL;
    float* sh_d = smem + 2 * D_VAL;
    int* sh_i = (int*)(sh_d + 2 * BLOCK * K_VAL);

    for (int j = tid; j < D_VAL; j += BLOCK) {
        q0_sh[j] = Q[q0 * D_VAL + j];
        if (q1 < n) q1_sh[j] = Q[q1 * D_VAL + j];
    }
    __syncthreads();

    float ld0[K_VAL], ld1[K_VAL];
    int li0[K_VAL], li1[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) {
        ld0[i] = 3.402823e+38f; li0[i] = -1;
        ld1[i] = 3.402823e+38f; li1[i] = -1;
    }
    int ln0 = 0, ln1 = 0;
    float lmax0 = 3.402823e+38f, lmax1 = 3.402823e+38f;
    int lmax_pos0 = 0, lmax_pos1 = 0;

    for (int ci = tid; ci < m; ci += BLOCK) {
        const float* xp = X + (long long)ci * D_VAL;
        float dist0 = 0.f;
        float dist1 = 0.f;
#if D_VAL <= 96
        #pragma unroll
#endif
        for (int di = 0; di < D_VAL; di++) {
            float x = xp[di];
            float d0 = q0_sh[di] - x;
            dist0 += d0 * d0;
            if (q1 < n) {
                float d1 = q1_sh[di] - x;
                dist1 += d1 * d1;
            }
        }
        heap_push(ld0, li0, &ln0, &lmax0, &lmax_pos0, dist0, ci, K_VAL);
        if (q1 < n) heap_push(ld1, li1, &ln1, &lmax1, &lmax_pos1, dist1, ci, K_VAL);
    }

    #pragma unroll
    for (int i = 0; i < K_VAL; i++) {
        sh_d[(tid * 2 + 0) * K_VAL + i] = ld0[i];
        sh_i[(tid * 2 + 0) * K_VAL + i] = li0[i];
        sh_d[(tid * 2 + 1) * K_VAL + i] = ld1[i];
        sh_i[(tid * 2 + 1) * K_VAL + i] = li1[i];
    }
    __syncthreads();

    if (tid == 0) {
        float fd0[K_VAL], fd1[K_VAL];
        int fi0[K_VAL], fi1[K_VAL];
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) {
            fd0[i] = 3.402823e+38f; fi0[i] = -1;
            fd1[i] = 3.402823e+38f; fi1[i] = -1;
        }
        int fn0 = 0, fn1 = 0;
        float fmax0 = 3.402823e+38f, fmax1 = 3.402823e+38f;
        int fmax_pos0 = 0, fmax_pos1 = 0;

        for (int t = 0; t < BLOCK; t++) {
            #pragma unroll
            for (int i = 0; i < K_VAL; i++) {
                float d0 = sh_d[(t * 2 + 0) * K_VAL + i];
                int id0 = sh_i[(t * 2 + 0) * K_VAL + i];
                if (id0 >= 0) heap_push(fd0, fi0, &fn0, &fmax0, &fmax_pos0, d0, id0, K_VAL);
                if (q1 < n) {
                    float d1 = sh_d[(t * 2 + 1) * K_VAL + i];
                    int id1 = sh_i[(t * 2 + 1) * K_VAL + i];
                    if (id1 >= 0) heap_push(fd1, fi1, &fn1, &fmax1, &fmax_pos1, d1, id1, K_VAL);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) {
            out_ids[q0 * K_VAL + i] = fi0[i];
            out_dists[q0 * K_VAL + i] = fd0[i];
            if (q1 < n) {
                out_ids[q1 * K_VAL + i] = fi1[i];
                out_dists[q1 * K_VAL + i] = fd1[i];
            }
        }
    }
}
"""


def run_bf_fp32_q2(Q_gpu: cp.ndarray, X_gpu: cp.ndarray, k: int, reps: int = 10):
    n, d = Q_gpu.shape
    m = X_gpu.shape[0]
    limit = 47 * 1024
    reg_cap = max(32, 65536 // max(64, 2 * d + 2 * k + 10))
    block = 32
    for cand in [128, 64, 32, 256]:
        if cand > reg_cap:
            continue
        if (2 * d + 4 * cand * k) * 4 < limit:
            block = cand
            break
    kern = compile_kernel(_BF32_Q2_SRC, "bf_fp32_q2", dict(K=k, D=d, BLOCK=block))
    out_ids = cp.zeros((n, k), cp.int32)
    out_dists = cp.zeros((n, k), cp.float32)
    shared_mem = (2 * d + 4 * block * k) * 4
    n_pairs = (n + 1) // 2

    def fn():
        kern(
            (n_pairs,),
            (block,),
            (Q_gpu, X_gpu, out_ids, out_dists, np.int32(m), np.int32(n)),
            shared_mem=shared_mem,
        )

    ms = tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize()
    fn()
    attrs = kernel_attrs(kern)
    attrs["kernel_variant"] = "two_queries_per_block"
    return {
        "ids": out_ids.get(),
        "ms": ms,
        "block": block,
        "shared_mem_bytes": shared_mem,
        "kernel_attrs": attrs,
    }
