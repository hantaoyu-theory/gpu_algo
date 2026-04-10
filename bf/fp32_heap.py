from __future__ import annotations

import cupy as cp
import numpy as np

from bf.fp32 import compile_kernel, choose_block, tms, kernel_attrs


_BF32_HEAP_SRC = r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__

__device__ __forceinline__ void heap_swap(float* ld, int* li, int a, int b)
{
    float td = ld[a]; ld[a] = ld[b]; ld[b] = td;
    int ti = li[a]; li[a] = li[b]; li[b] = ti;
}

__device__ __forceinline__ void sift_up(float* ld, int* li, int idx)
{
    while (idx > 0) {
        int parent = (idx - 1) >> 1;
        if (ld[parent] >= ld[idx]) break;
        heap_swap(ld, li, parent, idx);
        idx = parent;
    }
}

__device__ __forceinline__ void sift_down(float* ld, int* li, int idx, int n)
{
    while (true) {
        int left = idx * 2 + 1;
        int right = left + 1;
        int largest = idx;
        if (left < n && ld[left] > ld[largest]) largest = left;
        if (right < n && ld[right] > ld[largest]) largest = right;
        if (largest == idx) break;
        heap_swap(ld, li, idx, largest);
        idx = largest;
    }
}

__device__ __forceinline__ void heap_push_max(
    float* ld, int* li, int* ln, float dist, int idx)
{
    if (*ln < K_VAL) {
        int pos = *ln;
        ld[pos] = dist;
        li[pos] = idx;
        (*ln)++;
        sift_up(ld, li, pos);
    } else if (dist < ld[0]) {
        ld[0] = dist;
        li[0] = idx;
        sift_down(ld, li, 0, K_VAL);
    }
}

__device__ __forceinline__ void heap_sort_asc(float* ld, int* li, int n)
{
    for (int end = n - 1; end > 0; --end) {
        heap_swap(ld, li, 0, end);
        sift_down(ld, li, 0, end);
    }
}

extern "C" __global__ void bf_fp32_heap(
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
        q_sh[j] = Q[qi * D_VAL + j];
    __syncthreads();

    float ld[K_VAL];
    int li[K_VAL];
    #pragma unroll
    for (int i = 0; i < K_VAL; i++) { ld[i] = -3.402823e+38f; li[i] = -1; }
    int ln = 0;

    for (int ci = tid; ci < m; ci += BLOCK) {
        const float* xp = X + (long long)ci * D_VAL;
        float dist = 0.f;
#if D_VAL <= 96
        #pragma unroll
#endif
        for (int di = 0; di < D_VAL; di++) {
            float dif = q_sh[di] - xp[di];
            dist += dif * dif;
        }
        heap_push_max(ld, li, &ln, dist, ci);
    }

    if (ln < K_VAL) {
        for (int i = ln; i < K_VAL; i++) {
            ld[i] = 3.402823e+38f;
            li[i] = -1;
        }
    } else {
        heap_sort_asc(ld, li, K_VAL);
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
        for (int i = 0; i < K_VAL; i++) { fd[i] = -3.402823e+38f; fi[i] = -1; }
        int fn = 0;
        for (int t = 0; t < BLOCK; t++) {
            #pragma unroll
            for (int i = 0; i < K_VAL; i++) {
                int id = sh_i[t * K_VAL + i];
                float dist = sh_d[t * K_VAL + i];
                if (id < 0) continue;
                heap_push_max(fd, fi, &fn, dist, id);
            }
        }
        if (fn == K_VAL) heap_sort_asc(fd, fi, K_VAL);
        for (int i = fn; i < K_VAL; i++) {
            fd[i] = 3.402823e+38f;
            fi[i] = -1;
        }
        #pragma unroll
        for (int i = 0; i < K_VAL; i++) {
            out_ids[qi * K_VAL + i] = fi[i];
            out_dists[qi * K_VAL + i] = fd[i];
        }
    }
}
"""


def run_bf_fp32_heap(Q_gpu: cp.ndarray, X_gpu: cp.ndarray, k: int, reps: int = 10):
    n, d = Q_gpu.shape
    m = X_gpu.shape[0]
    block = choose_block(k, d)
    kern = compile_kernel(_BF32_HEAP_SRC, "bf_fp32_heap", dict(K=k, D=d, BLOCK=block))
    out_ids = cp.zeros((n, k), cp.int32)
    out_dists = cp.zeros((n, k), cp.float32)
    shared_mem = (d + 2 * block * k) * 4

    def fn():
        kern(
            (n,),
            (block,),
            (Q_gpu, X_gpu, out_ids, out_dists, np.int32(m)),
            shared_mem=shared_mem,
        )

    ms = tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize()
    fn()
    attrs = kernel_attrs(kern)
    attrs["kernel_variant"] = "true_max_heap"
    return {
        "ids": out_ids.get(),
        "ms": ms,
        "block": block,
        "shared_mem_bytes": shared_mem,
        "kernel_attrs": attrs,
    }

