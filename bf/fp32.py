from __future__ import annotations

import cupy as cp
import numpy as np


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


_BF32_SMALLK_SRC = r"""
#define K_VAL  __K__
#define D_VAL  __D__
#define BLOCK  __BLOCK__

__device__ __forceinline__ void insert_sorted(
    float* ld, int* li, float dist, int idx)
{
    if (dist >= ld[K_VAL - 1]) return;
    int pos = K_VAL - 1;
    while (pos > 0 && dist < ld[pos - 1]) {
        ld[pos] = ld[pos - 1];
        li[pos] = li[pos - 1];
        pos--;
    }
    ld[pos] = dist;
    li[pos] = idx;
}

extern "C" __global__ void bf_fp32_smallk(
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
    for (int i = 0; i < K_VAL; i++) {
        ld[i] = 3.402823e+38f;
        li[i] = -1;
    }

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
        insert_sorted(ld, li, dist, ci);
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
        for (int i = 0; i < K_VAL; i++) {
            fd[i] = 3.402823e+38f;
            fi[i] = -1;
        }

        for (int t = 0; t < BLOCK; t++) {
            #pragma unroll
            for (int i = 0; i < K_VAL; i++) {
                float dist = sh_d[t * K_VAL + i];
                int idx = sh_i[t * K_VAL + i];
                if (idx < 0) continue;
                if (dist >= fd[K_VAL - 1]) continue;
                int pos = K_VAL - 1;
                while (pos > 0 && dist < fd[pos - 1]) {
                    fd[pos] = fd[pos - 1];
                    fi[pos] = fi[pos - 1];
                    pos--;
                }
                fd[pos] = dist;
                fi[pos] = idx;
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


_KERN_CACHE: dict = {}
_KERNEL_SRC_VERSION = 1


def compile_kernel(src: str, name: str, subs: dict) -> cp.RawKernel:
    key = (_KERNEL_SRC_VERSION, name, tuple(sorted(subs.items())))
    if key in _KERN_CACHE:
        return _KERN_CACHE[key]
    code = src
    for k, v in subs.items():
        code = code.replace(f"__{k}__", str(v))
    kern = cp.RawKernel(code, name)
    _KERN_CACHE[key] = kern
    return kern


def choose_block(k: int, d: int) -> int:
    """Choose a block size that fits the per-query shared-memory footprint."""
    limit = 47 * 1024
    reg_cap = max(32, 65536 // max(64, d + k + 10))
    for b in [128, 64, 32, 256, 512]:
        if b > reg_cap:
            continue
        if (d + 2 * b * k) * 4 < limit:
            return b
    return 32


def tms(fn, warmup: int = 10, reps: int = 20) -> float:
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record()
    ev1.synchronize()
    return float(cp.cuda.get_elapsed_time(ev0, ev1)) / reps


def kernel_attrs(kern: cp.RawKernel) -> dict:
    attrs = dict(kern.attributes)
    keys = [
        "num_regs",
        "shared_size_bytes",
        "local_size_bytes",
        "const_size_bytes",
        "max_threads_per_block",
        "binary_version",
        "ptx_version",
    ]
    return {key: int(attrs[key]) for key in keys if key in attrs}


def build_bf_fp32_runner(Q_gpu: cp.ndarray, X_gpu: cp.ndarray, k: int):
    n, d = Q_gpu.shape
    m = X_gpu.shape[0]
    block = choose_block(k, d)
    use_smallk = k in {2, 4, 8, 16, 32}
    if use_smallk:
        kern = compile_kernel(_BF32_SMALLK_SRC, "bf_fp32_smallk", dict(K=k, D=d, BLOCK=block))
    else:
        kern = compile_kernel(_BF32_SRC, "bf_fp32", dict(K=k, D=d, BLOCK=block))
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

    attrs = kernel_attrs(kern)
    attrs["kernel_variant"] = "smallk_sorted" if use_smallk else "generic_heap"
    return fn, out_ids, out_dists, block, shared_mem, attrs


def run_bf_fp32(Q_gpu: cp.ndarray, X_gpu: cp.ndarray, k: int, reps: int = 10):
    fn, out_ids, _, block, shared_mem, attrs = build_bf_fp32_runner(Q_gpu, X_gpu, k)
    ms = tms(fn, reps=reps)
    cp.cuda.Stream.null.synchronize()
    fn()
    return {
        "ids": out_ids.get(),
        "ms": ms,
        "block": block,
        "shared_mem_bytes": shared_mem,
        "kernel_attrs": attrs,
    }
