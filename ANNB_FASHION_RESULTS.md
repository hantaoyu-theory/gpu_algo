# Fashion-MNIST-784-euclidean on A100 (explore_annb)

**Host:** RunPod A100 (`explore.py` with cooperative `q_sh` load + `_KERNEL_SRC_VERSION=4`).  
**Command:** `python3 -u explore_annb.py --hdf5 fashion-mnist-784-euclidean.hdf5 --k 10 --max-test 150 --no-lsh`  
**Setup:** `m=60_000`, `d=784`, `n=150`, `k=10`, ground truth from HDF5 `neighbors`.  
**Log on pod:** `/tmp/annb_fashion_v4.log` (full run ~125 lines).

## Bug fix (why BF/IVF were wrong before)

Kernels used `if (tid < D) q_sh[tid] = …`. With **`d=784`** and **`BLOCK=128`**, only part of the query was loaded into shared memory; L2 distances were garbage. **Fix:** `for (int j = tid; j < D; j += BLOCK) q_sh[j] = …` (and the INT8 `D_Q` variant). See `ANNB_STATUS.md`.

## Measured L2 bandwidth (warm-up)

~**1669 GB/s** (16 MB resident array).

## Main timings and recall@10

| algorithm | ms | recall@10 |
|-----------|---:|----------:|
| BF-FP32 | 47.76 | 1.0000 |
| BF-FP16 | 42.62 | 1.0000 |
| BF-INT8 | 10.59 | 0.9913 |
| BF-GEMM | 7.75 | 1.0000 |
| IVF1(C=64,p=4) | 8.72 | 0.9907 |
| IVF1(C=64,p=16) | 22.85 | 1.0000 |
| IVF2(C=128,p=16) | 17.32 | 1.0000 |
| IVF-INT8(C=64,p=16) | 5.53 | 0.9913 |
| BF-PQ(M=8) | 0.58 | 0.4073 |
| IVF-PQ(C=64,p=8) | 3.29 | 0.3913 |
| IVF-PQ(C=256,p=16) | 12.14 | 0.3927 |

**PQ build:** ~**383 s** for 8× K-means-256 on 60k×784 (CPU `sklearn`); search is sub-ms but recall ~0.39–0.41 at this setting.

## Notes

- LSH was skipped (`--no-lsh`).
- BF-GEMM is fastest exact-style baseline here; BF-FP32/FP16 are correct after the shared-memory load fix.
- IVF recalls in the high-0.99s reflect approximate indexing (not a distance bug).
