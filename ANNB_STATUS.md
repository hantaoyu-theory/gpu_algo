# ann-benchmarks HDF5 experiments — status

## What ran (A100, Fashion-MNIST-784-euclidean)

- **Command:** `explore_annb.py --hdf5 fashion-mnist-784-euclidean.hdf5 --k 10 --max-test 200`
- **Corpus:** `m=60_000`, `d=784`, `n=200`, ground truth from HDF5 `neighbors`.
- **Log (partial):** `/tmp/annb_fashion.log` on pod — run **stopped before PQ / summary** (only 45 lines captured in one pull; may have been killed or still running).

### Numbers from partial log (before fix)

| Method | ms | recall@10 |
|--------|---:|----------:|
| BF-FP32 | 64.8 | **0.0005** (broken) |
| BF-FP16 | 40.8 | **0.0005** (broken) |
| BF-INT8 | 10.4 | 0.053 |
| **BF-GEMM** | **6.6** | **1.000** (trust this as exact) |
| LSH (L32–L64) | 26–80 | ~0.001–0.004 |
| IVF1 | 8–29 | ~0.007–0.014 |
| IVF2 | 9–13 | ~0.005–0.020 |
| IVF-INT8 | 6–9 | ~0.04–0.07 |

**Diagnosis (root cause):** BF / IVF search kernels load the query into shared memory with `if (tid < D_VAL) q_sh[tid] = …`. When **`D_VAL > BLOCK`** (e.g. **d=784**, **BLOCK=128**), only the first **BLOCK** dimensions are loaded; the rest of `q_sh` is **uninitialized**, so L2 distances are garbage. **BF-GEMM** never used that pattern, so it stayed correct.

Secondary: `#pragma unroll` on huge inner loops can hurt codegen; those loops are only unrolled when `D_VAL <= 96` (and `D_Q` / `KSUB_VAL` similarly).

## Fix applied (explore.py)

1. **Cooperative query load:** `for (int j = tid; j < D_VAL; j += BLOCK) q_sh[j] = …` (and the same for `D_Q` in INT8 paths) in BF-FP32/16/INT8, IVF search, IVF2 partial, IVF-INT8 search.
2. **Conditional unroll** on inner dimension loops when `D_VAL` / `D_Q` / `KSUB_VAL` ≤ 96.
3. **Kernel cache:** `_KERNEL_SRC_VERSION` bumped so CuPy does not reuse stale JIT modules after edits.

## Subsampling train (`--max-train`)

- If `max_train < |train|`, file neighbors are invalid; loader now sets **`gt=None`** and recomputes exact k-NN on the prefix (see `explore_annb.py`).

## Re-run on pod (no sleep; one SSH)

```bash
scp -P 42505 -i ~/.ssh/id_ed25519 explore.py explore_annb.py root@HOST:/workspace/gpu_algo/
ssh root@HOST -p 42505 -i ~/.ssh/id_ed25519 \
  'cd /workspace/gpu_algo && python3 -u explore_annb.py --hdf5 fashion-mnist-784-euclidean.hdf5 --k 10 --max-test 200 2>&1 | tee /tmp/annb_fashion_v2.log'
```

Expect **BF-FP32 / BF-FP16 recall ≈ 1.0** after the **cooperative `q_sh` load** fix (see `ANNB_FASHION_RESULTS.md` for a full A100 run).

## Results write-up

- **`ANNB_FASHION_RESULTS.md`** — Fashion-MNIST-784, `n=150`, full table + PQ build time.

## recall_at_k

- Uses **`int(...)`** on ids so int32 / int64 sets intersect correctly (defensive).
