# A100 80GB — `explore.py` benchmark results

**Hardware:** NVIDIA A100 80GB PCIe, driver 565.x, CUDA 12.x, CuPy 14.x (`cupy-cuda12x`)  
**Workload:** Random Gaussian `X ∈ R^{m×d}`, `Q ∈ R^{n×d}`, k-NN with `k=10`, `m=500_000`, `n=1000`, `seed=42`  
**Script:** `python3 -u explore.py --m 500000 --d <8|128> --n 1000 --k 10`  
**Logs on pod:** `/tmp/explore_a100_d8.log`, `/tmp/explore_a100_d128.log`

---

## d = 8 (X = 16 MB FP32, fits many GPUs’ L2)

| Algorithm | ms | recall@10 |
|-----------|---:|----------:|
| BF-FP32 | 8.46 | 1.000 |
| BF-FP16 | 5.98 | 0.999 |
| BF-INT8 | 2.76 | 0.934 |
| BF-GEMM | 127.48 | 1.000 |
| **LSH L100 K3 w3 m2k** | **16.98** | **0.929** |
| LSH L100 K3 w3 m500 | 7.52 | 0.565 |
| **IVF1(C=256,p=8)** | **0.88** | **0.981** |
| IVF1(C=128,p=8) | 1.16 | 0.992 |
| IVF1(C=64,p=8) | 1.75 | 0.997 |
| IVF1(C=64,p=16) | 2.84 | 1.000 |
| IVF2(C=256,p=8) | 1.72 | 0.981 |
| IVF-INT8(C=128,p=8) | 0.69 | 0.930 |
| BF-PQ(M=2) | 2.37 | 0.136 |
| IVF-PQ(C=256,p=8) | 0.54 | 0.149 |

**Takeaways (d=8)**

- **IVF1** is fastest at high recall (~0.98–1.0): **~0.9–1.8 ms** vs **BF-FP32 ~8.5 ms**.
- **LSH** reaches **~0.93 recall** at **~17 ms** (config `L=100,K=3,w=3,max_cands=2000`); a faster LSH row (~7.5 ms) drops recall to **~0.56**.
- **BF-GEMM** is **slow here (~127 ms)** despite exact recall — CuPy `argpartition`/merge overhead is much smaller than on SM120 Blackwell, but still dominates vs custom BF kernels.
- **BF-PQ / IVF-PQ** use **M=2** subspaces for `d=8` (script rule: `M = min(8, d//4)`), so **recall ~0.14** is expected (PQ is not meaningful at this `d`).

---

## d = 128 (X = 256 MB FP32, does not fit 40 MB A100 L2)

| Algorithm | ms | recall@10 |
|-----------|---:|----------:|
| BF-FP32 | 457.90 | 1.000 |
| BF-FP16 | 615.74 | 0.999 |
| BF-INT8 | 126.89 | 0.957 |
| **BF-GEMM** | **136.65** | **1.000** |
| LSH L32 w0 m500 | 18.18 | 0.044 |
| LSH L64 w0 m800 | 51.81 | 0.128 |
| IVF1(C=64,p=16) | 116.11 | 0.548 |
| IVF1(C=128,p=16) | 58.53 | 0.388 |
| IVF1(C=256,p=16) | 33.79 | 0.285 |
| IVF2(C=256,p=8) | 17.12 | 0.177 |
| IVF-INT8(C=128,p=8) | 10.25 | 0.244 |
| BF-PQ(M=8) | 6.98 | 0.023 |
| IVF-PQ(C=256,p=8) | 1.36 | 0.018 |

**Takeaways (d=128, i.i.d. Gaussian)**

- **Exact methods:** **BF-GEMM (~137 ms) < BF-INT8 (~127 ms) < BF-FP32 (~458 ms)** on this A100 stack — GEMM path is strong; FP16 BF regression is likely compute/format overhead in the custom kernel.
- **IVF / LSH / PQ** show **low recall** on **unstructured** high-dimensional Gaussian data unless `nprobe`, `L`, `max_cands`, or PQ training are scaled up a lot (curse of dimensionality + weak cluster structure).
- **BF-PQ(M=8)** is **fast (~7 ms)** but **~2% recall** on this data: standard PQ without OPQ / learned rotation / more codebooks is a poor match to i.i.d. Gaussians.
- **IVF-PQ** is **sub-2 ms** but recall **~2%** for the same reason.

---

## d-sweep (m=100k, n=1000, k=10) — both runs

| d | BF-FP32 | BF-INT8 | BF-PQ |
|--:|--------:|--------:|------:|
| 4 | 1.22 ms | 1.12 ms | — |
| 8 | 2.15 ms | 1.16 ms | 1.07 ms |
| 16 | 6.86 ms | 1.30 ms | 1.15 ms |
| 32 | 23.4 ms | 2.16 ms | 2.08 ms |
| 64 | 47–48 ms | 6.58 ms | 2.07 ms |
| 128 | 92.3 ms | 24.6 ms | 2.20 ms |

(INT8 and PQ stay attractive as `d` grows for this scan-based BF implementation.)

---

## How to re-run (short polling, no long local sleeps)

On the pod:

```bash
# follow log
tail -f /tmp/explore_a100_d8.log

# or wait until finished (checks every 15s)
while ! grep -q '^  Done\.' /tmp/explore_a100_d128.log; do sleep 15; done; tail -40 /tmp/explore_a100_d128.log
```

---

## Repo fix applied during this run

- `explore.py` d-sweep: **`M_d` UnboundLocalError** when `d=4` — fixed by initializing `M_d` and only using it when PQ runs.
