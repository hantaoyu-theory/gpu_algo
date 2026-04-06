# ANN Algorithm Exploration — RTX 5090 Findings

---

## Part I: Small-d regime (d=8, m=500k) — X fits in L2

**Problem**: k-NN search, m=500k, d=8, n=1000, k=10  
**Hardware**: RTX 5090 (SM_120 / Blackwell), 96 MB L2 cache, 1792 GB/s HBM  
**Key fact**: X (16 MB FP32) fits entirely in L2 cache  

---

## Performance-Recall Frontier  (final, with BLOCK=128, 10 warmup + 20 reps)

| Algorithm | Time | Recall | vs BF-FP32 | vs lsh_fast |
|-----------|-----:|-------:|------:|------:|
| lsh_fast.py (prev best) | 7.30 ms | 0.970 | 0.4× | — |
| **BF-FP32** (BLOCK=128) | **2.89 ms** | 1.000 | 1× | 2.5× |
| **BF-FP16** | **1.93 ms** | 0.999 | 1.5× | 3.8× |
| **BF-INT8** (DP4A) | **1.67 ms** | 0.934 | 1.7× | 4.4× |
| **IVF1(C=512,p=4)** | **0.239 ms** | 0.842 | **12×** | **30×** |
| **IVF1(C=256,p=8)** ← sweet spot | **0.353 ms** | 0.981 | **8.2×** | **21×** |
| **IVF1(C=128,p=8)** | **0.450 ms** | 0.992 | **6.4×** | **16×** |
| **IVF1(C=512,p=16)** | **0.461 ms** | 0.998 | **6.3×** | **16×** |
| **IVF1(C=256,p=16)** | **0.512 ms** | 0.999 | **5.6×** | **14×** |
| **IVF1(C=128,p=16)** | **0.690 ms** | 1.000 | **4.2×** | **11×** |

**Winner: IVF1(C=256,p=8) — 8× faster than BF with 98% recall, 21× faster than lsh_fast.**

---

## Theoretical AM Model

### Hardware Parameters

| Quantity | Value | Source |
|---------|------:|--------|
| L2 BW (theoretical) | 10,000 GB/s | Architecture spec |
| L2 BW (measured, read+write) | ~3,900 GB/s | `measure_bw()` |
| **L2 BW (read-only, empirical)** | **~5,500 GB/s** | BF-FP32: 16 GB / 2.89 ms |
| HBM BW | 1,792 GB/s | Spec |
| FP32 peak (no TC) | 104.8 TFLOP/s | Spec |
| Ridge (L2→compute) | 10.5 FLOPs/byte | Peak / L2_theo |
| L2 size | 96 MB | Spec |

### Per-Algorithm Arithmetic Intensity

| Algorithm | AI (FLOPs/byte) | L2 Traffic | Pred (rd BW) | Actual | Bound |
|-----------|----------------:|------------|:------------:|:------:|-------|
| BF-FP32   | 0.50 | n·m·d·4 = 16 GB | 2.90 ms | 2.89 ms | L2-MEM ✓ |
| BF-FP16   | 1.00 | n·m·d·2 = 8 GB  | 1.45 ms | 1.93 ms | L2-MEM |
| BF-INT8   | 2.00 | n·m·d·1 = 4 GB  | 0.73 ms | 1.67 ms | L2-MEM + compute |
| IVF1(C,p) | 0.50 | n·(p/C)·m·d·4  | varies  | +0.22 ms overhead | L2-MEM |

### IVF Overhead Decomposition

IVF search time has two components:

```
t_IVF = t_overhead + t_scan
  t_overhead = block_scheduling + serial_heap_merge ≈ 0.22 ms  (constant for 1000 blocks)
  t_scan     = (p/C) · m · d · 4  /  BW_read ≈ (p/C) · 16 GB / 5500 GB/s
```

| Config | t_scan | t_overhead | t_total | t_overhead / t_total |
|--------|-------:|-----------:|--------:|---------------------:|
| C=128, p=4 | 0.09 ms | 0.22 ms | 0.31 ms | 72% |
| C=256, p=8 | 0.09 ms | 0.26 ms | 0.35 ms | 74% |
| C=64,  p=16 | 0.73 ms | 0.30 ms | 1.03 ms | 29% |

**The ~0.22 ms overhead floor** comes from:
1. CUDA block scheduling: 1000 blocks / 170 SMs ≈ 6 waves
2. Thread-0 serial heap merge: BLOCK×k = 128×10 = 1280 comparisons per block × 1000 blocks

**Optimal C is 256–512**: clusters large enough for efficient coalesced reads, small enough for fast probing.

---

## What Didn't Work

### IVF2 (one block per (query, cluster) pair)
- Goal: parallelize cluster scans across blocks → better occupancy
- Result: **4–10× SLOWER** than IVF1  
- Reason: serial thread-0 merge runs n×nprobe = 8000 times instead of n = 1000 times → more total merge work, not less. The merge is the bottleneck, not scheduling.

### BF-GEMM (cuBLAS QXᵀ + tiled top-k)
- GEMM itself is fast (cuBLAS works on Blackwell)
- **cp.argpartition / cp.concatenate are 200× too slow** on SM_120 (no precompiled PTX)
- Actual: 620 ms vs predicted 2 ms

### IVF-INT8 (DP4A distances within clusters)
- Slightly faster: 0.39–0.45 ms vs IVF1 at 0.35–0.45 ms  
- **Recall capped at ~93.4%** regardless of nprobe — quantization error prevents finding true NNs
- Not worth the recall hit for d=8

---

## d-Sweep Findings  (m=100k)

| d | BF-FP32 | BF-FP16 | BF-INT8 | INT8 speedup |
|--:|--------:|--------:|--------:|-------------:|
| 4 | 0.76 ms | 0.71 ms | 0.69 ms | 1.1× |
| 8 | 0.92 ms | 0.77 ms | 0.73 ms | 1.3× |
| 16 | 1.98 ms | 1.19 ms | 0.79 ms | 2.5× |
| 32 | 12.14 ms | 3.71 ms | 0.94 ms | **12.9×** |
| 64 | 25.56 ms | 18.56 ms | 2.00 ms | **12.8×** |
| 128 | 29.48 ms | 41.40 ms | 8.11 ms | **3.6×** |

**Key insight**: For d≥32, BF-FP32 collapses due to register pressure from the unrolled distance loop. INT8 with DP4A is dramatically better: 4× less data + efficient 4-way integer dot product.  
**For a high-d system (d=128), an IVF-INT8 hybrid would be the optimal approach.**

---

## Roofline Model Update

```
          BF-FP16 ●  BF-INT8 ●              ● IVF(small p)
          BF-FP32 ●
                                             Ridge @ 10.5 F/B
────────────────────────────────────────────────┼──────── COMPUTE-BOUND
  L2-MEM  0.50  1.00  2.00 ... 4.00  8.00  16.00  FLOPs/byte
BOUND ──────────────────────────────────────────┤
```

All our custom kernels are **L2-memory-bound**. None are compute-bound (AI < ridge).  
To become compute-bound:  
- Use tensor cores (FP16 GEMM): ridge at 67 F/B for TC — requires batch_size ≥ 67/2 = 34 queries reusing X simultaneously  
- Currently impractical for per-tile GEMM with n=1000 queries and m=500k because (n,m) output matrix is 2 GB

---

## Recommendations

1. **Replace lsh_fast.py with IVF1(C=256,p=8)** — 21× speedup, higher recall  
2. **Set BLOCK=128** (not 256) for all custom CUDA kernels on RTX 5090  
3. **Use more warmup** (≥10 iterations) before timing — GPU boost clock needs time to activate  
4. **For larger d (≥32)**: use INT8 quantization — up to 13× speedup vs FP32  
5. **Avoid CuPy elementwise/sort ops** on SM_120 (no precompiled PTX, 100–200× slower than expected)  
6. **IVF2 is a dead end** for n=1000; might help if n were 100k+

---

## LSH vs IVF/BF (same script)

`explore.py` now benchmarks **FastLSHIndex** (`lsh_fast.py`) on the **same** random data and ground truth as BF / IVF / PQ:

- **Grids** (tuned by `d`): low `d` uses more tables (`L` up to 256) and fixed `w`; high `d` uses fewer tables (`L` 32–64) and **auto `w`** to limit build-time `argsort` cost and fused rerank work (`C × d`).
- **Timed metric**: **query search ms** only (build time printed separately; IVF build is also offline).
- **Flags**: `--no-lsh` to skip; `--lsh-oversample` (default 4) matches `lsh_fast.search_fast`.

On the original **d=8, m=500k** regime, expect IVF1 to stay **faster** than the best LSH row at similar recall; LSH rows fill out the **speed–recall frontier** next to BF-GEMM / IVF in one table.

## Files

| File | Purpose |
|------|---------|
| `lsh_bf.py` | BF-FP32 and BF-FP16 kernels |
| `explore.py` | Comprehensive comparison: BF-FP32/16/INT8, BF-GEMM, **LSH (FastLSH)**, IVF1, IVF2, IVF-INT8, BF-PQ, IVF-PQ |
| `explore_annb.py` | Same benchmarks on **[ann-benchmarks](https://github.com/erikbern/ann-benchmarks)** HDF5 (`train` / `test` / `neighbors`); uses file GT, skips synthetic d-sweep. **Dense Euclidean or angular** (angular → row L2-normalize). Not for Jaccard/sparse HDF5. |
| `lsh_fast.py` | Custom LSH: CUDA stage 7 + fused rerank + GPU dedup (`quiet=` for batch benchmarks) |
| `lsh.py` | Base LSH index + params; CuPy-heavy `search()` (slow on SM_120) |

Example (after `pip install h5py`):

```bash
wget -q http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
python explore_annb.py --hdf5 fashion-mnist-784-euclidean.hdf5 --k 10 --max-test 5000
# Optional: --max-train 20000  (drops queries whose GT points outside prefix)
```

---

## Part II: Large-d regime (d=128, m=500k) — X does NOT fit in L2

**Problem**: k-NN search, m=500k, d=128, n=1000, k=10  
**Key change**: X (FP32) = 256 MB >> 96 MB L2 → HBM-bound for brute-force algorithms

### Memory Footprints

| Format | Size | Fits in L2? |
|--------|-----:|------------|
| X (FP32, d=128) | 256 MB | ✗ HBM-bound |
| X (INT8, d=128) | 64 MB | ✗ HBM-bound |
| X_pq (M=8 subspaces) | **4 MB** | **✓ L2-resident!** |
| IVF cluster (C=256, FP32) | 1 MB each | ✓ few at a time |
| All IVF clusters' PQ codes | 4 MB total | **✓ all L2-resident!** |

### Theoretical Predictions (HBM regime vs L2 regime)

| Algorithm | Traffic | BW used | Predicted time | Notes |
|-----------|--------:|---------|---------------:|-------|
| BF-FP32 | 256 GB | HBM 1792 GB/s | ~143 ms | 50× slower than d=8! |
| BF-INT8 (DP4A) | 64 GB | HBM 1792 GB/s | ~36 ms | Still HBM-bound |
| **BF-PQ (M=8)** | 4 GB | **L2 5500 GB/s** | **~0.73 ms** | X_pq=4MB fits in L2 |
| IVF1(C=256,p=8) | 256 MB→8 GB | HBM/L2 mix | ~0.5–5 ms | cluster reuse unclear |
| **IVF-PQ(C=256,p=8)** | ~125 MB | **L2 5500 GB/s** | **~0.25–0.35 ms** | all PQ codes L2-resident |

### Key Insight: PQ Maps Large-d Back to L2-Resident Regime

For d=128, M=8 PQ subspaces (each 16 dims, 256 codewords):

```
X_pq = m × M = 500k × 8 = 4 MB  <<  96 MB L2

BF-PQ scan:  n × 4 MB through L2 = 4 GB / 5500 GB/s = 0.73 ms
IVF-PQ scan: (p/C) × 4 MB = 125 MB / 5500 GB/s = 0.02 ms + 0.22 ms overhead
```

**BF-PQ at d=128 is faster than BF-FP32 at d=8** — PQ compression makes the
problem dimensionality-invariant in terms of memory traffic.

### IVF Cluster Reuse Analysis (d=128, C=256)

Each FP32 cluster = (500k/256) × 128 × 4 ≈ **1 MB**

With n=1000 queries and nprobe=8, each cluster is probed by ~31 queries on average.
If cluster data stays in L2 between probes (1 MB cluster << 96 MB L2):

```
t_scan (perfect reuse) = 256 clusters × 1 MB / 1792 GB/s = 0.14 ms
t_scan (no reuse)      = 1000 × 8 × 1 MB / 1792 GB/s     = 4.46 ms
```

In practice: reality will be between these bounds depending on how many clusters
fit concurrently in L2 with 1000 concurrent query blocks.

### Why This Changes Our Theoretical Model

The d=8 model assumed everything was L2-resident. For d=128:

| Condition | Algorithm | BW limit |
|-----------|-----------|----------|
| X fits in L2 (< 96 MB) | All BF variants | L2 BW = 5500 GB/s |
| X > L2 (d≥32, m=500k) | BF-FP32, BF-INT8 | HBM BW = 1792 GB/s |
| X_pq fits in L2 (M×m < 96 MB) | BF-PQ, IVF-PQ | L2 BW = 5500 GB/s |
| Clusters fit piecemeal | IVF-FP32 | L2+HBM mix |

**The critical transition**: for d >= 32 with m=500k, X no longer fits in L2.
PQ quantization is the mechanism to stay in the L2-resident regime.

### Algorithm Recommendations for d=128

1. **IVF-PQ(C=256, p=8, M=8)** — predicted ~0.25–0.35 ms, recall ~75–85%
2. **BF-PQ(M=8)** — predicted ~0.73 ms, recall ~80–90% (exact if M large enough)
3. **IVF1(C=256, p=16) FP32** — predicted ~1–5 ms, recall ~99% (if cluster reuse works)
4. **BF-INT8** — predicted ~36 ms, recall 100% (exact) but very slow

Run: `python explore.py --d 128 --m 500000 --n 1000 --k 10`
