# GPU Model Experiments Results

Hardware:
- RunPod `NVIDIA A100 80GB PCIe`

Script:
- `gpu_model_experiments.py`

Artifact:
- `gpu_model_experiments.json`

## Purpose

These experiments test whether the proposed GPU cost model generalizes beyond
nearest-neighbor search.

The workloads were chosen to stress different terms:

- `copy`: regular streaming movement
- `saxpy`: movement-dominated elementwise math
- `reduce`: movement plus tree aggregation
- `compact`: irregular movement plus selection overhead
- `sort`: heavy orchestration / reordering
- `gemm`: high-reuse dense compute

## A100 measurements

### Vector size = 20,000,000

| workload | ms | effective BW | effective GFLOP/s | interpretation |
|---|---:|---:|---:|---|
| `copy` | `0.100` | `1601 GB/s` | `0` | close to streaming-HBM limit |
| `saxpy` | `0.365` | `657 GB/s` | `109.5` | movement-dominated, lower effective BW than copy |
| `reduce` | `0.053` | `1513 GB/s` | — | still mostly movement-bound |
| `compact` | `1.057` | `151 GB/s` | — | irregular writes / selection overhead dominate |
| `sort` | `7.416` | — | — | multi-pass reordering, not explainable by one-byte term |

### GEMM

| workload | ms | effective BW | effective GFLOP/s | interpretation |
|---|---:|---:|---:|---|
| `1024x1024 @ 1024x1024` | `0.039` | `162 GB/s` | `55.4 TFLOP/s` | compute-heavy |
| `2048x2048 @ 2048x2048` | `0.075` | `337 GB/s` | `230.2 TFLOP/s` | strongly compute/reuse-driven |

## What this means for the model

These runs support the proposed decomposition:

```text
T ~= T_move + T_orch + T_compute
```

Observations:

1. `copy` and `reduce` behave like movement-dominated kernels.
2. `saxpy` is still movement-dominated, but the effective BW is much lower than
   raw copy, so “one universal bandwidth constant” is already wrong.
3. `compact` and `sort` are the clearest examples where orchestration and
   irregular access dominate.
4. `gemm` clearly lives in a different regime, where reuse is high enough that
   dense compute throughput matters.

## Updated takeaway

The ANN-specific fitted model

```text
T ~= scanned_bytes / BW_fit(family, regime) + overhead_fit(family, regime)
```

is best viewed as one slice of a more general GPU model:

```text
T ~= bytes / BW_eff(pattern, level, family)
   + orchestration_overhead(family)
   + flops / P_eff(reuse, math_mode, family)
```

This is a better match to:

- ANN scan / IVF kernels
- streaming vector kernels
- reduction
- compaction / sorting
- GEMM
