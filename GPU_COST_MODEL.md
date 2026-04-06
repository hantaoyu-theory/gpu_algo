# GPU Cost Model

## Motivation

The current ANN results are enough to reject a pure FLOP-count model and also
reject a pure `time = bytes / peak_bandwidth` model.

The completed A100 `d=8` sweep shows:

- regular scan-style kernels are mostly explained by scanned bytes,
- but the fitted bandwidth is much lower than a simple memory-copy benchmark,
- and some families have a non-trivial fixed intercept,
- while other families stop correlating with bytes once quantization or control
  overhead dominates.

So the model needs three ingredients, not one.

## Proposed model

For a kernel or algorithm family `A`, model runtime as

```text
T(A, x) ~= T_move(A, x) + T_orch(A, x) + T_compute(A, x)
```

with:

```text
T_move(A, x)    = sum_l bytes_l(A, x) / BW_eff(A, l, pattern_l, reuse_l)
T_orch(A, x)    = launches + syncs + serial merge / selection + control overhead
T_compute(A, x) = flops(A, x) / P_eff(A, math_mode, reuse)
```

The key point is that `BW_eff` and `P_eff` are **effective** rates, not hardware
peak rates. They depend on:

- memory level: HBM, L2, shared, registers
- access pattern: sequential, gather/scatter, bucketed, reordered, sort-like
- reuse: whether data is streamed once or reused across many queries / tiles
- family-specific overhead: heap merge, top-k, hash lookup, compaction, etc.

## Decision rule

Instead of asking only “compute-bound or memory-bound?”, ask:

```text
Which of T_move, T_orch, T_compute dominates?
```

In practice:

- streaming map / scan kernels: `T_move` dominates
- IVF / approximate search: `T_move + T_orch`
- sort / selection / compaction: `T_orch + irregular T_move`
- GEMM / dense linear algebra with reuse: often `T_compute`, or balanced

## How ANN fits this model

For ANN, a useful family-level approximation is:

```text
T ~= scanned_bytes / BW_fit(family, regime) + overhead_fit(family, regime)
```

where:

- `family` might be `BF-FP32`, `IVF1`, `IVF2`, `IVF-INT8`, `BF-GEMM`
- `regime` encodes whether the database is L2-resident, HBM-driven, structured,
  quantized, etc.

This matches the A100 `d=8` observations:

- `IVF1` behaves roughly like bytes plus a tiny intercept
- `IVF2` needs a significant additive overhead term
- `IVF-INT8` is not captured by bytes alone
- `BF-GEMM` is dominated by orchestration / library pipeline behavior in the
  small-`d` regime

## Generalization beyond ANN

To validate that the model is not ANN-specific, benchmark several non-NN
problems with distinct structure:

1. streaming copy / SAXPY
   tests regular memory movement
2. reduction
   tests tree-style aggregation and synchronization
3. compaction / filtering
   tests irregular writes and selection overhead
4. sort
   tests multi-pass reordering with high orchestration cost
5. GEMM
   tests high-reuse dense compute

If the model is right, those workloads should separate into distinct dominant
terms:

- copy / SAXPY: `T_move`
- reduction: `T_move + T_orch`
- compaction / sort: `T_orch + irregular T_move`
- GEMM: `T_compute` or balanced `T_compute + T_move`

## What to fit experimentally

For each family, measure:

- total runtime
- effective bandwidth from estimated bytes moved
- effective math throughput from estimated FLOPs
- scaling with input size
- scaling with batch size / reuse

Then fit:

```text
T(size) ~= alpha * bytes + beta * flops + gamma * launches + delta
```

where some coefficients may collapse to zero for a given family.

This should be viewed as a **family-level predictive model**, not a universal
single constant.
