# Cost Component Summary

This table uses the largest completed `n` for each case so the decomposition is
less dominated by tiny-batch launch effects:

- `d=8 Gaussian`: `n=3000`
- `d=128 Gaussian`: `n=2000`
- `Fashion-MNIST-784`: `n=300`

The numbers come from the rough post-hoc profiler in `profile_cost_components.py`.

## d = 8 Gaussian (`n=3000`)

| algorithm | recall | total time (ms) | T_move (ms) | T_compute (ms) | T_orch (ms) | dominant |
|---|---:|---:|---:|---:|---:|---|
| `BF-FP32` | `1.0000` | `23.068` | `23.068` | `0.000` | `0.000` | move |
| `BF-FP16` | `0.9991` | `15.654` | `15.654` | `0.000` | `0.000` | move |
| `BF-INT8` | `0.9329` | `7.083` | `7.083` | `0.000` | `0.000` | move |
| `BF-GEMM` | `1.0000` | `731.707` | `160.237` | `0.104` | `571.365` | orch |
| `LSH L100 K3 w3 m2k` | `0.9290` | `45.382` | `45.382` | `0.000` | `0.000` | move |
| `IVF1(C=128,p=8)` | `0.9913` | `3.488` | `3.488` | `0.000` | `0.000` | move |
| `IVF1(C=256,p=8)` | `0.9807` | `2.388` | `2.388` | `0.000` | `0.000` | move |
| `IVF2(C=256,p=8)` | `0.9807` | `8.893` | `2.280` | `0.250` | `6.363` | orch |
| `IVF-INT8(C=128,p=8)` | `0.9278` | `4.849` | `1.364` | `0.188` | `3.298` | orch |

## d = 128 Gaussian (`n=2000`)

| algorithm | recall | total time (ms) | T_move (ms) | T_compute (ms) | T_orch (ms) | dominant |
|---|---:|---:|---:|---:|---:|---|
| `BF-FP32` | `1.0000` | `615.294` | `465.455` | `42.667` | `107.173` | move |
| `BF-FP16` | `0.99935` | `530.057` | `232.727` | `32.000` | `265.330` | orch |
| `BF-INT8` | `0.95935` | `116.687` | `116.364` | `0.323` | `0.000` | move |
| `BF-GEMM` | `1.0000` | `258.528` | `258.528` | `0.000` | `0.000` | move |
| `LSH L64 K3 w0 m800` | `0.24805` | `97.348` | `97.348` | `0.000` | `0.000` | move |
| `IVF1(C=64,p=16)` | `0.5879` | `209.219` | `116.152` | `10.667` | `82.400` | move |
| `IVF-INT8(C=128,p=8)` | `0.2562` | `13.291` | `7.273` | `1.000` | `5.018` | move |

## Fashion-MNIST-784 (`n=300`)

| algorithm | recall | total time (ms) | T_move (ms) | T_compute (ms) | T_orch (ms) | dominant |
|---|---:|---:|---:|---:|---:|---|
| `BF-FP32` | `1.0000` | `70.429` | `70.429` | `0.000` | `0.000` | move |
| `BF-FP16` | `1.0000` | `66.796` | `51.316` | `7.056` | `8.423` | move |
| `BF-INT8` | `0.9940` | `21.303` | `21.303` | `0.000` | `0.000` | move |
| `BF-GEMM` | `1.0000` | `8.644` | `8.644` | `0.000` | `0.000` | move |
| `LSH L64 K3 w0 m800` | `0.0010` | `125.538` | `125.538` | `0.000` | `0.000` | move |
| `IVF1(C=64,p=4)` | `0.9867` | `11.165` | `6.403` | `0.588` | `4.174` | move |
| `IVF-INT8(C=64,p=16)` | `0.9940` | `6.872` | `6.415` | `0.458` | `0.000` | move |

## Caveat

These are **rough** profiles, not hardware-counter measurements.

Recall values come from matching targeted recall runs when the final time-sweep
was executed with `--skip-recall`.

`BF-GEMM` is the least reliable split because the current estimator does not
cleanly separate:

- GEMM math
- output-tile materialization
- top-k selection / merge

So treat its row as directional only.
