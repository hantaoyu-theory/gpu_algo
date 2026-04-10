# Nearest-Neighbor Summary (RTX 5090)

Measured totals only. All numbers below were collected on the user-provided RTX 5090 VM.

Cell format: `query_ms / recall@10`.

Benchmark matrix:
- Distribution sweep: `distribution in {gaussian, uniform, laplace}`, `d in {8,16,32,64}`, `m=250,000`, `n=1000`
- Gaussian query scaling: `d in {8,16,32,64}`, `m=250,000`, `n in {100,1000,3000}`
- Gaussian database scaling: `d in {8,16,32,64}`, `n=1000`, `m in {100,000,250,000,500,000}`
- `k=10` throughout

Distribution definitions:
- `gaussian`: each coordinate is i.i.d. `N(0, 1)`
- `uniform`: each coordinate is i.i.d. `U(-sqrt(3), sqrt(3))`, so variance is `1`
- `laplace`: each coordinate is i.i.d. `Laplace(0, 1/sqrt(2))`, so variance is `1`

Main takeaways:
- `BF-INT8` is the fastest brute-force variant in every Gaussian `n`- and `m`-scaling case, with recall typically `0.93-0.96`.
- `BF-FP16` is the best high-recall brute-force path on this 5090 sweep; it stays near `0.999+` recall and is consistently faster than `BF-FP32`.
- `BF-GEMM` remains much slower than the custom brute-force kernels at these problem sizes, which is consistent with the CuPy tile-materialization and top-k overhead discussed elsewhere in the repo.
- `IVF-INT8(C=128,p=8)` is the fastest approximate method in the cross-distribution sweep, but its recall drops sharply as `d` increases.

## Distribution Sweep, d = 8

`m=250,000`, `n=1000`, `k=10`

| algorithm | gaussian | uniform | laplace |
|---|---:|---:|---:|
| `BF-FP32` | `1.674 ms / 1.0000` | `1.674 ms / 1.0000` | `1.670 ms / 1.0000` |
| `BF-FP16` | `1.256 ms / 0.9993` | `1.254 ms / 0.9990` | `1.250 ms / 0.9990` |
| `BF-INT8` | `1.146 ms / 0.9393` | `1.138 ms / 0.9768` | `1.133 ms / 0.8848` |
| `BF-GEMM` | `307.139 ms / 1.0000` | `305.382 ms / 1.0000` | `308.931 ms / 1.0000` |
| `IVF1(C=128,p=8)` | `0.310 ms / 0.9878` | `0.294 ms / 0.9940` | `0.366 ms / 0.9905` |
| `IVF-INT8(C=128,p=8)` | `0.276 ms / 0.9319` | `0.262 ms / 0.9718` | `0.328 ms / 0.8802` |

## Distribution Sweep, d = 16

`m=250,000`, `n=1000`, `k=10`

| algorithm | gaussian | uniform | laplace |
|---|---:|---:|---:|
| `BF-FP32` | `4.576 ms / 1.0000` | `4.565 ms / 1.0000` | `4.578 ms / 1.0000` |
| `BF-FP16` | `2.517 ms / 0.9994` | `2.509 ms / 0.9993` | `2.517 ms / 0.9996` |
| `BF-INT8` | `1.306 ms / 0.9512` | `1.294 ms / 0.9824` | `1.302 ms / 0.9078` |
| `BF-GEMM` | `310.720 ms / 1.0000` | `307.520 ms / 1.0000` | `311.713 ms / 1.0000` |
| `IVF1(C=128,p=8)` | `0.606 ms / 0.8090` | `0.600 ms / 0.8590` | `0.610 ms / 0.8844` |
| `IVF-INT8(C=128,p=8)` | `0.317 ms / 0.7975` | `0.316 ms / 0.8536` | `0.322 ms / 0.8348` |

## Distribution Sweep, d = 32

`m=250,000`, `n=1000`, `k=10`

| algorithm | gaussian | uniform | laplace |
|---|---:|---:|---:|
| `BF-FP32` | `30.235 ms / 1.0000` | `30.250 ms / 1.0000` | `30.267 ms / 0.9999` |
| `BF-FP16` | `8.936 ms / 0.9992` | `8.925 ms / 0.9996` | `8.925 ms / 0.9995` |
| `BF-INT8` | `1.794 ms / 0.9600` | `1.792 ms / 0.9866` | `1.801 ms / 0.9160` |
| `BF-GEMM` | `304.368 ms / 1.0000` | `308.162 ms / 1.0000` | `307.373 ms / 1.0000` |
| `IVF1(C=128,p=8)` | `2.010 ms / 0.5432` | `1.999 ms / 0.5762` | `1.959 ms / 0.7410` |
| `IVF-INT8(C=128,p=8)` | `0.403 ms / 0.5428` | `0.399 ms / 0.5760` | `0.408 ms / 0.7200` |

## Distribution Sweep, d = 64

`m=250,000`, `n=1000`, `k=10`

| algorithm | gaussian | uniform | laplace |
|---|---:|---:|---:|
| `BF-FP32` | `63.340 ms / 1.0000` | `63.290 ms / 1.0000` | `63.253 ms / 1.0000` |
| `BF-FP16` | `46.035 ms / 0.9998` | `46.051 ms / 0.9994` | `46.046 ms / 0.9992` |
| `BF-INT8` | `4.875 ms / 0.9609` | `4.866 ms / 0.9865` | `4.861 ms / 0.9181` |
| `BF-GEMM` | `306.191 ms / 1.0000` | `308.660 ms / 1.0000` | `308.144 ms / 1.0000` |
| `IVF1(C=128,p=8)` | `2.962 ms / 0.3471` | `2.946 ms / 0.3759` | `2.999 ms / 0.5268` |
| `IVF-INT8(C=128,p=8)` | `0.637 ms / 0.3471` | `0.643 ms / 0.3759` | `0.635 ms / 0.5244` |

## Gaussian Query Scaling, d = 8

`m=250,000`, `k=10`

| algorithm | n=100 | n=1000 | n=3000 |
|---|---:|---:|---:|
| `BF-FP32` | `0.799 ms / 1.0000` | `1.671 ms / 1.0000` | `5.833 ms / 1.0000` |
| `BF-FP16` | `0.734 ms / 0.9990` | `1.251 ms / 0.9994` | `3.952 ms / 0.9991` |
| `BF-INT8` | `0.681 ms / 0.9310` | `1.135 ms / 0.9383` | `3.389 ms / 0.9343` |
| `BF-GEMM` | `21.765 ms / 1.0000` | `309.280 ms / 1.0000` | `1104.445 ms / 1.0000` |

## Gaussian Query Scaling, d = 16

`m=250,000`, `k=10`

| algorithm | n=100 | n=1000 | n=3000 |
|---|---:|---:|---:|
| `BF-FP32` | `1.396 ms / 1.0000` | `4.579 ms / 1.0000` | `13.447 ms / 1.0000` |
| `BF-FP16` | `1.051 ms / 1.0000` | `2.525 ms / 0.9998` | `7.898 ms / 0.9993` |
| `BF-INT8` | `0.725 ms / 0.9460` | `1.314 ms / 0.9546` | `4.140 ms / 0.9540` |
| `BF-GEMM` | `19.819 ms / 1.0000` | `305.552 ms / 1.0000` | `1101.833 ms / 1.0000` |

## Gaussian Query Scaling, d = 32

`m=250,000`, `k=10`

| algorithm | n=100 | n=1000 | n=3000 |
|---|---:|---:|---:|
| `BF-FP32` | `3.306 ms / 1.0000` | `30.230 ms / 1.0000` | `97.237 ms / 1.0000` |
| `BF-FP16` | `1.890 ms / 1.0000` | `8.922 ms / 0.9998` | `26.636 ms / 0.9997` |
| `BF-INT8` | `0.941 ms / 0.9570` | `1.793 ms / 0.9531` | `6.075 ms / 0.9544` |
| `BF-GEMM` | `19.834 ms / 1.0000` | `312.738 ms / 1.0000` | `1103.857 ms / 1.0000` |

## Gaussian Query Scaling, d = 64

`m=250,000`, `k=10`

| algorithm | n=100 | n=1000 | n=3000 |
|---|---:|---:|---:|
| `BF-FP32` | `5.950 ms / 1.0000` | `63.216 ms / 1.0000` | `199.632 ms / 1.0000` |
| `BF-FP16` | `6.145 ms / 0.9990` | `46.059 ms / 0.9994` | `152.506 ms / 0.9994` |
| `BF-INT8` | `1.605 ms / 0.9650` | `4.860 ms / 0.9603` | `16.763 ms / 0.9596` |
| `BF-GEMM` | `22.831 ms / 1.0000` | `310.370 ms / 1.0000` | `1102.336 ms / 1.0000` |

## Gaussian Database Scaling, d = 8

`n=1000`, `k=10`

| algorithm | m=100k | m=250k | m=500k |
|---|---:|---:|---:|
| `BF-FP32` | `0.910 ms / 1.0000` | `1.671 ms / 1.0000` | `2.872 ms / 1.0000` |
| `BF-FP16` | `0.766 ms / 0.9988` | `1.246 ms / 0.9992` | `1.916 ms / 0.9994` |
| `BF-INT8` | `0.721 ms / 0.9419` | `1.132 ms / 0.9364` | `1.658 ms / 0.9326` |
| `BF-GEMM` | `123.117 ms / 1.0000` | `307.545 ms / 1.0000` | `615.470 ms / 1.0000` |

## Gaussian Database Scaling, d = 16

`n=1000`, `k=10`

| algorithm | m=100k | m=250k | m=500k |
|---|---:|---:|---:|
| `BF-FP32` | `1.966 ms / 1.0000` | `4.579 ms / 1.0000` | `8.867 ms / 1.0000` |
| `BF-FP16` | `1.185 ms / 0.9990` | `2.518 ms / 0.9992` | `4.644 ms / 0.9991` |
| `BF-INT8` | `0.783 ms / 0.9571` | `1.305 ms / 0.9509` | `2.060 ms / 0.9474` |
| `BF-GEMM` | `122.495 ms / 1.0000` | `307.562 ms / 1.0000` | `621.374 ms / 1.0000` |

## Gaussian Database Scaling, d = 32

`n=1000`, `k=10`

| algorithm | m=100k | m=250k | m=500k |
|---|---:|---:|---:|
| `BF-FP32` | `12.138 ms / 1.0000` | `30.235 ms / 1.0000` | `61.140 ms / 0.9999` |
| `BF-FP16` | `3.702 ms / 0.9994` | `8.924 ms / 0.9994` | `17.593 ms / 0.9995` |
| `BF-INT8` | `0.944 ms / 0.9611` | `1.796 ms / 0.9530` | `3.131 ms / 0.9518` |
| `BF-GEMM` | `123.333 ms / 1.0000` | `306.671 ms / 1.0000` | `607.094 ms / 0.9999` |

## Gaussian Database Scaling, d = 64

`n=1000`, `k=10`

| algorithm | m=100k | m=250k | m=500k |
|---|---:|---:|---:|
| `BF-FP32` | `25.271 ms / 1.0000` | `63.273 ms / 1.0000` | `127.624 ms / 1.0000` |
| `BF-FP16` | `18.517 ms / 0.9998` | `46.076 ms / 0.9998` | `93.164 ms / 0.9992` |
| `BF-INT8` | `1.999 ms / 0.9641` | `4.861 ms / 0.9574` | `10.063 ms / 0.9557` |
| `BF-GEMM` | `123.185 ms / 1.0000` | `302.707 ms / 1.0000` | `618.598 ms / 1.0000` |
