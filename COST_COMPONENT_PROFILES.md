# Cost Component Profiles

Rough decomposition of measured runtime into `T_move`, `T_compute`, and `T_orch`.

| experiment | algorithm | n/size | actual ms | T_move | T_compute | T_orch | dominant |
|---|---|---:|---:|---:|---:|---:|---|
|  | BF-FP32 | 100 | 2.075 | 2.075 | 0.000 | 0.000 | move |
|  | BF-FP16 | 100 | 1.760 | 1.455 | 0.200 | 0.106 | move |
|  | BF-INT8 | 100 | 1.707 | 0.727 | 0.100 | 0.880 | orch |
|  | BF-GEMM | 100 | 101.233 | 5.341 | 0.003 | 95.888 | orch |
|  | IVF1(C=128,p=8) | 100 | 0.456 | 0.181 | 0.017 | 0.258 | orch |
|  | IVF1(C=256,p=8) | 100 | 0.391 | 0.091 | 0.008 | 0.292 | orch |
|  | IVF2(C=256,p=8) | 100 | 0.376 | 0.076 | 0.008 | 0.291 | orch |
|  | IVF-INT8(C=128,p=8) | 100 | 0.373 | 0.045 | 0.006 | 0.322 | orch |
|  | BF-FP32 | 300 | 3.038 | 3.038 | 0.000 | 0.000 | move |
|  | BF-FP16 | 300 | 2.243 | 2.243 | 0.000 | 0.000 | move |
|  | BF-INT8 | 300 | 2.110 | 2.110 | 0.000 | 0.000 | move |
|  | BF-GEMM | 300 | 136.342 | 16.024 | 0.010 | 120.308 | orch |
|  | IVF1(C=128,p=8) | 300 | 0.607 | 0.544 | 0.050 | 0.013 | move |
|  | IVF1(C=256,p=8) | 300 | 0.508 | 0.272 | 0.025 | 0.210 | move |
|  | IVF2(C=256,p=8) | 300 | 0.685 | 0.228 | 0.025 | 0.432 | orch |
|  | IVF-INT8(C=128,p=8) | 300 | 0.459 | 0.136 | 0.019 | 0.304 | orch |
|  | BF-FP32 | 1000 | 8.456 | 8.456 | 0.000 | 0.000 | move |
|  | BF-FP16 | 1000 | 5.965 | 5.965 | 0.000 | 0.000 | move |
|  | BF-INT8 | 1000 | 2.763 | 2.763 | 0.000 | 0.000 | move |
|  | BF-GEMM | 1000 | 397.161 | 53.412 | 0.035 | 343.714 | orch |
|  | IVF1(C=128,p=8) | 1000 | 1.186 | 1.186 | 0.000 | 0.000 | move |
|  | IVF1(C=256,p=8) | 1000 | 0.880 | 0.880 | 0.000 | 0.000 | move |
|  | IVF2(C=256,p=8) | 1000 | 1.717 | 0.760 | 0.083 | 0.874 | orch |
|  | IVF-INT8(C=128,p=8) | 1000 | 0.686 | 0.455 | 0.062 | 0.169 | move |
|  | BF-FP32 | 3000 | 23.068 | 23.068 | 0.000 | 0.000 | move |
|  | BF-FP16 | 3000 | 15.654 | 15.654 | 0.000 | 0.000 | move |
|  | BF-INT8 | 3000 | 7.083 | 7.083 | 0.000 | 0.000 | move |
|  | BF-GEMM | 3000 | 731.707 | 160.237 | 0.104 | 571.365 | orch |
|  | IVF1(C=128,p=8) | 3000 | 3.488 | 3.488 | 0.000 | 0.000 | move |
|  | IVF1(C=256,p=8) | 3000 | 2.388 | 2.388 | 0.000 | 0.000 | move |
|  | IVF2(C=256,p=8) | 3000 | 8.893 | 2.280 | 0.250 | 6.363 | orch |
|  | IVF-INT8(C=128,p=8) | 3000 | 4.849 | 1.364 | 0.188 | 3.298 | orch |
| gaussian | BF-FP32 | 100 | 24.686 | 23.273 | 1.413 | 0.000 | move |
| gaussian | BF-FP16 | 100 | 26.029 | 11.636 | 1.600 | 12.793 | orch |
| gaussian | BF-INT8 | 100 | 6.661 | 5.818 | 0.800 | 0.043 | move |
| gaussian | BF-GEMM | 100 | 29.384 | 29.384 | 0.000 | 0.000 | move |
| gaussian | IVF1(C=64,p=16) | 100 | 8.758 | 5.808 | 0.533 | 2.417 | move |
| gaussian | IVF-INT8(C=128,p=8) | 100 | 2.199 | 0.364 | 0.050 | 1.785 | orch |
| gaussian | BF-FP32 | 300 | 72.257 | 69.818 | 2.438 | 0.000 | move |
| gaussian | BF-FP16 | 300 | 73.778 | 34.909 | 4.800 | 34.068 | move |
| gaussian | BF-INT8 | 300 | 18.096 | 17.455 | 0.642 | 0.000 | move |
| gaussian | BF-GEMM | 300 | 59.842 | 59.842 | 0.000 | 0.000 | move |
| gaussian | IVF1(C=64,p=16) | 300 | 28.379 | 17.423 | 1.600 | 9.356 | move |
| gaussian | IVF-INT8(C=128,p=8) | 300 | 3.363 | 1.091 | 0.150 | 2.122 | orch |
| gaussian | BF-FP32 | 1000 | 308.832 | 232.727 | 21.333 | 54.772 | move |
| gaussian | BF-FP16 | 1000 | 265.893 | 116.364 | 16.000 | 133.530 | orch |
| gaussian | BF-INT8 | 1000 | 61.195 | 58.182 | 3.013 | 0.000 | move |
| gaussian | BF-GEMM | 1000 | 135.112 | 135.112 | 0.000 | 0.000 | move |
| gaussian | IVF1(C=64,p=16) | 1000 | 106.740 | 58.076 | 5.333 | 43.331 | move |
| gaussian | IVF-INT8(C=128,p=8) | 1000 | 7.624 | 3.636 | 0.500 | 3.488 | move |
| gaussian | BF-FP32 | 2000 | 615.294 | 465.455 | 42.667 | 107.173 | move |
| gaussian | BF-FP16 | 2000 | 530.057 | 232.727 | 32.000 | 265.330 | orch |
| gaussian | BF-INT8 | 2000 | 116.687 | 116.364 | 0.323 | 0.000 | move |
| gaussian | BF-GEMM | 2000 | 258.528 | 258.528 | 0.000 | 0.000 | move |
| gaussian | IVF1(C=64,p=16) | 2000 | 209.219 | 116.152 | 10.667 | 82.400 | move |
| gaussian | IVF-INT8(C=128,p=8) | 2000 | 13.291 | 7.273 | 1.000 | 5.018 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP32 | 50 | 35.226 | 17.105 | 1.568 | 16.552 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP16 | 50 | 32.569 | 8.553 | 1.176 | 22.840 | orch |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-INT8 | 50 | 4.610 | 4.276 | 0.334 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-GEMM | 50 | 10.936 | 10.936 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF1(C=64,p=4) | 50 | 7.584 | 1.067 | 0.098 | 6.419 | orch |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF-INT8(C=64,p=16) | 50 | 4.149 | 1.069 | 0.147 | 2.933 | orch |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP32 | 100 | 35.858 | 34.211 | 1.647 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP16 | 100 | 32.743 | 17.105 | 2.352 | 13.286 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-INT8 | 100 | 4.589 | 4.589 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-GEMM | 100 | 5.223 | 5.223 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF1(C=64,p=4) | 100 | 7.920 | 2.134 | 0.196 | 5.590 | orch |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF-INT8(C=64,p=16) | 100 | 4.647 | 2.138 | 0.294 | 2.215 | orch |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP32 | 200 | 47.543 | 47.543 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP16 | 200 | 42.699 | 34.211 | 4.704 | 3.784 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-INT8 | 200 | 10.603 | 10.603 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-GEMM | 200 | 6.678 | 6.678 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF1(C=64,p=4) | 200 | 8.829 | 4.269 | 0.392 | 4.168 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF-INT8(C=64,p=16) | 200 | 5.655 | 4.276 | 0.588 | 0.791 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP32 | 300 | 70.429 | 70.429 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP16 | 300 | 66.796 | 51.316 | 7.056 | 8.423 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-INT8 | 300 | 21.303 | 21.303 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-GEMM | 300 | 8.644 | 8.644 | 0.000 | 0.000 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF1(C=64,p=4) | 300 | 11.165 | 6.403 | 0.588 | 4.174 | move |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF-INT8(C=64,p=16) | 300 | 6.872 | 6.415 | 0.458 | 0.000 | move |
| gpu_model_experiments | copy | 1000000 | 0.013 | 0.013 | 0.000 | 0.000 | move |
| gpu_model_experiments | saxpy | 1000000 | 0.314 | 0.019 | 0.000 | 0.295 | orch |
| gpu_model_experiments | reduce | 1000000 | 0.025 | 0.006 | 0.000 | 0.019 | orch |
| gpu_model_experiments | compact | 1000000 | 0.189 | 0.013 | 0.000 | 0.176 | orch |
| gpu_model_experiments | sort | 1000000 | 0.160 | 0.000 | 0.000 | 0.160 | orch |
| gpu_model_experiments | copy | 5000000 | 0.028 | 0.028 | 0.000 | 0.000 | move |
| gpu_model_experiments | saxpy | 5000000 | 0.102 | 0.043 | 0.000 | 0.059 | orch |
| gpu_model_experiments | reduce | 5000000 | 0.024 | 0.014 | 0.000 | 0.010 | move |
| gpu_model_experiments | compact | 5000000 | 0.229 | 0.028 | 0.000 | 0.201 | orch |
| gpu_model_experiments | sort | 5000000 | 0.352 | 0.000 | 0.000 | 0.352 | orch |
| gpu_model_experiments | copy | 20000000 | 0.100 | 0.100 | 0.000 | 0.000 | move |
| gpu_model_experiments | saxpy | 20000000 | 0.365 | 0.150 | 0.000 | 0.215 | orch |
| gpu_model_experiments | reduce | 20000000 | 0.053 | 0.050 | 0.000 | 0.003 | move |
| gpu_model_experiments | compact | 20000000 | 1.057 | 0.100 | 0.000 | 0.957 | orch |
| gpu_model_experiments | sort | 20000000 | 7.416 | 0.000 | 0.000 | 7.416 | orch |
| gpu_model_experiments | gemm:1024x1024 @ 1024x1024 | 1073741824 | 0.039 | 0.004 | 0.009 | 0.026 | orch |
| gpu_model_experiments | gemm:2048x2048 @ 2048x2048 | 8589934592 | 0.075 | 0.016 | 0.059 | 0.000 | compute |
| d8_gaussian | LSH L100 K3 w3 m2k | 3000 | 45.382 | 45.382 | 0.000 | 0.000 | move |
| d128_gaussian | LSH L64 K3 w0 m800 | 2000 | 97.348 | 97.348 | 0.000 | 0.000 | move |
| fashion784 | LSH L64 K3 w0 m800 | 300 | 125.538 | 125.538 | 0.000 | 0.000 | move |
