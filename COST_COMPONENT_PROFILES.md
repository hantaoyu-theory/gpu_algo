# Cost Component Profiles

| experiment | algorithm | n/size | actual ms |
|---|---|---:|---:|
|  | BF-FP32 | 100 | 2.075 |
|  | BF-FP16 | 100 | 1.760 |
|  | BF-INT8 | 100 | 1.707 |
|  | BF-GEMM | 100 | 101.233 |
|  | IVF1(C=128,p=8) | 100 | 0.456 |
|  | IVF1(C=256,p=8) | 100 | 0.391 |
|  | IVF2(C=256,p=8) | 100 | 0.376 |
|  | IVF-INT8(C=128,p=8) | 100 | 0.373 |
|  | BF-FP32 | 300 | 3.038 |
|  | BF-FP16 | 300 | 2.243 |
|  | BF-INT8 | 300 | 2.110 |
|  | BF-GEMM | 300 | 136.342 |
|  | IVF1(C=128,p=8) | 300 | 0.607 |
|  | IVF1(C=256,p=8) | 300 | 0.508 |
|  | IVF2(C=256,p=8) | 300 | 0.685 |
|  | IVF-INT8(C=128,p=8) | 300 | 0.459 |
|  | BF-FP32 | 1000 | 8.456 |
|  | BF-FP16 | 1000 | 5.965 |
|  | BF-INT8 | 1000 | 2.763 |
|  | BF-GEMM | 1000 | 397.161 |
|  | IVF1(C=128,p=8) | 1000 | 1.186 |
|  | IVF1(C=256,p=8) | 1000 | 0.880 |
|  | IVF2(C=256,p=8) | 1000 | 1.717 |
|  | IVF-INT8(C=128,p=8) | 1000 | 0.686 |
|  | BF-FP32 | 3000 | 23.068 |
|  | BF-FP16 | 3000 | 15.654 |
|  | BF-INT8 | 3000 | 7.083 |
|  | BF-GEMM | 3000 | 731.707 |
|  | IVF1(C=128,p=8) | 3000 | 3.488 |
|  | IVF1(C=256,p=8) | 3000 | 2.388 |
|  | IVF2(C=256,p=8) | 3000 | 8.893 |
|  | IVF-INT8(C=128,p=8) | 3000 | 4.849 |
| gaussian | BF-FP32 | 100 | 24.686 |
| gaussian | BF-FP16 | 100 | 26.029 |
| gaussian | BF-INT8 | 100 | 6.661 |
| gaussian | BF-GEMM | 100 | 29.384 |
| gaussian | IVF1(C=64,p=16) | 100 | 8.758 |
| gaussian | IVF-INT8(C=128,p=8) | 100 | 2.199 |
| gaussian | BF-FP32 | 300 | 72.257 |
| gaussian | BF-FP16 | 300 | 73.778 |
| gaussian | BF-INT8 | 300 | 18.096 |
| gaussian | BF-GEMM | 300 | 59.842 |
| gaussian | IVF1(C=64,p=16) | 300 | 28.379 |
| gaussian | IVF-INT8(C=128,p=8) | 300 | 3.363 |
| gaussian | BF-FP32 | 1000 | 308.832 |
| gaussian | BF-FP16 | 1000 | 265.893 |
| gaussian | BF-INT8 | 1000 | 61.195 |
| gaussian | BF-GEMM | 1000 | 135.112 |
| gaussian | IVF1(C=64,p=16) | 1000 | 106.740 |
| gaussian | IVF-INT8(C=128,p=8) | 1000 | 7.624 |
| gaussian | BF-FP32 | 2000 | 615.294 |
| gaussian | BF-FP16 | 2000 | 530.057 |
| gaussian | BF-INT8 | 2000 | 116.687 |
| gaussian | BF-GEMM | 2000 | 258.528 |
| gaussian | IVF1(C=64,p=16) | 2000 | 209.219 |
| gaussian | IVF-INT8(C=128,p=8) | 2000 | 13.291 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP32 | 50 | 35.226 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP16 | 50 | 32.569 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-INT8 | 50 | 4.610 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-GEMM | 50 | 10.936 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF1(C=64,p=4) | 50 | 7.584 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF-INT8(C=64,p=16) | 50 | 4.149 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP32 | 100 | 35.858 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP16 | 100 | 32.743 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-INT8 | 100 | 4.589 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-GEMM | 100 | 5.223 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF1(C=64,p=4) | 100 | 7.920 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF-INT8(C=64,p=16) | 100 | 4.647 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP32 | 200 | 47.543 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP16 | 200 | 42.699 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-INT8 | 200 | 10.603 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-GEMM | 200 | 6.678 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF1(C=64,p=4) | 200 | 8.829 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF-INT8(C=64,p=16) | 200 | 5.655 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP32 | 300 | 70.429 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-FP16 | 300 | 66.796 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-INT8 | 300 | 21.303 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | BF-GEMM | 300 | 8.644 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF1(C=64,p=4) | 300 | 11.165 |
| /workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5 (euclidean) | IVF-INT8(C=64,p=16) | 300 | 6.872 |
| gpu_model_experiments | copy | 1000000 | 0.013 |
| gpu_model_experiments | saxpy | 1000000 | 0.314 |
| gpu_model_experiments | reduce | 1000000 | 0.025 |
| gpu_model_experiments | compact | 1000000 | 0.189 |
| gpu_model_experiments | sort | 1000000 | 0.160 |
| gpu_model_experiments | copy | 5000000 | 0.028 |
| gpu_model_experiments | saxpy | 5000000 | 0.102 |
| gpu_model_experiments | reduce | 5000000 | 0.024 |
| gpu_model_experiments | compact | 5000000 | 0.229 |
| gpu_model_experiments | sort | 5000000 | 0.352 |
| gpu_model_experiments | copy | 20000000 | 0.100 |
| gpu_model_experiments | saxpy | 20000000 | 0.365 |
| gpu_model_experiments | reduce | 20000000 | 0.053 |
| gpu_model_experiments | compact | 20000000 | 1.057 |
| gpu_model_experiments | sort | 20000000 | 7.416 |
| gpu_model_experiments | gemm:1024x1024 @ 1024x1024 | 1073741824 | 0.039 |
| gpu_model_experiments | gemm:2048x2048 @ 2048x2048 | 8589934592 | 0.075 |
| d8_gaussian | LSH L100 K3 w3 m2k | 3000 | 45.382 |
| d128_gaussian | LSH L64 K3 w0 m800 | 2000 | 97.348 |
| fashion784 | LSH L64 K3 w0 m800 | 300 | 125.538 |
