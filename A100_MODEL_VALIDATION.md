# A100 Model Validation

Date: 2026-04-05

Hardware:
- RunPod `NVIDIA A100 80GB PCIe`

Remote workspace:
- `/workspace/gpu_algo`

Remote run logs:
- `/workspace/gpu_algo/runs/synth_d8.log`
- `/workspace/gpu_algo/runs/synth_d128.log`
- `/workspace/gpu_algo/runs/fashion300.log`

Remote JSON export:
- `/workspace/gpu_algo/runs/synth_d8.json`

Local copies pulled back:
- `gpu_algo/synth_d8.json`
- `gpu_algo/synth_d8.log`
- `gpu_algo/synth_d128.log`
- `gpu_algo/fashion300.log`

## What changed

Two code changes were needed before the A100 runs were usable:

1. `explore.py` now exports machine-readable results and fits a simple affine model
   `actual_ms ~= intercept_ms + traffic_GB / BW_fit`.
2. `explore.py` and `explore_annb.py` now support `--export-json`, and `explore.py`
   supports `--skip-pq` / `--skip-dsweep` for focused overnight runs.

There was also a portability fix: raw CUDA source strings had non-ASCII comments,
which caused CuPy/NVRTC to fail on the pod when writing kernel source.

## Finished run: synthetic `d=8`

Command:

```bash
python3 -u explore.py \
  --m 500000 --d 8 --n 1000 --k 10 \
  --no-lsh --skip-pq --skip-dsweep \
  --export-json /workspace/gpu_algo/runs/synth_d8.json
```

Measured microbenchmark bandwidth:
- `1685 GB/s` on a 16 MB resident array

Best completed rows at `recall >= 0.90`:

| algorithm | ms | recall@10 |
|---|---:|---:|
| `IVF1(C=256,p=8)` | `0.89` | `0.9814` |
| `IVF-INT8(C=128,p=8)` | `0.68` | `0.9295` |
| `IVF1(C=128,p=4)` | `1.63` | `0.9281` |
| `BF-INT8` | `7.52` | `0.9340` |
| `BF-FP32` | `27.15` | `1.0000` |

Model-fit output from the run:

| family | points | intercept (ms) | BW_fit (GB/s) | R^2 |
|---|---:|---:|---:|---:|
| `IVF1` | `8` | `-0.027` | `551` | `0.842` |
| `IVF2` | `5` | `1.152` | `658` | `0.829` |
| `IVF-INT8` | `5` | `1.431` | `inf` | `0.001` |

Interpretation:
- A pure bytes-moved model is too optimistic. The measured scan-family `BW_fit`
  is only `~550-660 GB/s`, far below the `1685 GB/s` microbenchmark.
- `IVF1` is close to a traffic-driven line, but not enough to justify
  `time = bytes / B` by itself.
- `IVF2` clearly needs a positive fixed term. The fitted intercept is about
  `1.15 ms`, which matches extra launch/merge/orchestration overhead.
- `IVF-INT8` does not fit a one-line traffic model at all. The near-zero `R^2`
  says quantization effects and fixed work dominate once bytes get small.

## Partial run: synthetic `d=128`

Command:

```bash
python3 -u explore.py \
  --m 250000 --d 128 --n 1000 --k 10 \
  --no-lsh --skip-pq --skip-dsweep \
  --export-json /workspace/gpu_algo/runs/synth_d128.json
```

Status:
- Still running on the pod as of this handoff.

Observed rows so far:

| algorithm | ms | recall@10 |
|---|---:|---:|
| `BF-FP32` | `319.72` | `1.0000` |
| `BF-FP16` | `268.27` | `0.9991` |
| `BF-INT8` | `61.19` | `0.9610` |
| `BF-GEMM` | `86.53` | `1.0000` |
| `IVF1(C=64,p=4)` | `28.20` | `0.2347` |
| `IVF1(C=64,p=8)` | `55.74` | `0.3803` |
| `IVF1(C=64,p=16)` | `106.60` | `0.5929` |
| `IVF1(C=128,p=4)` | `15.64` | `0.1594` |
| `IVF1(C=128,p=8)` | `29.38` | `0.2560` |

Interpretation:
- On unstructured high-dimensional Gaussian data, IVF is not competitive at your
  target recall regime.
- `BF-INT8` is currently the only non-exact scan family member already above
  `0.90` recall, at `61.19 ms`.
- This is consistent with the model transition you care about: once `X` no
  longer fits the relevant cache regime, clustering overhead does not buy enough
  reduction in scanned work on structureless data.

## Partial run: Fashion-MNIST-784

Command:

```bash
python3 -u explore_annb.py \
  --hdf5 fashion-mnist-784-euclidean.hdf5 \
  --k 10 --max-test 300 --no-lsh \
  --export-json /workspace/gpu_algo/runs/fashion300.json
```

Status:
- Still running on the pod as of this handoff.

Observed rows so far:

| algorithm | ms | recall@10 |
|---|---:|---:|
| `BF-FP32` | `163.82` | `1.0000` |
| `BF-FP16` | `130.00` | `1.0000` |
| `BF-INT8` | `52.29` | `0.9940` |
| `BF-GEMM` | `48.79` | `1.0000` |
| `IVF1(C=64,p=4)` | `11.17` | `0.9867` |
| `IVF1(C=64,p=8)` | `18.18` | `0.9983` |
| `IVF1(C=64,p=16)` | `64.13` | `1.0000` |
| `IVF1(C=128,p=4)` | `24.44` | `0.9667` |

Interpretation:
- On real structured data, IVF still dominates the speed/recall frontier.
- `IVF1(C=64,p=4)` already gives `0.9867` recall at `11.17 ms`, much faster
  than exact baselines.
- This is strong evidence that the right model must include data-structure-aware
  reduction in scanned bytes, but still preserve an additive fixed-cost term.

## Current model update

The current evidence supports replacing

```text
time ~= bytes_moved / B
```

with a family-aware affine model:

```text
time ~= alpha_family * scanned_bytes + beta_family + gamma_data
```

Where:
- `alpha_family` is an effective transfer coefficient, not raw peak bandwidth.
- `beta_family` captures launch, probe selection, heap merge, and scheduling.
- `gamma_data` captures failure modes not explained by traffic alone:
  quantization distortion, cache-regime changes, and low-structure data.

For the current repo, the practical version is:

```text
time ~= scanned_bytes / BW_fit(family, regime) + overhead(family, regime)
```

with recall modeled separately as a function of structure and approximation knobs.

## Practical ANN conclusion so far

If the target is `recall >= 0.90`:
- Small `d`, cache-resident regime: `IVF1` is the best current answer.
- Real Fashion-MNIST data: `IVF1` is also the best current answer.
- Large `d`, weak-structure Gaussian regime: scan-based `BF-INT8` is much more
  credible than IVF at the target recall.

This means the model should not rank methods by FLOPs. It should rank them by:
- bytes actually scanned,
- whether that scan is cache-resident or HBM-driven,
- and the non-negligible fixed orchestration term for the chosen family.
