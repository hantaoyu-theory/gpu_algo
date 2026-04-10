# BF Workspace

Focused workspace for brute-force ANN kernels and BF-only benchmarks.

Current scope:
- `fp32.py`: reusable BF-FP32 kernel wrapper extracted from `explore.py`
- `bench_fp32.py`: saturation sweep over query count `n`

Example:

```bash
python3 -m bf.bench_fp32 --m 250000 --d 128 --n-values 1000,2000,4000,8000,16000
```

Suggested use:
- Treat `n=1000` as a boundary point, not the headline throughput result.
- Use `n >= 4000` as the main range when studying BF kernel saturation.
