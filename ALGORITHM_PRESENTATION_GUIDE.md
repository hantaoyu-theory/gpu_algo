# Algorithm Presentation Guide

This file is a presentation-oriented explanation of the ANN algorithms used in
`gpu_algo`: what each algorithm does, how the code is structured into stages,
and which stage is taking most of the time in the current results.

Primary measurement summary:
- [COST_COMPONENT_SUMMARY.md](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/COST_COMPONENT_SUMMARY.md)

Primary implementation files:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py)
- [lsh_fast.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/lsh_fast.py)

## Common Setup

All algorithms solve:

```text
Given database X and queries Q, return the top-k nearest neighbors for each query.
```

In the tables:
- `k=10`
- recall means `recall@10`
- `C` = number of IVF clusters
- `p` = number of probed clusters (`nprobe`)
- `L` = number of LSH tables
- `K` = number of hash projections per LSH table

## 1. BF-FP32

Code:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py#L505)

Idea:
- brute-force scan of the full database
- store database in FP32
- compute exact L2 distance to every point
- keep only top-k during the scan

Concrete stages:
1. Start with database points `X in R^{m x d}` in HBM and queries `Q in R^{n x d}` in HBM.
2. For one query block, load one query vector `q` from HBM into on-chip shared memory so all threads in the block can reuse it.
3. Each thread streams a strided subset of database vectors `x_i`ho from HBM/L2, one candidate at a time.
4. For each candidate, compute

   ```text
   ||q - x_i||^2 = sum_j (q_j - x_{i,j})^2
   ```

   in FP32.
5. Keep only the current local top-k smallest distances for that thread; discard the rest immediately.
6. Write the thread-local top-k lists into shared memory.
7. One thread merges those partial top-k lists into the final top-k for the query and writes the result back to HBM.

Why it is fast/slow:
- exact and simple
- no large intermediate tensors
- but still scans all `m` points

Current bottleneck:
- `T_move`

Presentation takeaway:
- This is the cleanest exact baseline.
- It is mostly memory movement, not compute.

## 2. BF-FP16

Code:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py#L519)

Idea:
- same brute-force scan as BF-FP32
- database `X` is stored as FP16
- query stays FP32
- accumulation is still FP32

Concrete stages:
1. Store `X` in HBM as FP16; keep `Q` in FP32.
2. Load one query vector `q` into shared memory.
3. Stream FP16 database vectors `x_i` from HBM/L2.
4. Convert each loaded FP16 element of `x_i` to FP32 inside the kernel.
5. Compute

   ```text
   ||q - x_i||^2 = sum_j (q_j - float(x_{i,j}))^2
   ```

   with FP32 accumulation.
6. Maintain a streaming top-k exactly as in BF-FP32.
7. Merge the partial top-k lists and write the final top-k back to HBM.

Why it is fast/slow:
- cuts database traffic roughly in half
- still exact-style search
- still pays top-k merge cost

Current bottleneck:
- usually `T_move`
- in the `d=128` Gaussian case, `T_orch` becomes comparable or larger in the rough profile

Presentation takeaway:
- This is mainly a bandwidth optimization, not a low-precision compute algorithm.

## 3. BF-INT8

Code:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py#L536)

Idea:
- brute-force scan with quantized database and queries
- uses packed INT8 arithmetic (`dp4a`)
- computes approximate distances
- keeps top-k in streaming fashion

Concrete stages:
1. Quantize both `X` and `Q` from FP32 to INT8, using a shared scale.
2. Pack every 4 INT8 entries into one int32 word so the kernel can use `dp4a`.
3. Load one packed query `q` into shared memory.
4. Stream packed database vectors `x_i` from HBM/L2.
5. Compute an approximate dot product `q · x_i` with repeated `dp4a` instructions.
6. Recover an approximate squared distance using

   ```text
   ||q - x_i||^2 ~= ||q||^2 + ||x_i||^2 - 2 q·x_i
   ```

7. Update the streaming top-k for that query.
8. Merge the block’s partial top-k lists and write the final top-k to HBM.

Why it is fast/slow:
- much less memory traffic
- no full distance matrix is materialized
- recall is slightly below exact

Current bottleneck:
- mostly `T_move`

Presentation takeaway:
- This is the best brute-force-style compromise when high recall is needed and data are large.

## 4. BF-GEMM

Code:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py#L586)

Idea:
- brute-force search rewritten using matrix multiplication
- use

```text
||q - x||^2 = ||q||^2 + ||x||^2 - 2 q · x
```

- compute the `q · x` term with GEMM
- then recover top-k

Concrete stages:
1. Keep `X` and `Q` in HBM as dense FP32 matrices.
2. Precompute `||x_i||^2` for all database rows and `||q_j||^2` for all query rows.
3. For a tile `X_t` of the database, call GEMM to compute the matrix of dot products

   ```text
   S_t = Q X_t^T
   ```

4. Materialize the corresponding distance tile in global memory:

   ```text
   D_t = ||Q||^2 + ||X_t||^2 - 2 S_t
   ```

5. Run `argpartition` on `D_t` to get the best candidates inside that tile.
6. Gather those tile-local best ids/distances.
7. Concatenate them with the running top-k buffer from previous tiles.
8. Run another `argpartition` to merge back down to global top-k.
9. Repeat for the next tile of `X`.

Why it is fast/slow:
- the GEMM itself is efficient
- the surrounding top-k and merge pipeline can dominate
- especially bad when distance tiles are repeatedly materialized and reprocessed

Current bottleneck:
- `d=8`: strongly `T_orch`
- `d=128` / Fashion: the rough profiler is less trustworthy, but the practical issue is still the surrounding pipeline, not the GEMM core

Presentation takeaway:
- “GEMM is fast” does not imply “BF-GEMM is fast.”
- The bottleneck is top-k / merge / materialization around GEMM.

## 5. IVF1

Code:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py#L621)

Idea:
- cluster the database into `C` clusters
- for each query, find the `p` closest clusters
- only scan candidates from those clusters
- one CUDA block handles one query

Concrete stages:
1. Offline, cluster the database `X` into `C` clusters with K-means.
2. Reorder `X` in memory so points in the same cluster are contiguous in HBM.
3. For one query `q`, load `q` into shared memory.
4. Compare `q` against all `C` centroids and choose the `p` nearest clusters.
5. Compute the offsets of those clusters in the reordered database array.
6. One block scans only those cluster ranges, streaming the selected `x_i` values from HBM/L2.
7. For each scanned candidate, compute `||q - x_i||^2` and update a streaming top-k.
8. Merge the block-local top-k lists into the final top-k for the query.

Why it is fast:
- reduces scanned points from `m` to roughly `(p/C) * m`
- keeps reads cluster-ordered and simple
- avoids huge intermediate tensors

Current bottleneck:
- mostly `T_move`
- with a smaller but real orchestration term for probe selection and merge

Presentation takeaway:
- This is the best all-around ANN design in the repo.
- It gets candidate reduction with relatively low control overhead.

## 6. IVF2

Code:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py#L790)

Idea:
- same clustering idea as IVF1
- but parallelize more aggressively:
  one block handles one `(query, probed cluster)` pair

Concrete stages:
1. Build the same clustered/reordered database as IVF1.
2. For each query `q`, score `q` against centroids and choose the `p` nearest clusters.
3. Launch one CUDA block for each `(query, chosen-cluster)` pair.
4. Each block streams only the vectors from its assigned cluster range.
5. Each block computes distances `||q - x_i||^2` and produces a partial top-k for that one cluster.
6. Write those `p` partial top-k lists to global memory.
7. Launch a second merge kernel that reads the `p` partial lists for each query and merges them into one final top-k.

Why it can be slower than IVF1:
- more blocks
- more partial outputs
- more merging
- more scheduling / orchestration

Current bottleneck:
- `T_orch`

Presentation takeaway:
- IVF2 trades parallelism for much higher merge overhead.
- In this repo, IVF1 is usually the better design.

## 7. IVF-INT8

Code:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py#L1060)

Idea:
- IVF candidate reduction plus INT8 distance evaluation inside the cluster scan

Concrete stages:
1. Build the same IVF clustering structure as IVF1.
2. Quantize the cluster-ordered database vectors to INT8.
3. For each query `q`, choose the `p` nearest centroids.
4. Stream only the selected cluster ranges, but now read INT8 vectors instead of FP32.
5. Compute approximate distances inside those clusters using INT8 arithmetic / dequantized norms.
6. Update a streaming top-k while scanning.
7. Merge the partial top-k lists into the final top-k for the query.

Why it is fast:
- scans fewer candidates than brute force
- uses fewer bytes per candidate than FP32/FP16

Why it can fail:
- quantization error can cap recall
- if traffic becomes small, orchestration can become a larger fraction

Current bottleneck:
- `d=8`: orchestration is noticeable
- `d=128` / Fashion: mostly movement in the current rough profile

Presentation takeaway:
- This is often the fastest high-recall method on structured data.
- It is a candidate-reduction method plus a bandwidth-reduction method.

## 8. LSH (FastLSHIndex)

Code:
- [explore.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/explore.py#L479)
- [lsh_fast.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/lsh_fast.py#L318)
- [lsh_fast.py](/Users/hantaoyu/Desktop/AIResearch/gpu_algo/lsh_fast.py#L331)

Idea:
- hash each point into multiple tables
- queries only compare against candidates that collide in hash buckets

Concrete stages in `search_fast`:
1. Keep the database `X` already indexed into `L` sorted hash tables.
2. For each query matrix `Q`, project each query onto the LSH hyperplanes and hash it in all `L` tables.
3. Pack the hash codes into integer keys.
4. For each query and each table, lookup the matching bucket range in the sorted table arrays.
5. Gather up to `max_cands_per_table` candidate ids from each table, so each query gets a candidate list of size up to `L * max_cands_per_table`.
6. Run a fused rerank kernel: load the actual candidate vectors `x_i`, compute distances `||q - x_i||^2`, and keep only the top-k (or oversampled top-k) candidates.
7. Resolve duplicates lazily after reranking rather than doing a separate exact dedup stage up front.

Why it can be fast:
- avoids scanning the full database
- candidate count is capped by `L * max_cands_per_table`

Why it can be slow or poor quality:
- candidate set can still be large
- rerank step becomes irregular gather-heavy memory traffic
- hash collisions can be poor in high dimension or on structured data with the wrong config

Current bottleneck:
- mostly rerank-side `T_move` in the current rough model

Presentation takeaway:
- LSH is not competitive in the measured Fashion and `d=128` Gaussian settings.
- On `d=8`, it reaches the 0.9 recall regime but is still much slower than IVF.

## Stage-Level “What Takes Time?”

Use this simple story in slides:

1. `BF-FP32 / BF-FP16 / BF-INT8`
   The expensive stage is the streaming read of all database vectors `X` and the
   per-candidate distance update.

2. `BF-GEMM`
   The expensive stage is creating the distance tiles `D_t` and repeatedly doing
   tile-local top-k plus merge across tiles.

3. `IVF1`
   The expensive stage is scanning the selected cluster ranges of `X`.

4. `IVF2`
   The expensive stage is the second-stage merge of the `p` partial top-k lists.

5. `IVF-INT8`
   The expensive stage is usually the cluster scan, but once the scan gets small,
   the merge / control overhead becomes a large fraction.

6. `LSH`
   The expensive stage is loading candidate vectors after bucket lookup and
   reranking them, not the hash computation itself.

## Recommended Slide Summary

If you want one slide with the punchline:

- `BF-*` methods: “scan everything”
- `IVF1`: “scan only a few clusters”
- `IVF2`: “parallelize IVF1 more, but pay merge overhead”
- `IVF-INT8`: “IVF1 plus low-byte-distance evaluation”
- `LSH`: “hash to candidate buckets, then rerank”
- `BF-GEMM`: “use matrix multiply for brute force, but top-k pipeline dominates”

And the high-level runtime message:

```text
BF scan kernels: movement-dominated
IVF1: reduced movement
IVF2: orchestration-heavy
BF-GEMM: top-k/materialization-heavy
LSH: rerank movement-heavy
```
