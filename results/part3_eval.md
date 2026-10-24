# Part 3 Evaluation Results

## Set A: Automatic self-retrieval

Date: 2026-10-24  |  Seed: 42  |  Queries: 100  |  Device: cuda
Models: cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-base
n\_candidates=100, k=30

**Recall@100 (stage-1 ceiling): 0.880**

No reranker can recover a patent stage 1 missed.

| Config          | Hit@1 | Hit@10 | MRR@30 | nDCG@10 |
|-----------------|------:|-------:|-------:|--------:|
| Stage 1 top 30  | 0.730 | 0.870  | 0.788  | 0.808   |
| MiniLM pure     | 0.720 | 0.830  | 0.760  | 0.777   |
| MiniLM blend    | 0.750 | 0.830  | 0.785  | 0.794   |
| BGE pure        | 0.510 | 0.760  | 0.588  | 0.627   |
| BGE blend       | 0.600 | 0.800  | 0.671  | 0.701   |

> **Known bias:** claims from the same patent share vocabulary with that patent's
> abstract and description, making this easier than real prior-art search.

## Latency & Quality vs n\_candidates

Date: 2026-10-24  |  Device: cuda

### Reranker Latency

Queries: 8 NL + 12 claims = 20 total  |  Reps: 3  |  Warm-up: 2

| Model  | n\_candidates | n\_pairs (med) | rerank med ms | rerank p95 ms | total med ms | total p95 ms |
|--------|-------------:|--------------:|--------------:|--------------:|-------------:|-------------:|
| minilm |           25 |            44 |          59.7 |          98.0 |         81.9 |        121.2 |
| minilm |           50 |            87 |          98.9 |         174.0 |        120.1 |        197.5 |
| minilm |          100 |           176 |         149.3 |         322.3 |        172.3 |        346.7 |
| bge    |           25 |            44 |         322.8 |         468.1 |        348.3 |        491.2 |
| bge    |           50 |            87 |         494.8 |         890.9 |        516.2 |        914.7 |
| bge    |          100 |           176 |         866.2 |        1715.4 |        889.7 |       1741.5 |

### Quality vs n\_candidates (Set A, 100 queries)

| n\_candidates | Config          | Hit@1 | Hit@10 | MRR@30 | nDCG@10 |
|-------------:|-----------------|------:|-------:|-------:|--------:|
|           25 | Stage 1 top 30  | 0.730 | 0.870  | 0.788  | 0.808   |
|           25 | MiniLM pure     | 0.730 | 0.830  | 0.773  | 0.785   |
|           25 | MiniLM blend    | 0.760 | 0.850  | 0.792  | 0.804   |
|           25 | BGE pure        | 0.530 | 0.790  | 0.621  | 0.658   |
|           25 | BGE blend       | 0.610 | 0.830  | 0.693  | 0.724   |
|           50 | Stage 1 top 30  | 0.730 | 0.870  | 0.788  | 0.808   |
|           50 | MiniLM pure     | 0.720 | 0.830  | 0.765  | 0.780   |
|           50 | MiniLM blend    | 0.760 | 0.830  | 0.791  | 0.798   |
|           50 | BGE pure        | 0.520 | 0.770  | 0.602  | 0.640   |
|           50 | BGE blend       | 0.600 | 0.800  | 0.678  | 0.705   |
|          100 | Stage 1 top 30  | 0.730 | 0.870  | 0.788  | 0.808   |
|          100 | MiniLM pure     | 0.720 | 0.830  | 0.760  | 0.777   |
|          100 | MiniLM blend    | 0.750 | 0.830  | 0.785  | 0.794   |
|          100 | BGE pure        | 0.510 | 0.760  | 0.588  | 0.627   |
|          100 | BGE blend       | 0.600 | 0.800  | 0.671  | 0.701   |
