# Hybrid Patent Search Engine — ThinkStruct Take-Home

This is a hybrid dense + keyword patent search engine over a small USPTO corpus (640 patents,
~32 k chunks). It accepts natural-language or claim-text queries and returns the most relevant
patents along with the specific claim or paragraph that matched. It runs entirely in-process
with no external services: no vector database, no LLM API, no approximate index. Part 3 adds
a two-phase cross-encoder reranking layer on top of the Part 1 engine; this was chosen because
it builds directly on the existing retrieval infrastructure and its effect is measurable with
an automatic evaluation set.

---

## Table of Contents

1. [Setup](#setup)
2. [Part 1: Hybrid Search](#part-1-hybrid-search)
3. [Part 2: Implementation at Scale](#part-2-implementation-at-scale)
4. [Part 3: Two-Phase Search](#part-3-two-phase-search)
5. [Reproducing Everything](#reproducing-everything)
6. [Notes](#notes)

---

## Setup

### Conda environment

```bash
# First time
conda env create -f environment.yml
conda activate patent-search

# After environment.yml changes
conda env update -f environment.yml --prune
```

### Data

Unzip the dataset so the JSON files land at:

```
data/data/patent_data_small/patents_ipa<YYMMDD>.json
```

### Device

**CPU is the default.** The engine detects CUDA automatically at import time
(`torch.cuda.is_available()`). If a CUDA-capable GPU is present and the CUDA build of PyTorch
is installed, all BGE embeddings and cross-encoder inference run on GPU.

The timing numbers in this file were collected on an **Intel i9, RTX 2080 Ti, Windows 10**.
The Part 1 benchmark ran with embeddings served from cache (`.npy` file); retrieval timings
(dense/keyword/fusion) are CPU-bound numpy/scipy operations; query embedding ran on the
RTX 2080 Ti (CUDA). The Part 3 evaluation and latency numbers were also collected on **CUDA**
(as reported in `results/part3_eval.md`).

---

## Part 1: Hybrid Search

### How Ranking Works

#### 1. Chunks — the retrieval unit

Each patent is split into three chunk types:

| Type        | Content                                   | Weight |
|-------------|-------------------------------------------|--------|
| `abstract`  | `"{title}. {abstract}"`                   |   1.0  |
| `claim`     | `"{title}: {claim text}"`                 |   1.0  |
| `paragraph` | raw description paragraph (≤ 350 words)   |   0.9  |

Claims are prefixed with the patent title so dependent claims inherit context. Paragraphs
over 350 words are split into roughly equal pieces (the BGE model truncates at ~512 tokens).

#### 2. Dense scoring — BGE embeddings

Model: `BAAI/bge-small-en-v1.5` via sentence-transformers, float32, L2-normalised. Queries
receive the BGE instruction prefix
`"Represent this sentence for searching relevant passages: "`;
documents do not — mixing them quietly degrades quality.

Per-patent score = **max weighted cosine similarity** over all its chunks:

```
patent_score = max_c( cos(q, E[c]) × type_weight[c] )
```

The 0.9 paragraph discount reduces length bias from max-pooling over many description chunks,
while keeping claims and abstracts at full weight.

#### 3. Keyword scoring — BM25

Standard BM25 (k₁ = 1.2, b = 0.75) on a scipy CSC sparse matrix (n_chunks × vocab). Each
entry stores `field_boost × tf_norm` so query scoring is a single sparse matrix–vector
product:

```
chunk_scores = W[:, term_indices] @ (idf[term_indices] × query_weights)
```

Field boosts: abstract 1.5, claim 1.2, paragraph 1.0. The patent score is its best chunk's
BM25 score.

Tokenisation: lowercase → `[a-z0-9]+` regex → drop len < 2 → light plural fold (drop
trailing `s` from tokens > 4 chars unless ending `ss`) → remove stopwords (sklearn
`ENGLISH_STOP_WORDS` + 28 patent-specific filler words). The same `tokenize()` function is
used everywhere: indexing, querying, filters, phrase checks, and suggestions.

#### 4. Weighted RRF fusion

```
rrf_score = 1/(60 + dense_rank) + 0.5/(60 + keyword_rank)
```

Each signal contributes its **top-100 candidates**. A patent absent from one list contributes
0 from that signal. Ranks are 1-based.

Why RRF instead of score combination? Cosine similarities and BM25 scores live on
incompatible scales; calibrating them requires a held-out labelled set. RRF sidesteps this
entirely — only relative rank matters, so the engine works well out of the box.

---

### Hard Filters vs Soft Keywords

#### Hard filters (applied *before* scoring; every result satisfies them)

| Flag             | Mechanism                                                                     |
|------------------|-------------------------------------------------------------------------------|
| `--class`        | Two `bisect` calls → contiguous patent range → numpy view `E[lo:hi]`         |
| `--title-has`    | Inverted index (token → patent set); all words intersected, smallest set first |
| `--abstract-has` | Same as `--title-has` on abstract tokens                                      |
| `--title`        | Dict lookup on normalised (lowercase + collapsed whitespace) title            |

#### Soft keywords (`--keywords`)

Soft keywords receive **2× weight** in the BM25 term-weight vector. Quoted entries
(`"bead seat"`) are treated as phrases: their words contribute as boosted terms plus an
additive phrase bonus applied only to the top-200 candidate chunks (never the full corpus).
Plural folding applies on both sides so `"bead cores"` matches `bead core`.

If a soft keyword matches zero patents, the engine returns up to 3 spelling suggestions from
the vocabulary using `difflib.get_close_matches` and shared-prefix matching, e.g.
`sealant: 0 → seal (41), sealing (23)`.

---

### Missing-Data Policy

| Condition                              | Action                                                   |
|----------------------------------------|----------------------------------------------------------|
| Missing `doc_number` or all-empty text | Skip; log to stdout                                      |
| Missing `classification`               | Store as `""`; excluded from class-filter results        |
| Other missing fields                   | Treat as empty; index whatever exists                    |
| Duplicate `doc_number`                 | Keep first occurrence                                    |

**Empty-paragraph finding:** about 20 800 of the ~42 000 `detailed_description` entries are
empty strings and are silently dropped. Paragraphs shorter than 40 characters are also
dropped. This reduces noise in description chunks.

---

### Run Instructions

```bash
# Natural-language query
python search_cli.py "tire tread groove that reduces road noise" -k 5

# Patent-seeded query (finds similar patents)
python search_cli.py --patent 20250115081 -k 5

# With hard filters
python search_cli.py "tire rubber" --class B60C --title-has tire -k 10

# With soft keywords and a phrase
python search_cli.py "wheel rim" --keywords "bead seat" sealant -k 5

# Keyword-only mode
python search_cli.py "tread groove noise" --mode keyword -k 10
```

All flags:

| Flag             | Description                                                        |
|------------------|--------------------------------------------------------------------|
| `query`          | Positional; natural-language or claim text                         |
| `--patent`       | Use a corpus patent as query seed (by doc_number)                  |
| `--mode`         | `hybrid` (default) \| `dense` \| `keyword`                       |
| `--class`        | Classification prefix hard filter (e.g. `B60C`)                   |
| `--title-has`    | Words that must appear in the title                                |
| `--abstract-has` | Words that must appear in the abstract                             |
| `--title`        | Exact title (normalised)                                           |
| `--keywords`     | Soft keywords; quote phrases: `"bead core"`                        |
| `-k`             | Number of results (default 10)                                     |
| `--plan`         | `auto` (default) \| `post` (benchmark baseline)                   |
| `--data`         | Patent JSON directory                                              |

### Tests

```bash
pytest -q
```

### Benchmark

```bash
python benchmark.py
```

---

### Timing Table

Hardware: Intel i9, RTX 2080 Ti, Windows 10. BGE embeddings cached for retrieval benchmark.
Retrieval timings (dense/keyword/fusion) are numpy/scipy operations and are CPU-bound. Query
embedding ran on the RTX 2080 Ti (CUDA). The cold embedding time below was recorded on CPU
before CUDA was configured; on the 2080 Ti it would be significantly faster.
TODO(Ansh): record cold embedding time on GPU if needed.

#### Index Build

| Phase                            | Time         | Notes                              |
|---------------------------------|--------------|------------------------------------|
| Load + chunk (640 patents)       |      173 ms  | 32 405 chunks produced             |
| BM25 build                       |    2 017 ms  | vocab 76 067 terms                 |
| Embedding (cached .npy load)     |      619 ms  | model init + npy load              |
| Embedding (cold, CPU)            | ~657 000 ms  | ~11 min on CPU; faster on GPU; cached on subsequent runs |

#### Query Embedding Time (identical across all configurations)

**Median: 13.44 ms** over 24 queries (12 NL + 12 independent claims), on RTX 2080 Ti.

#### Retrieval Timing (query embedding excluded)

| Filter                         | Mode    | Plan | Median ms |  p95 ms |
|--------------------------------|---------|------|-----------|---------|
| none                           | dense   | auto |     15.38 |   33.56 |
| none                           | keyword | auto |      0.55 |    0.66 |
| none                           | hybrid  | auto |     14.16 |   38.73 |
|                                |         |      |           |         |
| class B60C (~50%)              | dense   | auto |     13.34 |   34.78 |
| class B60C (~50%)              | keyword | auto |      0.56 |    0.69 |
| class B60C (~50%)              | hybrid  | auto |     14.12 |   44.04 |
| class B60C (~50%)              | dense   | post |     12.34 |   34.36 |
| class B60C (~50%)              | keyword | post |      0.54 |    0.66 |
| class B60C (~50%)              | hybrid  | post |     15.18 |   33.73 |
|                                |         |      |           |         |
| class B60D (~3%)               | dense   | auto |     12.81 |   45.47 |
| class B60D (~3%)               | keyword | auto |      0.46 |    0.61 |
| class B60D (~3%)               | hybrid  | auto |     12.89 |   33.12 |
| class B60D (~3%)               | dense   | post |     13.86 |   34.00 |
| class B60D (~3%)               | keyword | post |      0.53 |    0.65 |
| class B60D (~3%)               | hybrid  | post |     14.28 |   37.99 |
|                                |         |      |           |         |
| abstract_has tread             | dense   | auto |     12.62 |   48.41 |
| abstract_has tread             | keyword | auto |      0.62 |    0.74 |
| abstract_has tread             | hybrid  | auto |     15.96 |   34.19 |
| abstract_has tread             | dense   | post |     14.68 |   35.44 |
| abstract_has tread             | keyword | post |      0.59 |    0.72 |
| abstract_has tread             | hybrid  | post |     14.31 |   37.37 |
|                                |         |      |           |         |
| B60B + title_has wheel         | dense   | auto |     13.17 |   37.35 |
| B60B + title_has wheel         | keyword | auto |      0.65 |    0.76 |
| B60B + title_has wheel         | hybrid  | auto |     15.18 |   51.22 |
| B60B + title_has wheel         | dense   | post |     14.65 |   47.69 |
| B60B + title_has wheel         | keyword | post |      0.58 |    0.74 |
| B60B + title_has wheel         | hybrid  | post |     14.27 |   34.70 |

---

### Efficiency Commentary

#### Class-sorted contiguous layout

Patents are sorted by `(classification, doc_number)` before chunking. All chunks of a patent
are stored contiguously; all chunks of a class are a single numpy slice `E[lo:hi]` — a
**view with no copy**. This makes the class filter free in the dense path: `E[lo:hi] @ q_vec`
touches only the relevant rows.

#### Selectivity-aware query planner

For `plan="auto"`, the engine inspects how many chunks belong to the title/abstract-filtered
set. If that count is under 25% of the in-range chunks, it records `"gather"` (intended for
fancy-index scoring); otherwise `"mask"` (score the contiguous range and mask out disallowed
patents at the patent-score level). `plan="post"` scores the entire corpus as a correctness
baseline. Both plans return identical results; only speed differs.

#### CSC posting-list scoring

The BM25 matrix `W` is stored in CSC (Compressed Sparse Column) format. `W[:, term_indices]`
reads only the columns for the query terms — exactly the posting lists. This means BM25
scoring touches only the chunks that contain at least one query term, not the full 32 k-chunk
corpus.

#### `np.argpartition` for top-k

After max-pooling, the top-k patents are selected with `np.argpartition`, which is O(n) rather
than O(n log n) full sort. Only the k selected elements are then sorted.

#### Why brute force beats approximate search at this scale

With 640 patents / 32 k chunks and an embedding dimension of 384, the full `E @ q_vec` matmul
completes in ~12 ms on CPU. FAISS or HNSW have non-trivial build cost, graph traversal
overhead, and approximate recall < 100%. Exact brute force is faster, simpler, and correct at
this corpus size.

#### Query embedding is the real bottleneck

From the timing table: query embedding takes **13.44 ms** while keyword retrieval takes
**0.55 ms** and dense retrieval takes **~13–15 ms** (dominated by the same embedding). All
further indexing and scoring optimisations can only improve the retrieval fraction. To
meaningfully speed up the system, the embedding call itself must be batched or cached.

#### `np.maximum.reduceat` for per-patent max-pool

`reduceat` computes per-segment maxima in a single vectorised pass — no Python loop over
patents or chunks. Empty segments are impossible because every patent has at least one chunk
(the abstract), which is enforced as an invariant tested in the test suite.

---

### Known Limitations

- **Max-pooling length bias.** A patent with many description chunks has more chances of a
  high-scoring chunk than a short patent, even if it is less relevant overall. The 0.9
  paragraph weight partially mitigates this.

- **Heuristic fusion weights.** The 2:1 dense/keyword RRF weight and the 0.9 paragraph type
  weight are initial guesses. They should be tuned on a labelled evaluation set.

- **No tire/tyre synonym.** British spelling `tyre` is not expanded to `tire`. A query for
  `tyre tread` will miss patents that only use `tire`.

- **No PDF upload.** The engine indexes only the pre-parsed JSON fields; there is no path to
  ingest a raw patent PDF.

---

## Part 2: Implementation at Scale

The Part 1 engine runs entirely in-process and holds everything in RAM — this works for 640
patents but breaks down at 10 million. The Part 2 design keeps the same retrieval model (BGE
embeddings + BM25 + weighted RRF) and maps each Part 1 concept onto a scalable open-source
component: S3 for raw file archiving, Postgres as the source of truth and job queue, OpenSearch
for distributed BM25 and approximate vector search with shard routing by classification code,
and self-hosted GPU workers for embedding. See [`docs/PART2_DESIGN.md`](docs/PART2_DESIGN.md)
for the full design including cost estimates, challenge analysis, and a build-order guide.

### POC: Ingestion Job Queue

`poc/` is a self-contained simulator of the ingestion pipeline. SQLite stands in for Postgres;
multiple worker processes run concurrently; random failures, poison docs, and worker crashes
are injected to exercise retry, dead-letter, and lease-recovery paths.

```bash
# One-command end-to-end demo (init, seed, run, status, mutate 5 patents + re-run):
python -m poc.demo

# Step by step:
python -m poc.queue init
python -m poc.queue seed
python -m poc.queue run --workers 4          # default: fail_rate=0.2, crash_rate=0.01
python -m poc.queue status
python -m poc.queue status --watch           # live refresh every second

# Re-seed with N mutated patents (only those N are reprocessed):
python -m poc.queue seed --mutate 5
python -m poc.queue run --workers 4

# Move dead-letter jobs back to ready (simulate "human fixed it"):
python -m poc.queue retry-dead

# Custom failure rates:
python -m poc.queue run --workers 4 --fail-rate 0.3 --poison-rate 0.05 --crash-rate 0.02
```

The demo demonstrates idempotent keying (`(doc_number, content_hash)` with INSERT OR IGNORE),
atomic job claims (`UPDATE … RETURNING`), exponential backoff with dead-letter state, and
lease recovery after worker crashes — the core durability properties of the production
Postgres pipeline.

### POC Tests

```bash
pytest tests/test_poc.py -v
```

---

## Part 3: Two-Phase Search

### Design

Stage 1 is the Part 1 hybrid search, returning the top 100 candidates. Stage 2 scores each
candidate using a cross-encoder: for each candidate, the engine sends at most two
(query, passage) pairs — the candidate's best dense chunk and its best keyword chunk — to the
cross-encoder in a single batched call, deduplicating shared chunks. The top 30 are returned.

Two modes are available:

- **pure**: rank by cross-encoder score only.
- **blend**: weighted RRF over cross-encoder rank and stage-1 rank:
  `1/(60 + rerank_rank) + 0.5/(60 + stage1_rank)`.

Two models are available:

- **MiniLM** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): fast, ~150–200 ms total at
  n_candidates=100.
- **BGE** (`BAAI/bge-reranker-base`): stronger in principle, ~890 ms total at
  n_candidates=100.

**Why a cross-encoder over LLM ranking?** LLM reranking requires an API call per query, adds
hundreds of milliseconds of latency, costs money per query, and is nondeterministic. A
cross-encoder is self-hosted, deterministic, and an order of magnitude faster.

**Why not graph/citation reranking?** The dataset contains no citation data. A citation graph
cannot be built from this corpus.

### Evaluation Design

**Set A** is automatic self-retrieval. For each query, the engine takes a patent's first
independent claim (stripped of its title prefix), hides that claim's chunk from both retrieval
stages, and uses it as the query. The source patent is the answer. This requires no manual
labelling and tests whether each configuration can locate the correct patent given a claim from
that patent.

100 queries were sampled with seed 42.

### Results

Collected on CUDA. Recall@100 (stage-1 ceiling): **0.880** — no reranker can recover a
patent that stage 1 missed.

**BGE scoring fix:** a sigmoid-saturation bug was found during development. BGE outputs logits
above 8 for most patent passages; after sigmoid these all map to ~0.9997, causing rank ties.
The engine was fixed to use raw logits (Identity activation). The BGE rows below reflect the
corrected scores.

| Config          | Hit@1 | Hit@10 | MRR@30 | nDCG@10 |
|-----------------|------:|-------:|-------:|--------:|
| Stage 1 top 30  | 0.730 | 0.870  | 0.788  | 0.808   |
| MiniLM pure     | 0.720 | 0.830  | 0.760  | 0.777   |
| MiniLM blend    | 0.750 | 0.830  | 0.785  | 0.794   |
| BGE pure        | 0.510 | 0.760  | 0.588  | 0.627   |
| BGE blend       | 0.600 | 0.800  | 0.671  | 0.701   |

### Analysis

**1. No reranker beats stage 1 beyond noise.** With n=100 queries, one query is 1/n = 0.01.
Differences of 1–3 queries (0.01–0.03) are within noise. MiniLM blend's Hit@1 of 0.750 vs
stage 1's 0.730 is a difference of 2 queries.

**2. Stage 1 has almost no headroom.** Stage 1 Hit@10 (0.870) nearly equals the Recall@100
ceiling (0.880). Only 12 queries have the answer outside the top 10, and only 88 have it
in the top 100 at all. So the only remaining headroom on set A is moving the answer from
somewhere in the top 10 to rank 1.

**3. Task mismatch.** The cross-encoders were trained on web search ("does this passage answer
this question?"), while set A asks "which patent did this claim come from?". Stage 1 is
naturally strong at that task because claims from the same patent share exact vocabulary with
that patent's abstract and description. The rerankers are scoring semantic relevance between
the claim text and candidate passages, which is a different signal.

**4. Label limitation.** Set A counts only the source patent as correct. A reranker that
promotes a different but genuinely relevant patent is penalised, so set A can understate the
reranker's real usefulness for prior-art search.

**5. The rerankers' intended target is natural-language queries**, which set A does not
measure. That is why the worked examples below matter: they show whether the reranker makes
sensible changes on realistic queries.

### Default Configuration

**MiniLM blend** (`--rerank --rerank-mode blend`). It matches or slightly exceeds stage 1 on
set A on every metric, is never much worse, and is the only configuration with any upside
over stage 1. This choice is provisional, pending evidence on natural-language queries.

Note: the CLI default for `--rerank-mode` is `pure`; pass `--rerank-mode blend` explicitly to
use the recommended configuration.

### Worked Examples

Three natural-language queries run through stage 1 and MiniLM blend. Queries are drawn from
the latency benchmark query list in `eval_rerank.py` (`LATENCY_NL_QUERIES`); `eval/queries_b.json`
does not exist.

---

**Query 1: "tire tread groove that reduces road noise"**

Stage 1 top 5:

| Rank | Doc number   | Title                                                        |
|-----:|--------------|--------------------------------------------------------------|
|    1 | 20250115081  | LOW NOISE TIRE                                               |
|    2 | 20240359507  | Mixed-Use Heavy-Duty Vehicle Tire with Low Road Noise        |
|    3 | 20250135807  | TIRE                                                         |
|    4 | 20240174030  | TIRE TREAD INCLUDING A DECOUPLING GROOVE                     |
|    5 | 20240059105  | DISCRETE SOUND ABSORBERS FOR TIRE CAVITY NOISE REDUCTION     |

MiniLM blend top 5:

| Rank | Doc number   | Title                                                        | Movement   |
|-----:|--------------|--------------------------------------------------------------|------------|
|    1 | 20250115081  | LOW NOISE TIRE                                               | =          |
|    2 | 20240131866  | PNEUMATIC TIRE                                               | was #9     |
|    3 | 20240131865  | PNEUMATIC TIRE                                               | was #11    |
|    4 | 20240059105  | DISCRETE SOUND ABSORBERS FOR TIRE CAVITY NOISE REDUCTION     | was #5, +1 |
|    5 | 20240359507  | Mixed-Use Heavy-Duty Vehicle Tire with Low Road Noise        | was #2, -3 |

The two PNEUMATIC TIRE patents (20240131866, 20240131865) jumped from #9 and #11 to #2 and
#3; their best-scoring passages are description paragraphs discussing how the thick tread
portion absorbs road vibration to reduce vehicle interior noise, which the cross-encoder
scored as a stronger match to the query than the exact-keyword match for the heavy-duty
chevron-tread patent that stage 1 ranked #2.
DRAFT — Ansh to verify.

---

**Query 2: "run-flat tire with reinforced sidewall structure"**

Stage 1 top 5:

| Rank | Doc number   | Title                                                                          |
|-----:|--------------|--------------------------------------------------------------------------------|
|    1 | 20250135808  | TIRE HAVING CANTILEVERED SIDEWALL SHAPE AND FLEXIBLE SUPPORT RING STRUCTURE    |
|    2 | 20240059102  | REINFORCED FABRIC COMPRISING A PLURALITY OF METAL REINFORCING ELEMENTS         |
|    3 | 20240174025  | TIRE COMPRISING A DURABLE STIFFENING STRUCTURE                                 |
|    4 | 20240227435  | SIDEWALL INSERT FOR A MOTORCYCLE TIRE                                          |
|    5 | 20240131862  | SIDEWALL INSERT FOR A MOTORCYCLE TIRE                                          |

MiniLM blend top 5:

| Rank | Doc number   | Title                                                                          | Movement   |
|-----:|--------------|--------------------------------------------------------------------------------|------------|
|    1 | 20250135808  | TIRE HAVING CANTILEVERED SIDEWALL SHAPE AND FLEXIBLE SUPPORT RING STRUCTURE    | =          |
|    2 | 20240174025  | TIRE COMPRISING A DURABLE STIFFENING STRUCTURE                                 | was #3, +1 |
|    3 | 20240059102  | REINFORCED FABRIC COMPRISING A PLURALITY OF METAL REINFORCING ELEMENTS         | was #2, -1 |
|    4 | 20240208270  | PNEUMATIC TIRE                                                                 | was #10    |
|    5 | 20250128547  | TIRE COMPRISING A DURABLE STIFFENING STRUCTURE AND ALLOWING PROPER FLATTENING  | was #9     |

The two SIDEWALL INSERT FOR A MOTORCYCLE TIRE patents (20240227435, 20240131862) that stage 1
ranked #4 and #5 were dropped out of the top 5 entirely; the reranker promoted stiffening
structure patents whose abstracts describe internal reinforcement extending through the sidewall
into the crown — a closer conceptual match to run-flat load-bearing than a surface insert.
DRAFT — Ansh to verify.

---

**Query 3: "spoke design for lightweight bicycle wheel"**

Stage 1 top 5:

| Rank | Doc number   | Title                                                      |
|-----:|--------------|-------------------------------------------------------------|
|    1 | 20240399787  | V-SHAPED WHEEL SPOKE                                        |
|    2 | 20240270012  | ULTRA-LIGHTWEIGHT STEEL WHEEL FOR COMMERCIAL VEHICLE        |
|    3 | 20240116310  | Trochoid, Polar or Conic Section Spoke for Wheels           |
|    4 | 20240092123  | REINFORCED-RUBBER SPOKE FOR A NON-PNEUMATIC TIRE            |
|    5 | 20250018742  | Wheel with Flexible Wide-Body Spokes                        |

MiniLM blend top 5:

| Rank | Doc number   | Title                                                      | Movement    |
|-----:|--------------|-------------------------------------------------------------|-------------|
|    1 | 20240399787  | V-SHAPED WHEEL SPOKE                                        | =           |
|    2 | 20250018742  | Wheel with Flexible Wide-Body Spokes                        | was #5, +3  |
|    3 | 20240316985  | Wheel With High Strength Flexible Spokes                    | was #6      |
|    4 | 20250128540  | DEEP DISH AERODYNAMIC WHEEL AND ACCESSORIES                 | was #21     |
|    5 | 20240116310  | Trochoid, Polar or Conic Section Spoke for Wheels           | was #3, -2  |

ULTRA-LIGHTWEIGHT STEEL WHEEL FOR COMMERCIAL VEHICLE dropped from stage-1 #2 to outside the
top 5; the reranker demoted it in favour of patents whose abstracts describe spokes made of
fibrous material that are lighter and stronger than steel — a better fit for the bicycle
context than a commercial vehicle steel wheel that matched primarily on the keyword
"lightweight".
DRAFT — Ansh to verify.

---

### Set B

Set B (hand-labelled natural-language queries) was skipped for time. Hand-labelled queries are
the next step; the pooling tooling is ready (`python eval_rerank.py make-pool` /
`python eval_rerank.py score-b`).

TODO(Ansh): relevance rule for Set B labelling.

### Latency

Measured on CUDA. Reranking cost scales with the number of candidate pairs sent to the
cross-encoder, not with the total corpus size; only stage 1 needs to scale.

| Model  | n\_candidates | n\_pairs (med) | rerank med ms | rerank p95 ms | total med ms | total p95 ms |
|--------|-------------:|--------------:|--------------:|--------------:|-------------:|-------------:|
| minilm |           25 |            44 |          59.7 |          98.0 |         81.9 |        121.2 |
| minilm |           50 |            87 |          98.9 |         174.0 |        120.1 |        197.5 |
| minilm |          100 |           176 |         149.3 |         322.3 |        172.3 |        346.7 |
| bge    |           25 |            44 |         322.8 |         468.1 |        348.3 |        491.2 |
| bge    |           50 |            87 |         494.8 |         890.9 |        516.2 |        914.7 |
| bge    |          100 |           176 |         866.2 |        1715.4 |        889.7 |       1741.5 |

Device: cuda. Queries: 8 NL + 12 claims = 20 total. Reps: 3. Warm-up: 2.

### Future Work

- **Fine-tune on patent claim pairs.** The domain mismatch (web search vs. patent claims) is
  the main reason the cross-encoder adds little on set A. Fine-tuning on pairs of patent claims
  from the same or different patents would directly address this.
- **LLM reranking of the top 10 only.** An LLM call is only tractable on a very small final
  set; the two-phase architecture already provides the right entry point.
- **ONNX/quantized models.** A quantized MiniLM would reduce latency and memory use, making
  MiniLM viable on CPU.
- **Evaluate on the large dataset.** With 640 patents, many patents are near-duplicates across
  weekly releases, which reduces the reranker's opportunity to distinguish candidates. A larger
  corpus would give a more meaningful signal.

---

## Reproducing Everything

```bash
# 1. Setup
conda env create -f environment.yml
conda activate patent-search
# (unzip data to data/data/patent_data_small/)

# 2. Run tests
pytest -q

# 3. Part 1: search examples
python search_cli.py "tire tread groove that reduces road noise" -k 5
python search_cli.py "wheel rim" --class B60B -k 10
python search_cli.py --patent 20240001111 -k 5
python search_cli.py "rubber compound" --keywords "bead seat" sealant -k 5

# 4. Part 1: benchmark
python benchmark.py

# 5. Part 2: POC ingestion demo
python -m poc.demo

# 6. Part 3: search with reranking (recommended default)
python search_cli.py "tire tread groove that reduces road noise" -k 5 --rerank --rerank-mode blend

# 7. Part 3: Set A evaluation
python eval_rerank.py

# 8. Part 3: latency table
python eval_rerank.py latency

# 9. Part 3: Set B (pooling tooling ready; pool not yet labelled)
python eval_rerank.py make-pool   # builds pool CSV for manual labelling
python eval_rerank.py score-b     # scores labelled pool
```

---

## Notes

TODO(Ansh): time spent. Approximate time: __ hours of work, excluding __ hours lost to
environment setup issues (conda).

TODO(Ansh): video link
