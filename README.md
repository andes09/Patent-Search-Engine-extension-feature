# Hybrid Patent Search Engine — ThinkStruct Take-Home, Part 1

A hybrid dense + keyword search engine over a small USPTO patent corpus
(640 patents, ~32 k chunks).  It runs entirely in-process with no external
services: no vector database, no LLM API, no approximate index.

---

## Problem Statement

Given a collection of patent JSON files, build a search engine that accepts
natural-language or claim-text queries and returns the most relevant patents,
ranked by a combination of semantic similarity and keyword relevance.  Hard
filters (classification class, title/abstract word presence, exact title) must
be applied *before* scoring so that every result satisfies them.

---

## How Ranking Works

### 1. Chunks — the retrieval unit

Each patent is split into three types of chunks:

| Type        | Content                                   | Weight |
|-------------|-------------------------------------------|--------|
| `abstract`  | `"{title}. {abstract}"`                   |   1.0  |
| `claim`     | `"{title}: {claim text}"`                 |   1.0  |
| `paragraph` | raw description paragraph (≤ 350 words)   |   0.9  |

Claims are prefixed with the patent title so dependent claims inherit context.
Paragraphs over 350 words are split into roughly equal pieces (the BGE model
truncates at ~512 tokens).

### 2. Dense scoring — BGE embeddings

Model: `BAAI/bge-small-en-v1.5` via sentence-transformers, float32, L2-normalised.
Queries receive the BGE instruction prefix
`"Represent this sentence for searching relevant passages: "`;
documents do not — mixing them quietly degrades quality.

Per-patent score = **max weighted cosine similarity** over all its chunks:

```
patent_score = max_c( cos(q, E[c]) × type_weight[c] )
```

The 0.9 paragraph discount reduces length bias from max-pooling over many
description chunks, while keeping claims and abstracts at full weight.

### 3. Keyword scoring — BM25

Standard BM25 (k₁ = 1.2, b = 0.75) on a scipy CSC sparse matrix
(n_chunks × vocab).  Each entry stores `field_boost × tf_norm` so query
scoring is a single sparse matrix–vector product:

```
chunk_scores = W[:, term_indices] @ (idf[term_indices] × query_weights)
```

Field boosts: abstract 1.5, claim 1.2, paragraph 1.0.  The patent score is
its best chunk's BM25 score.

Tokenisation: lowercase → `[a-z0-9]+` regex → drop len < 2 → light plural
fold (drop trailing `s` from tokens > 4 chars unless ending `ss`) → remove
stopwords (sklearn `ENGLISH_STOP_WORDS` + 28 patent-specific filler words).
The same `tokenize()` function is used everywhere: indexing, querying, filters,
phrase checks, and suggestions.

### 4. Weighted RRF fusion

```
rrf_score = 1/(60 + dense_rank) + 0.5/(60 + keyword_rank)
```

Each signal contributes its **top-100 candidates**.  A patent absent from one
list contributes 0 from that signal.  Ranks are 1-based.

Why RRF instead of score combination?  Cosine similarities and BM25 scores
live on incompatible scales; calibrating them requires a held-out labelled set.
RRF sidesteps this entirely — only relative rank matters, so the engine works
well out of the box.

---

## Hard Filters vs Soft Keywords

### Hard filters (applied *before* scoring; every result satisfies them)

| Flag            | Mechanism                                      |
|----------------|------------------------------------------------|
| `--class`       | Two `bisect` calls → contiguous patent range → numpy view `E[lo:hi]` |
| `--title-has`   | Inverted index (token → patent set); all words intersected, smallest set first |
| `--abstract-has`| Same as `--title-has` on abstract tokens       |
| `--title`       | Dict lookup on normalised (lowercase + collapsed whitespace) title |

### Soft keywords (`--keywords`)

Soft keywords receive **2× weight** in the BM25 term-weight vector.
Quoted entries (`"bead seat"`) are treated as phrases: their words contribute
as boosted terms plus an additive phrase bonus applied only to the top-200
candidate chunks (never the full corpus).  Plural folding applies on both
sides so `"bead cores"` matches `bead core`.

If a soft keyword matches zero patents, the engine returns up to 3 spelling
suggestions from the vocabulary using `difflib.get_close_matches` and
shared-prefix matching, e.g. `sealant: 0 → seal (41), sealing (23)`.

---

## Missing-Data Policy

| Condition                             | Action                                |
|--------------------------------------|---------------------------------------|
| Missing `doc_number` or all-empty text | Skip; log to stdout                  |
| Missing `classification`              | Store as `""`; excluded from class-filter results |
| Other missing fields                  | Treat as empty; index whatever exists |
| Duplicate `doc_number`               | Keep first occurrence                 |

**Empty-paragraph finding:** about 20 800 of the ~42 000 `detailed_description`
entries are empty strings and are silently dropped.  Paragraphs shorter than
40 characters are also dropped.  This reduces noise in description chunks.

---

## Run Instructions

### Setup

```bash
conda env create -f environment.yml   # first time only
conda activate patent-search
```

### Search CLI

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

| Flag            | Description                                        |
|----------------|----------------------------------------------------|
| `query`         | Positional; natural-language or claim text         |
| `--patent`      | Use a corpus patent as query seed (by doc_number)  |
| `--mode`        | `hybrid` (default) \| `dense` \| `keyword`        |
| `--class`       | Classification prefix hard filter (e.g. `B60C`)   |
| `--title-has`   | Words that must appear in the title                |
| `--abstract-has`| Words that must appear in the abstract             |
| `--title`       | Exact title (normalised)                           |
| `--keywords`    | Soft keywords; quote phrases: `"bead core"`        |
| `-k`            | Number of results (default 10)                     |
| `--plan`        | `auto` (default) \| `post` (benchmark baseline)   |
| `--data`        | Patent JSON directory                              |

### Tests

```bash
pytest -q
```

### Benchmark

```bash
python benchmark.py
```

---

## Timing Table

All times on an Intel i7, Windows 10, BGE embeddings cached.

### Index Build

| Phase                            | Time         | Notes                              |
|---------------------------------|--------------|------------------------------------|
| Load + chunk (640 patents)       |      173 ms  | 32 405 chunks produced             |
| BM25 build                       |    2 017 ms  | vocab 76 067 terms                 |
| Embedding (cached .npy load)     |      619 ms  | model init + npy load              |
| Embedding (cold, first run)      | ~657 000 ms  | ~11 min; cached on subsequent runs |

### Query Embedding Time (identical across all configurations)

**Median: 13.44 ms** over 24 queries (12 NL + 12 independent claims).

### Retrieval Timing (query embedding excluded)

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

## Efficiency Commentary

### Class-sorted contiguous layout

Patents are sorted by `(classification, doc_number)` before chunking.  All
chunks of a patent are stored contiguously; all chunks of a class are a single
numpy slice `E[lo:hi]` — a **view with no copy**.  This makes the class filter
free in the dense path: `E[lo:hi] @ q_vec` touches only the relevant rows.

### Selectivity-aware query planner

For `plan="auto"`, the engine inspects how many chunks belong to the
title/abstract-filtered set.  If that count is under 25% of the in-range
chunks, it records `"gather"` (intended for fancy-index scoring); otherwise
`"mask"` (score the contiguous range and mask out disallowed patents at the
patent-score level).  `plan="post"` scores the entire corpus as a
correctness baseline.  Both plans return identical results; only speed differs.

### CSC posting-list scoring

The BM25 matrix `W` is stored in CSC (Compressed Sparse Column) format.
`W[:, term_indices]` reads only the columns for the query terms — exactly
the posting lists.  This means BM25 scoring touches only the chunks that
contain at least one query term, not the full 32 k-chunk corpus.

### `np.argpartition` for top-k

After max-pooling, the top-k patents are selected with `np.argpartition`,
which is O(n) rather than O(n log n) full sort.  Only the k selected elements
are then sorted.

### Why brute force beats approximate search at this scale

With 640 patents / 32 k chunks and an embedding dimension of 384, the full
`E @ q_vec` matmul completes in ~12 ms on CPU.  FAISS or HNSW have non-trivial
build cost, graph traversal overhead, and approximate recall < 100%.  Exact
brute force is faster, simpler, and correct at this corpus size.

### Query embedding is the real bottleneck

From the timing table: query embedding takes **13.44 ms** while keyword
retrieval takes **0.55 ms** and dense retrieval takes **~13–15 ms** (dominated
by the same embedding).  All further indexing and scoring optimisations can
only improve the retrieval fraction.  To meaningfully speed up the system,
the embedding call itself must be batched or cached.

### `np.maximum.reduceat` for per-patent max-pool

`reduceat` computes per-segment maxima in a single vectorised pass — no Python
loop over patents or chunks.  Empty segments are impossible because every
patent has at least one chunk (the abstract), which is enforced as an invariant
tested in the test suite.

---

## Known Limitations

- **Max-pooling length bias.** A patent with many description chunks has more
  chances of a high-scoring chunk than a short patent, even if it is less
  relevant overall.  The 0.9 paragraph weight partially mitigates this.

- **Heuristic fusion weights.** The 2:1 dense/keyword RRF weight and the 0.9
  paragraph type weight are initial guesses.  They should be tuned on a
  labelled evaluation set in Part 3.

- **No tire/tyre synonym.** British spelling `tyre` is not expanded to `tire`.
  A query for `tyre tread` will miss patents that only use `tire`.

- **No PDF upload.** The engine indexes only the pre-parsed JSON fields; there
  is no path to ingest a raw patent PDF.

---

## Part 2 POC — Ingestion Job Queue

`poc/` is a self-contained simulator of the ingestion pipeline described in
`docs/PART2_DESIGN.md`.  SQLite stands in for Postgres; multiple worker
processes run concurrently; random failures, poison docs, and worker crashes
are injected to exercise retry, dead-letter, and lease-recovery paths.

### Quick start

```bash
# One-command demo (init, seed, run, status, mutate 5 patents + re-run):
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

### Tests

```bash
pytest tests/test_poc.py -v
```
