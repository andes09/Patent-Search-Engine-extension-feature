# Milestone Progress

## Milestone 1: Loading and Chunking (`search/data.py`) ✅

**What was built:**
- `load_patents(data_dir)` — scans `patents_ipa*.json` files, attaches `week` (YYMMDD) from filename, skips patents with no `doc_number` or no text, deduplicates keeping first occurrence, sorts by `(classification, doc_number)`.
- `PatentIndex` dataclass — holds all chunk arrays and patent metadata.
- `build_index(patents)` — produces contiguous per-patent chunk arrays:
  - **Abstract** chunk: `"{title}. {abstract}"` — one per patent.
  - **Claim** chunks: strips `^\s*\d+\s*[.)]\s*` prefix, prefixes with title. Claim 1 always independent; others dependent if body matches `\bclaims?\s+\d+`.
  - **Paragraph** chunks: drops empty and <40-char entries; paragraphs >350 words split into roughly equal pieces sharing the original paragraph index.
  - All arrays use `int64` / `bool` dtypes; chunks are contiguous per patent.

**Test result:** 40/40 passed in 0.45s.

---

## Milestone 2: Dense Search (`search/dense.py`) ✅

**What was built:**
- `build_dense_index(index)` — loads `BAAI/bge-small-en-v1.5` via sentence-transformers, embeds all chunks (float32, L2-normalised), caches to `cache/{model}_{sha1}.npy`. Cache key hashes all chunk texts so any chunking change auto-invalidates it.
- `DenseIndex` — holds embedding matrix `E` (n_chunks × dim), pre-computed per-chunk type-weight vector (abstract 1.0, claim 1.0, paragraph 0.9), and the model.
- `dense_search(dense, query, k, lo, hi, allowed_pids)` — embeds query with BGE instruction prefix, scores candidate slice via matmul, applies type weights, max-pools per patent with `np.maximum.reduceat`, selects top-k with `np.argpartition`.
- `embed_query` / `best_chunk_for_patent` — helpers for engine and result construction.

**Key design points:**
- BGE query prefix applied to queries only; never documents.
- `E[lo:hi]` is a numpy view (no copy) for the class-filter fast path.
- `np.maximum.reduceat` requires non-empty segments — guaranteed by M1 invariant.

**Test result:** 59/59 passed (657s first run — embedding 640 patents; cached thereafter).

---

## Milestone 3: Keyword Search (`search/keyword.py`) ✅

**What was built:**
- `tokenize(text)` — single canonical tokenizer: lowercase → `[a-z0-9]+` regex → drop len<2 → plural fold (drop trailing `s` on >4-char tokens unless ending `ss`) → drop stopwords (sklearn `ENGLISH_STOP_WORDS` + patent filler). Used everywhere: indexing, queries, filters, phrase checks, suggestions.
- `build_keyword_index(index)` — builds BM25 sparse matrix `W` (n_chunks × vocab, CSC format). Stores `field_boost × tf_norm` per entry; IDF computed as `log((N-df+0.5)/(df+0.5)+1)`. Tracks both chunk-level DF (for IDF) and patent-level DF (for suggestions).
- `keyword_search(...)` — query path: tokenize + IDF-trim to top-16 terms → sparse matmul `W[:,terms] @ (idf*weights)` → phrase bonus on top-200 candidates → `np.maximum.reduceat` per-patent max-pool → `np.argpartition` top-k positive-score patents.
- Phrase matching: quoted soft keywords (`"bead seat"`) matched via consecutive folded-token check on top candidates only.
- Zero-match suggestions: `difflib.get_close_matches` + shared-prefix search, returned only when a keyword term is absent from the vocab.
- Filter support: `lo/hi` contiguous slice (class filter) and `allowed_pids` boolean mask.

**Test result:** 103/103 passed in 11.4s.

---

## Milestone 4: Hard Filters, Fusion, and Public API (`search/engine.py`) ✅

**What was built:**
- `SearchEngine(data_dir)` — loads patents, builds dense + keyword indexes, builds inverted indexes for `title_has`/`abstract_has` (token → set of patent ids), title-exact dict, and doc_number lookup.
- Hard filters (applied before scoring): `class_prefix` via `bisect_left` → contiguous `[lo, hi)` chunk slice; `title_has` / `abstract_has` via posting-set intersection (smallest set first); `title_exact` via normalised-title dict lookup.
- Query planner: for `plan="auto"`, class filter uses `lo/hi` numpy view; word filters use `allowed_pids` boolean mask. If word-filter chunks < 25% of class range, planner notes "gather" (fast-path for future full implementation). `plan="post"` scores full corpus with all filters merged into a single mask. Both plans return identical results (invariant tested).
- Weighted RRF fusion: `1/(60+dense_rank) + 0.5/(60+kw_rank)` over top-100 from each signal. Missing list → 0 contribution. 1-based ranks.
- `mode`: hybrid | dense | keyword.
- `patent_id` query: builds query from title + abstract + first independent claim; excludes source patent from results.
- Results: `doc_number`, `title`, `classification`, `week`, `score`, `dense_rank`, `keyword_rank`, `found_by`, `matched_terms`, `matched_phrases`, `passage` (best chunk by dense similarity).
- Timings: `filter`, `query_embedding`, `dense`, `keyword`, `fusion`, `total`, `plan`.
- Bug fixed: `dense_search` now filters out `-inf` masked entries before `np.argpartition`.

**Test result:** 140/140 passed in 15.1s.

---

## Milestone 5: CLI (`search_cli.py`) ✅

**What was built:**
- `argparse` CLI with all required flags: positional `query`, `--patent`, `--mode`, `--class`, `--title-has`, `--abstract-has`, `--title`, `--keywords`, `-k`, `--plan`, `--data`.
- Readable output: header with query/mode/filters, numbered results showing every field (doc_number, title, class, week, score, dense/keyword ranks, found_by, matched terms/phrases, passage with type/ref/independence), zero-match suggestions, and a timings table.
- Smoke test: `python search_cli.py "tire tread groove that reduces road noise" -k 5` returns 5 relevant B60C patents in ~25ms (cached embeddings).

**Test result:** 140/140 passed (no new tests needed — CLI wraps the engine).

---

## Milestone 6: Benchmark (`benchmark.py`) ✅

**What was built:**
- `benchmark.py` — standalone timing script; prints Markdown tables for the README.
- **Section 1 — Index build:** times `load_patents()+build_index()` (load+chunk), `build_keyword_index()` (BM25), and `build_dense_index()` (cached .npy load). Reports cold embedding as "~657 s (first run only)."
- **Section 2 — Query embedding baseline:** warm-up then 5-rep median over 24 queries; reported separately since it's identical across all retrieval configurations.
- **Section 3 — Retrieval benchmark:** 5 filter configs × (3 modes, plus plan="post" for filtered) = 27 configurations; 2 warm-up + 5 reps × 24 queries each; reports median and p95 retrieval time (total − embedding).
- 12 NL queries (editable) + 12 independent claims sampled with fixed seed 42.

**Key results (this machine):**

| Phase | Time |
|---|---|
| Load + chunk (640 patents, 32 405 chunks) | 173 ms |
| BM25 build (vocab 76 067 terms) | 2 017 ms |
| Embedding (cached .npy load) | 619 ms |
| Query embedding (median, 24 queries) | 13.44 ms |
| Dense retrieval, no filter | 15.4 ms median / 33.6 ms p95 |
| Keyword retrieval, no filter | 0.55 ms median / 0.66 ms p95 |
| Hybrid retrieval, no filter | 14.2 ms median / 38.7 ms p95 |

**Test result:** 140/140 passed.

---

## Milestone 7: README ✅

**What was written:**
- Problem statement
- Ranking explanation: chunks + type weights, BGE dense scoring, BM25 with field boosts, weighted RRF and why it avoids score calibration
- Hard filters vs soft keywords (phrase bonus, zero-match suggestions)
- Missing-data policy and empty-paragraph finding (~20.8k dropped)
- Run instructions (setup, CLI flags, tests, benchmark)
- Full timing table from the benchmark run
- Efficiency commentary: contiguous class layout, selectivity-aware planner, CSC posting lists, `np.argpartition`, brute-force correctness at this scale, `np.maximum.reduceat`, query embedding as the real bottleneck
- Known limitations: max-pooling length bias, heuristic fusion weights, no tire/tyre synonym, no PDF upload
