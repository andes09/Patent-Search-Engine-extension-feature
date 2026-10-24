# Task: write the Part 2 design doc (`docs/PART2_DESIGN.md`)

Write a markdown design doc describing how the Part 1 search engine in this repo would run at
scale: 10 million patents. **Do not modify any code.** This task produces only the markdown file.

## Requirements from the brief
- An intern should be able to read it and start implementing.
- It must cover: system components, major pipelines, a breakdown of expected cost, error
  handling, how system contents and status are tracked, and major challenges at scale.
- It should pick something simple and openly acknowledge its weaknesses. It is not meant to be
  a perfect system.

## Before writing
1. Read `README.md` and the `search/` modules, so the doc refers to Part 1's real design:
   chunking, best-chunk scoring with type weights, BM25 with field boosts, weighted RRF
   (1.0 / 0.5), class-sorted contiguous layout, gather-vs-mask planner, hard filters vs soft
   keywords, and phrase bonus.
2. Wherever the doc says "in Part 1 we…", it must match the actual code. If the code differs from
   what's described below, flag the difference to me instead of guessing.

## Decisions (final — describe them, don't change them)

### 1. What gets embedded
- **v1:** claims and abstracts only, about 20 chunks per patent, so about 200M vectors.
  Descriptions stay keyword-searchable but are not embedded.
- **Storage:** int8 instead of float32 (384 bytes vs ~1.5 KB per vector), about 75 GB of vectors
  total.
- **Show the math:**
  - Embedding every chunk would be about 500M vectors and about 770 GB in float32.
  - Claims + abstract in float32 is about 300 GB.
  - Claims + abstract in int8 is about 75 GB.
  - Explain why it matters: vector indexes must sit in RAM to be fast, and RAM is the main cost.
- **The known weakness, stated plainly:** anything disclosed anywhere in an earlier patent counts
  as prior art, not only its claims. Claims-only embeddings miss semantic matches that exist only
  in a description paragraph. Include a short example: a claim says "damping member", while the
  description says it "suppresses vibration at highway speeds".
- **Phase 2 fix:** binary quantization (48 bytes per vector, so all ~500M chunks fit in ~24 GB).
  Binary vectors find candidates fast, then the top few hundred are re-scored with
  full-precision vectors kept on disk.

### 2. Components (explain simply, then technically)
- **S3:** raw USPTO files exactly as downloaded. Cheap and permanent.
- **Postgres:** the source of truth for patent metadata, chunks, and pipeline status. If the
  search index breaks, it is rebuilt from Postgres.
- **OpenSearch:** the search engine. One system handles BM25, phrase queries (it stores word
  positions), field boosts, filters, fuzzy matching and suggestions, and vector (k-NN) search.
  This maps almost every Part 1 feature onto native functionality.
- **Shard routing by classification code:** class-filtered queries only touch the relevant
  shards. This is the scale version of Part 1's class-sorted contiguous slice. Call out that
  parallel explicitly.
- **Query embedding service:** a small CPU service running the same `BAAI/bge-small-en-v1.5`
  model as indexing.
- **Embedding workers:** GPU workers that consume embedding jobs.

### 3. Ingestion pipeline
- Flow: weekly USPTO bulk files → S3 → parse workers → Postgres (metadata + chunks) → embedding
  workers → bulk writes into OpenSearch.
- **Job queue:** a Postgres table. Workers claim jobs with `SELECT ... FOR UPDATE SKIP LOCKED`.
  No separate queue system is needed, and the queue table doubles as the status table.
- **Job states:** `pending → parsing → parsed → embedding → embedded → indexing → indexed`,
  plus `failed` and `dead`.
- **Idempotency:** every job is keyed by `(doc_number, content_hash)`, so reprocessing never
  creates duplicates.
- **Retries:** exponential backoff with a maximum attempt count. After that, the job moves to a
  dead-letter state, with its error message stored for human review.
- **Two modes:** a one-time backfill of 10M patents, and weekly incremental runs.

### 4. Embedding
- Self-hosted bge-small on rented GPUs. Throughput is roughly a few thousand chunks per second
  per mid-range GPU, so the 200M-chunk backfill takes about a day.
- An API would cost roughly 30B tokens' worth (hundreds of dollars) and comes with rate limits.
- Store the model version with every vector.
- A model change means re-embedding into a **new** index. Run the regression query set against
  it, then switch traffic via an index alias (blue/green). Search never goes down during a
  migration.

### 5. Query path
- Flow: embed the query → one OpenSearch hybrid query (filters + BM25 + k-NN) → top 100 → fuse
  → return results with matched passage, matched terms, and found-by labels, as in Part 1.
- Scale-specific notes (these were parked during Part 1):
  - **Query trimming becomes mandatory.** Keep a long query's ~16 rarest terms, because common
    terms have index lists with millions of entries.
  - **Phrase checking** is native in OpenSearch via positional indexes. This replaces Part 1's
    candidate-only check, which was right for small data.
  - **Similar-word suggestions** use OpenSearch's built-in suggesters instead of scanning the
    vocabulary.
  - **Fusion:** describe weighted RRF. Note that the intern must verify which hybrid and fusion
    options the deployed OpenSearch version supports natively. If RRF isn't supported, the
    fallback is to run two queries and fuse them in the application, exactly as Part 1 does.
- **Future feature: upload a patent.** The patent is parsed, broken down (independent claims as
  intent, abstract, distinctive terms, suggested class code shown but not auto-applied), and its
  independent claims are searched separately. Because it runs several searches, it is an async
  job with a progress indicator, not a blocking request.

### 6. Cost (keep this short)
A small table of **rough orders of magnitude**, explicitly labeled as estimates to verify against
current cloud pricing. Do not invent precise prices.

| Item | Estimate |
|---|---|
| OpenSearch cluster (largest cost: vectors in RAM) | low thousands of dollars/month |
| Managed Postgres | a few hundred dollars/month |
| S3 raw storage (~1 TB) | tens of dollars/month |
| Embedding backfill | tens of dollars, one-time |
| Query embedding service | tens of dollars/month |

### 7. Challenges
- **Filtered approximate vector search can miss valid results** under very selective filters.
  The mitigation is switching to exact search when the filter is selective, which is the same
  idea as Part 1's gather-vs-mask planner. Say so. The intern must verify how the deployed
  OpenSearch version handles filtered k-NN.
- **Full re-embeds on model change** (handled by blue/green).
- **Shard skew:** some classification codes are huge. Split large classes into sub-shards.
- **Data quality:** old patents have OCR errors and missing fields. Parse failures go to the
  dead-letter state instead of crashing the pipeline.
- **The description-embedding gap** from section 1.
- **Regression risk:** a fixed benchmark query set runs after every reindex to catch silent
  quality drops.

### 8. Monitoring and status
- Job counts per state.
- Ingestion lag (newest USPTO week vs newest indexed week).
- Dead-letter count.
- Query p50/p95 latency.
- Regression-set scores after each reindex.

### 9. Proof of concept
A short section pointing to `poc/`: a SQLite-based ingestion job queue with worker processes,
injected random failures, retries with backoff, a dead-letter state, and a status report.
- Note that SQLite stands in for Postgres. Workers claim jobs with an atomic `UPDATE … RETURNING`
  because SQLite has no `SKIP LOCKED`.
- Production would use Postgres as described above.
- Leave a placeholder for the run command; it will be filled in once the POC is built.

### 10. Build order for an intern
Phased steps, each one testable on its own:
1. Postgres schema + job table
2. Parser for USPTO files
3. Embedding worker
4. OpenSearch index mapping + bulk loader
5. Query service
6. Monitoring
7. Backfill
8. Weekly incremental runs

## Format
- Open with a one-paragraph overview and a Mermaid diagram of the components and both pipelines
  (ingestion and query). GitHub renders Mermaid.
- Explain each component in plain words first, then technically.
- Aim for 1,500–2,500 words, using tables where they genuinely help.
- Mark every estimate as an estimate.
- Don't present unverified product capabilities as fact. Use "verify against current docs"
  wherever the decisions above say so.
- When done, summarize what you wrote and list anything you flagged as inconsistent with the
  Part 1 code.
