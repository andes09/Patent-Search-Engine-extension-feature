# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ThinkStruct interview problem repository. It currently contains only patent data — no source code, build system, or tests exist yet.

## Data

Patent JSON files are in `data/data/patent_data_small/`. Files are named `patents_ipa<YYMMDD>.json` (e.g., `patents_ipa240905.json`), covering weekly USPTO IPA releases from early 2024 through mid-2025 (~63 files).

Each JSON file is an array of patent objects with these fields:
- `title` — patent title
- `doc_number` — USPTO document number (e.g., `"20240293901"`)
- `filename` — source XML filename
- `abstract` — patent abstract text
- `detailed_description` — array of paragraph strings

# CLAUDE.md

Hybrid patent search engine (Thinkstruct take-home, Part 1). The full build spec is
`PART1_PROMPT.md`. It is the source of truth for every design decision. This file holds the
working rules that apply to every session.

## Ground rules
- **Never change a design decision silently.** If the spec seems wrong, ambiguous, or blocks
  you, stop and ask. Do not "improve" weights, thresholds, models, or architecture on your own.
- **Work one milestone at a time.** Finish the milestone, run the checks below, summarize, and
  wait for approval before starting the next.
- **Make small diffs.** Don't refactor code outside the current milestone unless asked.
- **Don't add dependencies.** Allowed: numpy, scipy, scikit-learn, sentence-transformers,
  pytest, plus the standard library. Banned: rank_bm25, faiss, any vector DB, any LLM API,
  pandas on query paths.
- **Never edit files in `data/`.** Treat the dataset as read-only.
- When unsure how a library behaves, write a 3-line check in a scratch script and run it.
  Don't guess.

## Environment (conda)
- The environment is defined in `environment.yml`. The env is named `patent-search`.
- Always run commands inside `patent-search`. Never install into `base`.
- Install packages from conda-forge only. Don't mix in `pip install`, since mixing conda and pip
  is a common source of broken environments. If a package is truly unavailable on conda-forge,
  stop and ask.
- To add a dependency (only if the ground rules allow it): add it to `environment.yml`, then run
  `conda env update -f environment.yml --prune`. Never install ad hoc.

## Commands
```bash
conda env create -f environment.yml         # first time
conda activate patent-search
conda env update -f environment.yml --prune # after environment.yml changes
pytest -q                                   # run after EVERY change
python search_cli.py "tire tread groove that reduces road noise" -k 5   # smoke test
python benchmark.py                         # Milestone 6 onward
```

## Definition of done (every milestone)
1. `pytest -q` passes, with new tests covering the milestone.
2. The smoke test runs end to end once the CLI exists.
3. No new warnings or errors in the output.
4. You report: what changed, the test output, and any deviation from the spec (there should be
   none).

## Invariants: assert these in tests, never break them
- Patents are sorted by `(classification, doc_number)`, and `codes` is sorted.
- Each patent's chunks are contiguous, with `starts[i] < ends[i]`. Every patent has at least one
  chunk; patents with zero chunks are dropped before indexing.
- `len(texts) == len(chunk_pid) == E.shape[0] == W.shape[0]`.
- Embeddings are float32 and L2-normalized (row norms ≈ 1).
- No empty or whitespace-only chunk text.
- `doc_number` is unique across the index.
- A source patent used as the query (`patent_id`) never appears in its own results.
- With any hard filter, every result satisfies that filter. Test this for all filter types.
- `plan="auto"` and `plan="post"` return identical results. Only speed may differ.

## Known pitfalls in this codebase
- **`np.maximum.reduceat` with an empty segment** silently returns the wrong value. Never let a
  patent have zero chunks.
- **`np.argpartition(x, k)` requires `k < len(x)`.** Clamp k, and handle an empty candidate set
  by returning no results instead of crashing.
- **Class prefix range:** use `bisect_left(codes, p)` and `bisect_left(codes, p + "\uffff")`.
  Uppercase the prefix first.
- **Sparse matrices:** `W[:, cols]` is fast only in CSC format, so keep W as CSC. Converting
  CSR to CSC on every query defeats the purpose. Use `.toarray().ravel()` or `np.asarray(...)`;
  `.A1` was removed in recent scipy.
- **BGE query prefix:** add it to queries only, never to documents. Mixing this up quietly
  degrades results.
- **The tokenizer must be identical** for indexing, queries, filters, phrase checks, and
  suggestions. There is exactly one `tokenize()` function, and everything imports it.
- **Embedding cache:** the cache key hashes the chunk texts, so any chunking change invalidates
  it automatically. Never key the cache on the filename alone.
- **Claim regex:** handle both "2 . text" and "10 ) text" formats.
- **Index dtypes:** use int64 for any index arithmetic (starts, ends, cumulative sums) to avoid
  overflow when the dataset grows.
- **RRF ranks are 1-based.** A patent missing from a list contributes 0, not 1/(60+∞) computed
  with a sentinel rank.

## Performance rules
- No Python loops over chunks or patents on a per-query path. Loops over query terms, filter
  keywords, or the final top-k results are fine.
- Scoring with only a class filter must use the contiguous slice `E[lo:hi]` (a numpy view), not
  fancy indexing.
- Phrase checks and zero-match suggestions run only on candidates or on demand, never over the
  full corpus per query.
- Build every index structure once at load time. Never rebuild anything per query.

## Timing rules (benchmark)
- Use `time.perf_counter()` only.
- Do warm-up runs before measuring.
- Report median and 95th percentile across queries.
- Report index-build time and query-embedding time separately from retrieval time.
- Use a fixed random seed for the sampled claim queries, so runs are reproducible.
- Don't "optimize" the benchmark itself. It must measure the real `search()` path.

## Data facts (verified)
- The dataset has 640 patents in 64 files. All fields are present.
- About 20.8k of the ~42k description entries are empty strings.
- Classification codes look like `B60B104FI`. The split is about 318 B60C, 298 B60B, 20 B60D,
  and a few others.
- If the loader reports different counts, stop and flag it before continuing.

## Style
- Python 3.10+ with type hints on public functions.
- Short comments explaining *why*, especially next to the non-obvious performance tricks. These
  comments feed the README and the video walkthrough.
- Keep the module layout from the spec: `search/data.py`, `search/dense.py`,
  `search/keyword.py`, `search/engine.py`, `search_cli.py`, `benchmark.py`, `tests/`.
- Commit after each approved milestone with the message `milestone N: <summary>`.
