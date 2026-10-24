"""benchmark.py — Milestone 6: index build and query timing benchmark.

Run:
    python benchmark.py

Prints Markdown-formatted tables for the README.
"""

from __future__ import annotations

import random
from time import perf_counter

import numpy as np

from search.data import build_index, load_patents
from search.dense import build_dense_index
from search.engine import SearchEngine
from search.keyword import build_keyword_index

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "data/data/patent_data_small"

# ── 12 natural-language vehicle queries — edit to taste ────────────────────
NL_QUERIES: list[str] = [
    "tire tread groove that reduces road noise",
    "rubber compound for high performance racing tire",
    "steel wheel rim with bead seat for tubeless tire",
    "spoke design for lightweight bicycle wheel",
    "pneumatic tire inner liner to prevent air leakage",
    "run-flat tire with reinforced sidewall structure",
    "wheel balancing weight attachment mechanism",
    "snow tire tread pattern for ice traction",
    "tire vulcanization mold with groove pattern",
    "alloy wheel with decorative spoke design",
    "tire pressure monitoring system sensor integration",
    "bicycle tire bead retention on rim hook",
]

CLAIM_SEED      = 42   # fixed seed → reproducible claim sample across runs
N_CLAIM_QUERIES = 12
N_REPS          = 5    # measured reps per query; median is taken
N_WARMUP        = 2    # discarded warm-up reps before measuring each config
K               = 10   # results per search call

# ---------------------------------------------------------------------------
# Filter configurations to benchmark
# ---------------------------------------------------------------------------

# (display_label, kwargs forwarded to engine.search())
FILTER_CONFIGS: list[tuple[str, dict]] = [
    ("none",                       {}),
    ("class B60C (~50%)",          {"class_prefix": "B60C"}),
    ("class B60D (~3%)",           {"class_prefix": "B60D"}),
    ("abstract_has tread",         {"abstract_has": ["tread"]}),
    ("B60B + title_has wheel",     {"class_prefix": "B60B", "title_has": ["wheel"]}),
]

MODES = ["dense", "keyword", "hybrid"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_claim_queries(engine: SearchEngine) -> list[str]:
    """Return N_CLAIM_QUERIES independent-claim texts, sampled with CLAIM_SEED."""
    rng = random.Random(CLAIM_SEED)
    idx = engine.index
    indep = [
        i for i, (t, ind) in enumerate(zip(idx.chunk_type, idx.chunk_independent))
        if t == "claim" and ind
    ]
    chosen = rng.sample(indep, N_CLAIM_QUERIES)
    return [idx.texts[i] for i in chosen]


def _bench_config(
    engine: SearchEngine,
    queries: list[str],
    mode: str,
    plan: str,
    filter_kwargs: dict,
) -> tuple[float, float, float]:
    """Run the benchmark for one (mode, plan, filter) configuration.

    Returns
    -------
    embed_med_s  : median query-embedding time over queries (seconds)
    retr_med_s   : median retrieval time (total - embed) over queries (seconds)
    retr_p95_s   : 95th-percentile retrieval time over queries (seconds)

    Retrieval time = total - query_embedding, so query embedding is excluded.
    """
    # Warm up on the first query — heats the JIT and any lazy-init paths.
    for _ in range(N_WARMUP):
        engine.search(queries[0], mode=mode, plan=plan, k=K, **filter_kwargs)

    embed_meds: list[float] = []
    retr_meds:  list[float] = []

    for query in queries:
        emb_reps  = []
        retr_reps = []
        for _ in range(N_REPS):
            resp   = engine.search(query, mode=mode, plan=plan, k=K, **filter_kwargs)
            t_emb  = resp.timings["query_embedding"]
            t_retr = resp.timings["total"] - t_emb
            emb_reps.append(t_emb)
            retr_reps.append(t_retr)
        embed_meds.append(float(np.median(emb_reps)))
        retr_meds.append(float(np.median(retr_reps)))

    return (
        float(np.median(embed_meds)),
        float(np.median(retr_meds)),
        float(np.percentile(retr_meds, 95)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ── Section 1: index build timing ─────────────────────────────────────
    print("## Index Build Timing\n")

    t = perf_counter()
    patents_b = load_patents(DATA_DIR)
    idx_b     = build_index(patents_b)
    t_load_chunk = perf_counter() - t

    t = perf_counter()
    kidx_b = build_keyword_index(idx_b)
    t_bm25 = perf_counter() - t

    # build_dense_index hits the on-disk cache; measures model-init + npy load.
    t = perf_counter()
    build_dense_index(idx_b)
    t_embed_cached = perf_counter() - t

    n_patents = len(idx_b.patents)
    n_chunks  = len(idx_b.texts)
    vocab_sz  = len(kidx_b.vocab)

    print(f"| Phase                | Time (ms)   | Notes                                  |")
    print(f"|---------------------|-------------|----------------------------------------|")
    print(
        f"| Load + chunk         | {t_load_chunk*1e3:>9.1f}  "
        f"| {n_patents} patents, {n_chunks} chunks                  |"
    )
    print(
        f"| BM25 build           | {t_bm25*1e3:>9.1f}  "
        f"| vocab {vocab_sz} terms                       |"
    )
    print(
        f"| Embedding (cached)   | {t_embed_cached*1e3:>9.1f}  "
        f"| load from .npy cache                   |"
    )
    print(
        f"| Embedding (cold)     | ~657 000.0  "
        f"| measured on first run; cached thereafter |"
    )
    print()

    # ── Build engine for query benchmark ──────────────────────────────────
    print("Building SearchEngine …", flush=True)
    engine  = SearchEngine(DATA_DIR)
    queries = NL_QUERIES + _sample_claim_queries(engine)
    assert len(queries) == 24, f"Expected 24 queries, got {len(queries)}"
    print(
        f"Ready: 24 queries "
        f"(12 NL + {N_CLAIM_QUERIES} claims, seed={CLAIM_SEED}).\n"
    )

    # ── Section 2: query embedding baseline ───────────────────────────────
    # Embedding time is independent of filter/mode; measure once on the
    # no-filter hybrid config and report it separately.
    print("Measuring query-embedding baseline …", flush=True)
    emb_med, _, _ = _bench_config(engine, queries, "hybrid", "auto", {})

    print(f"\n## Query Embedding Time (excluded from all retrieval timings below)\n")
    print(f"Median over 24 queries: **{emb_med*1e3:.2f} ms**\n")

    # ── Section 3: retrieval benchmark ────────────────────────────────────
    print("Running retrieval benchmark …\n", flush=True)

    # Row: (filter_label, mode, plan, median_ms, p95_ms)
    rows: list[tuple[str, str, str, float, float]] = []

    for filt_label, filt_kwargs in FILTER_CONFIGS:
        is_filtered = bool(filt_kwargs)
        # For filtered runs include plan="post" as the scoring baseline.
        plans = ["auto", "post"] if is_filtered else ["auto"]

        for plan in plans:
            for mode in MODES:
                print(
                    f"  {filt_label:<30}  mode={mode:<7}  plan={plan} …",
                    flush=True,
                )
                _, med, p95 = _bench_config(
                    engine, queries, mode, plan, filt_kwargs
                )
                rows.append((filt_label, mode, plan, med * 1e3, p95 * 1e3))

    # ── Markdown table ─────────────────────────────────────────────────────
    print()
    print("## Retrieval Timing (query embedding excluded)\n")
    hdr  = f"| {'Filter':<30} | {'Mode':<7} | {'Plan':<4} | {'Median ms':>9} | {'p95 ms':>7} |"
    sep  = f"|{'-'*32}|{'-'*9}|{'-'*6}|{'-'*11}|{'-'*9}|"
    print(hdr)
    print(sep)
    prev_filt = None
    for filt, mode, plan, med, p95 in rows:
        # Blank row between filter groups for readability
        if prev_filt is not None and filt != prev_filt:
            print(f"|{' '*32}|{' '*9}|{' '*6}|{' '*11}|{' '*9}|")
        print(
            f"| {filt:<30} | {mode:<7} | {plan:<4} | {med:>9.2f} | {p95:>7.2f} |"
        )
        prev_filt = filt

    print()


if __name__ == "__main__":
    main()
