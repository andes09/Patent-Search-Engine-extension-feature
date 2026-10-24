"""eval_rerank.py — Part 3: evaluation for the two-phase reranking system.

Commands
--------
  python eval_rerank.py              # run Set A (automatic self-retrieval)
  python eval_rerank.py latency      # latency table + quality-vs-n (Milestone 6)
  python eval_rerank.py make-pool    # build pool CSV for Set B  (Milestone 5)
  python eval_rerank.py score-b      # score labeled Set B pool (Milestone 5)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import date
from typing import Any

import numpy as np
import torch

from search.engine import SearchEngine
from search.rerank import _DEVICE, _get_model, two_phase_search

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR    = "data/data/patent_data_small"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "part3_eval.md")

SEED         = 42
N_QUERIES    = 100
N_CANDIDATES = 100
K            = 30

# ---------------------------------------------------------------------------
# Latency benchmark constants
# ---------------------------------------------------------------------------

# 8 NL queries substitute for the skipped Set B queries.
LATENCY_NL_QUERIES: list[str] = [
    "tire tread groove that reduces road noise",
    "rubber compound for high performance racing tire",
    "steel wheel rim with bead seat for tubeless tire",
    "spoke design for lightweight bicycle wheel",
    "pneumatic tire inner liner to prevent air leakage",
    "run-flat tire with reinforced sidewall structure",
    "wheel balancing weight attachment mechanism",
    "snow tire tread pattern for ice traction",
]

LATENCY_N_CANDIDATES: list[int] = [25, 50, 100]
N_LATENCY_REPS: int    = 3
N_LATENCY_WARMUP: int  = 2
N_LATENCY_CLAIMS: int  = 12
LATENCY_CLAIM_SEED: int = 42


# ---------------------------------------------------------------------------
# Query sampling (Set A)
# ---------------------------------------------------------------------------

def sample_queries(
    engine: SearchEngine,
    seed: int = SEED,
    n: int = N_QUERIES,
) -> list[dict[str, Any]]:
    """Sample *n* patents that have at least one independent claim.

    Returns a list of dicts with keys:
        pid         — patent index
        doc_number  — USPTO doc number
        query       — first independent claim body (no title prefix, no claim number)
        chunk_id    — chunk id of that claim (will be hidden during eval)
    """
    index = engine.index
    eligible: list[tuple[int, dict, int]] = []

    for pid, p in enumerate(index.patents):
        lo, hi = int(index.starts[pid]), int(index.ends[pid])
        for ci in range(lo, hi):
            if index.chunk_type[ci] == "claim" and index.chunk_independent[ci]:
                eligible.append((pid, p, ci))
                break   # first independent claim only

    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(n, len(eligible)))

    queries: list[dict[str, Any]] = []
    for pid, p, ci in chosen:
        title  = (p.get("title") or "").strip()
        prefix = f"{title}: "
        chunk_text = index.texts[ci]
        # Strip title prefix the engine prepended when building the chunk.
        query = chunk_text[len(prefix):] if chunk_text.startswith(prefix) else chunk_text
        queries.append({
            "pid":        pid,
            "doc_number": p["doc_number"],
            "query":      query,
            "chunk_id":   ci,
        })
    return queries


# ---------------------------------------------------------------------------
# Metric helpers (single relevant document per query)
# ---------------------------------------------------------------------------

def _find_rank(results: list, doc_number: str) -> int | None:
    """Return 1-based rank of *doc_number* in *results*, or None."""
    for i, r in enumerate(results):
        if r.doc_number == doc_number:
            return i + 1
    return None


def _hit(rank: int | None, k: int) -> float:
    return 1.0 if (rank is not None and rank <= k) else 0.0


def _mrr(rank: int | None, k: int) -> float:
    return (1.0 / rank) if (rank is not None and rank <= k) else 0.0


def _ndcg(rank: int | None, k: int) -> float:
    return (1.0 / math.log2(rank + 1)) if (rank is not None and rank <= k) else 0.0


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Set A evaluation
# ---------------------------------------------------------------------------

_CONFIGS: list[tuple[str, str | None, str | None]] = [
    # (display_name, model, mode)   model/mode=None → stage-1 only
    ("Stage 1 top 30", None,      None),
    ("MiniLM pure",    "minilm",  "pure"),
    ("MiniLM blend",   "minilm",  "blend"),
    ("BGE pure",       "bge",     "pure"),
    ("BGE blend",      "bge",     "blend"),
]


def run_set_a(
    engine: SearchEngine,
    queries: list[dict],
    n_candidates: int = N_CANDIDATES,
) -> tuple[float, dict]:
    """Evaluate all 5 configs on *queries*.

    Parameters
    ----------
    n_candidates : stage-1 pool size passed to both _search_candidates and two_phase_search.
                   Final k is clamped to min(K, n_candidates).

    Returns
    -------
    recall_n  : float  — fraction of queries where answer is in stage-1 top-n_candidates
    summary   : dict   — config_name → {hit1, hit10, mrr30, ndcg10}
    """
    # Pre-load models so the first query doesn't pay the load cost.
    print("  Pre-loading reranker models …")
    _get_model("minilm")
    _get_model("bge")
    print("  Models ready.\n")

    recall_hits: list[float] = []
    accum: dict[str, dict[str, list[float]]] = {
        name: {"hit1": [], "hit10": [], "mrr30": [], "ndcg10": []}
        for name, _, _ in _CONFIGS
    }

    n = len(queries)
    for i, q in enumerate(queries, 1):
        print(f"  Query {i:3d}/{n}  {q['doc_number']}", end="\r", flush=True)

        doc   = q["doc_number"]
        qtext = q["query"]
        excl  = {q["chunk_id"]}

        # Stage-1 candidates (run once; rerankers re-run stage-1 internally
        # but that's fine — the benchmark measures the real search path).
        s1 = engine._search_candidates(query=qtext, k=n_candidates, exclude_chunks=excl)
        s1_docs = {r.doc_number for r in s1.response.results}
        recall_hits.append(1.0 if doc in s1_docs else 0.0)

        # Stage-1 top-K metrics (K capped to available candidates)
        k_eff  = min(K, n_candidates)
        top_k  = s1.response.results[:k_eff]
        rank   = _find_rank(top_k, doc)
        accum["Stage 1 top 30"]["hit1"].append(_hit(rank, 1))
        accum["Stage 1 top 30"]["hit10"].append(_hit(rank, 10))
        accum["Stage 1 top 30"]["mrr30"].append(_mrr(rank, k_eff))
        accum["Stage 1 top 30"]["ndcg10"].append(_ndcg(rank, 10))

        # Reranker configs
        for name, model, mode in _CONFIGS[1:]:
            resp = two_phase_search(
                engine, query=qtext,
                n_candidates=n_candidates, k=k_eff,
                model=model, mode=mode,
                exclude_chunks=excl,
            )
            rank = _find_rank(resp.results, doc)
            accum[name]["hit1"].append(_hit(rank, 1))
            accum[name]["hit10"].append(_hit(rank, 10))
            accum[name]["mrr30"].append(_mrr(rank, k_eff))
            accum[name]["ndcg10"].append(_ndcg(rank, 10))

    print()  # end progress line

    recall100 = _mean(recall_hits)
    summary: dict[str, dict[str, float]] = {
        name: {
            "hit1":   _mean(accum[name]["hit1"]),
            "hit10":  _mean(accum[name]["hit10"]),
            "mrr30":  _mean(accum[name]["mrr30"]),
            "ndcg10": _mean(accum[name]["ndcg10"]),
        }
        for name, _, _ in _CONFIGS
    }
    return recall100, summary


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _md_table_a(summary: dict[str, dict[str, float]]) -> str:
    rows = [
        "| Config          | Hit@1 | Hit@10 | MRR@30 | nDCG@10 |",
        "|-----------------|------:|-------:|-------:|--------:|",
    ]
    for name, _ in [(n, None) for n, _, _ in _CONFIGS]:
        m = summary[name]
        rows.append(
            f"| {name:<15} | {m['hit1']:.3f} | {m['hit10']:.3f}  "
            f"| {m['mrr30']:.3f}  | {m['ndcg10']:.3f}   |"
        )
    return "\n".join(rows)


def _build_set_a_report(
    recall100: float,
    summary: dict[str, dict[str, float]],
    n_queries: int,
    device: str,
) -> str:
    s1_hit1 = summary["Stage 1 top 30"]["hit1"]
    lines: list[str] = [
        f"## Set A: Automatic self-retrieval",
        f"",
        f"Date: {date.today()}  |  Seed: {SEED}  |  Queries: {n_queries}  |  Device: {device}",
        f"Models: cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-base",
        f"n\\_candidates={N_CANDIDATES}, k={K}",
        f"",
        f"**Recall@100 (stage-1 ceiling): {recall100:.3f}**",
        f"",
        f"No reranker can recover a patent stage 1 missed.",
        f"",
    ]

    if s1_hit1 > 0.85:
        lines += [
            f"> **Headroom note:** Stage 1 Hit@1 is already {s1_hit1:.3f} (above 0.85).",
            f"> Set A has little room left for improvement, so a small or zero gain from",
            f"> reranking reflects the test set's limits, not necessarily the reranker's quality.",
            f"",
        ]

    lines += [
        _md_table_a(summary),
        f"",
        f"> **Known bias:** claims from the same patent share vocabulary with that patent's",
        f"> abstract and description, making this easier than real prior-art search.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Latency benchmark (Milestone 6)
# ---------------------------------------------------------------------------

def _latency_queries(engine: SearchEngine) -> list[str]:
    """Return 20 latency benchmark queries: 8 NL + 12 sampled independent claims."""
    queries: list[str] = list(LATENCY_NL_QUERIES)
    rng = random.Random(LATENCY_CLAIM_SEED)
    idx = engine.index
    indep = [
        i for i, (t, ind) in enumerate(zip(idx.chunk_type, idx.chunk_independent))
        if t == "claim" and ind
    ]
    chosen = rng.sample(indep, N_LATENCY_CLAIMS)
    queries.extend(idx.texts[ci] for ci in chosen)
    return queries


def run_latency_table(
    engine: SearchEngine,
    queries: list[str],
) -> dict[str, dict[int, dict[str, float]]]:
    """Measure rerank_ms and total_ms for both models at each n_candidates value.

    Returns nested dict: model → n_candidates → {rerank_med, rerank_p95,
    total_med, total_p95, n_pairs_med}.
    """
    out: dict[str, dict[int, dict[str, float]]] = {}
    for model_name in ("minilm", "bge"):
        out[model_name] = {}
        for n_cand in LATENCY_N_CANDIDATES:
            print(
                f"  {model_name:<6}  n_candidates={n_cand:<3}  "
                f"warm-up ({N_LATENCY_WARMUP} reps) …",
                end=" ", flush=True,
            )
            k_eff = min(K, n_cand)
            for _ in range(N_LATENCY_WARMUP):
                two_phase_search(
                    engine, query=queries[0],
                    n_candidates=n_cand, k=k_eff,
                    model=model_name,
                )

            rerank_ms_vals: list[float] = []
            total_ms_vals:  list[float] = []
            n_pairs_vals:   list[float] = []

            for q in queries:
                for _ in range(N_LATENCY_REPS):
                    resp = two_phase_search(
                        engine, query=q,
                        n_candidates=n_cand, k=k_eff,
                        model=model_name,
                    )
                    rerank_ms_vals.append(resp.timings["rerank_ms"])
                    total_ms_vals.append(resp.timings["total_ms"])
                    n_pairs_vals.append(float(resp.timings["n_pairs"]))

            out[model_name][n_cand] = {
                "rerank_med":  float(np.median(rerank_ms_vals)),
                "rerank_p95":  float(np.percentile(rerank_ms_vals, 95)),
                "total_med":   float(np.median(total_ms_vals)),
                "total_p95":   float(np.percentile(total_ms_vals, 95)),
                "n_pairs_med": float(np.median(n_pairs_vals)),
            }
            s = out[model_name][n_cand]
            print(
                f"done  "
                f"rerank_med={s['rerank_med']:.1f} ms  "
                f"n_pairs_med={s['n_pairs_med']:.0f}"
            )

    return out


def _build_latency_report(
    lat: dict[str, dict[int, dict[str, float]]],
) -> str:
    lines = [
        "### Reranker Latency",
        "",
        f"Queries: {len(LATENCY_NL_QUERIES)} NL + {N_LATENCY_CLAIMS} claims = "
        f"{len(LATENCY_NL_QUERIES) + N_LATENCY_CLAIMS} total  |  "
        f"Reps: {N_LATENCY_REPS}  |  Warm-up: {N_LATENCY_WARMUP}",
        "",
        "| Model  | n\\_candidates | n\\_pairs (med) | rerank med ms | rerank p95 ms |"
        " total med ms | total p95 ms |",
        "|--------|-------------:|--------------:|--------------:|--------------:|"
        "-------------:|-------------:|",
    ]
    for model_name in ("minilm", "bge"):
        for n_cand in LATENCY_N_CANDIDATES:
            s = lat[model_name][n_cand]
            lines.append(
                f"| {model_name:<6} | {n_cand:>12} | {s['n_pairs_med']:>13.0f} "
                f"| {s['rerank_med']:>13.1f} | {s['rerank_p95']:>13.1f} "
                f"| {s['total_med']:>12.1f} | {s['total_p95']:>12.1f} |"
            )
    return "\n".join(lines)


def _build_quality_vs_n_report(
    qvn: dict[int, dict[str, dict[str, float]]],
) -> str:
    lines = [
        "### Quality vs n\\_candidates (Set A, 100 queries)",
        "",
        "| n\\_candidates | Config          | Hit@1 | Hit@10 | MRR@30 | nDCG@10 |",
        "|-------------:|-----------------|------:|-------:|-------:|--------:|",
    ]
    for n_cand in sorted(qvn.keys()):
        summary = qvn[n_cand]
        for name, _, _ in _CONFIGS:
            m = summary[name]
            lines.append(
                f"| {n_cand:>12} | {name:<15} | {m['hit1']:.3f} | {m['hit10']:.3f}  "
                f"| {m['mrr30']:.3f}  | {m['ndcg10']:.3f}   |"
            )
    return "\n".join(lines)


def cmd_latency(engine: SearchEngine) -> None:
    """Run the latency table and quality-vs-n tables; append to results file."""
    print(f"Device: {_DEVICE}\n")

    print("Pre-loading reranker models …")
    _get_model("minilm")
    _get_model("bge")
    print("Models ready.\n")

    # ---- Latency table -------------------------------------------------------
    lq = _latency_queries(engine)
    print(
        f"Latency queries: {len(lq)} "
        f"({len(LATENCY_NL_QUERIES)} NL + {N_LATENCY_CLAIMS} claims)\n"
    )
    print("Running latency table …\n")
    lat = run_latency_table(engine, lq)

    # ---- Quality vs n_candidates ---------------------------------------------
    print(f"\nSampling {N_QUERIES} Set A queries (seed={SEED}) …")
    set_a_queries = sample_queries(engine)
    print(f"Sampled {len(set_a_queries)} queries.\n")

    print(f"Running Set A at n_candidates = {LATENCY_N_CANDIDATES} …\n")
    qvn: dict[int, dict[str, dict[str, float]]] = {}
    for n_cand in LATENCY_N_CANDIDATES:
        print(f"  n_candidates={n_cand}:")
        _, summary = run_set_a(engine, set_a_queries, n_candidates=n_cand)
        qvn[n_cand] = summary

    # ---- Format and print ----------------------------------------------------
    lat_report = _build_latency_report(lat)
    qvn_report = _build_quality_vs_n_report(qvn)

    print("\n" + lat_report + "\n")
    print(qvn_report + "\n")

    # ---- Append to results file ----------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as fh:
        fh.write(f"\n## Latency & Quality vs n\\_candidates\n\n")
        fh.write(f"Date: {date.today()}  |  Device: {_DEVICE}\n\n")
        fh.write(lat_report + "\n\n")
        fh.write(qvn_report + "\n")
    print(f"Results appended to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Set B stubs (Milestone 5)
# ---------------------------------------------------------------------------

def cmd_make_pool(engine: SearchEngine) -> None:  # noqa: ARG001
    print("make-pool: not yet implemented (Milestone 5)")
    sys.exit(0)


def cmd_score_b() -> None:
    print("score-b: not yet implemented (Milestone 5)")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the two-phase reranking system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (none)      Run Set A automatic self-retrieval evaluation
  latency     Run latency table and quality-vs-n_candidates table
  make-pool   Build the candidate pool CSV for Set B labelling  (Milestone 5)
  score-b     Score the labeled Set B pool                      (Milestone 5)
""",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="set-a",
        choices=["set-a", "latency", "make-pool", "score-b"],
        help="Evaluation command (default: set-a)",
    )
    parser.add_argument(
        "--data",
        default=DATA_DIR,
        metavar="DIR",
        help=f"Patent JSON directory (default: {DATA_DIR})",
    )
    args = parser.parse_args(argv)

    if args.command == "latency":
        print("Loading index …")
        engine = SearchEngine(args.data)
        print("Index ready.\n")
        cmd_latency(engine)
        return

    if args.command == "make-pool":
        print("Loading index …")
        engine = SearchEngine(args.data)
        cmd_make_pool(engine)
        return

    if args.command == "score-b":
        cmd_score_b()
        return

    # --- Set A ---------------------------------------------------------------
    print(f"Device: {_DEVICE}")
    print("Loading index …")
    engine = SearchEngine(args.data)
    print("Index ready.\n")

    print(f"Sampling {N_QUERIES} queries (seed={SEED}) …")
    queries = sample_queries(engine)
    print(f"Sampled {len(queries)} queries.\n")

    print("Running Set A evaluation (all 5 configs) …")
    recall100, summary = run_set_a(engine, queries)

    report = _build_set_a_report(recall100, summary, len(queries), _DEVICE)
    print("\n" + report + "\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        fh.write(f"# Part 3 Evaluation Results\n\n")
        fh.write(report + "\n")
    print(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
