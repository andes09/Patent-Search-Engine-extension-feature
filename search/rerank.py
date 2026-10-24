"""search/rerank.py — Part 3 Milestone 2: two-phase cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from search.engine import Passage, SearchEngine, Stage1Candidates

# Detect once at import time so all models share the same device.
_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model registry — each model is loaded once and reused across all queries
# ---------------------------------------------------------------------------

_MODEL_NAMES: dict[str, str] = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge":    "BAAI/bge-reranker-base",
}

# Module-level cache: model name → CrossEncoder instance
_MODEL_CACHE: dict[str, CrossEncoder] = {}


def _get_model(name: str) -> CrossEncoder:
    """Return the cached CrossEncoder for *name*, loading it once on first call."""
    if name not in _MODEL_NAMES:
        raise ValueError(f"Unknown reranker model {name!r}; choose 'minilm' or 'bge'")
    if name not in _MODEL_CACHE:
        ce = CrossEncoder(_MODEL_NAMES[name], max_length=512, device=_DEVICE)
        # Set eval mode on the underlying PyTorch model (no dropout during inference).
        ce.model.eval()
        print(f"[reranker] {name} loaded on device: {_DEVICE}")
        _MODEL_CACHE[name] = ce
    return _MODEL_CACHE[name]


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    doc_number: str
    title: str
    classification: str
    week: str
    score: float            # final ranking score (rerank_score for pure; blend score for blend)
    dense_rank: int | None
    keyword_rank: int | None
    found_by: str
    matched_terms: list[str]
    matched_phrases: list[str]
    passage: Passage        # the chunk that scored highest under the reranker
    stage1_rank: int        # 1-based position in stage-1 top-n_candidates
    final_rank: int         # 1-based position after reranking
    moved: int              # stage1_rank − final_rank  (positive = moved up)
    rerank_score: float     # raw cross-encoder score


@dataclass
class TwoPhaseResponse:
    results: list[RerankResult]
    suggestions: dict[str, list[tuple[str, int]]]
    timings: dict[str, Any]   # stage1_ms, rerank_ms, n_pairs (int), total_ms


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def two_phase_search(
    engine: SearchEngine,
    query: str | None = None,
    patent_id: str | None = None,
    *,
    n_candidates: int = 100,
    k: int = 30,
    model: str = "minilm",
    mode: str = "pure",
    exclude_chunks: set[int] | None = None,
    **stage1_filters: Any,
) -> TwoPhaseResponse:
    """Two-phase patent search: stage-1 hybrid retrieval followed by cross-encoder reranking.

    Parameters
    ----------
    engine         : loaded SearchEngine
    query          : natural-language or claim query text
    patent_id      : doc_number of a seed patent (expands to title+abstract+claim 1)
    n_candidates   : number of stage-1 candidates to rerank
    k              : final results to return (≤ n_candidates)
    model          : "minilm"  — cross-encoder/ms-marco-MiniLM-L-6-v2 (fast)
                     "bge"     — BAAI/bge-reranker-base (stronger, slower)
    mode           : "pure"   — rank by reranker score only
                     "blend"  — weighted RRF: 1/(60+rerank_rank) + 0.5/(60+stage1_rank)
    exclude_chunks : chunk ids invisible to both stages
    **stage1_filters : class_prefix, title_has, abstract_has, title_exact, keywords, plan
    """
    if mode not in ("pure", "blend"):
        raise ValueError(f"mode must be 'pure' or 'blend', got {mode!r}")

    t_total = perf_counter()

    # ---- Stage 1: hybrid retrieval of n_candidates --------------------------
    t_s1 = perf_counter()
    s1: Stage1Candidates = engine._search_candidates(
        query=query,
        patent_id=patent_id,
        k=n_candidates,
        exclude_chunks=exclude_chunks,
        mode="hybrid",   # stage 1 is always hybrid
        **stage1_filters,
    )
    stage1_ms = (perf_counter() - t_s1) * 1000.0

    resp = s1.response
    if not resp.results:
        return TwoPhaseResponse(
            results=[],
            suggestions=resp.suggestions,
            timings={
                "stage1_ms": stage1_ms,
                "rerank_ms": 0.0,
                "n_pairs":   0,
                "total_ms":  (perf_counter() - t_total) * 1000.0,
            },
        )

    # Resolve query text for (query, passage) pairs.
    resolved_query = _resolve_query_text(engine, query, patent_id)

    # ---- Build (query, chunk_text) pairs for the cross-encoder --------------
    # Each candidate contributes at most two texts: best dense chunk and best
    # keyword chunk.  If both point to the same chunk, or only one signal was
    # active, we send one pair.  Texts are deduplicated across all candidates
    # so shared chunks are scored exactly once.
    index = engine.index

    # unique_texts: ordered list of distinct chunk texts for this batch
    unique_texts: list[str] = []
    seen_text_idx: dict[str, int] = {}   # text → index in unique_texts

    # per-candidate: which chunk ids to score (1 or 2)
    cand_chunk_ids: list[list[int]] = []

    for sr in resp.results:
        pid = engine._doc_number_index[sr.doc_number]
        dense_ci = s1.dense_chunk.get(pid)
        kw_ci    = s1.keyword_chunk.get(pid)

        chunks: list[int] = []
        if dense_ci is not None:
            chunks.append(dense_ci)
        # Only add keyword chunk if it differs from the dense chunk.
        if kw_ci is not None and kw_ci != dense_ci:
            chunks.append(kw_ci)
        if not chunks:
            # Fallback: use absolute start chunk (should not happen given invariants)
            chunks.append(int(index.starts[pid]))

        cand_chunk_ids.append(chunks)

        # Register each text once in the dedup table.
        for ci in chunks:
            text = index.texts[ci]
            if text not in seen_text_idx:
                seen_text_idx[text] = len(unique_texts)
                unique_texts.append(text)

    # ---- Score all unique pairs in a single batched predict call ------------
    t_rerank = perf_counter()
    ce = _get_model(model)
    pairs = [[resolved_query, t] for t in unique_texts]
    # Use raw logits (Identity activation) for all models.
    # Sigmoid is monotonic in theory, but float32 precision causes ties when
    # logits cluster in the high range (e.g. BGE outputs logits > 8 for most
    # patent passages, which all round to ~0.9997 after sigmoid).
    raw_scores = ce.predict(
        pairs,
        show_progress_bar=False,
        activation_fn=torch.nn.Identity(),
    )
    n_pairs = len(pairs)
    rerank_ms = (perf_counter() - t_rerank) * 1000.0

    # Map chunk text → score (O(1) lookup below)
    text_score: dict[str, float] = {
        t: float(raw_scores[i]) for i, t in enumerate(unique_texts)
    }

    # ---- Per-candidate: best reranker score + winning chunk -----------------
    cand_rerank: list[float] = []
    cand_best_ci: list[int] = []

    for chunk_ids in cand_chunk_ids:
        best_score = float("-inf")
        best_ci    = chunk_ids[0]
        for ci in chunk_ids:
            s = text_score[index.texts[ci]]
            if s > best_score:
                best_score = s
                best_ci    = ci
        cand_rerank.append(best_score)
        cand_best_ci.append(best_ci)

    # ---- Ranking ------------------------------------------------------------
    n_cands = len(resp.results)

    # rerank_order: candidate indices sorted best→worst by reranker score
    rerank_order = sorted(range(n_cands), key=lambda i: -cand_rerank[i])

    # 1-based rerank rank for each candidate index
    rerank_rank: list[int] = [0] * n_cands
    for rank_pos, cand_i in enumerate(rerank_order):
        rerank_rank[cand_i] = rank_pos + 1

    if mode == "pure":
        final_order = rerank_order
    else:
        # blend: 1/(60+rerank_rank) + 0.5/(60+stage1_rank)
        # stage1_rank for candidate i is i+1 (resp.results already sorted by stage-1 score)
        blend: list[float] = [
            1.0 / (60 + rerank_rank[i]) + 0.5 / (60 + (i + 1))
            for i in range(n_cands)
        ]
        final_order = sorted(range(n_cands), key=lambda i: -blend[i])

    final_order = final_order[:k]

    # ---- Build result objects -----------------------------------------------
    results: list[RerankResult] = []
    for final_pos, cand_i in enumerate(final_order):
        sr             = resp.results[cand_i]
        stage1_rank_v  = cand_i + 1
        final_rank_v   = final_pos + 1

        best_ci = cand_best_ci[cand_i]
        passage = Passage(
            type=index.chunk_type[best_ci],
            ref=int(index.chunk_ref[best_ci]),
            independent=bool(index.chunk_independent[best_ci]),
            text=index.texts[best_ci],
        )

        if mode == "pure":
            final_score = cand_rerank[cand_i]
        else:
            i = cand_i
            final_score = 1.0 / (60 + rerank_rank[i]) + 0.5 / (60 + (i + 1))

        results.append(RerankResult(
            doc_number=sr.doc_number,
            title=sr.title,
            classification=sr.classification,
            week=sr.week,
            score=final_score,
            dense_rank=sr.dense_rank,
            keyword_rank=sr.keyword_rank,
            found_by=sr.found_by,
            matched_terms=sr.matched_terms,
            matched_phrases=sr.matched_phrases,
            passage=passage,
            stage1_rank=stage1_rank_v,
            final_rank=final_rank_v,
            moved=stage1_rank_v - final_rank_v,
            rerank_score=cand_rerank[cand_i],
        ))

    total_ms = (perf_counter() - t_total) * 1000.0
    return TwoPhaseResponse(
        results=results,
        suggestions=resp.suggestions,
        timings={
            "stage1_ms": stage1_ms,
            "rerank_ms": rerank_ms,
            "n_pairs":   n_pairs,
            "total_ms":  total_ms,
        },
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resolve_query_text(
    engine: SearchEngine,
    query: str | None,
    patent_id: str | None,
) -> str:
    """Return the resolved query string that was used for stage-1 retrieval."""
    if patent_id is not None:
        pid = engine._doc_number_index.get(patent_id)
        if pid is None:
            raise ValueError(f"Patent {patent_id!r} not found in index")
        p = engine.index.patents[pid]
        title    = (p.get("title")    or "").strip()
        abstract = (p.get("abstract") or "").strip()
        q = f"{title}. {abstract}"
        idx = engine.index
        for ci in range(int(idx.starts[pid]), int(idx.ends[pid])):
            if idx.chunk_type[ci] == "claim" and idx.chunk_independent[ci]:
                q += " " + idx.texts[ci]
                break
        return q
    if query:
        return query
    raise ValueError("Provide query text or a valid patent_id")
