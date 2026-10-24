"""search/dense.py — Milestone 2: dense (semantic) search with BGE embeddings."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from search.data import PatentIndex

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE instruction prefix is added to *queries only*, never to documents.
# Mixing it into documents quietly degrades retrieval quality.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Type weights applied to each chunk's cosine score before per-patent max-pool.
# Abstract and claims carry full weight; paragraphs get a small discount to
# reduce the length bias from max-pooling over many description chunks.
_TYPE_WEIGHTS: dict[str, float] = {
    "abstract": 1.0,
    "claim": 1.0,
    "paragraph": 0.9,
}

_CACHE_DIR = "cache"


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _cache_path(texts: list[str]) -> str:
    """Return the .npy cache path for *texts*, keyed on a SHA-1 of all content.

    Any change to chunking (text content, order, count) automatically
    invalidates the cache — we never key on filename alone.
    """
    sha = hashlib.sha1("\n".join(texts).encode()).hexdigest()
    model_slug = _MODEL_NAME.replace("/", "_")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{model_slug}_{sha}.npy")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DenseIndex:
    """Holds document embeddings and supports per-query dense retrieval."""

    def __init__(self, E: np.ndarray, index: PatentIndex, model: SentenceTransformer) -> None:
        # E: float32, shape (n_chunks, dim), L2-normalised
        self.E = E
        self.index = index
        self.model = model

        # Pre-compute type weight vector — one scalar per chunk — so scoring
        # is a single elementwise multiply, not a Python loop per query.
        weights = np.array(
            [_TYPE_WEIGHTS[t] for t in index.chunk_type], dtype=np.float32
        )
        self.weights = weights  # shape (n_chunks,)


def build_dense_index(index: PatentIndex) -> DenseIndex:
    """Embed all chunks (with caching) and return a DenseIndex."""
    model = SentenceTransformer(_MODEL_NAME)

    cache_file = _cache_path(index.texts)
    if os.path.exists(cache_file):
        E = np.load(cache_file)
    else:
        # Documents must NOT get the query prefix.
        E = model.encode(
            index.texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(cache_file, E)

    return DenseIndex(E, index, model)


def dense_search(
    dense: DenseIndex,
    query: str,
    k: int = 10,
    lo: int = 0,
    hi: int | None = None,
    allowed_pids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k patent indices and their dense scores.

    Parameters
    ----------
    dense       : DenseIndex built by build_dense_index
    query       : natural-language or claim text
    k           : number of results to return
    lo, hi      : chunk slice to search (contiguous class-filter range)
    allowed_pids: optional boolean mask over patents for non-contiguous filters

    Returns
    -------
    patent_ids  : int64 array of length ≤ k, sorted by descending score
    scores      : float32 array of matching scores
    """
    index = dense.index

    # Embed query with the BGE instruction prefix (documents never get this).
    q_vec = dense.model.encode(
        [_QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)[0]  # shape (dim,)

    if hi is None:
        hi = len(index.texts)

    # --- Score the candidate chunk slice -----------------------------------
    E_sub = dense.E[lo:hi]                         # numpy view, no copy
    raw_scores = E_sub @ q_vec                      # (n_chunks_in_range,)
    weighted = raw_scores * dense.weights[lo:hi]   # apply type discount

    # --- Per-patent max-pool using contiguous ranges -----------------------
    # starts/ends are absolute; convert to relative offsets within [lo, hi).
    n_patents = len(index.patents)

    # Find which patents fall (at least partially) in [lo, hi).
    # Because chunks are contiguous, a patent is in range iff
    # index.starts[pid] >= lo and index.ends[pid] <= hi (class filter guarantees this).
    # We use reduceat over the relative positions.

    # Build relative starts for patents whose chunks are inside [lo, hi).
    # For a class filter lo/hi covers exactly the right patents.
    # For a full search lo=0, hi=n_chunks, so all patents are included.

    # Map absolute starts → relative, clamped to [0, hi-lo].
    rel_starts = np.clip(index.starts - lo, 0, hi - lo)
    rel_ends   = np.clip(index.ends   - lo, 0, hi - lo)

    # Identify patents that have at least one chunk in [lo, hi).
    in_range = (rel_starts < rel_ends)
    pids_in_range = np.where(in_range)[0]

    if len(pids_in_range) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # np.maximum.reduceat: for each segment [rel_starts[pid], rel_starts[next_pid])
    # take the running max. We pass rel_starts for the in-range patents only.
    seg_starts = rel_starts[pids_in_range]
    patent_scores = np.maximum.reduceat(weighted, seg_starts)  # one value per in-range patent

    # Apply allowed_pids mask — masked patents get -inf and are excluded below.
    if allowed_pids is not None:
        mask = allowed_pids[pids_in_range]
        patent_scores = np.where(mask, patent_scores, -np.inf)

    # Keep only finite (non-masked) patents before top-k selection.
    # Dense scores are cosine similarities; masked entries are set to -inf.
    valid = np.isfinite(patent_scores)
    valid_pids   = pids_in_range[valid]
    valid_scores = patent_scores[valid]

    if len(valid_scores) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # --- Top-k selection ---------------------------------------------------
    k_actual = min(k, len(valid_scores))
    top_rel  = np.argpartition(valid_scores, -k_actual)[-k_actual:]
    top_rel  = top_rel[np.argsort(valid_scores[top_rel])[::-1]]

    return valid_pids[top_rel].astype(np.int64), valid_scores[top_rel].astype(np.float32)


def embed_query(dense: DenseIndex, query: str) -> np.ndarray:
    """Return the normalised float32 query vector (with BGE prefix applied)."""
    return dense.model.encode(
        [_QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)[0]


def best_chunk_for_patent(
    dense: DenseIndex,
    pid: int,
    q_vec: np.ndarray,
) -> int:
    """Return the chunk index with the highest weighted cosine score for patent *pid*."""
    index = dense.index
    lo, hi = int(index.starts[pid]), int(index.ends[pid])
    raw = dense.E[lo:hi] @ q_vec
    weighted = raw * dense.weights[lo:hi]
    return lo + int(np.argmax(weighted))
