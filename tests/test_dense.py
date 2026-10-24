"""Tests for search/dense.py — Milestone 2: dense search."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from search.data import build_index, load_patents
from search.dense import DenseIndex, build_dense_index, dense_search, embed_query

# ---------------------------------------------------------------------------
# Shared fixture: build dense index over real dataset once per session.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_dense():
    patents = load_patents("data/data/patent_data_small")
    index = build_index(patents)
    return build_dense_index(index)


# ---------------------------------------------------------------------------
# Embedding shape and dtype invariants
# ---------------------------------------------------------------------------

class TestEmbeddingInvariants:
    def test_shape_matches_chunks(self, real_dense):
        idx = real_dense.index
        assert real_dense.E.shape[0] == len(idx.texts)

    def test_dtype_is_float32(self, real_dense):
        assert real_dense.E.dtype == np.float32

    def test_embeddings_are_l2_normalised(self, real_dense):
        # Row norms should be ≈ 1 for all chunks.
        norms = np.linalg.norm(real_dense.E, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), \
            f"min norm={norms.min():.4f}, max norm={norms.max():.4f}"

    def test_weights_length_matches_chunks(self, real_dense):
        assert len(real_dense.weights) == len(real_dense.index.texts)

    def test_weights_are_float32(self, real_dense):
        assert real_dense.weights.dtype == np.float32


# ---------------------------------------------------------------------------
# Type weight values
# ---------------------------------------------------------------------------

class TestTypeWeights:
    def test_abstract_weight_is_1(self, real_dense):
        idx = real_dense.index
        i = next(i for i, t in enumerate(idx.chunk_type) if t == "abstract")
        assert real_dense.weights[i] == pytest.approx(1.0)

    def test_claim_weight_is_1(self, real_dense):
        idx = real_dense.index
        i = next(i for i, t in enumerate(idx.chunk_type) if t == "claim")
        assert real_dense.weights[i] == pytest.approx(1.0)

    def test_paragraph_weight_is_09(self, real_dense):
        idx = real_dense.index
        i = next(i for i, t in enumerate(idx.chunk_type) if t == "paragraph")
        assert real_dense.weights[i] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Cache: second call should be fast (loaded from .npy) and identical
# ---------------------------------------------------------------------------

class TestCache:
    def test_second_build_returns_identical_embeddings(self, real_dense):
        idx = real_dense.index
        dense2 = build_dense_index(idx)  # should hit the cache
        assert np.array_equal(real_dense.E, dense2.E)


# ---------------------------------------------------------------------------
# dense_search results
# ---------------------------------------------------------------------------

class TestDenseSearch:
    def test_returns_k_results(self, real_dense):
        pids, scores = dense_search(real_dense, "tire tread groove", k=5)
        assert len(pids) == 5
        assert len(scores) == 5

    def test_scores_descending(self, real_dense):
        _, scores = dense_search(real_dense, "tire tread groove", k=10)
        assert list(scores) == sorted(scores, reverse=True)

    def test_no_duplicate_patents(self, real_dense):
        pids, _ = dense_search(real_dense, "tire tread groove", k=20)
        assert len(pids) == len(set(pids.tolist()))

    def test_pids_in_valid_range(self, real_dense):
        n = len(real_dense.index.patents)
        pids, _ = dense_search(real_dense, "steel wheel rim", k=10)
        assert all(0 <= p < n for p in pids)

    def test_class_filter_lo_hi_respected(self, real_dense):
        """Results with a lo/hi slice must all come from patents in that range."""
        idx = real_dense.index
        # Use the first 100 patents' chunk range as a filter.
        lo = int(idx.starts[0])
        hi = int(idx.ends[99])
        pids, _ = dense_search(real_dense, "rubber compound", k=5, lo=lo, hi=hi)
        assert all(p < 100 for p in pids)

    def test_k_larger_than_corpus_returns_all(self, real_dense):
        n = len(real_dense.index.patents)
        pids, scores = dense_search(real_dense, "wheel", k=n + 100)
        assert len(pids) == n

    def test_relevant_result_in_top5(self, real_dense):
        """A tread-noise query should surface at least one B60C patent."""
        idx = real_dense.index
        pids, _ = dense_search(real_dense, "tire tread groove reduces road noise", k=5)
        classes = [idx.patents[p]["classification"] for p in pids]
        assert any(c.startswith("B60C") for c in classes)


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_shape(self, real_dense):
        vec = embed_query(real_dense, "test query")
        assert vec.shape == (real_dense.E.shape[1],)

    def test_dtype(self, real_dense):
        vec = embed_query(real_dense, "test query")
        assert vec.dtype == np.float32

    def test_normalised(self, real_dense):
        vec = embed_query(real_dense, "test query")
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4
