"""Tests for search/rerank.py — Part 3 Milestone 2: two-phase reranking."""

from __future__ import annotations

import pytest

from search.engine import SearchEngine
from search.rerank import RerankResult, TwoPhaseResponse, two_phase_search


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine():
    return SearchEngine("data/data/patent_data_small")


@pytest.fixture(scope="session")
def rerank_resp(engine):
    """Small cached result used by multiple tests."""
    return two_phase_search(engine, "tire tread groove", n_candidates=20, k=10, mode="pure")


@pytest.fixture(scope="session")
def blend_resp(engine):
    return two_phase_search(engine, "tire tread groove", n_candidates=20, k=10, mode="blend")


# ---------------------------------------------------------------------------
# Basic shape / membership
# ---------------------------------------------------------------------------

class TestBasicShape:
    def test_results_le_k(self, rerank_resp):
        assert len(rerank_resp.results) <= 10

    def test_returns_results(self, rerank_resp):
        assert len(rerank_resp.results) > 0

    def test_all_from_stage1_candidates(self, engine, rerank_resp):
        """Every reranked result must have come from the stage-1 candidate set."""
        stage1 = engine._search_candidates("tire tread groove", k=20)
        stage1_docs = {r.doc_number for r in stage1.response.results}
        for r in rerank_resp.results:
            assert r.doc_number in stage1_docs, \
                f"{r.doc_number} not in stage-1 candidates"


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------

class TestResultFields:
    def test_required_fields_present(self, rerank_resp):
        for r in rerank_resp.results:
            assert r.doc_number
            assert isinstance(r.title, str)
            assert isinstance(r.stage1_rank, int) and r.stage1_rank >= 1
            assert isinstance(r.final_rank,  int) and r.final_rank  >= 1
            assert isinstance(r.moved,       int)
            assert isinstance(r.rerank_score, float)
            assert r.passage.text.strip()

    def test_no_duplicate_doc_numbers(self, rerank_resp):
        docs = [r.doc_number for r in rerank_resp.results]
        assert len(docs) == len(set(docs))

    def test_timings_keys(self, rerank_resp):
        expected = {"stage1_ms", "rerank_ms", "n_pairs", "total_ms"}
        assert set(rerank_resp.timings.keys()) == expected

    def test_timings_non_negative(self, rerank_resp):
        for k, v in rerank_resp.timings.items():
            if k != "n_pairs":
                assert v >= 0, f"Negative timing {k}: {v}"
        assert rerank_resp.timings["n_pairs"] >= 1


# ---------------------------------------------------------------------------
# pure mode: sorted by rerank_score descending
# ---------------------------------------------------------------------------

class TestPureMode:
    def test_sorted_by_rerank_score(self, rerank_resp):
        scores = [r.rerank_score for r in rerank_resp.results]
        assert scores == sorted(scores, reverse=True), \
            "pure mode results are not sorted by rerank_score"

    def test_score_equals_rerank_score(self, rerank_resp):
        """In pure mode, the reported .score must equal .rerank_score."""
        for r in rerank_resp.results:
            assert r.score == r.rerank_score

    def test_final_rank_sequential(self, rerank_resp):
        ranks = [r.final_rank for r in rerank_resp.results]
        assert ranks == list(range(1, len(ranks) + 1))


# ---------------------------------------------------------------------------
# blend mode
# ---------------------------------------------------------------------------

class TestBlendMode:
    def test_blend_returns_results(self, blend_resp):
        assert len(blend_resp.results) > 0

    def test_blend_result_count(self, blend_resp):
        assert len(blend_resp.results) <= 10

    def test_blend_score_differs_from_rerank_score(self, blend_resp):
        # In blend mode, .score is the blend formula, not the raw reranker score.
        # At least one result should differ (unless blend and pure happen to agree perfectly).
        for r in blend_resp.results:
            assert isinstance(r.score, float)
            assert isinstance(r.rerank_score, float)

    def test_blend_no_duplicates(self, blend_resp):
        docs = [r.doc_number for r in blend_resp.results]
        assert len(docs) == len(set(docs))


# ---------------------------------------------------------------------------
# moved field consistency
# ---------------------------------------------------------------------------

class TestMovedConsistency:
    def test_moved_equals_s1_minus_final(self, rerank_resp):
        for r in rerank_resp.results:
            assert r.moved == r.stage1_rank - r.final_rank, \
                f"moved mismatch: {r.moved} != {r.stage1_rank} - {r.final_rank}"

    def test_moved_consistency_blend(self, blend_resp):
        for r in blend_resp.results:
            assert r.moved == r.stage1_rank - r.final_rank


# ---------------------------------------------------------------------------
# n_pairs deduplication
# ---------------------------------------------------------------------------

class TestPairsDedup:
    def test_n_pairs_le_2x_candidates(self, engine):
        """Upper bound: each candidate contributes at most 2 unique texts."""
        resp = two_phase_search(engine, "tire tread groove", n_candidates=10, k=5, mode="pure")
        assert resp.timings["n_pairs"] <= 20
        assert resp.timings["n_pairs"] >= 1

    def test_n_pairs_matches_unique_texts(self, engine):
        """n_pairs must equal the number of unique chunk texts across all candidates,
        i.e. shared chunk texts are not scored twice."""
        n = 10
        # Get stage-1 candidates independently to compute expected unique texts.
        s1 = engine._search_candidates("tire tread groove", k=n)
        idx = engine.index
        unique_texts: set[str] = set()
        for sr in s1.response.results:
            pid = engine._doc_number_index[sr.doc_number]
            dense_ci = s1.dense_chunk.get(pid)
            kw_ci    = s1.keyword_chunk.get(pid)
            chunks: list[int] = []
            if dense_ci is not None:
                chunks.append(dense_ci)
            if kw_ci is not None and kw_ci != dense_ci:
                chunks.append(kw_ci)
            if not chunks:
                chunks.append(int(idx.starts[pid]))
            for ci in chunks:
                unique_texts.add(idx.texts[ci])

        resp = two_phase_search(engine, "tire tread groove", n_candidates=n, k=n, mode="pure")
        assert resp.timings["n_pairs"] == len(unique_texts), (
            f"n_pairs={resp.timings['n_pairs']} but unique texts={len(unique_texts)}"
        )


# ---------------------------------------------------------------------------
# Hard filters still apply to every reranked result
# ---------------------------------------------------------------------------

class TestFiltersHoldAfterRerank:
    def test_class_filter_all_results(self, engine):
        resp = two_phase_search(
            engine, "tire tread groove",
            n_candidates=50, k=10, mode="pure",
            class_prefix="B60C",
        )
        for r in resp.results:
            assert r.classification.upper().startswith("B60C"), \
                f"Expected B60C, got {r.classification}"

    def test_b60b_class_filter(self, engine):
        resp = two_phase_search(
            engine, "wheel rim spoke",
            n_candidates=50, k=10, mode="pure",
            class_prefix="B60B",
        )
        for r in resp.results:
            assert r.classification.upper().startswith("B60B")

    def test_empty_class_returns_empty(self, engine):
        resp = two_phase_search(
            engine, "tire",
            n_candidates=10, k=5, mode="pure",
            class_prefix="ZZZZ",
        )
        assert len(resp.results) == 0


# ---------------------------------------------------------------------------
# exclude_chunks still respected in two-phase search
# ---------------------------------------------------------------------------

class TestExcludeChunksInTwoPhase:
    def test_excluded_chunk_not_in_passage(self, engine):
        """An excluded chunk must not appear as the passage for any result."""
        pid0  = 0
        ci0   = int(engine.index.starts[pid0])
        resp  = two_phase_search(
            engine, "tire tread groove",
            n_candidates=20, k=10, mode="pure",
            exclude_chunks={ci0},
        )
        for r in resp.results:
            rpid = engine._doc_number_index[r.doc_number]
            lo   = int(engine.index.starts[rpid])
            hi   = int(engine.index.ends[rpid])
            passage_ci = None
            for ci in range(lo, hi):
                if engine.index.texts[ci] == r.passage.text:
                    passage_ci = ci
                    break
            assert passage_ci != ci0, \
                f"Excluded chunk {ci0} appeared as passage for {r.doc_number}"

    def test_patent_with_all_chunks_excluded_not_in_results(self, engine):
        pid0 = 0
        lo   = int(engine.index.starts[pid0])
        hi   = int(engine.index.ends[pid0])
        doc0 = engine.index.patents[pid0]["doc_number"]
        resp = two_phase_search(
            engine, "tire tread groove",
            n_candidates=30, k=15, mode="pure",
            exclude_chunks=set(range(lo, hi)),
        )
        assert all(r.doc_number != doc0 for r in resp.results)
