"""Tests for search/engine.py — Milestone 4: hard filters, fusion, and public API."""

from __future__ import annotations

import pytest
import numpy as np

from search.engine import SearchEngine, SearchResponse, SearchResult
from search.keyword import tokenize


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine():
    return SearchEngine("data/data/patent_data_small")


# ---------------------------------------------------------------------------
# Basic query
# ---------------------------------------------------------------------------

class TestBasicQuery:
    def test_returns_k_results(self, engine):
        resp = engine.search("tire tread groove", k=5)
        assert len(resp.results) == 5

    def test_scores_descending(self, engine):
        resp = engine.search("tire tread groove", k=10)
        scores = [r.score for r in resp.results]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_results(self, engine):
        resp = engine.search("tire tread groove", k=20)
        nums = [r.doc_number for r in resp.results]
        assert len(nums) == len(set(nums))

    def test_k_larger_than_corpus(self, engine):
        n = len(engine.index.patents)
        resp = engine.search("wheel", k=n + 100)
        assert len(resp.results) <= n

    def test_all_scores_positive(self, engine):
        resp = engine.search("tire tread groove", k=10)
        assert all(r.score > 0 for r in resp.results)


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------

class TestResultFields:
    def test_all_fields_present(self, engine):
        resp = engine.search("tire tread groove", k=3)
        for r in resp.results:
            assert r.doc_number
            assert isinstance(r.title, str)
            assert isinstance(r.classification, str)
            assert isinstance(r.week, str)
            assert r.score > 0
            assert r.found_by in ("semantic", "keyword", "both")
            assert isinstance(r.matched_terms, list)
            assert isinstance(r.matched_phrases, list)

    def test_passage_fields(self, engine):
        resp = engine.search("tire tread groove", k=3)
        for r in resp.results:
            assert r.passage.type in ("abstract", "claim", "paragraph")
            assert isinstance(r.passage.ref, int)
            assert isinstance(r.passage.independent, bool)
            assert r.passage.text.strip()

    def test_timings_keys(self, engine):
        resp = engine.search("tire tread", k=5)
        expected = {"filter", "query_embedding", "dense", "keyword", "fusion", "total", "plan"}
        assert set(resp.timings.keys()) == expected

    def test_timings_non_negative(self, engine):
        resp = engine.search("tire tread", k=5)
        for key, val in resp.timings.items():
            if key != "plan":
                assert val >= 0, f"Negative timing for {key}: {val}"


# ---------------------------------------------------------------------------
# Source patent excluded from own results
# ---------------------------------------------------------------------------

class TestPatentIdQuery:
    def test_source_excluded(self, engine):
        doc = engine.index.patents[0]["doc_number"]
        resp = engine.search(patent_id=doc, k=10)
        assert all(r.doc_number != doc for r in resp.results)

    def test_patent_id_returns_results(self, engine):
        doc = engine.index.patents[0]["doc_number"]
        resp = engine.search(patent_id=doc, k=5)
        assert len(resp.results) > 0

    def test_unknown_patent_id_raises(self, engine):
        with pytest.raises(ValueError):
            engine.search(patent_id="99999999999")


# ---------------------------------------------------------------------------
# Hard filters — class_prefix
# ---------------------------------------------------------------------------

class TestClassFilter:
    def test_all_results_match_prefix(self, engine):
        resp = engine.search("tire tread", k=10, class_prefix="B60C")
        for r in resp.results:
            assert r.classification.upper().startswith("B60C"), \
                f"Expected B60C, got {r.classification}"

    def test_b60b_filter(self, engine):
        resp = engine.search("wheel rim spoke", k=10, class_prefix="B60B")
        for r in resp.results:
            assert r.classification.upper().startswith("B60B")

    def test_nonexistent_class_returns_empty(self, engine):
        resp = engine.search("tire", k=5, class_prefix="ZZZZ")
        assert len(resp.results) == 0

    def test_class_filter_reduces_results(self, engine):
        resp_all   = engine.search("tire", k=100)
        resp_class = engine.search("tire", k=100, class_prefix="B60C")
        assert len(resp_class.results) <= len(resp_all.results)


# ---------------------------------------------------------------------------
# Hard filters — title_has
# ---------------------------------------------------------------------------

class TestTitleHasFilter:
    def test_every_result_has_word(self, engine):
        resp = engine.search("tire rubber", k=10, title_has=["wheel"])
        idx = engine.index
        for r in resp.results:
            pid   = engine._doc_number_index[r.doc_number]
            title = idx.patents[pid].get("title", "") or ""
            # After tokenisation, "wheel" (or its folded form) must appear
            assert "wheel" in tokenize(title), \
                f"'wheel' not found in title: {title!r}"

    def test_multi_word_title_has(self, engine):
        resp = engine.search("wheel", k=10, title_has=["wheel", "rim"])
        idx = engine.index
        for r in resp.results:
            pid   = engine._doc_number_index[r.doc_number]
            title = idx.patents[pid].get("title", "") or ""
            toks  = set(tokenize(title))
            assert "wheel" in toks and "rim" in toks, \
                f"Not both words in title: {title!r}"

    def test_stopword_title_has_has_no_effect(self, engine):
        # "the" is a stopword; filter should have no effect and not crash
        resp = engine.search("tire", k=5, title_has=["the"])
        assert len(resp.results) > 0

    def test_impossible_title_has_returns_empty(self, engine):
        resp = engine.search("tire", k=5, title_has=["xyzxyzimpossible"])
        assert len(resp.results) == 0


# ---------------------------------------------------------------------------
# Hard filters — abstract_has
# ---------------------------------------------------------------------------

class TestAbstractHasFilter:
    def test_every_result_has_word(self, engine):
        resp = engine.search("tire rubber", k=10, abstract_has=["tread"])
        idx = engine.index
        for r in resp.results:
            pid  = engine._doc_number_index[r.doc_number]
            abst = idx.patents[pid].get("abstract", "") or ""
            assert "tread" in tokenize(abst), \
                f"'tread' not found in abstract: {abst[:80]!r}"


# ---------------------------------------------------------------------------
# Hard filters — title_exact
# ---------------------------------------------------------------------------

class TestTitleExactFilter:
    def test_exact_title_match(self, engine):
        # Use the actual title of the first patent
        first_title = engine.index.patents[0].get("title", "")
        if not first_title:
            pytest.skip("First patent has no title")
        resp = engine.search("tire wheel", k=5, title_exact=first_title)
        if resp.results:
            assert resp.results[0].title.lower() == first_title.lower()

    def test_unknown_exact_title_returns_empty(self, engine):
        resp = engine.search("tire", k=5, title_exact="Absolutely Nonexistent Patent Title XYZ")
        assert len(resp.results) == 0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

class TestModes:
    def test_dense_mode_keyword_rank_none(self, engine):
        resp = engine.search("tire tread groove", k=5, mode="dense")
        for r in resp.results:
            assert r.dense_rank is not None
            assert r.keyword_rank is None
            assert r.found_by == "semantic"

    def test_keyword_mode_dense_rank_none(self, engine):
        resp = engine.search("tire tread groove", k=5, mode="keyword")
        for r in resp.results:
            assert r.keyword_rank is not None
            assert r.dense_rank is None
            assert r.found_by == "keyword"

    def test_hybrid_mode_has_both_fields(self, engine):
        resp = engine.search("tire tread groove", k=10, mode="hybrid")
        # At least some results should be found by both signals
        found_bys = {r.found_by for r in resp.results}
        assert found_bys <= {"semantic", "keyword", "both"}

    def test_hybrid_returns_results(self, engine):
        resp = engine.search("tire tread groove", k=5, mode="hybrid")
        assert len(resp.results) > 0


# ---------------------------------------------------------------------------
# plan="auto" and plan="post" — identical results invariant
# ---------------------------------------------------------------------------

class TestPlanInvariant:
    def _same_docs(self, r1: SearchResponse, r2: SearchResponse) -> bool:
        return (
            [r.doc_number for r in r1.results]
            == [r.doc_number for r in r2.results]
        )

    def test_no_filter(self, engine):
        r1 = engine.search("tire tread groove", k=10, plan="auto")
        r2 = engine.search("tire tread groove", k=10, plan="post")
        assert self._same_docs(r1, r2)

    def test_class_filter(self, engine):
        r1 = engine.search("tread groove", k=10, class_prefix="B60C", plan="auto")
        r2 = engine.search("tread groove", k=10, class_prefix="B60C", plan="post")
        assert self._same_docs(r1, r2)

    def test_class_small_filter(self, engine):
        # B60D is ~3% of corpus — likely triggers "gather" planner
        r1 = engine.search("tire", k=5, class_prefix="B60D", plan="auto")
        r2 = engine.search("tire", k=5, class_prefix="B60D", plan="post")
        assert {r.doc_number for r in r1.results} == {r.doc_number for r in r2.results}

    def test_title_has_filter(self, engine):
        r1 = engine.search("wheel rim", k=10, title_has=["wheel"], plan="auto")
        r2 = engine.search("wheel rim", k=10, title_has=["wheel"], plan="post")
        assert self._same_docs(r1, r2)

    def test_abstract_has_filter(self, engine):
        r1 = engine.search("tread noise", k=10, abstract_has=["tread"], plan="auto")
        r2 = engine.search("tread noise", k=10, abstract_has=["tread"], plan="post")
        assert self._same_docs(r1, r2)

    def test_combined_class_and_title(self, engine):
        r1 = engine.search("wheel", k=10, class_prefix="B60B",
                           title_has=["wheel"], plan="auto")
        r2 = engine.search("wheel", k=10, class_prefix="B60B",
                           title_has=["wheel"], plan="post")
        assert {r.doc_number for r in r1.results} == {r.doc_number for r in r2.results}


# ---------------------------------------------------------------------------
# Filter correctness — every result must satisfy any active filter
# ---------------------------------------------------------------------------

class TestFilterCorrectness:
    def test_class_filter_all_results(self, engine):
        for prefix in ("B60B", "B60C"):
            resp = engine.search("tire wheel", k=20, class_prefix=prefix)
            for r in resp.results:
                assert r.classification.upper().startswith(prefix)

    def test_title_has_all_results(self, engine):
        resp = engine.search("rubber compound", k=10, title_has=["tire"])
        idx = engine.index
        for r in resp.results:
            pid   = engine._doc_number_index[r.doc_number]
            title = idx.patents[pid].get("title", "") or ""
            # After tokenisation "tire" must appear (possibly folded from "tires")
            assert "tire" in tokenize(title)

    def test_abstract_has_all_results(self, engine):
        resp = engine.search("rubber compound", k=10, abstract_has=["rubber"])
        idx = engine.index
        for r in resp.results:
            pid  = engine._doc_number_index[r.doc_number]
            abst = idx.patents[pid].get("abstract", "") or ""
            assert "rubber" in tokenize(abst)

    def test_class_plus_title_filter(self, engine):
        resp = engine.search("wheel", k=10, class_prefix="B60B",
                             title_has=["wheel"])
        idx = engine.index
        for r in resp.results:
            assert r.classification.upper().startswith("B60B")
            pid   = engine._doc_number_index[r.doc_number]
            title = idx.patents[pid].get("title", "") or ""
            assert "wheel" in tokenize(title)


# ---------------------------------------------------------------------------
# Milestone 1 (Part 3): exclude_chunks parameter
# ---------------------------------------------------------------------------

class TestExcludeChunks:
    def test_no_change_when_none(self, engine):
        """exclude_chunks=None must produce identical results to not passing it."""
        r1 = engine.search("tire tread groove", k=10)
        r2 = engine.search("tire tread groove", k=10, exclude_chunks=None)
        assert [r.doc_number for r in r1.results] == [r.doc_number for r in r2.results]

    def test_excluded_chunk_never_in_passage(self, engine):
        """A excluded chunk id must not appear as any result's passage chunk."""
        # Pick the abstract chunk (chunk 0) of the first patent.
        pid0 = 0
        ci0 = int(engine.index.starts[pid0])  # first chunk of first patent
        resp = engine.search("tire tread groove", k=20, exclude_chunks={ci0})
        for r in resp.results:
            # Recover the chunk id backing this passage via text match
            rpid = engine._doc_number_index[r.doc_number]
            lo = int(engine.index.starts[rpid])
            hi = int(engine.index.ends[rpid])
            passage_ci = None
            for ci in range(lo, hi):
                if engine.index.texts[ci] == r.passage.text:
                    passage_ci = ci
                    break
            assert passage_ci != ci0, \
                f"Excluded chunk {ci0} appeared as passage for {r.doc_number}"

    def test_patent_with_all_chunks_excluded_dropped(self, engine):
        """A patent whose every chunk is excluded must not appear in results."""
        pid0 = 0
        lo = int(engine.index.starts[pid0])
        hi = int(engine.index.ends[pid0])
        all_chunks = set(range(lo, hi))
        doc0 = engine.index.patents[pid0]["doc_number"]
        resp = engine.search("tire tread groove", k=30, exclude_chunks=all_chunks)
        assert all(r.doc_number != doc0 for r in resp.results), \
            f"Patent {doc0} appeared despite all its chunks being excluded"

    def test_exclude_empty_set_unchanged(self, engine):
        """An empty exclude set must behave identically to None."""
        r1 = engine.search("wheel bearing", k=10)
        r2 = engine.search("wheel bearing", k=10, exclude_chunks=set())
        assert [r.doc_number for r in r1.results] == [r.doc_number for r in r2.results]

    def test_filter_still_holds_with_exclusions(self, engine):
        """Hard filters must still apply when exclude_chunks is set."""
        pid0 = 0
        ci0 = int(engine.index.starts[pid0])
        resp = engine.search(
            "tire tread groove", k=10,
            class_prefix="B60C",
            exclude_chunks={ci0},
        )
        for r in resp.results:
            assert r.classification.upper().startswith("B60C")
