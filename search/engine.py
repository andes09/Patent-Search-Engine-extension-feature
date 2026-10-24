"""search/engine.py — Milestone 4: hard filters, fusion, and the public API."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from search.data import PatentIndex, build_index, load_patents
from search.dense import (
    DenseIndex,
    best_chunk_for_patent,
    build_dense_index,
    dense_search,
    embed_query,
)
from search.keyword import (
    KeywordIndex,
    KeywordResult,
    best_keyword_chunk_for_patent,
    build_keyword_index,
    keyword_search,
    tokenize,
)

# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    type: str           # "abstract" | "claim" | "paragraph"
    ref: int            # 1-based claim number, or original paragraph index
    independent: bool   # True for independent claims; False otherwise
    text: str           # the chunk text


@dataclass
class SearchResult:
    doc_number: str
    title: str
    classification: str
    week: str
    score: float
    dense_rank: int | None        # 1-based rank in dense top-100 (None if absent)
    keyword_rank: int | None      # 1-based rank in keyword top-100 (None if absent)
    found_by: str                 # "semantic" | "keyword" | "both"
    matched_terms: list[str]
    matched_phrases: list[str]
    passage: Passage


@dataclass
class SearchResponse:
    results: list[SearchResult]
    suggestions: dict[str, list[tuple[str, int]]]
    timings: dict[str, float]


@dataclass
class Stage1Candidates:
    """Rich result from stage-1 retrieval, carrying per-candidate chunk data for stage 2.

    ``response`` is the normal SearchResponse (returned directly by search()).
    ``d_rank`` / ``k_rank`` are the 1-based dense / keyword ranks for all candidates
    before the top-k cut (used by the blend RRF formula in stage 2).
    ``dense_chunk`` / ``keyword_chunk`` map each result patent id to its best
    chunk id under the respective signal; None entries if the signal wasn't active.
    ``q_vec`` is the normalised query embedding, reused by stage 2 for passage lookup.
    """
    response: SearchResponse
    d_rank: dict[int, int]          # pid → 1-based rank in dense top-n_candidates
    k_rank: dict[int, int]          # pid → 1-based rank in keyword top-n_candidates
    dense_chunk: dict[int, int]     # pid → best dense chunk id (exclude_chunks applied)
    keyword_chunk: dict[int, int]   # pid → best keyword chunk id (exclude_chunks applied)
    q_vec: np.ndarray               # float32 query embedding


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SearchEngine:
    """Hybrid patent search engine over a directory of patent JSON files."""

    def __init__(self, data_dir: str) -> None:
        patents = load_patents(data_dir)
        self.index: PatentIndex = build_index(patents)
        self.dense: DenseIndex = build_dense_index(self.index)
        self.kidx: KeywordIndex = build_keyword_index(self.index)
        self._build_filter_indexes()

    # ------------------------------------------------------------------
    # Filter index construction (called once at load time)
    # ------------------------------------------------------------------

    def _build_filter_indexes(self) -> None:
        self._title_index:    dict[str, set[int]] = {}
        self._abstract_index: dict[str, set[int]] = {}
        self._title_exact_index: dict[str, int]   = {}
        self._doc_number_index:  dict[str, int]   = {}

        for pid, p in enumerate(self.index.patents):
            self._doc_number_index[p["doc_number"]] = pid

            title    = (p.get("title")    or "").strip()
            abstract = (p.get("abstract") or "").strip()

            for tok in tokenize(title):
                self._title_index.setdefault(tok, set()).add(pid)

            for tok in tokenize(abstract):
                self._abstract_index.setdefault(tok, set()).add(pid)

            # Normalise: lowercase + collapse whitespace
            self._title_exact_index[" ".join(title.lower().split())] = pid

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _class_range(self, prefix: str) -> tuple[int, int]:
        """Return (lo_pid, hi_pid) patent indices for *prefix* via bisect."""
        codes = self.index.codes
        return bisect_left(codes, prefix), bisect_left(codes, prefix + "\uffff")

    def _filter_words(
        self, words: list[str], tok_index: dict[str, set[int]]
    ) -> set[int]:
        """Patent ids where every word in *words* is present (intersect posting sets).

        Intersects from the smallest set first for efficiency.  A stopword that
        tokenises to nothing is silently ignored; any unknown non-stop token
        immediately returns an empty set (no patent can match).
        """
        sets: list[set[int] | frozenset[int]] = []
        for w in words:
            for tok in tokenize(w):
                s = tok_index.get(tok)
                if s is None:
                    return set()           # unknown token → zero results
                sets.append(s)

        if not sets:
            # All words were stopwords — filter has no effect
            return set(range(len(self.index.patents)))

        sets.sort(key=len)                 # start from smallest posting list
        result: set[int] = set(sets[0])
        for s in sets[1:]:
            result &= s
            if not result:
                return set()
        return result

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str | None = None,
        patent_id: str | None = None,
        mode: str = "hybrid",
        class_prefix: str | None = None,
        title_has: list[str] | None = None,
        abstract_has: list[str] | None = None,
        title_exact: str | None = None,
        keywords: list[str] | None = None,
        k: int = 10,
        plan: str = "auto",
        exclude_chunks: set[int] | None = None,
    ) -> SearchResponse:
        """Search the patent index.

        Parameters
        ----------
        query          : natural-language or claim query text
        patent_id      : doc_number of a seed patent (query = title+abstract+claim 1)
        mode           : "hybrid" | "dense" | "keyword"
        class_prefix   : hard filter — classification must start with this prefix
        title_has      : hard filter — every word must appear in the title
        abstract_has   : hard filter — every word must appear in the abstract
        title_exact    : hard filter — exact (normalised) title match
        keywords       : soft keywords; quoted entries (``"bead seat"``) are phrases
        k              : number of results to return
        plan           : "auto" (class via lo/hi, word filter via mask) or
                         "post" (score full corpus, then filter — benchmark baseline)
        exclude_chunks : chunk ids invisible to both dense and keyword scoring;
                         patents left with zero chunks are dropped from results
        """
        return self._search_candidates(
            query=query, patent_id=patent_id, mode=mode,
            class_prefix=class_prefix, title_has=title_has,
            abstract_has=abstract_has, title_exact=title_exact,
            keywords=keywords, k=k, plan=plan,
            exclude_chunks=exclude_chunks,
        ).response

    def _search_candidates(
        self,
        query: str | None = None,
        patent_id: str | None = None,
        mode: str = "hybrid",
        class_prefix: str | None = None,
        title_has: list[str] | None = None,
        abstract_has: list[str] | None = None,
        title_exact: str | None = None,
        keywords: list[str] | None = None,
        k: int = 10,
        plan: str = "auto",
        exclude_chunks: set[int] | None = None,
    ) -> "Stage1Candidates":
        """Full stage-1 retrieval; returns the SearchResponse plus per-candidate
        best chunk ids and rank dicts needed by two_phase_search in stage 2."""
        t0 = perf_counter()
        n_patents = len(self.index.patents)
        n_chunks  = len(self.index.texts)

        # ---- 1. Resolve query ------------------------------------------------
        exclude_pid: int | None = None

        if patent_id is not None:
            pid = self._doc_number_index.get(patent_id)
            if pid is None:
                raise ValueError(f"Patent {patent_id!r} not found in index")
            p = self.index.patents[pid]
            title    = (p.get("title")    or "").strip()
            abstract = (p.get("abstract") or "").strip()
            query    = f"{title}. {abstract}"
            # Append first independent claim for context
            for ci in range(int(self.index.starts[pid]), int(self.index.ends[pid])):
                if (self.index.chunk_type[ci] == "claim"
                        and self.index.chunk_independent[ci]):
                    query += " " + self.index.texts[ci]
                    break
            exclude_pid = pid

        if not query:
            raise ValueError("Provide query text or a valid patent_id")

        # ---- 2. Hard filters -------------------------------------------------
        t_filter = perf_counter()

        # Class prefix → contiguous chunk slice
        lo_pid, hi_pid = 0, n_patents
        lo, hi         = 0, n_chunks

        if class_prefix:
            lo_pid, hi_pid = self._class_range(class_prefix.upper())
            if lo_pid >= hi_pid:
                dt = perf_counter() - t_filter
                empty_resp = SearchResponse(
                    results=[],
                    suggestions={},
                    timings={"filter": dt, "total": perf_counter() - t0, "plan": plan},
                )
                return Stage1Candidates(
                    response=empty_resp,
                    d_rank={}, k_rank={},
                    dense_chunk={}, keyword_chunk={},
                    q_vec=np.zeros(1, dtype=np.float32),
                )
            lo = int(self.index.starts[lo_pid])
            hi = int(self.index.ends[hi_pid - 1])

        # Word filters → set of allowed patent ids
        word_set: set[int] | None = None

        if title_has:
            s = self._filter_words(title_has, self._title_index)
            word_set = s if word_set is None else word_set & s

        if abstract_has:
            s = self._filter_words(abstract_has, self._abstract_index)
            word_set = s if word_set is None else word_set & s

        if title_exact is not None:
            norm  = " ".join(title_exact.lower().split())
            p_ex  = self._title_exact_index.get(norm)
            s     = {p_ex} if p_ex is not None else set()
            word_set = s if word_set is None else word_set & s

        # Intersect word filter with class range
        if word_set is not None and class_prefix:
            word_set = {p for p in word_set if lo_pid <= p < hi_pid}

        # Boolean mask for word filters
        word_pids: np.ndarray | None = None
        if word_set is not None:
            word_pids = np.zeros(n_patents, dtype=bool)
            for p in word_set:
                word_pids[p] = True

        # Query planner: decide gather vs mask for diagnostic / future optimisation.
        # Both code paths give identical results; gather is faster for very
        # selective filters because it only scores the matching patent chunks.
        planner = "mask"
        if plan == "auto" and word_set is not None and (hi - lo) > 0:
            n_allowed_chunks = sum(
                int(self.index.ends[p]) - int(self.index.starts[p])
                for p in word_set
                if lo_pid <= p < hi_pid
            )
            if n_allowed_chunks < 0.25 * (hi - lo):
                planner = "gather"

        # Determine final search parameters
        if plan == "post":
            # Score full corpus; merge all filters into a single mask.
            s_lo, s_hi = 0, n_chunks
            if class_prefix or word_set is not None:
                post_mask        = np.zeros(n_patents, dtype=bool)
                post_mask[lo_pid:hi_pid] = True
                if word_pids is not None:
                    post_mask &= word_pids
                s_allowed: np.ndarray | None = post_mask
            else:
                s_allowed = None
            actual_plan = "post"
        else:
            # plan="auto": class range via lo/hi; word filter via allowed_pids mask.
            # (The "gather" path would use fancy chunk indexing for a speed boost;
            # the mask path gives identical results with slightly more work.)
            s_lo, s_hi = lo, hi
            s_allowed  = word_pids
            actual_plan = f"auto_{planner}"

        dt_filter = perf_counter() - t_filter

        # ---- 3. Embed query --------------------------------------------------
        t_embed = perf_counter()
        q_vec   = embed_query(self.dense, query)
        dt_embed = perf_counter() - t_embed

        # ---- 4. Dense retrieval ---------------------------------------------
        t_dense = perf_counter()
        dense_pids   = np.array([], dtype=np.int64)
        dense_scores = np.array([], dtype=np.float32)

        if mode in ("hybrid", "dense"):
            dense_pids, dense_scores = dense_search(
                self.dense, query, k=100,
                lo=s_lo, hi=s_hi, allowed_pids=s_allowed,
                exclude_chunks=exclude_chunks,
            )
        dt_dense = perf_counter() - t_dense

        # ---- 5. Keyword retrieval -------------------------------------------
        t_kw     = perf_counter()
        kw_result: KeywordResult | None = None

        if mode in ("hybrid", "keyword"):
            kw_result = keyword_search(
                self.kidx, query, k=100,
                soft_keywords=keywords,
                lo=s_lo, hi=s_hi, allowed_pids=s_allowed,
                exclude_chunks=exclude_chunks,
            )
        dt_kw = perf_counter() - t_kw

        # ---- 6. Weighted RRF fusion -----------------------------------------
        # score = 1/(60 + dense_rank) + 0.5/(60 + keyword_rank)
        # A patent absent from one list contributes 0 from that list.
        # Ranks are 1-based; the spec explicitly requires this.
        t_fusion = perf_counter()

        kw_pids = kw_result.patent_ids if kw_result else np.array([], dtype=np.int64)

        d_rank: dict[int, int] = {int(p): r + 1 for r, p in enumerate(dense_pids)}
        k_rank: dict[int, int] = {int(p): r + 1 for r, p in enumerate(kw_pids)}

        all_cands: set[int] = set(d_rank) | set(k_rank)
        if exclude_pid is not None:
            all_cands.discard(exclude_pid)

        rrf: dict[int, float] = {}
        for pid in all_cands:
            sc  = 0.0
            dr  = d_rank.get(pid)
            kr  = k_rank.get(pid)
            if dr is not None:
                sc += 1.0 / (60 + dr)
            if kr is not None:
                sc += 0.5 / (60 + kr)
            rrf[pid] = sc

        top_pids = sorted(rrf, key=lambda p: -rrf[p])[:k]
        dt_fusion = perf_counter() - t_fusion

        # ---- 7. Build result objects ----------------------------------------
        mt  = kw_result.matched_terms   if kw_result else []
        mp  = kw_result.matched_phrases if kw_result else []
        sug = kw_result.suggestions     if kw_result else {}

        results: list[SearchResult] = []
        dense_chunk_map: dict[int, int] = {}
        keyword_chunk_map: dict[int, int] = {}

        for pid in top_pids:
            p = self.index.patents[pid]

            in_d = pid in d_rank
            in_k = pid in k_rank
            found_by = "both" if (in_d and in_k) else ("semantic" if in_d else "keyword")

            # Best passage by dense similarity; excludes hidden chunks.
            ci = best_chunk_for_patent(
                self.dense, pid, q_vec, exclude_chunks=exclude_chunks
            )
            dense_chunk_map[pid] = ci
            passage = Passage(
                type=self.index.chunk_type[ci],
                ref=int(self.index.chunk_ref[ci]),
                independent=bool(self.index.chunk_independent[ci]),
                text=self.index.texts[ci],
            )

            # Best keyword chunk for stage-2 pair deduplication.
            if kw_result is not None and kw_result.chunk_scores is not None:
                kci = best_keyword_chunk_for_patent(
                    self.kidx, pid, kw_result.chunk_scores,
                    exclude_chunks=exclude_chunks,
                )
                keyword_chunk_map[pid] = kci

            results.append(SearchResult(
                doc_number=p["doc_number"],
                title=(p.get("title")          or ""),
                classification=(p.get("classification") or ""),
                week=(p.get("week")             or ""),
                score=float(rrf[pid]),
                dense_rank=d_rank.get(pid),
                keyword_rank=k_rank.get(pid),
                found_by=found_by,
                matched_terms=mt,
                matched_phrases=mp,
                passage=passage,
            ))

        dt_total = perf_counter() - t0
        timings: dict[str, float] = {
            "filter":          dt_filter,
            "query_embedding": dt_embed,
            "dense":           dt_dense,
            "keyword":         dt_kw,
            "fusion":          dt_fusion,
            "total":           dt_total,
            "plan":            actual_plan,
        }
        response = SearchResponse(results=results, suggestions=sug, timings=timings)
        return Stage1Candidates(
            response=response,
            d_rank=d_rank,
            k_rank=k_rank,
            dense_chunk=dense_chunk_map,
            keyword_chunk=keyword_chunk_map,
            q_vec=q_vec,
        )
