"""search/keyword.py — Milestone 3: BM25 keyword search."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from search.data import PatentIndex

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_PATENT_STOP: frozenset[str] = frozenset({
    "claim", "claims", "wherein", "said", "comprising", "comprises", "comprise",
    "include", "includes", "including", "according", "thereof", "therein",
    "thereto", "configured", "plurality", "embodiment", "embodiments",
    "present", "disclosure", "invention", "example", "fig", "figs", "figure",
    "respectively", "having", "formed", "mean", "means", "provided", "such",
})

STOP_WORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS) | _PATENT_STOP

# ---------------------------------------------------------------------------
# BM25 parameters & constants
# ---------------------------------------------------------------------------

_K1 = 1.2
_B  = 0.75

_FIELD_BOOST: dict[str, float] = {
    "abstract":  1.5,
    "claim":     1.2,
    "paragraph": 1.0,
}

_MAX_QUERY_TERMS = 16    # for long queries (claims, full patents), keep top-IDF terms
_PHRASE_BONUS    = 2.0   # additive bonus added to a chunk score when a phrase matches
_PHRASE_CANDS    = 200   # max chunks to inspect for phrase presence per query

# ---------------------------------------------------------------------------
# Tokenizer — single canonical function used everywhere (indexing, queries,
# filters, phrase checks, suggestions).  Import it; never re-implement it.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'[a-z0-9]+')


def _fold(token: str) -> str:
    """Light plural fold: drop trailing 's' if token > 4 chars and not ending 'ss'."""
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Return the canonical token list for *text*.

    Steps: lowercase → ``[a-z0-9]+`` regex → drop len<2 → fold plural →
    drop stopwords.
    """
    result: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if len(raw) < 2:
            continue
        tok = _fold(raw)
        if tok in STOP_WORDS:
            continue
        result.append(tok)
    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KeywordIndex:
    """Sparse BM25 index for keyword retrieval."""
    W: sp.csc_matrix          # (n_chunks, vocab_size) BM25 TF-norm × field_boost
    idf: np.ndarray           # float32 (vocab_size,) — used at query time
    vocab: dict[str, int]     # token → column index
    vocab_list: list[str]     # column index → token (for suggestions)
    patent_df: np.ndarray     # int32 (vocab_size,) patents that contain each term
    index: PatentIndex


@dataclass
class KeywordResult:
    patent_ids: np.ndarray                         # int64, top-k patent indices
    scores: np.ndarray                             # float32, descending
    suggestions: dict[str, list[tuple[str, int]]]  # zero-match: term → [(similar, count)]
    matched_terms: list[str]
    matched_phrases: list[str]
    chunk_scores: np.ndarray | None = None         # per-chunk BM25 scores; used by stage 2


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_keyword_index(index: PatentIndex) -> KeywordIndex:
    """Build the BM25 sparse matrix (CSC) and vocabulary from *index*.

    The matrix W stores ``field_boost × tf_norm`` per (chunk, term) so that
    query scoring is a single sparse matrix–vector product:
        chunk_scores = W[:, term_indices] @ (idf[term_indices] * query_weights)
    """
    tokenized: list[list[str]] = [tokenize(t) for t in index.texts]

    # Build vocabulary (insertion order = column index)
    vocab: dict[str, int] = {}
    for toks in tokenized:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)
    N = len(index.texts)

    lengths  = np.array([len(toks) for toks in tokenized], dtype=np.float32)
    avgdl    = float(lengths.mean()) if N > 0 else 1.0

    # Accumulate COO entries for the sparse matrix
    rows:      list[int]   = []
    cols_data: list[int]   = []
    vals:      list[float] = []

    df          = np.zeros(V, dtype=np.int32)   # chunk-level DF (for IDF)
    pat_term_sets: list[set[int]] = [set() for _ in range(len(index.patents))]

    for i, (toks, ctype) in enumerate(zip(tokenized, index.chunk_type)):
        if not toks:
            continue
        boost    = _FIELD_BOOST[ctype]
        dl       = float(lengths[i])
        norm_denom = 1.0 - _B + _B * dl / avgdl
        tf = Counter(toks)
        seen: set[int] = set()
        for tok, cnt in tf.items():
            j = vocab[tok]
            tf_norm = (cnt * (_K1 + 1)) / (cnt + _K1 * norm_denom)
            rows.append(i)
            cols_data.append(j)
            vals.append(boost * tf_norm)
            if j not in seen:
                df[j] += 1
                seen.add(j)
        pid = int(index.chunk_pid[i])
        pat_term_sets[pid].update(seen)

    # Patent-level DF (for suggestion display)
    patent_df = np.zeros(V, dtype=np.int32)
    for term_set in pat_term_sets:
        for j in term_set:
            patent_df[j] += 1

    # IDF: log((N - df + 0.5) / (df + 0.5) + 1) — guaranteed non-negative
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

    W = sp.coo_matrix((vals, (rows, cols_data)), shape=(N, V), dtype=np.float32).tocsc()

    vocab_list = [""] * V
    for tok, j in vocab.items():
        vocab_list[j] = tok

    return KeywordIndex(
        W=W, idf=idf, vocab=vocab, vocab_list=vocab_list,
        patent_df=patent_df, index=index,
    )


# ---------------------------------------------------------------------------
# Zero-match suggestions
# ---------------------------------------------------------------------------

def _suggestions_for(term: str, kidx: KeywordIndex, n: int = 3) -> list[tuple[str, int]]:
    """Up to *n* vocab terms similar to *term*, with their patent counts."""
    vlist = kidx.vocab_list

    close = difflib.get_close_matches(term, vlist, n=n * 2, cutoff=0.7)

    # Shared-prefix matches (prefix = first max(3, len-1) chars)
    prefix = term[:max(3, len(term) - 1)]
    prefix_hits = [v for v in vlist if v.startswith(prefix) and v != term]

    # Merge, de-duplicate, preserve order
    seen: set[str] = set()
    cands: list[str] = []
    for c in close + prefix_hits:
        if c not in seen:
            seen.add(c)
            cands.append(c)

    result: list[tuple[str, int]] = []
    for cand in cands[:n]:
        j = kidx.vocab[cand]
        result.append((cand, int(kidx.patent_df[j])))
    return result


# ---------------------------------------------------------------------------
# Phrase matching helper
# ---------------------------------------------------------------------------

def _has_phrase(chunk_toks: list[str], phrase: list[str]) -> bool:
    """True if *phrase* appears as a consecutive sub-sequence in *chunk_toks*."""
    m = len(phrase)
    if m == 0 or m > len(chunk_toks):
        return False
    for i in range(len(chunk_toks) - m + 1):
        if chunk_toks[i: i + m] == phrase:
            return True
    return False


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

def keyword_search(
    kidx: KeywordIndex,
    query: str,
    k: int = 10,
    soft_keywords: list[str] | None = None,
    lo: int = 0,
    hi: int | None = None,
    allowed_pids: np.ndarray | None = None,
    exclude_chunks: set[int] | None = None,
) -> KeywordResult:
    """Return top-k patents by BM25 score.

    Parameters
    ----------
    kidx          : KeywordIndex from build_keyword_index
    query         : natural-language or claim query text
    k             : results to return
    soft_keywords : optional; entries wrapped in ``"..."`` are phrases
    lo, hi        : contiguous chunk slice for class-filter fast path
    allowed_pids  : bool mask (n_patents,) for non-contiguous filters
    """
    index = kidx.index
    if hi is None:
        hi = len(index.texts)

    # ---- Parse soft keywords -------------------------------------------
    phrases: list[list[str]] = []   # tokenised phrase sequences
    soft_terms: list[str]    = []   # all tokens from soft_keywords
    suggestions: dict[str, list[tuple[str, int]]] = {}

    for kw in (soft_keywords or []):
        kw = kw.strip()
        is_phrase = kw.startswith('"') and kw.endswith('"') and len(kw) > 2
        inner = kw[1:-1] if is_phrase else kw
        toks  = tokenize(inner)
        if is_phrase and toks:
            phrases.append(toks)
        soft_terms.extend(toks)

        # Zero-match detection for each token in this keyword
        for tok in toks:
            if tok not in kidx.vocab or kidx.patent_df[kidx.vocab[tok]] == 0:
                sugg = _suggestions_for(tok, kidx)
                if sugg:
                    suggestions[tok] = sugg

    # ---- Build term weight map -----------------------------------------
    # Query text → 1.0;  soft keywords (and phrase words) → 2.0
    query_toks  = tokenize(query)
    unique_query = list(dict.fromkeys(t for t in query_toks if t in kidx.vocab))

    # Trim long queries to the top-IDF terms (avoids noise from whole-claim queries)
    if len(unique_query) > _MAX_QUERY_TERMS:
        unique_query.sort(key=lambda t: -float(kidx.idf[kidx.vocab[t]]))
        unique_query = unique_query[:_MAX_QUERY_TERMS]

    term_weights: dict[str, float] = {t: 1.0 for t in unique_query}
    for t in soft_terms:
        if t in kidx.vocab:
            term_weights[t] = 2.0  # soft keyword always overrides query weight

    if not term_weights:
        return KeywordResult(
            patent_ids=np.array([], dtype=np.int64),
            scores=np.array([], dtype=np.float32),
            suggestions=suggestions,
            matched_terms=[],
            matched_phrases=[],
        )

    # ---- Score all chunks (CSC: only query-term columns are read) --------
    t_list    = list(term_weights)
    t_indices = np.array([kidx.vocab[t] for t in t_list], dtype=np.int64)
    t_weights = np.array(
        [kidx.idf[j] * term_weights[t] for t, j in zip(t_list, t_indices)],
        dtype=np.float32,
    )

    # Shape: (n_chunks,).  Sparse × dense → dense via scipy.
    all_scores: np.ndarray = np.asarray(
        kidx.W[:, t_indices].dot(t_weights)
    ).ravel()

    # ---- Phrase bonus: check only top candidates in [lo, hi) ------------
    if phrases:
        sub = all_scores[lo:hi]
        n_cands = min(_PHRASE_CANDS, len(sub))
        if n_cands > 0:
            top_rel = np.argpartition(sub, -n_cands)[-n_cands:]
            for ph in phrases:
                for rel_i in top_rel:
                    abs_i = lo + int(rel_i)
                    # tokenize re-applies plural folding to chunk text
                    if _has_phrase(tokenize(index.texts[abs_i]), ph):
                        sub[rel_i] += _PHRASE_BONUS
            all_scores[lo:hi] = sub

    # Mask excluded chunks so they cannot win or become a passage.
    if exclude_chunks:
        for ci in exclude_chunks:
            if 0 <= ci < len(all_scores):
                all_scores[ci] = -np.inf

    # ---- Per-patent max-pool over [lo, hi) using contiguous layout ------
    rel_starts = np.clip(index.starts - lo, 0, hi - lo)
    rel_ends   = np.clip(index.ends   - lo, 0, hi - lo)
    in_range   = rel_starts < rel_ends
    pids_in_range = np.where(in_range)[0]

    if len(pids_in_range) == 0:
        return KeywordResult(
            patent_ids=np.array([], dtype=np.int64),
            scores=np.array([], dtype=np.float32),
            suggestions=suggestions,
            matched_terms=list(term_weights),
            matched_phrases=[" ".join(ph) for ph in phrases],
            chunk_scores=all_scores,
        )

    chunk_scores_sub = all_scores[lo:hi]
    seg_starts       = rel_starts[pids_in_range]
    patent_scores    = np.maximum.reduceat(chunk_scores_sub, seg_starts)

    # Mask out disallowed patents (non-contiguous filter)
    if allowed_pids is not None:
        mask = allowed_pids[pids_in_range]
        patent_scores = np.where(mask, patent_scores, -np.inf)

    # ---- Return only patents with positive score (matched at least one term)
    pos_mask          = patent_scores > 0
    positive_pids     = pids_in_range[pos_mask]
    positive_scores   = patent_scores[pos_mask]

    if len(positive_scores) == 0:
        return KeywordResult(
            patent_ids=np.array([], dtype=np.int64),
            scores=np.array([], dtype=np.float32),
            suggestions=suggestions,
            matched_terms=list(term_weights),
            matched_phrases=[" ".join(ph) for ph in phrases],
            chunk_scores=all_scores,
        )

    k_actual  = min(k, len(positive_scores))
    top_rel   = np.argpartition(positive_scores, -k_actual)[-k_actual:]
    top_rel   = top_rel[np.argsort(positive_scores[top_rel])[::-1]]

    matched_terms   = [t for t in term_weights if t in kidx.vocab
                       and kidx.patent_df[kidx.vocab[t]] > 0]
    matched_phrases = [" ".join(ph) for ph in phrases]

    return KeywordResult(
        patent_ids=positive_pids[top_rel].astype(np.int64),
        scores=positive_scores[top_rel].astype(np.float32),
        suggestions=suggestions,
        matched_terms=matched_terms,
        matched_phrases=matched_phrases,
        chunk_scores=all_scores,
    )


def best_keyword_chunk_for_patent(
    kidx: KeywordIndex,
    pid: int,
    chunk_scores: np.ndarray,
    *,
    exclude_chunks: set[int] | None = None,
) -> int:
    """Return the chunk index with highest BM25 score for patent *pid*.

    *chunk_scores* is the flat per-chunk score array from KeywordResult.chunk_scores.
    Excluded chunks are masked to -inf before argmax, so they cannot win.
    """
    index = kidx.index
    lo, hi = int(index.starts[pid]), int(index.ends[pid])
    scores = chunk_scores[lo:hi].copy()
    if exclude_chunks:
        for ci in exclude_chunks:
            rel = ci - lo
            if 0 <= rel < hi - lo:
                scores[rel] = -np.inf
    return lo + int(np.argmax(scores))
