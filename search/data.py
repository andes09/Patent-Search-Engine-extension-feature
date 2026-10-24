"""search/data.py — Milestone 1: patent loading and chunking."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches claim number prefixes in both formats: "2 . " and "10 ) "
_CLAIM_PREFIX_RE = re.compile(r'^\s*\d+\s*[.)]\s*')

# A claim is dependent if its text references another claim number.
_DEPENDENT_RE = re.compile(r'\bclaims?\s+\d+', re.IGNORECASE)

_MAX_PARA_WORDS = 350   # paragraphs longer than this are split
_MIN_PARA_CHARS = 40    # paragraphs shorter than this are dropped


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class PatentIndex:
    """All in-memory structures produced by build_index."""
    patents: list[dict[str, Any]]   # sorted by (classification, doc_number)
    codes: list[str]                # classification code per patent (parallel to patents)
    texts: list[str]                # one entry per chunk
    chunk_pid: np.ndarray           # int64 — which patent each chunk belongs to
    chunk_type: list[str]           # "abstract" | "claim" | "paragraph"
    chunk_ref: np.ndarray           # int64 — 1-based claim number, or original para index
    chunk_independent: np.ndarray   # bool — True only for independent claims
    starts: np.ndarray              # int64 — first chunk index for patent i
    ends: np.ndarray                # int64 — one past last chunk index for patent i


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_patents(data_dir: str) -> list[dict[str, Any]]:
    """Load all patents_ipa*.json files from *data_dir*.

    - Attaches ``week`` (YYMMDD string) parsed from the filename.
    - Skips patents with no ``doc_number`` or with no text in any field.
    - Deduplicates by ``doc_number``, keeping the first occurrence.
    - Sorts the result by ``(classification, doc_number)``.
    - Missing ``classification`` is stored as ``""``.
    """
    seen: set[str] = set()
    raw: list[dict[str, Any]] = []
    skipped = 0

    fnames = sorted(
        f for f in os.listdir(data_dir)
        if re.match(r'patents_ipa\d+\.json$', f)
    )

    for fname in fnames:
        week_m = re.search(r'patents_ipa(\d+)\.json$', fname)
        week = week_m.group(1) if week_m else ""

        with open(os.path.join(data_dir, fname), encoding="utf-8") as fh:
            entries: list[dict] = json.load(fh)

        for p in entries:
            doc = (p.get("doc_number") or "").strip()
            if not doc:
                skipped += 1
                continue

            # Skip only if every text field is empty.
            has_text = (
                (p.get("title") or "").strip()
                or (p.get("abstract") or "").strip()
                or any((s or "").strip() for s in (p.get("detailed_description") or []))
                or any((s or "").strip() for s in (p.get("claims") or []))
            )
            if not has_text:
                skipped += 1
                continue

            if doc in seen:
                continue  # keep first occurrence only
            seen.add(doc)

            p = dict(p)  # shallow copy — don't mutate caller's data
            p["week"] = week
            if not (p.get("classification") or "").strip():
                p["classification"] = ""
            raw.append(p)

    if skipped:
        print(f"[load_patents] skipped {skipped} patent(s) (no doc_number or no text)")

    raw.sort(key=lambda p: (p["classification"], p["doc_number"]))
    return raw


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _split_paragraph(text: str) -> list[str]:
    """Split *text* into roughly equal pieces, each at most _MAX_PARA_WORDS words."""
    words = text.split()
    n = len(words)
    n_pieces = math.ceil(n / _MAX_PARA_WORDS)
    piece_size = math.ceil(n / n_pieces)
    pieces = [
        " ".join(words[i * piece_size: (i + 1) * piece_size])
        for i in range(n_pieces)
    ]
    # Guard: discard any empty tail piece that could appear from rounding.
    return [p for p in pieces if p]


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(patents: list[dict[str, Any]]) -> PatentIndex:
    """Build the contiguous chunk index from a sorted patent list.

    Chunk order within each patent: abstract → claims → paragraphs.
    Every patent produces at least one chunk (the abstract), so
    ``np.maximum.reduceat`` over ``starts``/``ends`` is always safe.
    """
    texts: list[str] = []
    pids: list[int] = []
    types: list[str] = []
    refs: list[int] = []
    indep: list[bool] = []
    starts: list[int] = []
    ends: list[int] = []

    for pid, p in enumerate(patents):
        start = len(texts)
        title = (p.get("title") or "").strip()

        # -- Abstract chunk -------------------------------------------------
        abstract = (p.get("abstract") or "").strip()
        texts.append(f"{title}. {abstract}")
        pids.append(pid)
        types.append("abstract")
        refs.append(0)
        indep.append(False)  # non-claim chunks are never marked independent

        # -- Claim chunks ---------------------------------------------------
        for claim_num, raw_claim in enumerate(p.get("claims") or [], start=1):
            # Strip numeric prefix present on claims 2+; claim 1 often lacks it.
            body = _CLAIM_PREFIX_RE.sub("", raw_claim or "")
            # Claim 1 is unconditionally independent; others depend on text.
            is_indep = (claim_num == 1) or not bool(_DEPENDENT_RE.search(body))
            texts.append(f"{title}: {body}")
            pids.append(pid)
            types.append("claim")
            refs.append(claim_num)
            indep.append(is_indep)

        # -- Paragraph chunks -----------------------------------------------
        for para_idx, para in enumerate(p.get("detailed_description") or []):
            if not para or not para.strip():
                continue  # drop empty strings
            if len(para) < _MIN_PARA_CHARS:
                continue  # drop very short paragraphs

            # Split long paragraphs so the embedding model sees complete text.
            pieces = (
                _split_paragraph(para)
                if len(para.split()) > _MAX_PARA_WORDS
                else [para]
            )
            for piece in pieces:
                texts.append(piece)
                pids.append(pid)
                types.append("paragraph")
                refs.append(para_idx)   # original index, same for all split pieces
                indep.append(False)

        starts.append(start)
        ends.append(len(texts))

    return PatentIndex(
        patents=patents,
        codes=[p.get("classification") or "" for p in patents],
        texts=texts,
        chunk_pid=np.array(pids, dtype=np.int64),
        chunk_type=types,
        chunk_ref=np.array(refs, dtype=np.int64),
        chunk_independent=np.array(indep, dtype=bool),
        starts=np.array(starts, dtype=np.int64),
        ends=np.array(ends, dtype=np.int64),
    )
