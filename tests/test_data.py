"""Tests for search/data.py — Milestone 1: loading and chunking."""

import json
import os
import tempfile

import numpy as np
import pytest

from search.data import PatentIndex, build_index, load_patents

# ---------------------------------------------------------------------------
# Minimal patent fixtures
# ---------------------------------------------------------------------------

# B60B sorts before B60C, so PATENT_B comes first after sorting.
PATENT_A = {
    "title": "Rubber Tire",
    "doc_number": "20240002222",
    "filename": "a.xml",
    "abstract": "A tire with improved tread for wet roads.",
    "detailed_description": [
        "",  # empty — must be dropped
        "Short.",  # under 40 chars — must be dropped
        (
            "This is a detailed description paragraph about the tire tread pattern "
            "that is definitely long enough to pass the forty-character threshold and be indexed."
        ),
    ],
    "claims": [
        # claim 1 — preamble cut off (no number prefix)
        "a rubber tire comprising a tread portion with grooves.",
        # claim 2 — dot format, dependent
        "2 . The tire according to claim 1 , wherein the tread portion has a specific depth.",
        # claim 3 — paren format, dependent
        "3 ) The tire according to claim 2 , wherein the grooves reduce road noise.",
    ],
    "bibtex": "",
    "classification": "B60C104FI",
}

PATENT_B = {
    "title": "Steel Wheel",
    "doc_number": "20240001111",
    "filename": "b.xml",
    "abstract": "A wheel made of high-strength steel for passenger vehicles.",
    "detailed_description": [
        "A paragraph describing the steel wheel hub assembly in sufficient detail to be indexed here.",
    ],
    "claims": [
        "a steel wheel having a rim and a plurality of spokes.",
        "2 . The wheel according to claim 1 , wherein the rim has a bead seat.",
    ],
    "bibtex": "",
    "classification": "B60B104FI",
}

# Patent with a very long paragraph that must be split.
LONG_PARA = " ".join([f"word{i}" for i in range(400)])  # 400 words
PATENT_LONG = {
    "title": "Long Patent",
    "doc_number": "20240003333",
    "filename": "c.xml",
    "abstract": "Abstract of a patent with a very long description paragraph.",
    "detailed_description": [LONG_PARA],
    "claims": ["an apparatus comprising a first component and a second component."],
    "bibtex": "",
    "classification": "B60D104FI",
}


def make_tmpdir_with_patents(*patent_lists):
    """Write each list to its own JSON file in a temp dir. Returns (tmpdir, path)."""
    tmpdir = tempfile.mkdtemp()
    for i, patents in enumerate(patent_lists):
        fpath = os.path.join(tmpdir, f"patents_ipa24010{i}.json")
        with open(fpath, "w") as f:
            json.dump(patents, f)
    return tmpdir


# ---------------------------------------------------------------------------
# load_patents
# ---------------------------------------------------------------------------

class TestLoadPatents:
    def test_returns_sorted_by_classification_then_doc_number(self):
        tmpdir = make_tmpdir_with_patents([PATENT_A, PATENT_B])
        patents = load_patents(tmpdir)
        assert len(patents) == 2
        # B60B < B60C, so PATENT_B (B60B) must come first
        assert patents[0]["doc_number"] == "20240001111"
        assert patents[1]["doc_number"] == "20240002222"

    def test_deduplicates_by_doc_number_keeps_first(self):
        dup = {**PATENT_B, "abstract": "duplicate version"}
        tmpdir = make_tmpdir_with_patents([PATENT_B, dup])
        patents = load_patents(tmpdir)
        assert len(patents) == 1
        assert patents[0]["abstract"] == PATENT_B["abstract"]

    def test_skips_patents_without_doc_number(self):
        bad = {**PATENT_A, "doc_number": ""}
        tmpdir = make_tmpdir_with_patents([bad, PATENT_B])
        patents = load_patents(tmpdir)
        assert len(patents) == 1
        assert patents[0]["doc_number"] == PATENT_B["doc_number"]

    def test_attaches_week_from_filename(self):
        tmpdir = make_tmpdir_with_patents([PATENT_B])
        patents = load_patents(tmpdir)
        # Filename is patents_ipa240100.json → week "240100"
        assert patents[0]["week"] == "240100"


# ---------------------------------------------------------------------------
# build_index — structural invariants
# ---------------------------------------------------------------------------

class TestBuildIndexInvariants:
    @pytest.fixture(autouse=True)
    def idx(self):
        tmpdir = make_tmpdir_with_patents([PATENT_A, PATENT_B])
        patents = load_patents(tmpdir)
        self.idx = build_index(patents)

    def test_parallel_arrays_same_length(self):
        idx = self.idx
        n = len(idx.texts)
        assert len(idx.chunk_pid) == n
        assert len(idx.chunk_type) == n
        assert len(idx.chunk_ref) == n
        assert len(idx.chunk_independent) == n

    def test_no_empty_or_whitespace_chunks(self):
        for text in self.idx.texts:
            assert text.strip(), f"Empty/whitespace chunk: {text!r}"

    def test_starts_ends_contiguous(self):
        idx = self.idx
        assert len(idx.starts) == len(idx.patents)
        assert len(idx.ends) == len(idx.patents)
        for i in range(len(idx.patents)):
            assert idx.starts[i] < idx.ends[i], f"Patent {i} has no chunks"
        for i in range(1, len(idx.patents)):
            assert idx.starts[i] == idx.ends[i - 1], "Chunks not contiguous"

    def test_starts_end_covers_all_chunks(self):
        idx = self.idx
        assert idx.starts[0] == 0
        assert idx.ends[-1] == len(idx.texts)

    def test_chunk_pid_matches_starts_ends(self):
        idx = self.idx
        for pid in range(len(idx.patents)):
            for ci in range(idx.starts[pid], idx.ends[pid]):
                assert idx.chunk_pid[ci] == pid

    def test_codes_sorted(self):
        codes = self.idx.codes
        assert codes == sorted(codes)

    def test_arrays_are_correct_dtype(self):
        idx = self.idx
        assert idx.starts.dtype == np.int64
        assert idx.ends.dtype == np.int64
        assert idx.chunk_pid.dtype == np.int64
        assert idx.chunk_ref.dtype == np.int64
        assert idx.chunk_independent.dtype == bool

    def test_doc_numbers_unique(self):
        doc_numbers = [p["doc_number"] for p in self.idx.patents]
        assert len(doc_numbers) == len(set(doc_numbers))


# ---------------------------------------------------------------------------
# Chunk content: abstract
# ---------------------------------------------------------------------------

class TestAbstractChunk:
    @pytest.fixture(autouse=True)
    def setup(self):
        tmpdir = make_tmpdir_with_patents([PATENT_B])
        patents = load_patents(tmpdir)
        self.idx = build_index(patents)

    def test_exactly_one_abstract_chunk(self):
        abstract_chunks = [t for t, typ in zip(self.idx.texts, self.idx.chunk_type) if typ == "abstract"]
        assert len(abstract_chunks) == 1

    def test_abstract_chunk_text(self):
        i = next(i for i, t in enumerate(self.idx.chunk_type) if t == "abstract")
        expected = f"{PATENT_B['title']}. {PATENT_B['abstract']}"
        assert self.idx.texts[i] == expected

    def test_abstract_chunk_is_not_independent_claim(self):
        i = next(i for i, t in enumerate(self.idx.chunk_type) if t == "abstract")
        # chunk_independent is False for non-claims
        assert not self.idx.chunk_independent[i]


# ---------------------------------------------------------------------------
# Chunk content: claims
# ---------------------------------------------------------------------------

class TestClaimChunks:
    @pytest.fixture(autouse=True)
    def setup(self):
        tmpdir = make_tmpdir_with_patents([PATENT_A])
        patents = load_patents(tmpdir)
        self.idx = build_index(patents)
        self.claim_indices = [i for i, t in enumerate(self.idx.chunk_type) if t == "claim"]

    def test_three_claim_chunks(self):
        assert len(self.claim_indices) == 3

    def test_claim1_has_no_number_prefix(self):
        # claim 1 text starts with title prefix, no "1 ." or "1 )" in the chunk
        i = self.claim_indices[0]
        text = self.idx.texts[i]
        assert text.startswith(f"{PATENT_A['title']}: ")
        # the actual claim text should start with "a rubber tire"
        assert "a rubber tire" in text

    def test_dot_format_prefix_stripped(self):
        # claim 2 had "2 . The tire..."
        i = self.claim_indices[1]
        text = self.idx.texts[i]
        assert text.startswith(f"{PATENT_A['title']}: ")
        assert "The tire" in text
        # "2 ." must not appear in the chunk text
        assert "2 ." not in text

    def test_paren_format_prefix_stripped(self):
        # claim 3 had "3 ) The tire..."
        i = self.claim_indices[2]
        text = self.idx.texts[i]
        assert text.startswith(f"{PATENT_A['title']}: ")
        assert "The tire" in text
        assert "3 )" not in text

    def test_claim1_is_independent(self):
        i = self.claim_indices[0]
        assert self.idx.chunk_independent[i]

    def test_claim2_is_dependent(self):
        i = self.claim_indices[1]
        assert not self.idx.chunk_independent[i]

    def test_claim3_is_dependent(self):
        i = self.claim_indices[2]
        assert not self.idx.chunk_independent[i]

    def test_claim_ref_stores_claim_number(self):
        # claim_ref should be 1-based claim number
        for k, i in enumerate(self.claim_indices):
            assert self.idx.chunk_ref[i] == k + 1


# ---------------------------------------------------------------------------
# Chunk content: paragraphs
# ---------------------------------------------------------------------------

class TestParagraphChunks:
    @pytest.fixture(autouse=True)
    def setup(self):
        tmpdir = make_tmpdir_with_patents([PATENT_A])
        patents = load_patents(tmpdir)
        self.idx = build_index(patents)
        self.para_indices = [i for i, t in enumerate(self.idx.chunk_type) if t == "paragraph"]

    def test_empty_paragraphs_dropped(self):
        # PATENT_A has 1 empty, 1 short, 1 valid → only 1 paragraph chunk
        assert len(self.para_indices) == 1

    def test_short_paragraphs_dropped(self):
        # "Short." (7 chars) must not appear
        for i in self.para_indices:
            assert "Short." not in self.idx.texts[i]

    def test_valid_paragraph_text(self):
        i = self.para_indices[0]
        assert "tire tread" in self.idx.texts[i]

    def test_paragraph_ref_is_original_index(self):
        # The valid paragraph was at index 2 in detailed_description
        i = self.para_indices[0]
        assert self.idx.chunk_ref[i] == 2


# ---------------------------------------------------------------------------
# Long paragraph splitting
# ---------------------------------------------------------------------------

class TestLongParagraphSplit:
    @pytest.fixture(autouse=True)
    def setup(self):
        tmpdir = make_tmpdir_with_patents([PATENT_LONG])
        patents = load_patents(tmpdir)
        self.idx = build_index(patents)
        self.para_indices = [i for i, t in enumerate(self.idx.chunk_type) if t == "paragraph"]

    def test_long_paragraph_is_split(self):
        # 400 words > 350 threshold → must produce 2+ chunks
        assert len(self.para_indices) >= 2

    def test_split_chunks_are_not_empty(self):
        for i in self.para_indices:
            assert self.idx.texts[i].strip()

    def test_split_chunks_together_cover_all_words(self):
        combined = " ".join(self.idx.texts[i] for i in self.para_indices)
        # All 400 words must appear somewhere
        for word in LONG_PARA.split():
            assert word in combined

    def test_split_chunk_ref_is_original_paragraph_index(self):
        # Both chunks from the split paragraph should reference index 0
        for i in self.para_indices:
            assert self.idx.chunk_ref[i] == 0


# ---------------------------------------------------------------------------
# Real dataset smoke test
# ---------------------------------------------------------------------------

class TestRealDataset:
    @pytest.fixture(autouse=True, scope="class")
    def idx(self, tmp_path_factory):
        patents = load_patents("data/data/patent_data_small")
        self.__class__._idx = build_index(patents)

    def test_patent_count(self):
        assert len(self._idx.patents) == 640

    def test_total_chunks_positive(self):
        assert len(self._idx.texts) > 640  # at least one chunk per patent

    def test_no_empty_chunks(self):
        for text in self._idx.texts:
            assert text.strip()

    def test_patents_sorted_by_classification_doc_number(self):
        patents = self._idx.patents
        keys = [(p["classification"], p["doc_number"]) for p in patents]
        assert keys == sorted(keys)

    def test_codes_sorted(self):
        codes = self._idx.codes
        assert codes == sorted(codes)

    def test_doc_numbers_unique(self):
        doc_numbers = [p["doc_number"] for p in self._idx.patents]
        assert len(doc_numbers) == len(set(doc_numbers))

    def test_all_chunk_texts_non_empty(self):
        assert all(t.strip() for t in self._idx.texts)

    def test_parallel_arrays_same_length(self):
        idx = self._idx
        n = len(idx.texts)
        assert len(idx.chunk_pid) == n
        assert len(idx.chunk_type) == n
        assert len(idx.chunk_ref) == n
        assert len(idx.chunk_independent) == n

    def test_contiguous_chunks(self):
        idx = self._idx
        assert idx.starts[0] == 0
        assert idx.ends[-1] == len(idx.texts)
        for i in range(1, len(idx.patents)):
            assert idx.starts[i] == idx.ends[i - 1]
