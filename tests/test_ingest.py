"""
test_ingest.py
==============
Smoke tests for the ingestion pipeline.

These tests verify that:
1. The knowledge-base files can be loaded and produce at least one Document.
2. The chunking step produces at least as many chunks as source documents,
   and that every chunk is non-empty.
3. The chunk size constraint is respected (no chunk exceeds CHUNK_SIZE * 1.2
   characters, allowing for a small overrun due to the keep_separator option).

These tests do NOT call the watsonx.ai API — they test the pure Python logic
so they run instantly without credentials.

Run with:
    python -m pytest tests/ -v
"""

import pathlib
import sys

import pytest

# Ensure src/ is importable without installing as a package.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import the two pure-Python functions from ingest.py directly.
# We cannot import build_vector_store (requires watsonx credentials).
from src.ingest import load_documents, chunk_documents, KB_DIR, CHUNK_SIZE


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_documents() -> list[Document]:
    """Load raw documents from the sample knowledge base."""
    return load_documents(KB_DIR)


@pytest.fixture(scope="module")
def chunks(raw_documents) -> list[Document]:
    """Chunk the raw documents."""
    return chunk_documents(raw_documents)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadDocuments:
    def test_loads_at_least_one_document(self, raw_documents):
        """The KB directory must contain at least one .txt file."""
        assert len(raw_documents) >= 1, (
            f"No documents loaded from {KB_DIR}. "
            "Make sure data/sample_kb/ contains .txt files."
        )

    def test_all_documents_have_content(self, raw_documents):
        """Every loaded document must have non-empty text."""
        for doc in raw_documents:
            assert doc.page_content.strip(), (
                f"Document from '{doc.metadata.get('source')}' has empty content."
            )

    def test_documents_have_source_metadata(self, raw_documents):
        """Each document should carry a 'source' metadata key."""
        for doc in raw_documents:
            assert "source" in doc.metadata, (
                f"Document missing 'source' metadata: {doc.metadata}"
            )

    def test_documents_have_category_metadata(self, raw_documents):
        """Each document should carry a 'category' metadata key."""
        for doc in raw_documents:
            assert "category" in doc.metadata


class TestChunkDocuments:
    def test_produces_more_chunks_than_documents(self, raw_documents, chunks):
        """
        Chunking must increase (or equal) the number of units — each source
        document is long enough to be split into multiple chunks.
        """
        assert len(chunks) >= len(raw_documents), (
            f"Expected at least {len(raw_documents)} chunks, got {len(chunks)}."
        )

    def test_chunk_count_is_positive(self, chunks):
        """There must be at least one chunk."""
        assert len(chunks) > 0, "chunk_documents() returned an empty list."

    def test_all_chunks_are_non_empty(self, chunks):
        """Every chunk must contain at least some non-whitespace text."""
        empty = [c for c in chunks if not c.page_content.strip()]
        assert not empty, f"Found {len(empty)} empty chunk(s)."

    def test_chunk_size_within_bounds(self, chunks):
        """
        No chunk should exceed CHUNK_SIZE by more than 20% (the slight overrun
        is caused by the keep_separator=True option, which retains the
        separator character in the preceding chunk).
        """
        max_allowed = int(CHUNK_SIZE * 1.2)
        oversized = [
            c for c in chunks
            if len(c.page_content) > max_allowed
        ]
        assert not oversized, (
            f"{len(oversized)} chunk(s) exceed {max_allowed} characters. "
            f"Longest: {max(len(c.page_content) for c in chunks)} chars."
        )

    def test_chunks_inherit_source_metadata(self, chunks):
        """Chunks should inherit the 'source' metadata from their parent document."""
        for chunk in chunks:
            assert "source" in chunk.metadata, (
                f"Chunk missing 'source' metadata: {chunk.metadata}"
            )


class TestJsonRepair:
    """
    Unit tests for the _extract_json() helper in agent.py — no API calls needed.
    """

    def setup_method(self):
        from src.agent import _extract_json
        self._extract_json = _extract_json

    def test_clean_json_passthrough(self):
        raw = '{"technical_questions": [], "behavioral_questions": [], "confidence_checklist": []}'
        result = self._extract_json(raw)
        import json
        parsed = json.loads(result)
        assert "technical_questions" in parsed

    def test_strips_markdown_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        result = self._extract_json(raw)
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_strips_preamble(self):
        raw = 'Here is your JSON output:\n\n{"key": "value"}'
        result = self._extract_json(raw)
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="Could not locate"):
            self._extract_json("This response contains no JSON object at all.")
