"""
retriever.py
============
Loads the persisted Chroma vector store and exposes a single function,
retrieve_context(), that accepts a natural-language query and returns the
top-k most relevant text chunks from the knowledge base.

This module is imported by agent.py at query time — it does NOT rebuild
or re-embed anything; it just loads the existing ./db/ and runs a similarity
search against it.
"""

import os
import pathlib
from typing import Optional

from langchain_chroma import Chroma

from src.watsonx_client import get_embedding_model

# Path to the persisted Chroma database (built by ingest.py)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_DIR = str(PROJECT_ROOT / "db")

# Module-level cache — the Chroma client is opened once and reused across
# multiple retrieve_context() calls within the same process.
_vector_store: Optional[Chroma] = None


def _load_vector_store() -> Chroma:
    """
    Load (or return the cached) Chroma vector store from disk.

    Raises a clear RuntimeError if the DB directory does not exist, reminding
    the user to run `python src/ingest.py` first.
    """
    global _vector_store

    if _vector_store is not None:
        # Already loaded — return the cached instance.
        return _vector_store

    if not os.path.exists(DB_DIR):
        raise RuntimeError(
            f"Vector store not found at {DB_DIR}.\n"
            "Please run:  python src/ingest.py\n"
            "to build the knowledge-base index before starting the app."
        )

    # Initialise the same embedding model used during ingestion.
    # Chroma needs the embedding function to convert the query string into a
    # vector before running the ANN (approximate nearest-neighbour) search.
    embedding_model = get_embedding_model()

    # Chroma(persist_directory, embedding_function) opens the existing SQLite
    # database without re-embedding any documents.
    _vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_name="interview_kb",
    )

    return _vector_store


def retrieve_context(query: str, k: int = 5) -> list[str]:
    """
    Run a similarity search against the knowledge-base vector store.

    Parameters
    ----------
    query : Natural-language retrieval query (e.g., role + skills + level).
    k     : Number of top-matching chunks to return.

    Returns
    -------
    List of raw text strings (page_content of the top-k Documents), ordered
    by descending similarity to the query.  These strings are injected into
    the Granite prompt as contextual grounding.
    """
    vector_store = _load_vector_store()

    # similarity_search() converts the query to a vector via the embedding
    # model, then retrieves the k nearest neighbours from the Chroma index
    # using cosine similarity (Chroma's default).
    docs = vector_store.similarity_search(query, k=k)

    # Return just the text content — metadata (source filename, category)
    # is available on doc.metadata but not needed in the prompt.
    return [doc.page_content for doc in docs]


def retrieve_context_with_scores(query: str, k: int = 5) -> list[tuple[str, float]]:
    """
    Same as retrieve_context() but also returns the similarity score for
    each chunk.  Useful for debugging and for the Streamlit debug panel.

    Returns
    -------
    List of (chunk_text, similarity_score) tuples, ordered by descending score.
    """
    vector_store = _load_vector_store()

    results = vector_store.similarity_search_with_score(query, k=k)
    return [(doc.page_content, float(score)) for doc, score in results]
