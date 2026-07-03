"""
ingest.py
=========
One-time script that builds the Chroma vector store from the local
knowledge-base text files in data/sample_kb/.

Run once before starting the Streamlit app:
    python src/ingest.py

What it does
------------
1. Walks data/sample_kb/ and reads every .txt file.
2. Splits each document into overlapping chunks with LangChain's
   RecursiveCharacterTextSplitter (chunk_size=500, overlap=50).
3. Embeds each chunk using IBM Slate (WatsonxEmbeddings from watsonx_client).
4. Persists the resulting Chroma vector store to ./db/.

Re-running the script drops and recreates the ./db/ directory so the index
always reflects the current state of the knowledge base.
"""

import os
import sys
import shutil
import pathlib

# ── Ensure project root is on sys.path so `src.*` imports work whether
# this file is run as `python src/ingest.py` OR as a module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# LangChain text splitter — splits on paragraph, then sentence, then word
# boundaries in order so chunks never break mid-word.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain document schema
from langchain_core.documents import Document

# ChromaDB via LangChain integration — persists to disk automatically.
from langchain_chroma import Chroma

# Our centralised watsonx client helpers
from src.watsonx_client import get_embedding_model

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve paths relative to the project root (one level above src/).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "sample_kb"
DB_DIR = str(PROJECT_ROOT / "db")

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE = 500      # characters per chunk (not tokens)
CHUNK_OVERLAP = 50    # overlap between consecutive chunks to preserve context


def load_documents(kb_dir: pathlib.Path) -> list[Document]:
    """
    Read all .txt files from kb_dir and return them as LangChain Documents.

    Each Document carries the raw text and metadata (source filename and
    role/category derived from the filename stem).
    """
    documents: list[Document] = []

    txt_files = sorted(kb_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {kb_dir}. "
            "Make sure data/sample_kb/ contains the knowledge-base files."
        )

    for path in txt_files:
        print(f"  Loading: {path.name}")
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    # Use the filename stem as a category tag for filtering.
                    "category": path.stem,
                },
            )
        )

    print(f"  → Loaded {len(documents)} document(s) from {kb_dir}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    RecursiveCharacterTextSplitter tries to split on paragraph boundaries
    first (\n\n), then line breaks (\n), then spaces, then individual
    characters — so chunks respect natural text structure whenever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Keep separators as part of the preceding chunk for readability.
        keep_separator=True,
        # Measure length in characters (not tokens) for simplicity.
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"  → Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
    """
    Embed chunks and persist to Chroma on disk at DB_DIR.

    Chroma.from_documents() calls embedding_function.embed_documents() on
    each chunk's page_content, then stores the resulting vectors alongside
    the text and metadata in the SQLite-backed Chroma database.

    This uses the IBM Slate model (ibm/slate-125m-english-rtrvr) via the
    WatsonxEmbeddings wrapper — each embed call hits the watsonx.ai API.
    """
    print("  Initialising IBM Slate embedding model …")
    embedding_model = get_embedding_model()

    # If a previous DB exists, remove it so we start fresh.
    if os.path.exists(DB_DIR):
        print(f"  Removing existing vector store at {DB_DIR} …")
        shutil.rmtree(DB_DIR)

    print(f"  Embedding {len(chunks)} chunks and persisting to {DB_DIR} …")
    print("  (This may take a minute — each chunk requires an API call to watsonx.ai)")

    # Chroma.from_documents():
    # - Calls WatsonxEmbeddings.embed_documents() to get vectors.
    # - Stores text + vectors + metadata in a Chroma SQLite DB at persist_directory.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR,
        collection_name="interview_kb",
    )

    print(f"  ✓ Vector store persisted to {DB_DIR}")
    return vector_store


def main() -> None:
    print("=" * 60)
    print("Interview Trainer Agent — Knowledge Base Ingestion")
    print("=" * 60)

    print("\n[1/3] Loading documents …")
    documents = load_documents(KB_DIR)

    print("\n[2/3] Chunking documents …")
    chunks = chunk_documents(documents)

    print("\n[3/3] Building vector store …")
    build_vector_store(chunks)

    print("\n✓ Ingestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
