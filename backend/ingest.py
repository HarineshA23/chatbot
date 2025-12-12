import os
import faiss
import numpy as np
import pickle
import re

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

# ----------- CONFIGURATION ---------------- #

DOCUMENT_DIR = "data/documents"      # <-- your local docs folder
INDEX_DIR = "data/index_data"        # <-- FAISS + metadata saved here

os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_INDEX_PATH = f"{INDEX_DIR}/faiss.index"
META_PATH = f"{INDEX_DIR}/meta.pkl"

# Embedding model
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# FAISS + metadata in memory
index = None
meta = []
# ------------------------------------------ #


# ---------- Utility Functions ------------- #

def load_index():
    """Load FAISS index and metadata, or initialize new index."""
    global index, meta

    dim = EMB_MODEL.get_sentence_embedding_dimension()

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
        meta = []


def save_index():
    """Persist FAISS index + metadata to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)


def read_file(filepath):
    """Extract text from PDF, DOCX or TXT."""
    if filepath.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if filepath.lower().endswith(".docx"):
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)

    # txt or other text files
    return open(filepath, "r", errors="ignore").read()


def chunk_text(text):
    """Split long documents into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []

    i = 0
    while i < len(text):
        chunk = text[i:i + CHUNK_SIZE]
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# -------------- Ingestion Function ------------ #

def ingest_all():
    """
    Reads all files in DOCUMENT_DIR,
    produces embeddings, stores them in FAISS.
    """
    print("Loading vector index...")
    load_index()

    files = os.listdir(DOCUMENT_DIR)
    if not files:
        print("❌ No files found in data/documents/")
        return 0

    print(f"Found {len(files)} files to ingest...")

    for fname in files:
        fpath = os.path.join(DOCUMENT_DIR, fname)

        print(f"\n📄 Processing: {fname}")

        text = read_file(fpath)
        chunks = chunk_text(text)

        print(f" → {len(chunks)} chunks extracted")

        # Create embeddings
        embeddings = EMB_MODEL.encode(chunks, batch_size=16)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

        # Add to FAISS
        index.add(embeddings.astype("float32"))

        # Save metadata
        for ch in chunks:
            meta.append({
                "file": fname,
                "chunk": ch
            })

    # Persist index
    save_index()

    print("\n✅ Ingestion completed!")
    print(f"📦 Total chunks stored: {len(meta)}")

    return len(meta)


# -------------- Retrieval Function ------------- #

def retrieve(question, k=5):
    """Retrieve top-K relevant document chunks for the query."""
    load_index()

    # Embed query
    q_emb = EMB_MODEL.encode([question])
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # Search FAISS
    D, I = index.search(q_emb.astype("float32"), k)

    results = [meta[i] for i in I[0] if i < len(meta)]
    return results
