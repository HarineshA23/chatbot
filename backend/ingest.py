import os
import faiss
import numpy as np
import pickle
import re
import fitz
from sentence_transformers import SentenceTransformer
from docx import Document

# ---------------- CONFIG ---------------- #

DOCUMENT_DIR = "data/documents"
INDEX_DIR = "data/index_data"

FAISS_INDEX_PATH = f"{INDEX_DIR}/faiss.index"
META_PATH = f"{INDEX_DIR}/meta.pkl"

EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

index = None
meta = []

os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------------------- #


def load_index():
    global index, meta
    dim = EMB_MODEL.get_sentence_embedding_dimension()

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(dim)
        meta = []


def save_index():
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)


def read_file(filepath):
    if filepath.lower().endswith(".pdf"):
        doc = fitz.open(filepath)
        pages = [p.get_text("text") for p in doc]
        return "\n".join(pages)

    if filepath.lower().endswith(".docx"):
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)

    return open(filepath, "r", errors="ignore").read()


# ✅ SECTION-AWARE CHUNKING (CRITICAL FIX)
def chunk_text(text):
    text = re.sub(r"\s+", " ", text).strip()

    sections = re.split(r"\n\s*\d+\.\s+", text)
    chunks = []

    for sec in sections:
        sec = sec.strip()
        if len(sec) < 80:
            continue

        if len(sec) <= CHUNK_SIZE:
            chunks.append(sec)
        else:
            i = 0
            while i < len(sec):
                chunks.append(sec[i:i + CHUNK_SIZE])
                i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest_all():
    load_index()

    files = os.listdir(DOCUMENT_DIR)
    if not files:
        print("❌ No documents found")
        return

    for fname in files:
        fpath = os.path.join(DOCUMENT_DIR, fname)

        text = read_file(fpath)
        chunks = chunk_text(text)

        embeddings = EMB_MODEL.encode(chunks, batch_size=16)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

        index.add(embeddings.astype("float32"))

        for ch in chunks:
            meta.append({
                "file": fname,
                "chunk": ch
            })

    save_index()
    print(f"✅ Ingested {len(meta)} chunks")


def retrieve(question, k=5):
    load_index()

    q_emb = EMB_MODEL.encode([question])
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    _, I = index.search(q_emb.astype("float32"), k)

    return [meta[i] for i in I[0] if i < len(meta)]
