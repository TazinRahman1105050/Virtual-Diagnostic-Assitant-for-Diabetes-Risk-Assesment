import os
import faiss
import pickle
import PyPDF2
from tqdm import tqdm
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
import numpy as np

# --- Configuration ---
DOCS_DIR = "data/documents"
INDEX_DIR = "data/index"
CHUNK_SIZE = 500  # words

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def ingest_documents(doc_folder):
    chunks = []
    metadata = []
    for file in tqdm(Path(doc_folder).rglob("*")):
        if file.suffix.lower() == ".txt":
            content = load_txt(file)
        elif file.suffix.lower() == ".pdf":
            content = load_pdf(file)
        else:
            continue

        doc_chunks = chunk_text(content)
        chunks.extend(doc_chunks)
        metadata.extend([{"source": str(file), "chunk_id": i} for i in range(len(doc_chunks))])
    return chunks, metadata

def index_documents(chunks, metadata, index_dir=INDEX_DIR):
    print("Loading OpenAI Embeddings...")
    embedding_model = OpenAIEmbeddings()

    print("Generating embeddings...")
    embeddings = []
    for chunk in tqdm(chunks):
        emb = embedding_model.embed_query(chunk)  # or embed_documents([chunk]) if you prefer batch
        embeddings.append(emb)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "docs.faiss"))

    with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
        pickle.dump(chunks, f)  # Save chunks (not metadata) so retrieval can get text

    print(f"Indexing complete. {len(chunks)} chunks indexed.")

if __name__ == "__main__":
    print("Ingesting and indexing documents...")
    texts, metadata = ingest_documents(DOCS_DIR)
    index_documents(texts, metadata)

