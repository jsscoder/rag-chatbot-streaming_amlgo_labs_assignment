import os
import pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# ---------- 1. LOAD PDF ----------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text


# ---------- 2. CLEAN TEXT ----------
def clean_text(text):
    return text.replace("\n", " ").strip()


# ---------- 3. CHUNKING ----------
def chunk_text(text, chunk_size=150):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


# ---------- 4. EMBEDDINGS ----------
def create_embeddings(chunks):
    model = SentenceTransformer("BAAI/bge-small-en")
    embeddings = model.encode(chunks)
    return model, embeddings


# ---------- 5. FAISS INDEX ----------
def create_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index


# ---------- MAIN PIPELINE ----------
def main():
    pdf_path =r"C:\Users\omkar\Desktop\bite-speed-assignment\java\rag-pipeline\data\training_docs.pdf" 


    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Cleaning text...")
    text = clean_text(text)

    print("Chunking...")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings...")
    model, embeddings = create_embeddings(chunks)

    print("Creating FAISS index...")
    index = create_faiss(embeddings)

    # ---------- SAVE ----------
    print("Saving data...")

    os.makedirs("vectordb", exist_ok=True)
    os.makedirs("chunks", exist_ok=True)

    faiss.write_index(index, "vectordb/index.faiss")

    with open("chunks/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Done ✅")


if __name__ == "__main__":
    main()