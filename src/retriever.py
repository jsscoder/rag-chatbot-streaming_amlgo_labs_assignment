import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------- LOAD EVERYTHING ----------
def load_data():
    index = faiss.read_index("vectordb/index.faiss")

    with open("chunks/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer("BAAI/bge-small-en")

    return index, chunks, model


# ---------- RETRIEVE ----------
def retrieve(query, index, chunks, model, k=5):
    query_embedding = model.encode([query])

    D, I = index.search(np.array(query_embedding), k)

    results = [chunks[i] for i in I[0]]
    return results


# ---------- TEST ----------
if __name__ == "__main__":
    index, chunks, model = load_data()

    while True:
        query = input("\nAsk something: ")

        results = retrieve(query, index, chunks, model)

        print("\nTop Results:\n")
        for i, r in enumerate(results):
            print(f"{i+1}. {r[:200]}...\n")