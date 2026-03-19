import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------- LOAD EVERYTHING ----------
def load_all():
    index = faiss.read_index("vectordb/index.faiss")

    with open("chunks/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embed_model = SentenceTransformer("BAAI/bge-small-en")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return index, chunks, embed_model, tokenizer, model


# ---------- RETRIEVE WITH THRESHOLD ----------
def retrieve(query, index, chunks, model, k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)

    # 🔥 Threshold check (tune if needed)
    if D[0][0] > 1.5:
        return []

    return [chunks[i] for i in I[0]]


# ---------- GENERATE ----------
def generate_answer(query, index, chunks, embed_model, tokenizer, model):
    context_chunks = retrieve(query, index, chunks, embed_model)

    if len(context_chunks) == 0:
        return "Not found in document.", []

    context = context_chunks[0]

    # 🔥 KEYWORD FILTER (important)
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())

    if len(query_words.intersection(context_words)) < 1:
        return "Not found in document.", []

    prompt = f"""
Extract the answer ONLY from the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    answer = answer.replace("•", "").strip()

    if answer == "":
        answer = "Not found in document."

    return answer, context_chunks

# ---------- TEST LOOP ----------
if __name__ == "__main__":
    index, chunks, embed_model, tokenizer, model = load_all()

    while True:
        query = input("\nAsk: ")

        answer, sources = generate_answer(query, index, chunks, embed_model, tokenizer, model)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:\n")
        if len(sources) == 0:
            print("No relevant sources found.\n")
        else:
            for i, s in enumerate(sources):
                print(f"{i+1}. {s[:200]}...\n")