import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------- LOAD MODELS ----------
@st.cache_resource
def load_all():
    index = faiss.read_index("vectordb/index.faiss")

    with open("chunks/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embed_model = SentenceTransformer("BAAI/bge-small-en")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return index, chunks, embed_model, tokenizer, model


# ---------- RETRIEVE ----------
def retrieve(query, index, chunks, model, k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)

    if D[0][0] > 0.6:
        return []

    return [chunks[i] for i in I[0]]


# ---------- GENERATE ----------
def generate_answer(query, index, chunks, embed_model, tokenizer, model):
    context_chunks = retrieve(query, index, chunks, embed_model)

    if len(context_chunks) == 0:
        return "Not found in document.", []

    context = context_chunks[0]

    # 🔥 KEYWORD FILTER (THIS WAS MISSING)
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())

    overlap = query_words.intersection(context_words)

    if len(overlap) < 1:
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


# ---------- UI CONFIG ----------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# ---------- LOAD ----------
index, chunks, embed_model, tokenizer, model = load_all()

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("💬 Chat History")

    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {chat['query'][:40]}...")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.write("### ⚙️ Info")
    st.write("Model: FLAN-T5")
    st.write(f"Chunks: {len(chunks)}")


# ---------- MAIN ----------
st.title("📄 RAG Chatbot")

query = st.chat_input("Ask something...")

if query:
    # Generate answer
    answer, sources = generate_answer(
        query, index, chunks, embed_model, tokenizer, model
    )

    # Save history
    st.session_state.chat_history.append({
        "query": query,
        "answer": answer,
        "sources": sources
    })

# ---------- DISPLAY CHAT ----------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["query"])

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        # 🔥 streaming effect
        for word in chat["answer"].split():
            full_response += word + " "
            response_container.markdown(full_response)

        # Sources
        if chat["sources"]:
            with st.expander("📚 Sources"):
                for s in chat["sources"]:
                    st.write("- " + s[:200])