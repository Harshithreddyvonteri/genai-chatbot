import streamlit as st
import google.generativeai as genai
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# ================================
# CONFIGURATION
# ================================
FAISS_PATH = "data/vectorstore/v2_faiss.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ================================
# LOAD EMBEDDINGS + MODEL
# ================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()

@st.cache_resource
def load_faiss():
    with open(FAISS_PATH, "rb") as f:
        faiss_data = pickle.load(f)
    return faiss_data["index"], faiss_data["documents"], faiss_data["metadata"]

index, documents, metadata = load_faiss()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ================================
# HELPER FUNCTIONS
# ================================
def embed_fn(text):
    return embedder.encode([text])[0]

def retrieve_context(query, k=3):
    query_emb = np.array(embed_fn(query), dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_emb, k)
    results = []
    for idx in I[0]:
        results.append((documents[idx], metadata[idx]))
    return results

def context_strength(chunks, min_words=50):
    total_words = sum(len(c[0].split()) for c in chunks)
    return total_words >= min_words

def generate_answer(query: str, context_chunks: list):
    model = genai.GenerativeModel("gemini-1.5-flash")
    if not context_chunks or not context_strength(context_chunks):
        prompt = f"""
        You are an AI assistant for GitLab.
        The retrieved context was weak or insufficient.
        Use your general knowledge to answer clearly and concisely:
        Question: {query}
        """
        response = model.generate_content(prompt)
        return response.text, ["General knowledge (no handbook context)"]

    context_text = "\n\n".join([c[0] for c in context_chunks])
    prompt = f"""
    You are a helpful assistant answering questions about GitLab.

    Context:
    {context_text}

    Question: {query}

    Use the context to answer. If something is missing, add reasoning from general GitLab knowledge.
    Answer clearly and concisely.
    """
    response = model.generate_content(prompt)
    source_urls = [meta.get("url", "unknown") for (_, meta) in context_chunks]
    return response.text, source_urls

# ================================
# STREAMLIT APP
# ================================
st.set_page_config(page_title="GitLab Handbook Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 GitLab Handbook Chatbot")

# Dark mode chat bubble style
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #EDEDED;
    }
    .user-msg {
        background-color: #2F2F2F;
        color: white;
        padding: 10px 15px;
        border-radius: 12px;
        margin: 5px 0;
        display: inline-block;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.4);
    }
    .bot-msg {
        background-color: #1E1E1E;
        color: #EDEDED;
        padding: 10px 15px;
        border-radius: 12px;
        margin: 5px 0;
        display: inline-block;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.4);
    }
    </style>
    """, unsafe_allow_html=True
)


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to handle input
def handle_input():
    user_text = st.session_state.user_input
    if user_text.strip():
        with st.spinner("Thinking..."):
            context = retrieve_context(user_text)
            answer, sources = generate_answer(user_text, context)
            st.session_state.chat_history.append(("user", user_text))
            st.session_state.chat_history.append(("bot", f"{answer}\n\n**Sources:** {', '.join(sources)}"))
    st.session_state.user_input = ""  # Clear input

# Display chat bubbles
for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='user-msg'>🧑 {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {text}</div>", unsafe_allow_html=True)

# Input field
st.text_input(
    "Ask a question about GitLab:",
    key="user_input",
    on_change=handle_input,
    placeholder="Type your question here and press Enter...",
)
