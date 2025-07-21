import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --------------------------------
# Configuration
# --------------------------------
CHROMA_DIR = "data/vectorstore/v2_chroma"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # Number of chunks to retrieve

# Load Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not set. Run: setx GEMINI_API_KEY 'your_key'")
else:
    genai.configure(api_key=GEMINI_API_KEY)


# --------------------------------
# Initialize Models
# --------------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

embed_model = load_embed_model()

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("gitlab_v2")

collection = load_chroma()


# --------------------------------
# Helper Functions
# --------------------------------
def embed_query(query: str):
    """Convert query to embedding."""
    return embed_model.encode([query], convert_to_numpy=True)[0].tolist()

def retrieve_context(query: str, top_k: int = TOP_K):
    """Retrieve top-k relevant chunks."""
    q_emb = embed_query(query)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append((doc, meta))
    return chunks

def generate_answer(query: str, context_chunks: list):
    """Call Gemini with query + context."""
    context_text = "\n\n".join([c[0] for c in context_chunks])
    prompt = f"""
You are a helpful assistant answering questions based on the GitLab Handbook.

Context:
{context_text}

Question: {query}

Answer in a concise and clear way, referring only to the context above.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


# --------------------------------
# Streamlit UI
# --------------------------------
st.set_page_config(page_title="GitLab Handbook Chatbot", page_icon="🤖")
st.title("🤖 GitLab Handbook Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the GitLab Handbook:", "")

if st.button("Ask") and user_input.strip():
    # Retrieve context
    context = retrieve_context(user_input)

    # Generate answer
    answer = generate_answer(user_input, context)

    # Save in chat history
    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("Bot", answer))

# Display chat history
for speaker, text in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

