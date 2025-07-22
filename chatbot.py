import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -----------------------------
# Config
# -----------------------------
CHROMA_DIR = "data/vectorstore/v2_chroma"
CHROMA_COLLECTION = "gitlab_v2"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # retrieved chunks


# -----------------------------
# Gemini API setup
# -----------------------------
def configure_gemini():
    key = st.session_state.get("gemini_key") or os.getenv("GEMINI_API_KEY")
    if key:
        genai.configure(api_key=key)
        return key
    return None


# Sidebar API key input (optional override)
with st.sidebar:
    st.markdown("### API Key")
    st.text_input(
        "Gemini API key (overrides env)",
        value=os.getenv("GEMINI_API_KEY") or "",
        key="gemini_key",
        type="password",
        help="Get one free at makersuite.google.com",
    )
    if st.button("Use API Key"):
        st.success("API key set for this session.")


API_KEY_ACTIVE = configure_gemini()


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_chroma(path: str, collection_name: str):
    client = chromadb.PersistentClient(path=path)
    try:
        return client.get_collection(collection_name)
    except Exception:
        st.error(f"Collection '{collection_name}' not found in {path}. Did you run embedder.py?")
        st.stop()

embed_model = load_embed_model()
collection = load_chroma(CHROMA_DIR, CHROMA_COLLECTION)


# -----------------------------
# Retrieval & LLM helpers
# -----------------------------
def embed_query(query: str):
    return embed_model.encode([query], convert_to_numpy=True)[0].tolist()

def retrieve_context(query: str, top_k: int = TOP_K):
    q_emb = embed_query(query)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))

def build_prompt(query: str, context_chunks: list):
    context_blocks = []
    for chunk_text, meta in context_chunks:
        src = meta.get("url", "unknown")
        context_blocks.append(f"Source: {src}\n{chunk_text}")
    context_text = "\n\n---\n\n".join(context_blocks)

    return f"""You are a helpful assistant answering questions strictly from the GitLab Handbook & Direction documentation.

Use ONLY the information in Context below. If the answer is not in the context, say you do not know.

Return a clear, concise answer. When helpful, summarize and mention key GitLab concepts.

Context:
{context_text}

Question: {query}

Answer:"""

def call_gemini(prompt: str) -> str:
    if not API_KEY_ACTIVE:
        return "Error: Gemini API key not set. Please enter a key in the sidebar."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"[LLM Error] {e}"


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="GitLab Handbook Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 GitLab Handbook Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role: "user"/"assistant", "content": str, "sources": [...]}

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

# Chat input (auto-clears after send)
user_query = st.chat_input("Ask a question about the GitLab Handbook...")

if user_query:
    # 1. Echo user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Retrieve context
    context_chunks = retrieve_context(user_query, top_k=TOP_K)
    source_urls = [meta.get("url", "") for (_, meta) in context_chunks]

    # 3. Build prompt & call Gemini
    prompt = build_prompt(user_query, context_chunks)
    answer = call_gemini(prompt)

    # 4. Append bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_urls,
    })

    # 5. Force rerun to render new messages immediately
    st.rerun()
