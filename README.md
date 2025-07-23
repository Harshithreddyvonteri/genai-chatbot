# 🤖 GitLab Handbook Chatbot

This project is an **AI-powered chatbot** built to answer questions about the **GitLab Handbook**.  
It combines **semantic search** with **LLM reasoning** to provide precise, context-aware answers.

The chatbot leverages:
- **FAISS** for semantic vector search.
- **Sentence Transformers** for text embeddings.
- **Google Gemini 1.5 Flash** (free tier) for reasoning and fallback responses.
- **Streamlit** for a modern chat UI with a dark theme.

---

## ✨ Features
- **Handbook Q&A:** Retrieves and uses relevant handbook sections for answers.
- **Smart Fallback:** When context is insufficient, Gemini provides reasoning-based answers.
- **Optimized Search:** Uses FAISS for fast semantic lookups.
- **ChatGPT-like UI:** Dark mode, chat bubbles, and intuitive interaction.
- **Streamlit Cloud Ready:** Fully deployable with no server setup.

---

## ⚙️ Tech Stack
- **Language:** Python 3.13
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS (`v2_faiss.pkl`)
- **LLM:** Google Gemini 1.5 Flash
- **Frontend:** Streamlit with custom CSS for chat-like bubbles
- **Hosting:** Streamlit Cloud (Free tier)

---

## 🚀 Getting Started

### **1. Clone the Repository**
```bash
git clone <repo-url>
cd genai-chatbot
```

### **2. Create and Activate Virtual Environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Chatbot**
```bash
streamlit run chatbot.py
```

### **5. Set the Gemini API Key**
```bash
setx GEMINI_API_KEY "your-gemini-key"   # Windows
export GEMINI_API_KEY="your-gemini-key" # Linux/Mac
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this project to a **GitHub repository**.
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and click **New App**.
3. Select your repository and branch.
4. Add **Secrets**:
   ```
   GEMINI_API_KEY = your_gemini_api_key
   ```
5. Click **Deploy**.

---

## 📂 Project Structure
```
genai-chatbot/
│
├── chatbot.py             # Main Streamlit chatbot
├── convert_to_faiss.py    # Converts ChromaDB → FAISS
├── data/
│   └── vectorstore/       # FAISS index (v2_faiss.pkl)
├── requirements.txt
├── README.md
└── DESIGN_DECISIONS.md
```

---

## 🔍 How It Works (Overview)

```
User Question
      │
      ▼
[Sentence Transformer] –> Convert query to vector
      │
      ▼
[FAISS Index] –> Retrieve top relevant handbook chunks
      │
      ▼
[Gemini 1.5 Flash]
   │   └─> Uses retrieved context for Q&A
   │
   └─> Fallback: If context is weak, Gemini answers with reasoning
      │
      ▼
Answer Displayed in Streamlit Chat UI
```

---

## 🧩 Future Improvements
- Add a **scrollable chat area** with pinned input bar.
- Merge **v2 + curated v3** dataset for broader coverage.
- Implement **streaming responses** like ChatGPT.
- Export or save chat history.

---

## 📝 Acknowledgments
- **GitLab Handbook** (Open documentation).
- **Google Gemini API** for LLM reasoning.
- **Streamlit** for quick web app prototyping.
